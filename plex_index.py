import json
import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from itertools import cycle, islice
from typing import Any, Literal, NewType, TypedDict, cast

import click
import polars as pl
from tqdm import tqdm

logger = logging.getLogger("plex-index")

GUID_TYPE = Literal["episode", "movie", "season", "show"]
GUID_TYPES: set[GUID_TYPE] = {"episode", "movie", "season", "show"}
GUID_RE = r"plex://(?P<type>episode|movie|season|show)/(?P<key>[a-f0-9]{24})"

# 12 byte or 25c hex encoded media rating key
RatingKey = NewType("RatingKey", bytes)
TypedRatingKey = NewType("TypedRatingKey", tuple[GUID_TYPE, RatingKey])

DF_SCHEMA = pl.Schema(
    {
        "key": pl.Binary(),
        "type": pl.Categorical(),
        "success": pl.Boolean(),
        "retrieved_at": pl.Datetime(time_unit="ns"),
        "year": pl.UInt16(),
        "imdb_numeric_id": pl.UInt32(),
        "tmdb_id": pl.UInt32(),
        "tvdb_id": pl.UInt32(),
    }
)


def update_or_append(df: pl.DataFrame, other: pl.DataFrame) -> pl.DataFrame:
    output_schema = pl.Schema()
    for name in df.schema.names():
        output_schema[name] = df.schema[name]
    for name in other.schema.names():
        if name in output_schema:
            assert other.schema[name] == output_schema[name]
            continue
        output_schema[name] = other.schema[name]
    logger.debug(
        "update_or_append(df=%s, other=%s): output schema=%s",
        df.schema,
        other.schema,
        output_schema,
    )

    assert "key" in output_schema.names(), "output schema must have key column"

    if df.is_empty():
        return other.match_to_schema(output_schema, missing_columns="insert")

    df = df.match_to_schema(output_schema, missing_columns="insert")

    other = other.join(
        df.drop(set(other.columns) - {"key"}),
        on="key",
        how="left",
        coalesce=True,
    ).select(output_schema.names())

    return pl.concat([df, other]).unique(subset="key", keep="last", maintain_order=True)


def change_summary(df_old: pl.DataFrame, df_new: pl.DataFrame) -> tuple[int, int, int]:
    # Simplify complex schema in to two columns to compare
    df_old = pl.DataFrame({"key": df_old["key"], "hash": df_old.hash_rows()})
    df_new = pl.DataFrame({"key": df_new["key"], "hash": df_new.hash_rows()})
    df_com = df_old.join(df_new, on="key", how="inner", coalesce=True)

    new_len = len(df_new["key"])
    old_len = len(df_old["key"])

    if new_len > old_len:
        added = new_len - old_len
        removed = 0
    else:
        added = 0
        removed = old_len - new_len

    updated_s = df_com["hash"] != df_com["hash_right"]
    updated = int(updated_s.sum())

    return added, removed, updated


def compute_stats(df_old: pl.DataFrame, df_new: pl.DataFrame) -> pl.DataFrame:
    df = df_new
    row_count = df.height

    def fmt(n: int) -> str:
        if n == 0 or row_count == 0:
            return ""
        return f"{n:,} ({n / row_count:.1%})"

    rows = []
    for name, dtype in df.schema.items():
        updated = 0
        if name != "key":
            updated = change_summary(
                df_old.select("key", name),
                df_new.select("key", name),
            )[2]
        s = df[name]
        s_wo_null = s.drop_nulls()
        nulls = s.null_count()
        trues = int(s.sum()) if dtype == pl.Boolean else 0
        falses = int((~s).sum()) if dtype == pl.Boolean else 0
        unique = s_wo_null.n_unique() == s_wo_null.len()

        rows.append(
            {
                "name": name,
                "dtype": df.schema[name]._string_repr(),
                "null": fmt(nulls),
                "true": fmt(trues) if dtype == pl.Boolean else "",
                "false": fmt(falses) if dtype == pl.Boolean else "",
                "unique": "true" if unique else "",
                "updated": fmt(updated),
            }
        )

    return pl.DataFrame(rows)


def format_gh_step_summary(
    df_old: pl.DataFrame,
    df_new: pl.DataFrame,
    filename: str,
) -> str:
    df_stats = compute_stats(df_new=df_new, df_old=df_old)
    added, removed, updated = change_summary(df_old=df_old, df_new=df_new)

    with pl.Config() as cfg:
        cfg.set_fmt_str_lengths(100)
        cfg.set_tbl_cols(-1)
        cfg.set_tbl_column_data_type_inline(True)
        cfg.set_tbl_formatting("ASCII_MARKDOWN")
        cfg.set_tbl_hide_dataframe_shape(True)
        cfg.set_tbl_rows(-1)
        cfg.set_tbl_width_chars(500)

        buf = StringIO()
        print(f"## {filename}", file=buf)
        print("", file=buf)
        print(df_stats, file=buf)
        print("", file=buf)
        print(f"shape: ({df_new.shape[0]:,}, {df_new.shape[1]:,})", file=buf)
        print(f"changes: +{added:,} -{removed:,} ~{updated:,}", file=buf)
        print(f"rss: {df_new.estimated_size('mb'):,.1f}MB", file=buf)

        return buf.getvalue()


@dataclass
class PlexDevice:
    name: str
    public_address: str
    uri: str
    access_token: str


def plex_device_info(device_name: str, token: str) -> PlexDevice | None:
    headers = {"X-Plex-Token": token}
    req = urllib.request.Request(url="https://plex.tv/api/resources", headers=headers)
    logger.info('Fetching Plex "%s" device info', device_name)
    with urllib.request.urlopen(req, timeout=10) as response:
        tree = ET.parse(response)
        devices = tree.findall("./Device")
        for device in devices:
            if device.get("name") != device_name:
                continue
            uri = ""
            access_token = device.get("accessToken", "")
            public_address = device.get("publicAddress", "")
            for connection in device.findall("./Connection[@local='0']"):
                uri = connection.get("uri", "")
            return PlexDevice(
                name=device_name,
                public_address=public_address,
                uri=uri,
                access_token=access_token,
            )
    return None


def decode_plex_guid(guid: str) -> TypedRatingKey | None:
    match = re.match(GUID_RE, guid)
    if not match:
        logger.debug("Bad Plex GUID: %s", guid)
        return None
    type_str = match.group("type")
    key = bytes.fromhex(match.group("key"))
    assert len(key) == 12
    return cast(TypedRatingKey, (type_str, key))


def rating_key(key: str) -> RatingKey:
    assert len(key) == 24, f"expected 24c, got {len(key)}"
    return RatingKey(bytes.fromhex(key))


def typed_rating_key(type: str, key: str) -> TypedRatingKey:
    assert type in GUID_TYPES, f"invalid type: {type}"
    assert len(key) == 24, f"expected 24c, got {len(key)}"
    return cast(TypedRatingKey, (type, bytes.fromhex(key)))


def plex_library_guids(server_uri: str, token: str) -> Iterator[TypedRatingKey]:
    headers = {"X-Plex-Token": token}
    for section in [1, 2]:
        url = f"{server_uri}/library/sections/{section}/all"
        req = urllib.request.Request(url=url, headers=headers)
        logger.info("Fetching Plex library section=%s", section)
        with urllib.request.urlopen(req, timeout=60) as response:
            data = response.read().decode("utf-8", errors="ignore")
            for m in re.findall(GUID_RE, data):
                yield typed_rating_key(*m)


def sparql(query: str) -> Any:
    headers = {
        "Accept": "application/json",
        "User-Agent": "PlexIndex/0.0 (https://github.com/josh/plex-index)",
    }
    data = urllib.parse.urlencode({"query": query}).encode("utf-8")
    req = urllib.request.Request(
        "https://query.wikidata.org/sparql",
        data=data,
        headers=headers,
        method="POST",
    )
    logger.info("Querying Wikidata SPARQL: %s", query)
    with urllib.request.urlopen(req, timeout=90) as response:
        return json.load(response)


def wikidata_plex_media_keys() -> Iterator[RatingKey]:
    results = sparql("SELECT DISTINCT ?guid WHERE { ?item ps:P11460 ?guid. }")
    for binding in results["results"]["bindings"]:
        value = binding["guid"]["value"]
        if m := re.match(r"[a-f0-9]{24}", value):
            yield rating_key(m.group(0))


def plex_search_guids(query: str) -> Iterator[TypedRatingKey]:
    url = "https://discover.provider.plex.tv/library/search"
    params = {
        "query": query,
        "limit": "100",
        "searchTypes": "movies,tv",
        "includeMetadata": "1",
        "searchProviders": "discover",
    }
    url += "?" + urllib.parse.urlencode(params)
    headers = {
        "Accept": "application/json",
        "X-Plex-Provider-Version": "7.2.0",
    }
    logger.debug("Searching Plex: %s", query)
    req = urllib.request.Request(url=url, headers=headers)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                data = response.read().decode("utf-8", errors="ignore")
            for m in re.findall(GUID_RE, data):
                yield typed_rating_key(*m)
            return
        except urllib.error.URLError as e:
            if attempt == 2:
                logger.warning("search failed for %s: %s", query, e)
                return
            time.sleep(2**attempt)


_MOVIE_TITLE_QUERY = """
SELECT ?title WHERE {
  SERVICE bd:sample {
    ?item wdt:P4947 _:b1.
    bd:serviceParam bd:sample.limit ?limit ;
      bd:sample.sampleType "RANDOM".
  }
  ?item wdt:P1476 ?title.
  OPTIONAL { ?item wdt:P11460 ?plex_guid. }
  FILTER(!(BOUND(?plex_guid)))
}
"""

_TV_TITLE_QUERY = """
SELECT ?title WHERE {
  SERVICE bd:sample {
    ?item wdt:P4983 _:b1.
    bd:serviceParam bd:sample.limit ?limit ;
      bd:sample.sampleType "RANDOM".
  }
  ?item wdt:P1476 ?title.
  OPTIONAL { ?item wdt:P11460 ?plex_guid. }
  FILTER(!(BOUND(?plex_guid)))
}
"""


def roundrobin[T](*iterables: Iterable[T]) -> Iterator[T]:
    iterators: Iterator[Iterator[T]] = map(iter, iterables)
    for num_active in range(len(iterables), 0, -1):
        iterators = cycle(islice(iterators, num_active))
        yield from map(next, iterators)


def wd_random_titles(
    tmdb_type: Literal["movie", "tv"] | None = None,
    sample_size: int = 100,
) -> Iterator[str]:
    if tmdb_type is None:
        yield from roundrobin(
            wd_random_titles(tmdb_type="movie"),
            wd_random_titles(tmdb_type="tv"),
        )
        return
    elif tmdb_type == "movie":
        query = _MOVIE_TITLE_QUERY
    elif tmdb_type == "tv":
        query = _TV_TITLE_QUERY
    query = query.replace("?limit", str(sample_size))

    seen: set[str] = set()
    while True:
        results = sparql(query)
        for binding in results["results"]["bindings"]:
            title = binding["title"]["value"]
            if title in seen:
                logger.debug("skipping duplicate title: %s", title)
                continue
            seen.add(title)
            yield title


def wd_random_search_guids() -> Iterator[TypedRatingKey]:
    for title in wd_random_titles():
        query = (
            title.replace("#", "").replace("&", "").replace("'", "").replace('"', "")
        )
        yield from plex_search_guids(query)


def imdb_watchlist_guids(url: str) -> Iterator[TypedRatingKey]:
    df = pl.read_csv(url, columns="Title")
    for title in df["Title"]:
        query = (
            title.replace("#", "").replace("&", "").replace("'", "").replace('"', "")
        )
        yield from plex_search_guids(query)


def discover_media_keys(
    plex_server_name: str | None,
    plex_token: str | None,
    imdb_watchlist_url: str | None,
) -> Iterator[RatingKey]:
    yield from wikidata_plex_media_keys()

    if plex_server_name and plex_token:
        if device := plex_device_info(
            device_name=plex_server_name,
            token=plex_token,
        ):
            for _, key in plex_library_guids(
                server_uri=device.uri,
                token=device.access_token,
            ):
                yield key
        else:
            logger.error("failed to fetch Plex device info")
    else:
        logger.warning("skipping Plex library discovery")

    if imdb_watchlist_url:
        for _, key in imdb_watchlist_guids(url=imdb_watchlist_url):
            yield key

    for _, key in wd_random_search_guids():
        yield key


def discover_media_keys_df(
    df: pl.DataFrame,
    plex_server_name: str | None,
    plex_token: str | None,
    imdb_watchlist_url: str | None,
    limit: int,
) -> pl.DataFrame:
    seen_keys: set[RatingKey] = set(df["key"])
    keys_iter = (
        key
        for key in discover_media_keys(
            plex_server_name=plex_server_name,
            plex_token=plex_token,
            imdb_watchlist_url=imdb_watchlist_url,
        )
        if key not in seen_keys
    )
    df_new = pl.DataFrame(
        data={"key": islice(keys_iter, limit)},
        schema={"key": pl.Binary},
    )
    logger.debug("discovered %d new keys", len(df_new))
    return update_or_append(df, df_new)


class RowDict(TypedDict):
    key: bytes | None
    type: str | None
    success: bool | None
    retrieved_at: datetime | None
    year: int | None
    imdb_numeric_id: int | None
    tmdb_id: int | None
    tvdb_id: int | None


@dataclass
class PlexMetadata:
    type: GUID_TYPE
    key: RatingKey
    year: int | None
    imdb_numeric_id: int | None
    tmdb_id: int | None
    tvdb_id: int | None


def fetch_plex_metadata(
    key: RatingKey,
    token: str,
) -> tuple[PlexMetadata | None, list[TypedRatingKey]]:
    url = f"https://metadata.provider.plex.tv/library/metadata/{key.hex()}"
    headers = {
        "Accept": "application/json",
        "X-Plex-Token": token,
    }
    req = urllib.request.Request(url, headers=headers)
    try:
        logger.debug("fetching Plex metadata for key: %s", key.hex())
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.load(response)
            metadata = data["MediaContainer"]["Metadata"][0]
            assert metadata["ratingKey"] == key.hex()
            assert metadata["type"] in GUID_TYPES

            year: int | None = metadata.get("year", None)

            imdb: int | None = None
            tmdb: int | None = None
            tvdb: int | None = None
            for guid_obj in metadata.get("Guid", []):
                scheme, id = guid_obj["id"].split("://", 1)
                if scheme == "imdb":
                    imdb = int(id[2:])
                elif scheme == "tmdb":
                    tmdb = int(id)
                elif scheme == "tvdb":
                    tvdb = int(id)

            similar_guids: list[TypedRatingKey] = []
            for obj in metadata.get("Similar", []):
                if typed_key := decode_plex_guid(obj["guid"]):
                    similar_guids.append(typed_key)

            metadata = PlexMetadata(
                key=key,
                type=metadata["type"],
                year=year,
                imdb_numeric_id=imdb,
                tmdb_id=tmdb,
                tvdb_id=tvdb,
            )
            return metadata, similar_guids
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None, []
        raise


def backfill_metadata(
    df: pl.DataFrame,
    plex_token: str,
    refresh_limit: int,
) -> pl.DataFrame:
    updated_rows: list[RowDict] = []
    new_keys: set[RatingKey] = set()

    keys = df.filter(
        (pl.col("retrieved_at").rank("ordinal") <= refresh_limit)
        | (pl.col("retrieved_at").is_null())
    )["key"]

    for key in tqdm(keys, desc="Fetching Plex metadata"):
        metadata, similar_guids = fetch_plex_metadata(key, plex_token)

        if metadata:
            updated_rows.append(
                {
                    "key": metadata.key,
                    "type": metadata.type,
                    "success": True,
                    "retrieved_at": datetime.now(),
                    "year": metadata.year,
                    "imdb_numeric_id": metadata.imdb_numeric_id,
                    "tmdb_id": metadata.tmdb_id,
                    "tvdb_id": metadata.tvdb_id,
                }
            )
        else:
            updated_rows.append(
                {
                    "key": key,
                    "type": None,
                    "success": False,
                    "retrieved_at": datetime.now(),
                    "year": None,
                    "imdb_numeric_id": None,
                    "tmdb_id": None,
                    "tvdb_id": None,
                }
            )

        for _, key in similar_guids:
            new_keys.add(key)

    updated_df = pl.from_records(updated_rows, schema=DF_SCHEMA)
    new_df = pl.DataFrame(
        data={"key": list(new_keys)},
        schema={"key": pl.Binary},
    )

    return df.pipe(update_or_append, updated_df).pipe(update_or_append, new_df)


@click.command()
@click.argument(
    "filename",
    type=click.Path(),
    required=True,
)
@click.option(
    "--plex-token",
    type=str,
    required=True,
    envvar="PLEX_TOKEN",
    help="MyPlex authentication token",
)
@click.option(
    "--plex-server-name",
    type=str,
    required=False,
    envvar="PLEX_SERVER_NAME",
    help="Plex server name",
)
@click.option(
    "--imdb-watchlist-url",
    type=str,
    required=False,
    envvar="IMDB_WATCHLIST_URL",
    help="IMDB watchlist URL",
)
@click.option(
    "--discover-limit",
    type=int,
    default=10,
    envvar="DISCOVER_LIMIT",
    help="Wikidata search limit",
)
@click.option(
    "--refresh-limit",
    type=int,
    default=10,
    envvar="REFRESH_LIMIT",
    help="Refresh oldest items limit",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Dry run",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Verbose output",
)
def main(
    filename: str,
    plex_token: str,
    plex_server_name: str,
    imdb_watchlist_url: str,
    discover_limit: int,
    refresh_limit: int,
    dry_run: bool,
    verbose: bool,
) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)

    pl.enable_string_cache()

    if os.path.exists(filename):
        df = pl.read_parquet(filename)
        logger.debug("original df: %s", df)
    else:
        df = pl.DataFrame(schema=DF_SCHEMA)
        logger.warning("original df not found, initializing empty dataframe")

    df2 = discover_media_keys_df(
        df=df,
        plex_server_name=plex_server_name,
        plex_token=plex_token,
        imdb_watchlist_url=imdb_watchlist_url,
        limit=discover_limit,
    )
    df2 = backfill_metadata(
        df2,
        plex_token,
        refresh_limit=refresh_limit,
    )
    df2 = df2.sort(by=pl.col("key").bin.encode("hex"))

    if df2.height < df.height:
        logger.error(
            "df2 height %s is smaller than df height %s",
            df2.height,
            df.height,
        )
        exit(1)

    logger.debug(df2)

    summary_text = format_gh_step_summary(df_old=df, df_new=df2, filename=filename)
    logger.debug(summary_text)

    if "GITHUB_STEP_SUMMARY" in os.environ:
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(summary_text)

    if not dry_run:
        df2.write_parquet(
            filename,
            compression="zstd",
            statistics=True,
        )
    else:
        logger.debug("dry run, skipping write")


if __name__ == "__main__":
    main()
