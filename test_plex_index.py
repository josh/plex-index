import itertools
import os
from datetime import UTC, datetime

import polars as pl
import pytest

from plex_index import (
    change_summary,
    compute_stats,
    decode_plex_guid,
    discover_media_keys_df,
    fetch_plex_metadata,
    format_gh_step_summary,
    plex_device_info,
    plex_library_guids,
    plex_search_guids,
    rating_key,
    roundrobin,
    typed_rating_key,
    update_or_append,
    wd_random_search_guids,
    wd_random_titles,
    wikidata_plex_media_keys,
)


def test_update_or_append_merges_and_updates() -> None:
    df1 = pl.DataFrame(
        {
            "key": [
                rating_key("000000000000000000000000"),
                rating_key("000000000000000000000001"),
            ],
            "value": [10, 20],
        },
        schema={"key": pl.Binary, "value": pl.Int64},
    )
    df2 = pl.DataFrame(
        {
            "key": [
                rating_key("000000000000000000000001"),
                rating_key("000000000000000000000002"),
            ],
            "value": [200, 30],
        },
        schema={"key": pl.Binary, "value": pl.Int64},
    )
    result = update_or_append(df1, df2)
    assert result.columns == ["key", "value"]
    assert result.sort("key")["key"].to_list() == [
        rating_key("000000000000000000000000"),
        rating_key("000000000000000000000001"),
        rating_key("000000000000000000000002"),
    ]
    assert result.sort("key")["value"].to_list() == [10, 200, 30]


def test_update_or_append_mismatched_columns() -> None:
    df1 = pl.DataFrame(
        {
            "key": [
                rating_key("000000000000000000000001"),
                rating_key("000000000000000000000002"),
            ],
            "a": [10, 20],
            "b": [100, 200],
        },
        schema={"key": pl.Binary, "a": pl.Int64, "b": pl.Int64},
    )
    df2 = pl.DataFrame(
        {
            "key": [
                rating_key("000000000000000000000002"),
                rating_key("000000000000000000000003"),
            ],
            "b": [222, 333],
            "c": [42, 43],
        },
        schema={"key": pl.Binary, "b": pl.Int64, "c": pl.Int64},
    )
    result = update_or_append(df1, df2)
    assert result.columns == ["key", "a", "b", "c"]
    result = result.sort("key")
    assert result["key"].to_list() == [
        rating_key("000000000000000000000001"),
        rating_key("000000000000000000000002"),
        rating_key("000000000000000000000003"),
    ]
    assert result.row(0) == (
        rating_key("000000000000000000000001"),
        10,
        100,
        None,
    )
    assert result.row(1) == (
        rating_key("000000000000000000000002"),
        20,
        222,
        42,
    )
    assert result.row(2) == (
        rating_key("000000000000000000000003"),
        None,
        333,
        43,
    )


def test_update_or_append_empty_df() -> None:
    empty_df = pl.DataFrame()
    df2 = pl.DataFrame(
        {
            "key": [
                rating_key("000000000000000000000000"),
                rating_key("000000000000000000000001"),
            ],
            "value": [10, 20],
        },
        schema={"key": pl.Binary, "value": pl.Int64},
    )
    result = update_or_append(empty_df, df2)
    assert result.columns == ["key", "value"]
    assert result["key"].to_list() == [
        rating_key("000000000000000000000000"),
        rating_key("000000000000000000000001"),
    ]
    assert result["value"].to_list() == [10, 20]


def test_change_summary_added() -> None:
    df1 = pl.DataFrame(
        {
            "key": [
                rating_key("000000000000000000000000"),
                rating_key("000000000000000000000001"),
                rating_key("000000000000000000000002"),
            ],
            "value": [10, 20, 30],
        },
        schema={"key": pl.Binary, "value": pl.Int64},
    )
    df2 = pl.DataFrame(
        {
            "key": [
                rating_key("000000000000000000000000"),
                rating_key("000000000000000000000001"),
                rating_key("000000000000000000000002"),
                rating_key("000000000000000000000003"),
            ],
            "value": [10, 20, 30, 40],
        },
        schema={"key": pl.Binary, "value": pl.Int64},
    )
    added, removed, updated = change_summary(df1, df2)
    assert added == 1
    assert removed == 0
    assert updated == 0


def test_change_summary_removed() -> None:
    df1 = pl.DataFrame(
        {
            "key": [
                rating_key("000000000000000000000000"),
                rating_key("000000000000000000000001"),
                rating_key("000000000000000000000002"),
                rating_key("000000000000000000000003"),
            ],
            "value": [10, 20, 30, 40],
        },
        schema={"key": pl.Binary, "value": pl.Int64},
    )
    df2 = pl.DataFrame(
        {
            "key": [
                rating_key("000000000000000000000000"),
                rating_key("000000000000000000000001"),
                rating_key("000000000000000000000002"),
            ],
            "value": [10, 20, 30],
        },
        schema={"key": pl.Binary, "value": pl.Int64},
    )
    added, removed, updated = change_summary(df1, df2)
    assert added == 0
    assert removed == 1
    assert updated == 0


def test_change_summary_updated() -> None:
    df1 = pl.DataFrame(
        {
            "key": [
                rating_key("000000000000000000000000"),
                rating_key("000000000000000000000001"),
                rating_key("000000000000000000000002"),
            ],
            "value": [10, 20, 30],
        },
        schema={"key": pl.Binary, "value": pl.Int64},
    )
    df2 = pl.DataFrame(
        {
            "key": [
                rating_key("000000000000000000000000"),
                rating_key("000000000000000000000001"),
                rating_key("000000000000000000000002"),
            ],
            "value": [100, 200, 30],
        },
        schema={"key": pl.Binary, "value": pl.Int64},
    )
    added, removed, updated = change_summary(df1, df2)
    assert added == 0
    assert removed == 0
    assert updated == 2


def test_change_summary_added_updated() -> None:
    df1 = pl.DataFrame(
        {
            "key": [
                rating_key("000000000000000000000000"),
                rating_key("000000000000000000000002"),
            ],
            "value": [10, 30],
        },
        schema={"key": pl.Binary, "value": pl.Int64},
    )
    df2 = pl.DataFrame(
        {
            "key": [
                rating_key("000000000000000000000000"),
                rating_key("000000000000000000000001"),
                rating_key("000000000000000000000002"),
            ],
            "value": [100, 200, 30],
        },
        schema={"key": pl.Binary, "value": pl.Int64},
    )
    added, removed, updated = change_summary(df1, df2)
    assert added == 1
    assert removed == 0
    assert updated == 1


def test_change_summary_noop() -> None:
    df = pl.DataFrame(
        {"key": [rating_key("000000000000000000000000")], "value": [10]},
        schema={"key": pl.Binary, "value": pl.Int64},
    )
    added, removed, updated = change_summary(df, df)
    assert added == 0
    assert removed == 0
    assert updated == 0


def test_compute_stats() -> None:
    df_old = pl.DataFrame(
        [
            {
                "key": rating_key("000000000000000000000001"),
                "type": "movie",
                "success": True,
                "retrieved_at": datetime(2024, 1, 1, tzinfo=UTC),
                "year": 2024,
                "imdb_numeric_id": 111,
                "tmdb_id": 111,
                "tvdb_id": 111,
            },
            {
                "key": rating_key("000000000000000000000002"),
                "type": "movie",
                "success": True,
                "retrieved_at": None,
                "year": 2024,
                "imdb_numeric_id": 111,
                "tmdb_id": 111,
                "tvdb_id": 111,
            },
            {
                "key": rating_key("000000000000000000000003"),
                "type": "movie",
                "success": False,
                "retrieved_at": datetime(2024, 1, 2, tzinfo=UTC),
                "year": 2024,
                "imdb_numeric_id": 111,
                "tmdb_id": 111,
                "tvdb_id": 111,
            },
        ],
        schema={
            "key": pl.Binary,
            "type": pl.Categorical,
            "success": pl.Boolean,
            "retrieved_at": pl.Datetime("ns"),
            "year": pl.UInt16,
            "imdb_numeric_id": pl.UInt32,
            "tmdb_id": pl.UInt32,
            "tvdb_id": pl.UInt32,
        },
    )
    df_new = pl.DataFrame(
        [
            {
                "key": rating_key("000000000000000000000001"),
                "type": "movie",
                "success": True,
                "retrieved_at": datetime(2024, 1, 1, tzinfo=UTC),
                "year": 2024,
                "imdb_numeric_id": 111,
                "tmdb_id": 111,
                "tvdb_id": 111,
            },
            {
                "key": rating_key("000000000000000000000002"),
                "type": "movie",
                "success": True,
                "retrieved_at": None,
                "year": 2024,
                "imdb_numeric_id": 111,
                "tmdb_id": 111,
                "tvdb_id": 111,
            },
            {
                "key": rating_key("000000000000000000000003"),
                "type": "movie",
                "success": False,
                "retrieved_at": datetime(2024, 1, 3, tzinfo=UTC),
                "year": 2024,
                "imdb_numeric_id": 111,
                "tmdb_id": 111,
                "tvdb_id": 111,
            },
            {
                "key": rating_key("000000000000000000000004"),
                "type": "movie",
                "success": True,
                "retrieved_at": datetime(2024, 1, 4, tzinfo=UTC),
                "year": 2024,
                "imdb_numeric_id": 111,
                "tmdb_id": 111,
                "tvdb_id": 111,
            },
        ],
        schema={
            "key": pl.Binary,
            "type": pl.Categorical,
            "success": pl.Boolean,
            "retrieved_at": pl.Datetime("ns"),
            "imdb_numeric_id": pl.UInt32,
            "tmdb_id": pl.UInt32,
            "tvdb_id": pl.UInt32,
        },
    )
    df_stats = compute_stats(df_old=df_old, df_new=df_new)

    id_stats = df_stats.row(index=0, named=True)
    assert id_stats["name"] == "key"
    assert id_stats["dtype"] == "binary"
    assert id_stats["unique"] == "true"
    assert id_stats["updated"] == ""

    type_stats = df_stats.row(index=1, named=True)
    assert type_stats["name"] == "type"
    assert type_stats["dtype"] == "cat"
    assert type_stats["updated"] == ""


def test_compute_stats_empty() -> None:
    df = pl.DataFrame(schema={"key": pl.Binary, "type": pl.Categorical})
    df_stats = compute_stats(df_old=df, df_new=df)

    id_stats = df_stats.row(index=0, named=True)
    assert id_stats["name"] == "key"
    assert id_stats["dtype"] == "binary"
    assert id_stats["unique"] == "true"

    type_stats = df_stats.row(index=1, named=True)
    assert type_stats["name"] == "type"
    assert type_stats["dtype"] == "cat"


def test_format_gh_step_summary() -> None:
    schema = {"key": pl.Binary, "type": pl.Categorical}
    df_old = pl.DataFrame(
        data=[
            {"key": rating_key("000000000000000000000000"), "type": "movie"},
            {"key": rating_key("000000000000000000000001"), "type": "movie"},
        ],
        schema=schema,
    )
    df_new = pl.DataFrame(
        data=[
            {"key": rating_key("000000000000000000000000"), "type": "movie"},
            {"key": rating_key("000000000000000000000001"), "type": "movie"},
            {"key": rating_key("000000000000000000000002"), "type": "movie"},
        ],
        schema=schema,
    )
    actual = format_gh_step_summary(df_old, df_new, filename="plex-movie.parquet")
    expected = """
## plex-movie.parquet

| name (str) | dtype (str) | null (str) | true (str) | false (str) | unique (str) | updated (str) |
|------------|-------------|------------|------------|-------------|--------------|---------------|
| key        | binary      |            |            |             | true         |               |
| type       | cat         |            |            |             |              |               |

shape: (3, 2)
changes: +1 -0 ~0
rss: 0.0MB
    """
    assert actual.strip() == expected.strip()


@pytest.mark.skipif(
    not os.environ.get("PLEX_SERVER_NAME"),
    reason="PLEX_SERVER_NAME not set",
)
@pytest.mark.skipif(
    not os.environ.get("PLEX_TOKEN"),
    reason="PLEX_TOKEN not set",
)
def test_plex_device_info() -> None:
    device_name = os.environ["PLEX_SERVER_NAME"]
    auth_token = os.environ["PLEX_TOKEN"]
    device_info = plex_device_info(device_name=device_name, token=auth_token)
    assert device_info
    assert device_info.name == device_name
    assert device_info.uri.startswith("http://")
    assert device_info.access_token


@pytest.mark.skipif(
    not os.environ.get("PLEX_SERVER_NAME"),
    reason="PLEX_SERVER_NAME not set",
)
@pytest.mark.skipif(
    not os.environ.get("PLEX_TOKEN"),
    reason="PLEX_TOKEN not set",
)
def test_plex_library_guids() -> None:
    device_name = os.environ["PLEX_SERVER_NAME"]
    auth_token = os.environ["PLEX_TOKEN"]
    device_info = plex_device_info(device_name=device_name, token=auth_token)
    assert device_info

    itr = plex_library_guids(server_uri=device_info.uri, token=device_info.access_token)
    assert len(list(itertools.islice(itr, 10))) == 10


def test_wikidata_plex_media_keys() -> None:
    itr = wikidata_plex_media_keys()
    assert len(list(itertools.islice(itr, 10))) == 10


def test_plex_search_guids() -> None:
    itr = plex_search_guids(query="A Few Good Men")
    results = list(itertools.islice(itr, 10))
    assert len(results) == 10
    keys = [r[1] for r in results]
    # <https://app.plex.tv/desktop/#!/provider/tv.plex.provider.metadata/details?key=/library/metadata/5d7768288718ba001e3120b3>
    assert rating_key("5d7768288718ba001e3120b3") in keys


def test_wd_random_titles_movie() -> None:
    itr = wd_random_titles(tmdb_type="movie")
    results = list(itertools.islice(itr, 10))
    assert len(results) == 10


def test_wd_random_titles_tv() -> None:
    itr = wd_random_titles(tmdb_type="tv")
    results = list(itertools.islice(itr, 10))
    assert len(results) == 10


def test_wd_random_titles() -> None:
    itr = wd_random_titles()
    results = list(itertools.islice(itr, 10))
    assert len(results) == 10


def test_wd_random_search_guids() -> None:
    itr = wd_random_search_guids()
    assert len(list(itertools.islice(itr, 10))) == 10


@pytest.mark.skipif(
    not os.environ.get("PLEX_TOKEN"),
    reason="PLEX_TOKEN not set",
)
def test_fetch_plex_metadata_movie() -> None:
    token = os.environ["PLEX_TOKEN"]
    metadata, _ = fetch_plex_metadata(
        key=rating_key("5d776be17a53e9001e732ab9"),
        token=token,
    )
    assert metadata
    assert metadata.type == "movie"
    assert metadata.key == rating_key("5d776be17a53e9001e732ab9")
    assert metadata.year == 2022
    assert metadata.imdb_numeric_id == 1745960
    assert metadata.tmdb_id == 361743
    assert metadata.tvdb_id == 16721


@pytest.mark.skipif(
    not os.environ.get("PLEX_TOKEN"),
    reason="PLEX_TOKEN not set",
)
def test_fetch_plex_metadata_show() -> None:
    token = os.environ["PLEX_TOKEN"]
    metadata, _ = fetch_plex_metadata(
        key=rating_key("5d9c0874ffd9ef001e99607a"),
        token=token,
    )
    assert metadata
    assert metadata.type == "show"
    assert metadata.key == rating_key("5d9c0874ffd9ef001e99607a")
    assert metadata.year == 1975
    assert metadata.imdb_numeric_id == 72500
    assert metadata.tmdb_id == 2207
    assert metadata.tvdb_id == 75932


@pytest.mark.skipif(
    not os.environ.get("PLEX_TOKEN"),
    reason="PLEX_TOKEN not set",
)
def test_fetch_plex_metadata_missing() -> None:
    token = os.environ["PLEX_TOKEN"]
    metadata, _ = fetch_plex_metadata(
        key=rating_key("000000000000000000000000"),
        token=token,
    )
    assert metadata is None


@pytest.mark.skipif(
    not os.environ.get("PLEX_SERVER_NAME"),
    reason="PLEX_SERVER_NAME not set",
)
@pytest.mark.skipif(
    not os.environ.get("PLEX_TOKEN"),
    reason="PLEX_TOKEN not set",
)
def test_discover_media_keys_df() -> None:
    plex_server_name = os.environ["PLEX_SERVER_NAME"]
    plex_token = os.environ["PLEX_TOKEN"]
    df = discover_media_keys_df(
        df=pl.DataFrame({"key": []}, schema={"key": pl.Binary}),
        plex_server_name=plex_server_name,
        plex_token=plex_token,
        limit=10,
    )
    assert df.columns == ["key"]
    assert df["key"].dtype == pl.Binary
    assert len(df) == 10


def test_decode_plex_guid_valid() -> None:
    result = decode_plex_guid("plex://movie/5d776be17a53e9001e732ab9")
    assert result is not None
    assert result[0] == "movie"
    assert result[1] == rating_key("5d776be17a53e9001e732ab9")

    result = decode_plex_guid("plex://show/5d9c0874ffd9ef001e99607a")
    assert result is not None
    assert result[0] == "show"
    assert result[1] == rating_key("5d9c0874ffd9ef001e99607a")

    result = decode_plex_guid("plex://episode/5d9c0874ffd9ef001e99607b")
    assert result is not None
    assert result[0] == "episode"
    assert result[1] == rating_key("5d9c0874ffd9ef001e99607b")

    result = decode_plex_guid("plex://season/5d9c0874ffd9ef001e99607c")
    assert result is not None
    assert result[0] == "season"
    assert result[1] == rating_key("5d9c0874ffd9ef001e99607c")


def test_decode_plex_guid_invalid() -> None:
    result = decode_plex_guid("invalid_guid")
    assert result is None

    result = decode_plex_guid("plex://invalid/5d776be17a53e9001e732ab9")
    assert result is None

    result = decode_plex_guid("plex://movie/123456789012")
    assert result is None

    result = decode_plex_guid("plex://movie/5d776be17a53e9001e732ab9extra")
    assert result is not None
    assert result[0] == "movie"
    assert result[1] == rating_key("5d776be17a53e9001e732ab9")

    result = decode_plex_guid("")
    assert result is None


def test_rating_key_validation() -> None:
    key = rating_key("5d776be17a53e9001e732ab9")
    assert len(key) == 12
    assert key == bytes.fromhex("5d776be17a53e9001e732ab9")

    with pytest.raises(AssertionError):
        rating_key("123456789012345")

    with pytest.raises(AssertionError):
        rating_key("12345678901234567890123")


def test_typed_rating_key_validation() -> None:
    result = typed_rating_key("movie", "5d776be17a53e9001e732ab9")
    assert result[0] == "movie"
    assert result[1] == rating_key("5d776be17a53e9001e732ab9")

    with pytest.raises(AssertionError):
        typed_rating_key("invalid_type", "5d776be17a53e9001e732ab9")

    with pytest.raises(AssertionError):
        typed_rating_key("movie", "123456789012345")


def test_roundrobin_function() -> None:
    result = list(roundrobin([1, 2], [3, 4], [5]))
    assert result == [1, 3, 5, 2, 4]

    result = list(roundrobin([1, 2], [], [3, 4]))
    assert result == [1, 3, 2, 4]

    result = list(roundrobin([1, 2, 3]))
    assert result == [1, 2, 3]

    result = list(roundrobin())
    assert result == []
