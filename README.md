# plex-index

Index of Plex rating keys mapped to IMDb, TMDb and TVDb IDs, stored as a parquet file.

Each run grows the index by discovering new rating keys from a Plex Media Server library, an IMDb watchlist and random Plex Discover searches, then fetches metadata for new keys and refreshes the oldest rows.

```
>>> import polars as pl
>>> pl.read_parquet("https://example.com/plex-index/plex.parquet").filter(
    pl.col("imdb_numeric_id") == 111161
).select(
    pl.format(
        "https://app.plex.tv/desktop/#!/provider/tv.plex.provider.discover/details?key=/library/metadata/{}",
        pl.col("key").bin.encode("hex"),
    ),
    pl.col("imdb_numeric_id"),
    pl.col("tmdb_id"),
)
shape: (1, 3)
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┬─────────────────┬─────────┐
│ literal                                                                                                                  ┆ imdb_numeric_id ┆ tmdb_id │
│ ---                                                                                                                      ┆ ---             ┆ ---     │
│ str                                                                                                                      ┆ u32             ┆ u32     │
╞══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╪═════════════════╪═════════╡
│ https://app.plex.tv/desktop/#!/provider/tv.plex.provider.discover/details?key=/library/metadata/5d7768248a7581001f12bc77 ┆ 111161          ┆ 278     │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┴─────────────────┴─────────┘
```

## Schema

| Column | Type | Description |
| --- | --- | --- |
| `key` | binary | 12 byte Plex rating key. Hex encode it for the 24 character form. |
| `type` | string | `movie`, `show`, `season` or `episode`. |
| `success` | bool | Whether the last metadata fetch found the key. |
| `retrieved_at` | timestamp (ns) | When metadata was last fetched, in UTC. |
| `year` | uint16 | Release year. |
| `imdb_numeric_id` | uint32 | IMDb ID without the `tt` prefix. |
| `tmdb_id` | uint32 | TMDb ID. |
| `tvdb_id` | uint32 | TVDb ID. |

## Usage

```sh
go install github.com/josh/plex-index@latest
PLEX_INDEX_URL=https://example.com/plex-index/plex.parquet \
  PLEX_TOKEN=... \
  plex-index plex.parquet
```

Every Plex base URL can point at an authenticating proxy that adds `X-Plex-Token` itself, so the job never holds the token. With proxies for `PLEX_METADATA_URL` and `PLEX_SERVER_URL`, `PLEX_TOKEN` can be left unset.

The updated index is written to `FILE`. The existing index is read from `PLEX_INDEX_URL`, or from `FILE` itself when unset. A missing local file starts an empty index, but any HTTP error fails the run.

## Configuration

| Environment variable | Flag | Description |
| --- | --- | --- |
| | `FILE` | Output parquet path. Required. |
| `PLEX_INDEX_URL` | `-plex-index-url` | URL or local path of the existing index. Defaults to `FILE`. |
| `PLEX_TOKEN` | | Plex account token, sent as `X-Plex-Token`. Required unless `PLEX_METADATA_URL` points at a proxy that adds it. |
| `PLEX_DISCOVER_URL` | `-plex-discover-url` | Alternate Plex Discover base URL for search, such as an authenticating proxy. Defaults to `https://discover.provider.plex.tv`. |
| `PLEX_METADATA_URL` | `-plex-metadata-url` | Alternate Plex metadata provider base URL, such as an authenticating proxy. Defaults to `https://metadata.provider.plex.tv`. |
| `PLEX_SERVER_URL` | `-plex-server-url` | Plex Media Server base URL, such as an authenticating proxy, to discover library items from. |
| `PLEX_SERVER_NAME` | `-plex-server-name` | Plex Media Server name to look up with `PLEX_TOKEN` when `PLEX_SERVER_URL` is unset. |
| `IMDB_WATCHLIST_URL` | `-imdb-watchlist-url` | IMDb watchlist CSV URL or local path whose titles are searched on Plex. |
| `DISCOVER_LIMIT` | `-discover-limit` | Maximum new rating keys to add. Defaults to 10. |
| `REFRESH_LIMIT` | `-refresh-limit` | Number of oldest rows to refresh. Defaults to 10. |
| `SEARCH_LIMIT` | `-search-limit` | Maximum random Plex search queries per run. Defaults to 2000. |
| `DRY_RUN` | `-dry-run`, `-n` | Do everything except write the index. |
| | `-v` | Enable verbose logging. |
| | `-version`, `-V` | Print the version. |

When `GITHUB_STEP_SUMMARY` is set, a markdown summary of column stats and changes is appended to it.
