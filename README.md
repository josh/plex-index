# plex-index

```
>>> import polars as pl
>>> pl.read_parquet("https://josh.github.io/plex-index/plex.parquet").filter(
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
