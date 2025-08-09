# plex-index

TK

```
>>> import polars as pl
>>> pl.read_parquet("https://josh.github.io/plex-index/plex.parquet").filter(pl.col("imdb_numeric_id") == 111161)
```
