# Agents Guide

This project is written in Go and uses Go modules for dependency management. The repository assumes Go 1.26 or newer. Avoid adding dependencies beyond `github.com/parquet-go/parquet-go`.

## Setup

1. Install Go 1.26 or later.
2. Download dependencies with:

```sh
go mod download
```

## Testing

Run the test suite with:

```sh
go test ./...
```

Tests make real network requests to Plex. Avoid mocks and fakes. Tests that need credentials skip unless `PLEX_TOKEN`, `PLEX_SERVER_NAME`, `PLEX_SERVER_URL`, `PLEX_DISCOVER_URL`, `PLEX_METADATA_URL`, `IMDB_WATCHLIST_URL` or `PLEX_INDEX_URL` are set.

## Formatting

Format code with:

```sh
go fmt ./...
```

## Code Quality

Run vet and static analysis tools before committing:

```sh
go vet ./...
```

Optionally run `golangci-lint` for additional checks:

```sh
golangci-lint run ./...
```

## Building

Build the project with:

```sh
go build ./...
```

## Comments

Keep comments concise. Only add them when they clarify non-obvious logic.
