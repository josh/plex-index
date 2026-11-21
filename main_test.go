package main

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"
)

func mustKey(t *testing.T, s string) ratingKey {
	t.Helper()
	k, err := parseRatingKey(s)
	if err != nil {
		t.Fatal(err)
	}
	return k
}

func testPlex() plex {
	return plex{
		discoverURL: envString("PLEX_DISCOVER_URL", defaultDiscoverURL),
		metadataURL: envString("PLEX_METADATA_URL", defaultMetadataURL),
		serverURL:   os.Getenv("PLEX_SERVER_URL"),
		serverName:  os.Getenv("PLEX_SERVER_NAME"),
		token:       os.Getenv("PLEX_TOKEN"),
	}
}

func requireMetadataAccess(t *testing.T) plex {
	t.Helper()
	p := testPlex()
	if p.token == "" && p.metadataURL == defaultMetadataURL {
		t.Skip("PLEX_TOKEN or PLEX_METADATA_URL not set")
	}
	return p
}

func collect(t *testing.T, n int, fn func(yield func(ratingKey) bool) error) []ratingKey {
	t.Helper()
	var keys []ratingKey
	err := fn(func(k ratingKey) bool {
		keys = append(keys, k)
		return len(keys) < n
	})
	if err != nil {
		t.Fatal(err)
	}
	return keys
}

func TestParseRatingKey(t *testing.T) {
	k, err := parseRatingKey("5d776be17a53e9001e732ab9")
	if err != nil {
		t.Fatal(err)
	}
	if k.String() != "5d776be17a53e9001e732ab9" {
		t.Errorf("got %s", k)
	}
	for _, s := range []string{"123456789012345", "12345678901234567890123", "zzzzzzzzzzzzzzzzzzzzzzzz"} {
		if _, err := parseRatingKey(s); err == nil {
			t.Errorf("parseRatingKey(%q): expected error", s)
		}
	}
}

func TestDecodeGUID(t *testing.T) {
	valid := map[string]string{
		"plex://movie/5d776be17a53e9001e732ab9":         "movie",
		"plex://show/5d9c0874ffd9ef001e99607a":          "show",
		"plex://episode/5d9c0874ffd9ef001e99607b":       "episode",
		"plex://season/5d9c0874ffd9ef001e99607c":        "season",
		"plex://movie/5d776be17a53e9001e732ab9extra":    "movie",
		"plex://movie/5d776be17a53e9001e732ab9?a=b&c=d": "movie",
	}
	for guid, want := range valid {
		typ, key, ok := decodeGUID(guid)
		if !ok || typ != want || !strings.HasPrefix(guid[len("plex://"+want+"/"):], key.String()) {
			t.Errorf("decodeGUID(%q) = %q, %s, %v", guid, typ, key, ok)
		}
	}
	for _, guid := range []string{"", "invalid_guid", "plex://invalid/5d776be17a53e9001e732ab9", "plex://movie/123456789012", "x plex://movie/5d776be17a53e9001e732ab9"} {
		if _, _, ok := decodeGUID(guid); ok {
			t.Errorf("decodeGUID(%q): expected no match", guid)
		}
	}
}

func TestScanGUIDs(t *testing.T) {
	data := []byte(`<Video guid="plex://movie/5d776be17a53e9001e732ab9"/><Directory guid="plex://show/5d9c0874ffd9ef001e99607a"/>`)
	keys := collect(t, 10, func(yield func(ratingKey) bool) error {
		scanGUIDs(data, yield)
		return nil
	})
	want := []ratingKey{mustKey(t, "5d776be17a53e9001e732ab9"), mustKey(t, "5d9c0874ffd9ef001e99607a")}
	if !slices.Equal(keys, want) {
		t.Errorf("got %v, want %v", keys, want)
	}
	if keys := collect(t, 1, func(yield func(ratingKey) bool) error { scanGUIDs(data, yield); return nil }); len(keys) != 1 {
		t.Errorf("expected scan to stop after 1 key, got %d", len(keys))
	}
}

func TestIndexInsert(t *testing.T) {
	k1, k2 := mustKey(t, "000000000000000000000001"), mustKey(t, "000000000000000000000002")
	r := newRow(k1)
	r.Success = new(true)
	r.Year = new(uint16(2024))
	idx := index{k1: r}
	if idx.insert(k1) {
		t.Error("insert overwrote existing key")
	}
	if !idx.insert(k2) {
		t.Error("insert did not add new key")
	}
	if idx[k1].Year == nil || *idx[k1].Year != 2024 {
		t.Error("existing row was modified")
	}
	if !rowsEqual(idx[k2], newRow(k2)) {
		t.Error("new row has non-null fields")
	}
}

func TestBackfillKeys(t *testing.T) {
	base := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)
	idx := index{}
	add := func(hex string, at *time.Time) ratingKey {
		k := mustKey(t, hex)
		r := newRow(k)
		r.RetrievedAt = at
		idx[k] = r
		return k
	}
	newest := add("000000000000000000000001", new(base.Add(3*time.Hour)))
	oldest := add("000000000000000000000002", new(base))
	tieA := add("000000000000000000000003", new(base.Add(time.Hour)))
	tieB := add("000000000000000000000004", new(base.Add(time.Hour)))
	missing := add("000000000000000000000005", nil)

	if got, want := backfillKeys(idx, 2), []ratingKey{missing, oldest, tieA}; !slices.Equal(got, want) {
		t.Errorf("limit 2: got %v, want %v", got, want)
	}
	if got, want := backfillKeys(idx, 0), []ratingKey{missing}; !slices.Equal(got, want) {
		t.Errorf("limit 0: got %v, want %v", got, want)
	}
	if got, want := backfillKeys(idx, 100), []ratingKey{missing, oldest, tieA, tieB, newest}; !slices.Equal(got, want) {
		t.Errorf("limit 100: got %v, want %v", got, want)
	}
}

func testIndex(t *testing.T, years map[string]uint16) index {
	idx := index{}
	for hex, year := range years {
		k := mustKey(t, hex)
		r := newRow(k)
		r.Type = new("movie")
		r.Year = new(year)
		idx[k] = r
	}
	return idx
}

func TestChangeSummary(t *testing.T) {
	tests := []struct {
		name                    string
		old, cur                map[string]uint16
		added, removed, updated int
	}{
		{
			name:  "added",
			old:   map[string]uint16{"000000000000000000000000": 10, "000000000000000000000001": 20},
			cur:   map[string]uint16{"000000000000000000000000": 10, "000000000000000000000001": 20, "000000000000000000000002": 30},
			added: 1,
		},
		{
			name:    "removed",
			old:     map[string]uint16{"000000000000000000000000": 10, "000000000000000000000001": 20},
			cur:     map[string]uint16{"000000000000000000000000": 10},
			removed: 1,
		},
		{
			name:    "updated",
			old:     map[string]uint16{"000000000000000000000000": 10, "000000000000000000000001": 20, "000000000000000000000002": 30},
			cur:     map[string]uint16{"000000000000000000000000": 100, "000000000000000000000001": 200, "000000000000000000000002": 30},
			updated: 2,
		},
		{
			name:    "added and updated",
			old:     map[string]uint16{"000000000000000000000000": 10, "000000000000000000000002": 30},
			cur:     map[string]uint16{"000000000000000000000000": 100, "000000000000000000000001": 200, "000000000000000000000002": 30},
			added:   1,
			updated: 1,
		},
		{
			name: "noop",
			old:  map[string]uint16{"000000000000000000000000": 10},
			cur:  map[string]uint16{"000000000000000000000000": 10},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			added, removed, updated := changeSummary(testIndex(t, tt.old), testIndex(t, tt.cur))
			if added != tt.added || removed != tt.removed || updated != tt.updated {
				t.Errorf("got +%d -%d ~%d, want +%d -%d ~%d", added, removed, updated, tt.added, tt.removed, tt.updated)
			}
		})
	}
}

func TestColumnStats(t *testing.T) {
	old := testIndex(t, map[string]uint16{
		"000000000000000000000000": 2000,
		"000000000000000000000001": 2001,
	})
	cur := testIndex(t, map[string]uint16{
		"000000000000000000000000": 2000,
		"000000000000000000000001": 2011,
	})
	k2 := mustKey(t, "000000000000000000000002")
	r := newRow(k2)
	r.Success = new(false)
	cur[k2] = r
	k1 := mustKey(t, "000000000000000000000001")
	r = cur[k1]
	r.Success = new(true)
	cur[k1] = r

	want := []columnStat{
		{name: "key", dtype: "binary", unique: true},
		{name: "type", dtype: "str", nulls: 1},
		{name: "success", dtype: "bool", nulls: 1, trues: 1, falses: 1, unique: true, updated: 1},
		{name: "retrieved_at", dtype: "datetime[ns]", nulls: 3, unique: true},
		{name: "year", dtype: "u16", nulls: 1, unique: true, updated: 1},
		{name: "imdb_numeric_id", dtype: "u32", nulls: 3, unique: true},
		{name: "tmdb_id", dtype: "u32", nulls: 3, unique: true},
		{name: "tvdb_id", dtype: "u32", nulls: 3, unique: true},
	}
	if got := columnStats(old, cur); !slices.Equal(got, want) {
		t.Errorf("columnStats() = %+v, want %+v", got, want)
	}
}

func TestIndexRoundTrip(t *testing.T) {
	k1, k2 := mustKey(t, "5d776be17a53e9001e732ab9"), mustKey(t, "000000000000000000000000")
	r1 := newRow(k1)
	r1.Type = new("movie")
	r1.Success = new(true)
	r1.RetrievedAt = new(time.Date(2024, 5, 6, 7, 8, 9, 123456789, time.UTC))
	r1.Year = new(uint16(2022))
	r1.IMDbNumericID = new(uint32(1745960))
	r1.TMDbID = new(uint32(361743))
	r1.TVDbID = new(uint32(16721))
	r2 := newRow(k2)
	r2.Success = new(false)
	r2.RetrievedAt = new(time.Date(2024, 5, 6, 7, 8, 9, 0, time.FixedZone("PDT", -7*3600)))
	idx := index{k1: r1, k2: r2}

	path := filepath.Join(t.TempDir(), "plex.parquet")
	size, err := writeIndex(path, idx)
	if err != nil {
		t.Fatal(err)
	}
	if size == 0 {
		t.Error("expected non-zero file size")
	}
	got, err := loadIndex(t.Context(), path)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(idx) {
		t.Fatalf("got %d rows, want %d", len(got), len(idx))
	}
	for k, want := range idx {
		if !rowsEqual(got[k], want) {
			t.Errorf("%s: got %+v, want %+v", k, got[k], want)
		}
	}
	rows := got.sortedRows()
	if string(rows[0].Key) != string(k2[:]) {
		t.Error("rows not sorted by key")
	}
}

func TestLoadIndexMissingFile(t *testing.T) {
	idx, err := loadIndex(t.Context(), filepath.Join(t.TempDir(), "missing.parquet"))
	if err != nil {
		t.Fatal(err)
	}
	if len(idx) != 0 {
		t.Errorf("got %d rows, want 0", len(idx))
	}
}

func TestConfigValidate(t *testing.T) {
	tests := []struct {
		name    string
		cfg     config
		wantErr bool
		library bool
	}{
		{name: "missing output", cfg: config{token: "x", metadataURL: defaultMetadataURL}, wantErr: true},
		{name: "token", cfg: config{output: "o", token: "x", metadataURL: defaultMetadataURL}},
		{name: "no token default metadata", cfg: config{output: "o", metadataURL: defaultMetadataURL}, wantErr: true},
		{name: "no token proxy metadata", cfg: config{output: "o", metadataURL: "http://proxy"}},
		{name: "server name and token", cfg: config{output: "o", token: "x", metadataURL: defaultMetadataURL, serverName: "s"}, library: true},
		{name: "server name without token", cfg: config{output: "o", metadataURL: "http://proxy", serverName: "s"}},
		{name: "server url", cfg: config{output: "o", metadataURL: "http://proxy", serverURL: "http://server"}, library: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.cfg.validate(); (err != nil) != tt.wantErr {
				t.Errorf("validate() = %v, wantErr %v", err, tt.wantErr)
			}
			if got := tt.cfg.libraryEnabled(); got != tt.library {
				t.Errorf("libraryEnabled() = %v, want %v", got, tt.library)
			}
		})
	}
}

func TestSearch(t *testing.T) {
	keys, terms, err := testPlex().search(t.Context(), "star")
	if err != nil {
		t.Fatal(err)
	}
	if len(keys) == 0 {
		t.Error("expected search results")
	}
	if len(terms) == 0 {
		t.Error("expected suggested terms")
	}
}

func TestSearchKeys(t *testing.T) {
	keys := collect(t, 10, func(yield func(ratingKey) bool) error {
		_, err := testPlex().searchKeys(t.Context(), "A Few Good Men", yield)
		return err
	})
	if len(keys) != 10 {
		t.Errorf("got %d keys, want 10", len(keys))
	}
	if !slices.Contains(keys, mustKey(t, "5d7768288718ba001e3120b3")) {
		t.Errorf("A Few Good Men not in %v", keys)
	}
}

func TestRandomSearchKeys(t *testing.T) {
	keys := collect(t, 10, func(yield func(ratingKey) bool) error {
		return testPlex().randomSearchKeys(t.Context(), defaultSearchLimit, yield)
	})
	if len(keys) != 10 {
		t.Errorf("got %d keys, want 10", len(keys))
	}
}

func TestMetadata(t *testing.T) {
	p := requireMetadataAccess(t)
	tests := []struct {
		key              string
		typ              string
		year             uint16
		imdb, tmdb, tvdb uint32
	}{
		{key: "5d776be17a53e9001e732ab9", typ: "movie", year: 2022, imdb: 1745960, tmdb: 361743, tvdb: 16721},
		{key: "5d9c0874ffd9ef001e99607a", typ: "show", year: 1975, imdb: 72500, tmdb: 2207, tvdb: 75932},
	}
	for _, tt := range tests {
		t.Run(tt.key, func(t *testing.T) {
			r, similar, err := p.metadata(t.Context(), mustKey(t, tt.key), time.Now().UTC())
			if err != nil {
				t.Fatal(err)
			}
			if r.Success == nil || !*r.Success {
				t.Fatal("expected success")
			}
			if r.Type == nil || *r.Type != tt.typ {
				t.Errorf("type = %v, want %s", ptrValue(r.Type), tt.typ)
			}
			if r.Year == nil || *r.Year != tt.year {
				t.Errorf("year = %v, want %d", ptrValue(r.Year), tt.year)
			}
			if r.IMDbNumericID == nil || *r.IMDbNumericID != tt.imdb {
				t.Errorf("imdb = %v, want %d", ptrValue(r.IMDbNumericID), tt.imdb)
			}
			if r.TMDbID == nil || *r.TMDbID != tt.tmdb {
				t.Errorf("tmdb = %v, want %d", ptrValue(r.TMDbID), tt.tmdb)
			}
			if r.TVDbID == nil || *r.TVDbID != tt.tvdb {
				t.Errorf("tvdb = %v, want %d", ptrValue(r.TVDbID), tt.tvdb)
			}
			if len(similar) == 0 {
				t.Error("expected similar rating keys")
			}
		})
	}
}

func TestMetadataNotFound(t *testing.T) {
	p := requireMetadataAccess(t)
	r, similar, err := p.metadata(t.Context(), mustKey(t, "000000000000000000000000"), time.Now().UTC())
	if err != nil {
		t.Fatal(err)
	}
	if r.Success == nil || *r.Success {
		t.Error("expected success=false")
	}
	if r.RetrievedAt == nil {
		t.Error("expected retrieved_at")
	}
	if r.Type != nil || r.Year != nil || r.IMDbNumericID != nil || len(similar) != 0 {
		t.Errorf("expected empty row, got %+v", r)
	}
}

func TestDevice(t *testing.T) {
	p := testPlex()
	if p.serverName == "" || p.token == "" {
		t.Skip("PLEX_SERVER_NAME or PLEX_TOKEN not set")
	}
	dev, ok, err := p.device(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if !ok {
		t.Fatalf("device %q not found", p.serverName)
	}
	if dev.Name != p.serverName || dev.URI == "" || dev.AccessToken == "" {
		t.Errorf("incomplete device info: name=%q uri=%q token=%v", dev.Name, dev.URI, dev.AccessToken != "")
	}
}

func TestLibraryKeys(t *testing.T) {
	p := testPlex()
	cfg := config{serverURL: p.serverURL, serverName: p.serverName, token: p.token}
	if !cfg.libraryEnabled() {
		t.Skip("PLEX_SERVER_URL or PLEX_SERVER_NAME and PLEX_TOKEN not set")
	}
	keys := collect(t, 10, func(yield func(ratingKey) bool) error {
		return p.libraryKeys(t.Context(), yield)
	})
	if len(keys) != 10 {
		t.Errorf("got %d keys, want 10", len(keys))
	}
}

func TestWatchlistKeys(t *testing.T) {
	source := os.Getenv("IMDB_WATCHLIST_URL")
	if source == "" {
		t.Skip("IMDB_WATCHLIST_URL not set")
	}
	keys := collect(t, 10, func(yield func(ratingKey) bool) error {
		return testPlex().watchlistKeys(t.Context(), source, yield)
	})
	if len(keys) != 10 {
		t.Errorf("got %d keys, want 10", len(keys))
	}
}

func TestPublishedIndex(t *testing.T) {
	source := os.Getenv("PLEX_INDEX_URL")
	if source == "" {
		t.Skip("PLEX_INDEX_URL not set")
	}
	idx, err := loadIndex(t.Context(), source)
	if err != nil {
		t.Fatal(err)
	}
	r, ok := idx[mustKey(t, "5d7768248a7581001f12bc77")]
	if !ok {
		t.Fatal("The Shawshank Redemption missing from index")
	}
	if r.IMDbNumericID == nil || *r.IMDbNumericID != 111161 {
		t.Errorf("imdb_numeric_id = %v, want 111161", ptrValue(r.IMDbNumericID))
	}
	if r.TMDbID == nil || *r.TMDbID != 278 {
		t.Errorf("tmdb_id = %v, want 278", ptrValue(r.TMDbID))
	}
}
