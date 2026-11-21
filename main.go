package main

import (
	"bytes"
	"cmp"
	"context"
	"encoding/csv"
	"encoding/hex"
	"encoding/json"
	"encoding/xml"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"log/slog"
	"maps"
	"math/rand/v2"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/parquet-go/parquet-go"
)

const (
	Version                = "0.2.0"
	defaultDiscoverURL     = "https://discover.provider.plex.tv"
	defaultMetadataURL     = "https://metadata.provider.plex.tv"
	plexTVURL              = "https://plex.tv"
	maxAttempts            = 4
	maxConsecutiveFailures = 10
	defaultSearchLimit     = 2000
	rowGroupSize           = 128 * 1024
)

var retryStatuses = map[int]bool{
	http.StatusRequestTimeout:      true,
	http.StatusTooEarly:            true,
	http.StatusTooManyRequests:     true,
	http.StatusInternalServerError: true,
	http.StatusBadGateway:          true,
	http.StatusServiceUnavailable:  true,
	http.StatusGatewayTimeout:      true,
}

var (
	guidRE       = regexp.MustCompile(`plex://(episode|movie|season|show)/([a-f0-9]{24})`)
	guidPrefixRE = regexp.MustCompile(`^plex://(episode|movie|season|show)/([a-f0-9]{24})`)
)

type statusError struct {
	method string
	url    string
	status int
}

func (e *statusError) Error() string {
	return fmt.Sprintf("%s %s: HTTP %d", e.method, e.url, e.status)
}

func since(start time.Time) time.Duration {
	return time.Since(start).Round(time.Millisecond)
}

func fetch(req *http.Request, timeout time.Duration) ([]byte, error) {
	for attempt := 1; ; attempt++ {
		start := time.Now()
		body, status, err := fetchOnce(req, timeout)
		slog.Debug("http request",
			"method", req.Method,
			"url", req.URL.Redacted(),
			"attempt", attempt,
			"status", status,
			"bytes", len(body),
			"duration", since(start),
		)
		if err == nil {
			return body, nil
		}
		var se *statusError
		if (errors.As(err, &se) && !retryStatuses[se.status]) || attempt == maxAttempts || req.Context().Err() != nil {
			return nil, err
		}
		delay := time.Second << (attempt - 1)
		slog.Warn("retrying http request",
			"method", req.Method,
			"url", req.URL.Redacted(),
			"attempt", attempt,
			"delay", delay,
			"err", err,
		)
		select {
		case <-time.After(delay):
		case <-req.Context().Done():
			return nil, req.Context().Err()
		}
	}
}

func fetchOnce(req *http.Request, timeout time.Duration) ([]byte, int, error) {
	ctx, cancel := context.WithTimeout(req.Context(), timeout)
	defer cancel()
	resp, err := http.DefaultClient.Do(req.Clone(ctx))
	if err != nil {
		return nil, 0, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, resp.StatusCode, err
	}
	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		return body, resp.StatusCode, &statusError{method: req.Method, url: req.URL.Redacted(), status: resp.StatusCode}
	}
	return body, resp.StatusCode, nil
}

func isURL(source string) bool {
	u, err := url.Parse(source)
	return err == nil && (strings.EqualFold(u.Scheme, "http") || strings.EqualFold(u.Scheme, "https"))
}

func readSource(ctx context.Context, source string, timeout time.Duration) ([]byte, error) {
	if isURL(source) {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, source, nil)
		if err != nil {
			return nil, err
		}
		return fetch(req, timeout)
	}
	slog.Debug("reading local file", "path", source)
	return os.ReadFile(source)
}

type ratingKey [12]byte

func (k ratingKey) String() string {
	return hex.EncodeToString(k[:])
}

func parseRatingKey(s string) (ratingKey, error) {
	var k ratingKey
	if len(s) != 2*len(k) {
		return k, fmt.Errorf("invalid rating key %q: expected %d hex characters", s, 2*len(k))
	}
	if _, err := hex.Decode(k[:], []byte(s)); err != nil {
		return k, fmt.Errorf("invalid rating key %q: %w", s, err)
	}
	return k, nil
}

func decodeGUID(guid string) (string, ratingKey, bool) {
	m := guidPrefixRE.FindStringSubmatch(guid)
	if m == nil {
		return "", ratingKey{}, false
	}
	key, err := parseRatingKey(m[2])
	if err != nil {
		return "", ratingKey{}, false
	}
	return m[1], key, true
}

func scanGUIDs(data []byte, yield func(ratingKey) bool) bool {
	for _, m := range guidRE.FindAllSubmatch(data, -1) {
		key, err := parseRatingKey(string(m[2]))
		if err != nil {
			continue
		}
		if !yield(key) {
			return false
		}
	}
	return true
}

type row struct {
	Key           []byte     `parquet:"key,optional"`
	Type          *string    `parquet:"type,optional,dict"`
	Success       *bool      `parquet:"success,optional"`
	RetrievedAt   *time.Time `parquet:"retrieved_at,optional,timestamp(nanosecond:local)"`
	Year          *uint16    `parquet:"year,optional"`
	IMDbNumericID *uint32    `parquet:"imdb_numeric_id,optional"`
	TMDbID        *uint32    `parquet:"tmdb_id,optional"`
	TVDbID        *uint32    `parquet:"tvdb_id,optional"`
}

type index map[ratingKey]row

func newRow(key ratingKey) row {
	return row{Key: bytes.Clone(key[:])}
}

func (idx index) insert(key ratingKey) bool {
	if _, ok := idx[key]; ok {
		return false
	}
	idx[key] = newRow(key)
	return true
}

func readIndex(data []byte) (index, error) {
	rows, err := parquet.Read[row](bytes.NewReader(data), int64(len(data)))
	if err != nil {
		return nil, err
	}
	idx := make(index, len(rows))
	for _, r := range rows {
		var key ratingKey
		if len(r.Key) != len(key) {
			return nil, fmt.Errorf("invalid key length %d", len(r.Key))
		}
		copy(key[:], r.Key)
		idx[key] = r
	}
	return idx, nil
}

func loadIndex(ctx context.Context, source string) (index, error) {
	start := time.Now()
	data, err := readSource(ctx, source, 60*time.Second)
	if errors.Is(err, fs.ErrNotExist) {
		slog.Warn("Plex index not found, starting empty index", "path", source)
		return index{}, nil
	}
	if err != nil {
		return nil, err
	}
	idx, err := readIndex(data)
	if err != nil {
		return nil, fmt.Errorf("read Plex index: %w", err)
	}
	slog.Info("loaded Plex index", "rows", len(idx), "bytes", len(data), "duration", since(start))
	return idx, nil
}

func (idx index) sortedRows() []row {
	rows := slices.Collect(maps.Values(idx))
	slices.SortFunc(rows, func(a, b row) int { return bytes.Compare(a.Key, b.Key) })
	return rows
}

func writeIndex(path string, idx index) (int64, error) {
	f, err := os.CreateTemp(filepath.Dir(path), ".plex-index-*.parquet")
	if err != nil {
		return 0, err
	}
	defer os.Remove(f.Name())
	defer f.Close()
	w := parquet.NewGenericWriter[row](f,
		parquet.Compression(&parquet.Zstd),
		parquet.MaxRowsPerRowGroup(rowGroupSize),
	)
	if _, err := w.Write(idx.sortedRows()); err != nil {
		return 0, err
	}
	if err := w.Close(); err != nil {
		return 0, err
	}
	info, err := f.Stat()
	if err != nil {
		return 0, err
	}
	if err := f.Close(); err != nil {
		return 0, err
	}
	return info.Size(), os.Rename(f.Name(), path)
}

type plex struct {
	discoverURL string
	metadataURL string
	serverURL   string
	serverName  string
	token       string
}

func newPlexRequest(ctx context.Context, rawURL, token string) (*http.Request, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
	if err != nil {
		return nil, err
	}
	if token != "" {
		req.Header.Set("X-Plex-Token", token)
	}
	return req, nil
}

func (p plex) provider(ctx context.Context, baseURL, path string, query url.Values, timeout time.Duration) ([]byte, error) {
	u := strings.TrimRight(baseURL, "/") + path
	if len(query) > 0 {
		u += "?" + query.Encode()
	}
	req, err := newPlexRequest(ctx, u, p.token)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "application/json")
	req.Header.Set("X-Plex-Provider-Version", "7.2.0")
	return fetch(req, timeout)
}

type device struct {
	Name        string
	URI         string
	AccessToken string
}

type resourcesResponse struct {
	Devices []struct {
		Name        string `xml:"name,attr"`
		AccessToken string `xml:"accessToken,attr"`
		Connections []struct {
			URI   string `xml:"uri,attr"`
			Local string `xml:"local,attr"`
		} `xml:"Connection"`
	} `xml:"Device"`
}

func (p plex) device(ctx context.Context) (device, bool, error) {
	req, err := newPlexRequest(ctx, plexTVURL+"/api/resources", p.token)
	if err != nil {
		return device{}, false, err
	}
	slog.Info("fetching Plex device info", "name", p.serverName)
	body, err := fetch(req, 10*time.Second)
	if err != nil {
		return device{}, false, err
	}
	var resp resourcesResponse
	if err := xml.Unmarshal(body, &resp); err != nil {
		return device{}, false, fmt.Errorf("parse Plex resources: %w", err)
	}
	for _, d := range resp.Devices {
		if d.Name != p.serverName {
			continue
		}
		dev := device{Name: d.Name, AccessToken: d.AccessToken}
		for _, c := range d.Connections {
			if c.Local == "0" {
				dev.URI = c.URI
			}
		}
		return dev, true, nil
	}
	return device{}, false, nil
}

func libraryKeys(ctx context.Context, serverURL, token string, yield func(ratingKey) bool) error {
	for _, section := range []int{1, 2} {
		start := time.Now()
		u := fmt.Sprintf("%s/library/sections/%d/all", strings.TrimRight(serverURL, "/"), section)
		req, err := newPlexRequest(ctx, u, token)
		if err != nil {
			return err
		}
		body, err := fetch(req, 60*time.Second)
		if err != nil {
			return err
		}
		slog.Info("fetched Plex library section", "section", section, "bytes", len(body), "duration", since(start))
		if !scanGUIDs(body, yield) {
			return nil
		}
	}
	return nil
}

func (p plex) libraryKeys(ctx context.Context, yield func(ratingKey) bool) error {
	if p.serverURL != "" {
		return libraryKeys(ctx, p.serverURL, p.token, yield)
	}
	dev, ok, err := p.device(ctx)
	if err != nil {
		return err
	}
	if !ok {
		slog.Error("Plex device not found", "name", p.serverName)
		return nil
	}
	return libraryKeys(ctx, dev.URI, dev.AccessToken, yield)
}

type searchResponse struct {
	MediaContainer struct {
		SuggestedTerms []string `json:"suggestedTerms"`
		SearchResults  []struct {
			SearchResult []struct {
				Metadata struct {
					GUID string `json:"guid"`
				} `json:"Metadata"`
			} `json:"SearchResult"`
		} `json:"SearchResults"`
	} `json:"MediaContainer"`
}

func (p plex) search(ctx context.Context, query string) ([]ratingKey, []string, error) {
	body, err := p.provider(ctx, p.discoverURL, "/library/search", url.Values{
		"query":           {query},
		"limit":           {"100"},
		"searchTypes":     {"movies,tv"},
		"includeMetadata": {"1"},
		"searchProviders": {"discover"},
	}, 10*time.Second)
	if err != nil {
		return nil, nil, err
	}
	var resp searchResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, nil, fmt.Errorf("parse Plex search %q: %w", query, err)
	}
	var keys []ratingKey
	for _, sr := range resp.MediaContainer.SearchResults {
		for _, result := range sr.SearchResult {
			if _, key, ok := decodeGUID(result.Metadata.GUID); ok {
				keys = append(keys, key)
			}
		}
	}
	return keys, resp.MediaContainer.SuggestedTerms, nil
}

func (p plex) searchKeys(ctx context.Context, query string, yield func(ratingKey) bool) (bool, error) {
	keys, terms, err := p.search(ctx, query)
	if err != nil {
		if ctx.Err() != nil {
			return false, ctx.Err()
		}
		slog.Warn("Plex search failed", "query", query, "err", err)
		return true, nil
	}
	slog.Debug("searched Plex", "query", query, "results", len(keys), "suggested_terms", len(terms))
	for _, key := range keys {
		if !yield(key) {
			return false, nil
		}
	}
	return true, nil
}

func (p plex) randomSearchKeys(ctx context.Context, limit int, yield func(ratingKey) bool) error {
	queue := strings.Split("abcdefghijklmnopqrstuvwxyz", "")
	seen := make(map[string]bool)
	for queries := 0; len(queue) > 0; {
		// Random picks spread each run across the suggestion graph; FIFO
		// re-walks the same well-indexed terms first every run.
		i := rand.IntN(len(queue))
		query := queue[i]
		queue[i] = queue[len(queue)-1]
		queue = queue[:len(queue)-1]
		if seen[query] {
			continue
		}
		if queries >= limit {
			slog.Warn("reached Plex search query limit", "limit", limit)
			return nil
		}
		queries++
		seen[query] = true
		keys, terms, err := p.search(ctx, query)
		if err != nil {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			slog.Warn("Plex search failed", "query", query, "err", err)
			continue
		}
		for _, key := range keys {
			if !yield(key) {
				return nil
			}
		}
		for _, term := range terms {
			if !seen[term] {
				queue = append(queue, term)
			}
		}
	}
	return nil
}

var titleReplacer = strings.NewReplacer("#", "", "&", "", "'", "", `"`, "")

func (p plex) watchlistKeys(ctx context.Context, source string, yield func(ratingKey) bool) error {
	data, err := readSource(ctx, source, 10*time.Second)
	if err != nil {
		return err
	}
	records, err := csv.NewReader(bytes.NewReader(data)).ReadAll()
	if err != nil {
		return fmt.Errorf("parse IMDb watchlist: %w", err)
	}
	if len(records) == 0 {
		return errors.New("IMDb watchlist is empty")
	}
	titleCol := slices.Index(records[0], "Title")
	if titleCol < 0 {
		return errors.New("IMDb watchlist has no Title column")
	}
	slog.Info("fetched IMDb watchlist", "titles", len(records)-1, "bytes", len(data))
	for _, record := range records[1:] {
		ok, err := p.searchKeys(ctx, titleReplacer.Replace(record[titleCol]), yield)
		if !ok || err != nil {
			return err
		}
	}
	return nil
}

type metadataResponse struct {
	MediaContainer struct {
		Metadata []struct {
			RatingKey string `json:"ratingKey"`
			Type      string `json:"type"`
			Year      uint16 `json:"year"`
			PlexGUID  string `json:"guid"`
			Guid      []struct {
				ID string `json:"id"`
			} `json:"Guid"`
			Similar []struct {
				GUID string `json:"guid"`
			} `json:"Similar"`
		} `json:"Metadata"`
	} `json:"MediaContainer"`
}

func parseID(key ratingKey, guid, prefix, trim string) *uint32 {
	id, ok := strings.CutPrefix(guid, prefix)
	if !ok {
		return nil
	}
	n, err := strconv.ParseUint(strings.TrimPrefix(id, trim), 10, 32)
	if err != nil {
		slog.Warn("invalid external ID", "rating_key", key, "guid", guid)
		return nil
	}
	v := uint32(n)
	return &v
}

func (p plex) metadata(ctx context.Context, key ratingKey, now time.Time) (row, []ratingKey, error) {
	r := newRow(key)
	r.RetrievedAt = &now
	body, err := p.provider(ctx, p.metadataURL, "/library/metadata/"+key.String(), nil, 30*time.Second)
	var se *statusError
	if errors.As(err, &se) && se.status == http.StatusNotFound {
		r.Success = new(false)
		return r, nil, nil
	}
	if err != nil {
		return row{}, nil, err
	}
	var resp metadataResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return row{}, nil, fmt.Errorf("parse Plex metadata %s: %w", key, err)
	}
	if len(resp.MediaContainer.Metadata) == 0 {
		return row{}, nil, fmt.Errorf("Plex metadata %s: empty response", key)
	}
	m := resp.MediaContainer.Metadata[0]
	if m.RatingKey != key.String() {
		return row{}, nil, fmt.Errorf("Plex metadata %s: got rating key %s", key, m.RatingKey)
	}
	switch m.Type {
	case "episode", "movie", "season", "show":
	default:
		return row{}, nil, fmt.Errorf("Plex metadata %s: unknown type %q", key, m.Type)
	}
	r.Success = new(true)
	r.Type = &m.Type
	if m.Year != 0 {
		r.Year = &m.Year
	}
	for _, g := range m.Guid {
		if id := parseID(key, g.ID, "imdb://", "tt"); id != nil {
			r.IMDbNumericID = id
		} else if id := parseID(key, g.ID, "tmdb://", ""); id != nil {
			r.TMDbID = id
		} else if id := parseID(key, g.ID, "tvdb://", ""); id != nil {
			r.TVDbID = id
		}
	}
	var similar []ratingKey
	for _, s := range m.Similar {
		if _, k, ok := decodeGUID(s.GUID); ok {
			similar = append(similar, k)
		}
	}
	return r, similar, nil
}

func (p plex) discoverKeys(ctx context.Context, cfg config, idx index) error {
	if cfg.discoverLimit <= 0 {
		return nil
	}
	start := time.Now()
	added := 0
	add := func(key ratingKey) bool {
		if idx.insert(key) {
			added++
			slog.Debug("discovered rating key", "rating_key", key)
		}
		return added < cfg.discoverLimit
	}
	done := func() bool { return added >= cfg.discoverLimit }

	if cfg.libraryEnabled() {
		if err := p.libraryKeys(ctx, add); err != nil {
			return fmt.Errorf("Plex library discovery: %w", err)
		}
		slog.Info("finished Plex library discovery", "added", added)
	} else {
		slog.Warn("skipping Plex library discovery")
	}

	if !done() && cfg.watchlistURL != "" {
		before := added
		if err := p.watchlistKeys(ctx, cfg.watchlistURL, add); err != nil {
			return fmt.Errorf("IMDb watchlist discovery: %w", err)
		}
		slog.Info("finished IMDb watchlist discovery", "added", added-before)
	}

	if !done() {
		before := added
		if err := p.randomSearchKeys(ctx, cfg.searchLimit, add); err != nil {
			return fmt.Errorf("random search discovery: %w", err)
		}
		slog.Info("finished random search discovery", "added", added-before)
	}

	slog.Info("discovered new rating keys", "added", added, "limit", cfg.discoverLimit, "duration", since(start))
	return nil
}

func backfillKeys(idx index, refreshLimit int) []ratingKey {
	var missing, retrieved []ratingKey
	for key, r := range idx {
		if r.RetrievedAt == nil {
			missing = append(missing, key)
		} else {
			retrieved = append(retrieved, key)
		}
	}
	slices.SortFunc(missing, func(a, b ratingKey) int { return bytes.Compare(a[:], b[:]) })
	slices.SortFunc(retrieved, func(a, b ratingKey) int {
		return cmp.Or(idx[a].RetrievedAt.Compare(*idx[b].RetrievedAt), bytes.Compare(a[:], b[:]))
	})
	return append(missing, retrieved[:min(max(refreshLimit, 0), len(retrieved))]...)
}

func (p plex) backfill(ctx context.Context, idx index, refreshLimit int) error {
	start := time.Now()
	keys := backfillKeys(idx, refreshLimit)
	slog.Info("fetching Plex metadata", "keys", len(keys))
	var similar []ratingKey
	fetched, missing, failures, consecutive := 0, 0, 0, 0
	for i, key := range keys {
		if i > 0 && i%100 == 0 {
			slog.Info("fetching Plex metadata", "progress", i, "total", len(keys))
		}
		r, sim, err := p.metadata(ctx, key, time.Now().UTC())
		if err != nil {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			slog.Warn("failed to fetch Plex metadata", "rating_key", key, "err", err)
			failures++
			consecutive++
			if consecutive == maxConsecutiveFailures {
				slog.Error("giving up Plex metadata backfill", "consecutive_failures", consecutive)
				break
			}
			continue
		}
		consecutive = 0
		idx[key] = r
		if *r.Success {
			fetched++
		} else {
			missing++
		}
		similar = append(similar, sim...)
	}
	added := 0
	for _, key := range similar {
		if idx.insert(key) {
			added++
		}
	}
	slog.Info("fetched Plex metadata",
		"fetched", fetched,
		"not_found", missing,
		"failed", failures,
		"similar_added", added,
		"duration", since(start),
	)
	return nil
}

type column struct {
	name  string
	dtype string
	value func(row) any
}

func ptrValue[T comparable](p *T) any {
	if p == nil {
		return nil
	}
	return *p
}

var columns = []column{
	{"key", "binary", func(r row) any {
		if r.Key == nil {
			return nil
		}
		return string(r.Key)
	}},
	{"type", "str", func(r row) any { return ptrValue(r.Type) }},
	{"success", "bool", func(r row) any { return ptrValue(r.Success) }},
	{"retrieved_at", "datetime[ns]", func(r row) any {
		if r.RetrievedAt == nil {
			return nil
		}
		return r.RetrievedAt.UnixNano()
	}},
	{"year", "u16", func(r row) any { return ptrValue(r.Year) }},
	{"imdb_numeric_id", "u32", func(r row) any { return ptrValue(r.IMDbNumericID) }},
	{"tmdb_id", "u32", func(r row) any { return ptrValue(r.TMDbID) }},
	{"tvdb_id", "u32", func(r row) any { return ptrValue(r.TVDbID) }},
}

func rowsEqual(a, b row) bool {
	for _, c := range columns {
		if c.value(a) != c.value(b) {
			return false
		}
	}
	return true
}

func changeSummary(old, cur index) (added, removed, updated int) {
	added = max(len(cur)-len(old), 0)
	removed = max(len(old)-len(cur), 0)
	for key, r := range cur {
		if o, ok := old[key]; ok && !rowsEqual(o, r) {
			updated++
		}
	}
	return added, removed, updated
}

type columnStat struct {
	name    string
	dtype   string
	nulls   int
	trues   int
	falses  int
	unique  bool
	updated int
}

func columnStats(old, cur index) []columnStat {
	stats := make([]columnStat, len(columns))
	for i, c := range columns {
		s := columnStat{name: c.name, dtype: c.dtype}
		distinct := make(map[any]bool, len(cur))
		for key, r := range cur {
			v := c.value(r)
			if o, ok := old[key]; ok && i > 0 && c.value(o) != v {
				s.updated++
			}
			switch v {
			case nil:
				s.nulls++
				continue
			case true:
				s.trues++
			case false:
				s.falses++
			}
			distinct[v] = true
		}
		s.unique = len(distinct) == len(cur)-s.nulls
		stats[i] = s
	}
	return stats
}

func logColumnStats(old, cur index) {
	for _, s := range columnStats(old, cur) {
		attrs := []any{"name", s.name, "dtype", s.dtype, "nulls", s.nulls, "unique", s.unique, "updated", s.updated}
		if s.dtype == "bool" {
			attrs = append(attrs, "true", s.trues, "false", s.falses)
		}
		slog.Info("column stats", attrs...)
	}
}

type config struct {
	output        string
	indexURL      string
	token         string
	discoverURL   string
	metadataURL   string
	serverURL     string
	serverName    string
	watchlistURL  string
	discoverLimit int
	refreshLimit  int
	searchLimit   int
	dryRun        bool
	showVersion   bool
}

func (c config) libraryEnabled() bool {
	return c.serverURL != "" || (c.serverName != "" && c.token != "")
}

func (c config) validate() error {
	if c.output == "" {
		return errors.New("output FILE argument is required")
	}
	if c.token == "" && c.metadataURL == defaultMetadataURL {
		return errors.New("PLEX_TOKEN is required unless PLEX_METADATA_URL points at an authenticating proxy")
	}
	return nil
}

func envBool(name string) (bool, error) {
	v := os.Getenv(name)
	if v == "" {
		return false, nil
	}
	b, err := strconv.ParseBool(v)
	if err != nil {
		return false, fmt.Errorf("invalid %s: %q", name, v)
	}
	return b, nil
}

func envInt(name string, fallback int) (int, error) {
	v := os.Getenv(name)
	if v == "" {
		return fallback, nil
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return 0, fmt.Errorf("invalid %s: %q", name, v)
	}
	return n, nil
}

func envString(name, fallback string) string {
	if v := os.Getenv(name); v != "" {
		return v
	}
	return fallback
}

func parseConfig(level *slog.LevelVar) (config, error) {
	cfg := config{
		token:        os.Getenv("PLEX_TOKEN"),
		indexURL:     os.Getenv("PLEX_INDEX_URL"),
		discoverURL:  envString("PLEX_DISCOVER_URL", defaultDiscoverURL),
		metadataURL:  envString("PLEX_METADATA_URL", defaultMetadataURL),
		serverURL:    os.Getenv("PLEX_SERVER_URL"),
		serverName:   os.Getenv("PLEX_SERVER_NAME"),
		watchlistURL: os.Getenv("IMDB_WATCHLIST_URL"),
	}
	var err error
	if cfg.dryRun, err = envBool("DRY_RUN"); err != nil {
		return cfg, err
	}
	if cfg.discoverLimit, err = envInt("DISCOVER_LIMIT", 10); err != nil {
		return cfg, err
	}
	if cfg.refreshLimit, err = envInt("REFRESH_LIMIT", 10); err != nil {
		return cfg, err
	}
	if cfg.searchLimit, err = envInt("SEARCH_LIMIT", defaultSearchLimit); err != nil {
		return cfg, err
	}

	flag.Usage = func() {
		fmt.Fprintf(flag.CommandLine.Output(), "usage: %s [flags] FILE\n", filepath.Base(os.Args[0]))
		flag.PrintDefaults()
	}
	flag.StringVar(&cfg.indexURL, "plex-index-url", cfg.indexURL, "existing Plex index parquet URL or path, defaults to FILE (env PLEX_INDEX_URL)")
	flag.StringVar(&cfg.discoverURL, "plex-discover-url", cfg.discoverURL, "Plex Discover base URL for search (env PLEX_DISCOVER_URL)")
	flag.StringVar(&cfg.metadataURL, "plex-metadata-url", cfg.metadataURL, "Plex metadata provider base URL (env PLEX_METADATA_URL)")
	flag.StringVar(&cfg.serverURL, "plex-server-url", cfg.serverURL, "Plex Media Server base URL, skips device lookup (env PLEX_SERVER_URL)")
	flag.StringVar(&cfg.serverName, "plex-server-name", cfg.serverName, "Plex Media Server device name (env PLEX_SERVER_NAME)")
	flag.StringVar(&cfg.watchlistURL, "imdb-watchlist-url", cfg.watchlistURL, "IMDb watchlist CSV URL or path (env IMDB_WATCHLIST_URL)")
	flag.IntVar(&cfg.discoverLimit, "discover-limit", cfg.discoverLimit, "maximum new rating keys to discover (env DISCOVER_LIMIT)")
	flag.IntVar(&cfg.refreshLimit, "refresh-limit", cfg.refreshLimit, "number of oldest rows to refresh (env REFRESH_LIMIT)")
	flag.IntVar(&cfg.searchLimit, "search-limit", cfg.searchLimit, "maximum random Plex search queries (env SEARCH_LIMIT)")
	flag.BoolVar(&cfg.dryRun, "dry-run", cfg.dryRun, "skip writing the index (env DRY_RUN)")
	flag.BoolVar(&cfg.dryRun, "n", cfg.dryRun, "shorthand for -dry-run")
	verbose := flag.Bool("v", false, "enable verbose logging")
	flag.BoolVar(&cfg.showVersion, "version", false, "print version information")
	flag.BoolVar(&cfg.showVersion, "V", false, "shorthand for -version")
	flag.Parse()

	if cfg.showVersion {
		return cfg, nil
	}

	if *verbose {
		level.Set(slog.LevelDebug)
	}
	switch flag.NArg() {
	case 0:
	case 1:
		cfg.output = flag.Arg(0)
	default:
		return cfg, fmt.Errorf("unexpected arguments: %s", strings.Join(flag.Args()[1:], " "))
	}
	if cfg.indexURL == "" {
		cfg.indexURL = cfg.output
	}
	return cfg, cfg.validate()
}

func run(ctx context.Context, level *slog.LevelVar) error {
	start := time.Now()
	cfg, err := parseConfig(level)
	if err != nil {
		return err
	}
	if cfg.showVersion {
		fmt.Println(Version)
		return nil
	}
	slog.Info("starting index update",
		"version", Version,
		"output", cfg.output,
		"plex_index", cfg.indexURL,
		"plex_discover_url", cfg.discoverURL,
		"plex_metadata_url", cfg.metadataURL,
		"plex_server_url", cfg.serverURL,
		"plex_server_name", cfg.serverName,
		"plex_token", cfg.token != "",
		"imdb_watchlist", cfg.watchlistURL,
		"discover_limit", cfg.discoverLimit,
		"refresh_limit", cfg.refreshLimit,
		"search_limit", cfg.searchLimit,
		"dry_run", cfg.dryRun,
	)

	old, err := loadIndex(ctx, cfg.indexURL)
	if err != nil {
		return err
	}
	idx := maps.Clone(old)
	p := plex{discoverURL: cfg.discoverURL, metadataURL: cfg.metadataURL, serverURL: cfg.serverURL, serverName: cfg.serverName, token: cfg.token}
	if err := p.discoverKeys(ctx, cfg, idx); err != nil {
		return err
	}
	if err := p.backfill(ctx, idx, cfg.refreshLimit); err != nil {
		return err
	}
	if len(idx) < len(old) {
		return fmt.Errorf("new index has %d rows, fewer than the original %d", len(idx), len(old))
	}

	var size int64
	if cfg.dryRun {
		slog.Info("dry run, skipping write", "path", cfg.output)
	} else {
		if size, err = writeIndex(cfg.output, idx); err != nil {
			return fmt.Errorf("write Plex index: %w", err)
		}
		slog.Info("wrote Plex index", "path", cfg.output, "rows", len(idx), "bytes", size)
	}

	logColumnStats(old, idx)
	added, removed, updated := changeSummary(old, idx)
	slog.Info("finished index update",
		"rows", len(idx),
		"added", added,
		"removed", removed,
		"updated", updated,
		"duration", since(start),
	)
	return nil
}

func main() {
	level := new(slog.LevelVar)
	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: level})))
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	err := run(ctx, level)
	stop()
	if err != nil {
		slog.Error("index update failed", "err", err)
		os.Exit(1)
	}
}
