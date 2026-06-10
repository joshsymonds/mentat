package api_test

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"iter"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/Veraticus/mentat/internal/api"
	"github.com/Veraticus/mentat/internal/backend"
)

func newCassetteServer(t *testing.T, cassette string) *httptest.Server {
	t.Helper()
	bk, err := backend.NewCassette(
		filepath.Join("..", "..", "testdata", "cassettes", cassette))
	require.NoError(t, err)
	return newTestServer(t, bk)
}

func newTestServer(t *testing.T, bk backend.Backend) *httptest.Server {
	t.Helper()
	logger := slog.New(slog.DiscardHandler)
	server := httptest.NewServer(api.NewServer(bk, logger).Handler())
	t.Cleanup(server.Close)
	return server
}

type wireLine struct {
	Kind    string `json:"kind"`
	Text    string `json:"text"`
	Tool    string `json:"tool"`
	Message string `json:"message"`
	Done    *struct {
		Text      string  `json:"text"`
		IsError   bool    `json:"is_error"`
		SessionID string  `json:"session_id"`
		CostUSD   float64 `json:"cost_usd"`
	} `json:"done"`
}

func postTurn(t *testing.T, server *httptest.Server, sessionID, text string) (*http.Response, []wireLine) {
	t.Helper()
	body := strings.NewReader(`{"session_id":"` + sessionID + `","text":"` + text + `"}`)
	req, err := http.NewRequestWithContext(t.Context(), http.MethodPost,
		server.URL+"/v1/conversation", body)
	require.NoError(t, err)

	resp, err := server.Client().Do(req)
	require.NoError(t, err)
	t.Cleanup(func() { resp.Body.Close() })
	if resp.StatusCode != http.StatusOK {
		return resp, nil
	}

	var lines []wireLine
	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		var line wireLine
		require.NoError(t, json.Unmarshal(scanner.Bytes(), &line), "every line must be valid JSON")
		lines = append(lines, line)
	}
	require.NoError(t, scanner.Err())
	return resp, lines
}

func TestConversationStreamsNDJSON(t *testing.T) {
	t.Parallel()
	server := newCassetteServer(t, "simple_turn.ndjson")

	resp, lines := postTurn(t, server, "kitchen", "ping")
	require.Equal(t, http.StatusOK, resp.StatusCode)
	require.Equal(t, "application/x-ndjson", resp.Header.Get("Content-Type"))

	var streamed string
	for _, line := range lines {
		if line.Kind == "text_delta" {
			streamed += line.Text
		}
	}
	require.Equal(t, "PONG", streamed)

	last := lines[len(lines)-1]
	require.Equal(t, "done", last.Kind)
	require.NotNil(t, last.Done)
	require.Equal(t, "PONG", last.Done.Text)
	require.False(t, last.Done.IsError)
	require.Positive(t, last.Done.CostUSD)
}

func TestConversationMultiTurnSession(t *testing.T) {
	t.Parallel()
	server := newCassetteServer(t, "multi_turn.ndjson")

	for _, want := range []string{"OK", "TURBOLIFT"} {
		resp, lines := postTurn(t, server, "study", "next")
		require.Equal(t, http.StatusOK, resp.StatusCode)
		last := lines[len(lines)-1]
		require.Equal(t, "done", last.Kind)
		require.Equal(t, want, last.Done.Text)
	}
}

func TestConversationBackendRefusalIsAnHTTPError(t *testing.T) {
	t.Parallel()
	server := newCassetteServer(t, "simple_turn.ndjson")

	resp, _ := postTurn(t, server, "kitchen", "ping")
	require.Equal(t, http.StatusOK, resp.StatusCode)

	resp, _ = postTurn(t, server, "kitchen", "one too many")
	require.Equal(t, http.StatusBadGateway, resp.StatusCode,
		"a backend that refuses the turn outright is an upstream failure")
}

func TestConversationRejectsBadRequests(t *testing.T) {
	t.Parallel()
	server := newCassetteServer(t, "simple_turn.ndjson")

	for name, body := range map[string]string{
		"missing session_id": `{"text":"hi"}`,
		"missing text":       `{"session_id":"s"}`,
		"invalid json":       `{"session_id":`,
	} {
		req, err := http.NewRequestWithContext(t.Context(), http.MethodPost,
			server.URL+"/v1/conversation", strings.NewReader(body))
		require.NoError(t, err)
		resp, err := server.Client().Do(req)
		require.NoError(t, err)
		resp.Body.Close()
		require.Equal(t, http.StatusBadRequest, resp.StatusCode, name)
	}
}

func TestHealthz(t *testing.T) {
	t.Parallel()
	server := newCassetteServer(t, "simple_turn.ndjson")

	req, err := http.NewRequestWithContext(t.Context(), http.MethodGet,
		server.URL+"/healthz", nil)
	require.NoError(t, err)
	resp, err := server.Client().Do(req)
	require.NoError(t, err)
	defer resp.Body.Close()
	require.Equal(t, http.StatusOK, resp.StatusCode)
}

// stubBackend records CloseSession calls and can hold turns open so tests
// control exactly when a session is mid-turn.
type stubBackend struct {
	mu      sync.Mutex
	closed  []string
	started chan string
	release chan struct{}
}

func (s *stubBackend) Converse(_ context.Context, turn backend.Turn) (iter.Seq2[backend.Event, error], error) {
	return func(yield func(backend.Event, error) bool) {
		if s.started != nil {
			s.started <- turn.SessionID
		}
		if s.release != nil {
			<-s.release
		}
		yield(backend.Event{
			Kind: backend.KindDone,
			Done: &backend.Result{Text: "ok", SessionID: turn.SessionID},
		}, nil)
	}, nil
}

func (s *stubBackend) CloseSession(sessionID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.closed = append(s.closed, sessionID)
	return nil
}

func (s *stubBackend) closedSessions() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]string(nil), s.closed...)
}

func TestExpireIdleClosesIdleSessions(t *testing.T) {
	t.Parallel()
	stub := &stubBackend{}
	logger := slog.New(slog.DiscardHandler)
	server := api.NewServer(stub, logger)
	ts := httptest.NewServer(server.Handler())
	t.Cleanup(ts.Close)

	resp, err := ts.Client().Post(ts.URL+"/v1/conversation", "application/json",
		strings.NewReader(`{"session_id":"idle-one","text":"hi"}`))
	require.NoError(t, err)
	_, err = io.ReadAll(resp.Body)
	require.NoError(t, err)
	resp.Body.Close()

	expired := server.ExpireIdle(0)
	require.Equal(t, []string{"idle-one"}, expired)
	require.Equal(t, []string{"idle-one"}, stub.closedSessions())

	require.Empty(t, server.ExpireIdle(0), "an expired session is forgotten, not re-expired")
}

func TestExpireIdleSparesActiveTurns(t *testing.T) {
	t.Parallel()
	stub := &stubBackend{
		started: make(chan string, 1),
		release: make(chan struct{}),
	}
	logger := slog.New(slog.DiscardHandler)
	server := api.NewServer(stub, logger)
	ts := httptest.NewServer(server.Handler())
	t.Cleanup(ts.Close)

	turnDone := make(chan error, 1)
	go func() {
		resp, err := ts.Client().Post(ts.URL+"/v1/conversation", "application/json",
			strings.NewReader(`{"session_id":"busy","text":"hi"}`))
		if err == nil {
			_, err = io.ReadAll(resp.Body)
			resp.Body.Close()
		}
		turnDone <- err
	}()

	require.Equal(t, "busy", <-stub.started, "the turn is now mid-stream")
	require.Empty(t, server.ExpireIdle(0), "a session with an active turn must never be expired")

	close(stub.release)
	require.NoError(t, <-turnDone)
	require.Equal(t, []string{"busy"}, server.ExpireIdle(0),
		"once the turn completes the session is expirable")
}
