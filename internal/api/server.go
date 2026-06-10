// Package api exposes a Backend as a streaming HTTP conversation surface.
// One POST per user turn; the response streams the turn's events as NDJSON,
// one JSON object per line, flushed as they happen — the transport the HA
// conversation component and any other client consumes.
package api

import (
	"encoding/json"
	"log/slog"
	"net/http"
	"sync"
	"time"

	"github.com/Veraticus/mentat/internal/backend"
)

// Server routes conversation turns to a Backend and tracks per-session
// activity for idle expiry. Authentication is deliberately absent: the
// daemon binds localhost and trusts the deploy's tailnet ingress.
type Server struct {
	backend  backend.Backend
	logger   *slog.Logger
	mu       sync.Mutex
	sessions map[string]*sessionActivity
}

type sessionActivity struct {
	lastActive  time.Time
	activeTurns int
}

// NewServer wires a Backend behind the HTTP surface.
func NewServer(bk backend.Backend, logger *slog.Logger) *Server {
	return &Server{
		backend:  bk,
		logger:   logger,
		sessions: make(map[string]*sessionActivity),
	}
}

// Handler returns the daemon's HTTP routes.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/conversation", s.handleConversation)
	mux.HandleFunc("GET /healthz", s.handleHealthz)
	return mux
}

// ExpireIdle closes sessions whose last activity is older than maxIdle and
// returns their IDs. Sessions with a turn in flight are never expired.
// Expired sessions are forgotten; the backend may still restore their
// context if a new turn arrives later.
func (s *Server) ExpireIdle(maxIdle time.Duration) []string {
	cutoff := time.Now().Add(-maxIdle)

	s.mu.Lock()
	var expired []string
	for id, activity := range s.sessions {
		if activity.activeTurns == 0 && !activity.lastActive.After(cutoff) {
			expired = append(expired, id)
			delete(s.sessions, id)
		}
	}
	s.mu.Unlock()

	for _, id := range expired {
		if err := s.backend.CloseSession(id); err != nil {
			s.logger.Error("closing idle session failed",
				slog.String("session_id", id), slog.Any("error", err))
		}
	}
	return expired
}

type turnRequest struct {
	SessionID string            `json:"session_id"`
	Text      string            `json:"text"`
	Meta      map[string]string `json:"meta,omitempty"`
}

func (s *Server) handleConversation(w http.ResponseWriter, r *http.Request) {
	var req turnRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, `{"error":"invalid JSON body"}`, http.StatusBadRequest)
		return
	}
	if req.SessionID == "" || req.Text == "" {
		http.Error(w, `{"error":"session_id and text are required"}`, http.StatusBadRequest)
		return
	}
	if login := r.Header.Get("Tailscale-User-Login"); login != "" {
		s.logger.InfoContext(r.Context(), "turn received",
			slog.String("session_id", req.SessionID), slog.String("tailscale_user", login))
	}

	s.beginTurn(req.SessionID)
	defer s.endTurn(req.SessionID)

	stream, err := s.backend.Converse(r.Context(), backend.Turn{
		SessionID: req.SessionID,
		Text:      req.Text,
		Meta:      req.Meta,
	})
	if err != nil {
		s.logger.ErrorContext(r.Context(), "backend refused turn",
			slog.String("session_id", req.SessionID), slog.Any("error", err))
		http.Error(w, `{"error":"backend refused the turn"}`, http.StatusBadGateway)
		return
	}

	streamEvents(w, stream)
}

func (s *Server) handleHealthz(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write([]byte(`{"status":"ok"}` + "\n"))
}

func (s *Server) beginTurn(sessionID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	activity := s.sessions[sessionID]
	if activity == nil {
		activity = &sessionActivity{}
		s.sessions[sessionID] = activity
	}
	activity.activeTurns++
	activity.lastActive = time.Now()
}

func (s *Server) endTurn(sessionID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if activity := s.sessions[sessionID]; activity != nil {
		activity.activeTurns--
		activity.lastActive = time.Now()
	}
}
