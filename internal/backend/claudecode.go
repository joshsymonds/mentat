package backend

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"

	"github.com/Veraticus/mentat/internal/streamjson"
)

// shutdownGrace bounds how long Close waits for a child to exit after its
// stdin closes before killing it.
const shutdownGrace = 5 * time.Second

// eventBuffer is the per-session channel capacity between the reader
// goroutine and the turn consumer.
const eventBuffer = 256

// ClaudeCodeConfig configures the live Claude Code backend.
type ClaudeCodeConfig struct {
	// Bin is the absolute path to the claude binary. Required; there is
	// deliberately no PATH fallback — the deploy pins the binary.
	Bin string
	// Model selects the session model (e.g. "claude-haiku-4-5").
	Model string
	// Effort sets the session effort level (low|medium|high|xhigh|max).
	Effort string
	// SystemPrompt replaces the CLI's system prompt when set.
	SystemPrompt string
	// AddDirs grants the session access to additional directories
	// (the memory directory rides in here).
	AddDirs []string
	// MCPConfig is inline JSON or a path for --mcp-config.
	MCPConfig string
	// PermissionPromptTool names the MCP permission tool
	// (mcp__<server>__<tool>) consulted for gated tool calls.
	PermissionPromptTool string
	// AllowedTools and DisallowedTools restrict the CLI's tool surface.
	// The isolated child still carries the full built-in toolset, so the
	// daemon must choose a policy here.
	AllowedTools    []string
	DisallowedTools []string
	// MaxBudgetUSD caps a session's spend when positive.
	MaxBudgetUSD float64
	// RecordDir, when set, appends each session's raw NDJSON transcript to
	// <RecordDir>/<session-uuid>.ndjson. Recordings are future cassettes.
	RecordDir string
}

// ClaudeCode is a Backend supervising one persistent claude CLI child
// process per session, speaking the stream-json protocol over stdio.
type ClaudeCode struct {
	config   ClaudeCodeConfig
	mu       sync.Mutex
	sessions map[string]*session
}

// NewClaudeCode validates config and returns a live backend. No process is
// spawned until the first turn of a session arrives.
func NewClaudeCode(config ClaudeCodeConfig) (*ClaudeCode, error) {
	if config.Bin == "" {
		return nil, errors.New("claudecode: Bin is required (no PATH fallback)")
	}
	return &ClaudeCode{config: config, sessions: make(map[string]*session)}, nil
}

// Converse sends one user turn into the session's child process and streams
// the turn's events. Turns within a session are serialized; the returned
// stream must be consumed (or the range broken) to release the turn.
func (b *ClaudeCode) Converse(ctx context.Context, turn Turn) (iter.Seq2[Event, error], error) {
	if turn.SessionID == "" {
		return nil, errors.New("claudecode: turn requires a SessionID")
	}
	// The session deliberately outlives this turn's ctx (it owns its own
	// lifecycle context, canceled by Close), so the spawn path does not
	// inherit the request context.
	sess, err := b.sessionFor(turn.SessionID) //nolint:contextcheck // See above.
	if err != nil {
		return nil, err
	}

	sess.turnMu.Lock()
	if sendErr := sess.send(turn.Text); sendErr != nil {
		sess.turnMu.Unlock()
		return nil, sendErr
	}
	return func(yield func(Event, error) bool) {
		defer sess.turnMu.Unlock()
		streamTurn(ctx, sess, yield)
	}, nil
}

// CloseSession shuts down a session's child process while keeping its
// conversation identity: the next turn with the same SessionID respawns
// with --resume, restoring context. Unknown or already-dead sessions are
// harmless no-ops.
func (b *ClaudeCode) CloseSession(sessionID string) error {
	b.mu.Lock()
	sess := b.sessions[sessionID]
	b.mu.Unlock()
	if sess == nil {
		return nil
	}
	return sess.shutdown()
}

// Close shuts down every session: stdin closes, children get shutdownGrace
// to exit, stragglers are killed.
func (b *ClaudeCode) Close() error {
	b.mu.Lock()
	sessions := b.sessions
	b.sessions = make(map[string]*session)
	b.mu.Unlock()

	errs := make([]error, 0, len(sessions))
	for _, sess := range sessions {
		errs = append(errs, sess.shutdown())
	}
	return errors.Join(errs...)
}

// sessionFor returns the live session for id, spawning or respawning as
// needed. A dead session respawns with --resume so the CLI restores the
// conversation context (proven in docs/protocol.md scenario d).
func (b *ClaudeCode) sessionFor(id string) (*session, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	existing := b.sessions[id]
	if existing != nil && !existing.dead.Load() {
		return existing, nil
	}
	resumeUUID := ""
	if existing != nil {
		resumeUUID = existing.uuid
	}
	sess, err := b.startSession(resumeUUID)
	if err != nil {
		return nil, err
	}
	b.sessions[id] = sess
	return sess, nil
}

// startSession spawns a child whose lifecycle context the session owns: the
// process must outlive any single turn's request context, so cancellation
// comes from shutdown(), not from the first Converse's ctx.
func (b *ClaudeCode) startSession(resumeUUID string) (*session, error) {
	sessionUUID := resumeUUID
	if sessionUUID == "" {
		sessionUUID = uuid.NewString()
	}

	lifecycle, cancel := context.WithCancel(context.Background())
	//nolint:gosec // Bin and args come from operator config, not user input.
	cmd := exec.CommandContext(lifecycle, b.config.Bin, b.buildArgs(sessionUUID, resumeUUID != "")...)
	cmd.Env = childEnv()
	cmd.WaitDelay = shutdownGrace

	stdin, err := cmd.StdinPipe()
	if err != nil {
		cancel()
		return nil, fmt.Errorf("claudecode: opening stdin: %w", err)
	}
	// Graceful shutdown: cancellation closes stdin (the CLI's exit signal);
	// WaitDelay kills the child if it lingers past the grace period.
	cmd.Cancel = func() error {
		if closeErr := stdin.Close(); closeErr != nil {
			return fmt.Errorf("closing session stdin: %w", closeErr)
		}
		return nil
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		cancel()
		return nil, fmt.Errorf("claudecode: opening stdout: %w", err)
	}

	var transcript io.Reader = stdout
	var recorder *os.File
	if b.config.RecordDir != "" {
		path := filepath.Join(b.config.RecordDir, sessionUUID+".ndjson")
		//nolint:gosec // RecordDir is operator config; the filename is a daemon-generated UUID.
		recorder, err = os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o600)
		if err != nil {
			cancel()
			return nil, fmt.Errorf("claudecode: opening recording: %w", err)
		}
		transcript = io.TeeReader(stdout, recorder)
	}

	if startErr := cmd.Start(); startErr != nil {
		cancel()
		if recorder != nil {
			_ = recorder.Close()
		}
		return nil, fmt.Errorf("claudecode: starting %s: %w", b.config.Bin, startErr)
	}

	sess := &session{
		uuid:       sessionUUID,
		cmd:        cmd,
		stdin:      stdin,
		cancel:     cancel,
		events:     make(chan turnEvent, eventBuffer),
		readerDone: make(chan struct{}),
		translator: NewTranslator(),
	}
	go sess.readLoop(transcript, recorder)
	return sess, nil
}

// buildArgs assembles the invocation contract from docs/protocol.md. The
// isolation flags are unconditional: a bare child inherits the operator's
// interactive Claude Code configuration, which must never drive a daemon.
func (b *ClaudeCode) buildArgs(sessionUUID string, resume bool) []string {
	args := []string{
		"-p",
		"--input-format", "stream-json",
		"--output-format", "stream-json",
		"--verbose",
		"--include-partial-messages",
		"--setting-sources", "",
		"--strict-mcp-config",
	}
	if resume {
		args = append(args, "--resume", sessionUUID)
	} else {
		args = append(args, "--session-id", sessionUUID)
	}
	if b.config.Model != "" {
		args = append(args, "--model", b.config.Model)
	}
	if b.config.Effort != "" {
		args = append(args, "--effort", b.config.Effort)
	}
	if b.config.SystemPrompt != "" {
		args = append(args, "--system-prompt", b.config.SystemPrompt)
	}
	for _, dir := range b.config.AddDirs {
		args = append(args, "--add-dir", dir)
	}
	if b.config.MCPConfig != "" {
		args = append(args, "--mcp-config", b.config.MCPConfig)
	}
	if b.config.PermissionPromptTool != "" {
		args = append(args, "--permission-prompt-tool", b.config.PermissionPromptTool)
	}
	if len(b.config.AllowedTools) > 0 {
		args = append(args, "--allowedTools", strings.Join(b.config.AllowedTools, ","))
	}
	if len(b.config.DisallowedTools) > 0 {
		args = append(args, "--disallowedTools", strings.Join(b.config.DisallowedTools, ","))
	}
	if b.config.MaxBudgetUSD > 0 {
		args = append(args, "--max-budget-usd", strconv.FormatFloat(b.config.MaxBudgetUSD, 'f', -1, 64))
	}
	return args
}

// childEnv passes the daemon's environment through (auth tokens live there)
// minus Claude Code's own nesting markers, which would make the child
// believe it runs inside an interactive session.
func childEnv() []string {
	env := os.Environ()
	out := make([]string, 0, len(env))
	for _, entry := range env {
		if strings.HasPrefix(entry, "CLAUDECODE=") || strings.HasPrefix(entry, "CLAUDE_CODE_ENTRYPOINT=") {
			continue
		}
		out = append(out, entry)
	}
	return out
}

// turnEvent is one reader-to-consumer handoff.
type turnEvent struct {
	event Event
	err   error
}

// session is one supervised child process and its conversation identity.
type session struct {
	uuid       string
	cmd        *exec.Cmd
	stdin      io.WriteCloser
	cancel     context.CancelFunc
	turnMu     sync.Mutex
	events     chan turnEvent
	readerDone chan struct{}
	translator *Translator
	dead       atomic.Bool
	waitErr    error
}

// wireUserMessage is the stdin frame for one user turn.
type wireUserMessage struct {
	Type    string          `json:"type"`
	Message wireMessageBody `json:"message"`
}

type wireMessageBody struct {
	Role    string          `json:"role"`
	Content []wireTextBlock `json:"content"`
}

type wireTextBlock struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

func (s *session) send(text string) error {
	payload, err := json.Marshal(wireUserMessage{
		Type: "user",
		Message: wireMessageBody{
			Role:    "user",
			Content: []wireTextBlock{{Type: "text", Text: text}},
		},
	})
	if err != nil {
		return fmt.Errorf("claudecode: encoding turn: %w", err)
	}
	if _, writeErr := s.stdin.Write(append(payload, '\n')); writeErr != nil {
		return fmt.Errorf("claudecode: writing turn: %w", writeErr)
	}
	return nil
}

// readLoop owns the child's stdout for the session's lifetime: it parses,
// translates, and hands events to the current turn's consumer. On stdout
// EOF it reaps the child and closes the events channel, which consumers
// observe as session death.
func (s *session) readLoop(transcript io.Reader, recorder *os.File) {
	for line, err := range streamjson.Lines(transcript) {
		if err != nil {
			s.events <- turnEvent{err: err}
			break
		}
		for _, event := range s.translator.Translate(line) {
			s.events <- turnEvent{event: event}
		}
	}
	if recorder != nil {
		_ = recorder.Close()
	}
	s.waitErr = s.cmd.Wait()
	s.dead.Store(true)
	close(s.events)
	close(s.readerDone)
}

// streamTurn delivers one turn's events to yield, ending at the turn's Done,
// a stream error, context cancellation, or session death.
func streamTurn(ctx context.Context, sess *session, yield func(Event, error) bool) {
	for {
		select {
		case <-ctx.Done():
			yield(Event{}, fmt.Errorf("claudecode: turn interrupted: %w", ctx.Err()))
			return
		case received, ok := <-sess.events:
			if !ok {
				yield(Event{}, fmt.Errorf("claudecode: session ended mid-turn: %w",
					errors.Join(sess.waitErr, errSessionDied)))
				return
			}
			if received.err != nil {
				yield(Event{}, received.err)
				return
			}
			if !yield(received.event, nil) {
				return
			}
			if received.event.Kind == KindDone {
				return
			}
		}
	}
}

// errSessionDied marks a child process exiting before its turn completed.
var errSessionDied = errors.New("child process exited before the turn completed")

// shutdown cancels the session's lifecycle: cmd.Cancel closes stdin (the
// CLI's exit signal) and WaitDelay kills the child if it overstays the
// grace period. A session that died before shutdown returns nil: its death
// was already surfaced to the turn's consumer as a stream error.
func (s *session) shutdown() error {
	alreadyDead := s.dead.Load()
	s.cancel()
	<-s.readerDone

	// Wait reports the lifecycle context's cancellation even when the child
	// exits cleanly in response to it; our own cancel is not a failure.
	if !alreadyDead && s.waitErr != nil && !errors.Is(s.waitErr, context.Canceled) {
		return fmt.Errorf("claudecode: session %s shutdown: %w", s.uuid, s.waitErr)
	}
	return nil
}
