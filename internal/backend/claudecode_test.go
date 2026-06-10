package backend_test

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/Veraticus/mentat/internal/backend"
	"github.com/Veraticus/mentat/internal/streamjson"
)

// TestMain doubles as the fake claude binary: when MENTAT_FAKE_CLAUDE is
// set, the test executable becomes a protocol-speaking child process that
// replays a recorded cassette turn-by-turn. ClaudeCode tests point Bin at
// os.Args[0], so the supervisor is exercised against real recorded wire
// bytes without a claude binary or network.
func TestMain(m *testing.M) {
	if os.Getenv("MENTAT_FAKE_CLAUDE") != "" {
		fakeClaudeMain()
		return
	}
	if os.Getenv("MENTAT_FAKE_RAW") != "" {
		fakeRawMain()
		return
	}
	os.Exit(m.Run())
}

// fakeRawMain dumps a file's bytes to stdout verbatim on the first stdin
// turn, then stays alive draining stdin until the parent closes it. It is
// used to reproduce a child that keeps streaming after the parser chokes:
// the process exits only when stdin closes (the supervisor's cancel path).
func fakeRawMain() {
	bufio.NewScanner(os.Stdin).Scan()
	data, err := os.ReadFile(os.Getenv("MENTAT_FAKE_RAW"))
	if err != nil {
		os.Exit(2)
	}
	os.Stdout.Write(data)
	_, _ = io.Copy(io.Discard, os.Stdin)
}

func fakeClaudeMain() {
	if argsFile := os.Getenv("MENTAT_FAKE_ARGS_FILE"); argsFile != "" {
		f, err := os.OpenFile(argsFile, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o600)
		if err == nil {
			fmt.Fprintln(f, strings.Join(os.Args[1:], " "))
			f.Close()
		}
	}

	data, err := os.ReadFile(os.Getenv("MENTAT_FAKE_CLAUDE"))
	if err != nil {
		os.Exit(2)
	}
	turns := splitRawTurns(strings.Split(strings.TrimSpace(string(data)), "\n"))

	if marker := os.Getenv("MENTAT_FAKE_DIE_ONCE"); marker != "" {
		if _, statErr := os.Stat(marker); errors.Is(statErr, fs.ErrNotExist) {
			_ = os.WriteFile(marker, []byte("died"), 0o600)
			dieMidTurn(turns)
			return
		}
	}

	stdin := bufio.NewScanner(os.Stdin)
	turnIdx := 0
	for stdin.Scan() {
		if turnIdx >= len(turns) {
			os.Exit(3)
		}
		for _, line := range turns[turnIdx] {
			fmt.Println(line)
		}
		turnIdx++
	}
}

// dieMidTurn reads one stdin line, replays the turn only up to its first
// assistant line, then exits nonzero — a child crash mid-turn.
func dieMidTurn(turns [][]string) {
	bufio.NewScanner(os.Stdin).Scan()
	for _, line := range turns[0] {
		fmt.Println(line)
		var envelope struct {
			Type string `json:"type"`
		}
		if json.Unmarshal([]byte(line), &envelope) == nil && envelope.Type == "assistant" {
			break
		}
	}
	os.Exit(1)
}

func splitRawTurns(lines []string) [][]string {
	var turns [][]string
	var current []string
	for _, line := range lines {
		if line == "" {
			continue
		}
		current = append(current, line)
		var envelope struct {
			Type string `json:"type"`
		}
		if json.Unmarshal([]byte(line), &envelope) == nil && envelope.Type == "result" {
			turns = append(turns, current)
			current = nil
		}
	}
	if len(current) > 0 {
		turns = append(turns, current)
	}
	return turns
}

func cassettePath(t *testing.T, name string) string {
	t.Helper()
	abs, err := filepath.Abs(filepath.Join("..", "..", "testdata", "cassettes", name))
	require.NoError(t, err)
	return abs
}

func newFakeClaude(t *testing.T, cassette string, config backend.ClaudeCodeConfig) *backend.ClaudeCode {
	t.Helper()
	t.Setenv("MENTAT_FAKE_CLAUDE", cassettePath(t, cassette))
	config.Bin = os.Args[0]
	cc, err := backend.NewClaudeCode(config)
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, cc.Close()) })
	return cc
}

func collectTurn(t *testing.T, cc *backend.ClaudeCode, turn backend.Turn) ([]backend.Event, error) {
	t.Helper()
	stream, err := cc.Converse(t.Context(), turn)
	require.NoError(t, err)

	var events []backend.Event
	for ev, streamErr := range stream {
		if streamErr != nil {
			return events, streamErr
		}
		events = append(events, ev)
	}
	return events, nil
}

func TestClaudeCodeRequiresBin(t *testing.T) {
	t.Parallel()
	_, err := backend.NewClaudeCode(backend.ClaudeCodeConfig{})
	require.Error(t, err)
}

func TestClaudeCodeRequiresSessionID(t *testing.T) {
	t.Setenv("MENTAT_FAKE_CLAUDE", cassettePath(t, "simple_turn.ndjson"))
	cc, err := backend.NewClaudeCode(backend.ClaudeCodeConfig{Bin: os.Args[0]})
	require.NoError(t, err)
	defer cc.Close()

	_, err = cc.Converse(t.Context(), backend.Turn{Text: "no session"})
	require.Error(t, err)
}

func TestClaudeCodeStreamsATurn(t *testing.T) {
	cc := newFakeClaude(t, "simple_turn.ndjson", backend.ClaudeCodeConfig{})

	events, err := collectTurn(t, cc, backend.Turn{SessionID: "kitchen", Text: "ping"})
	require.NoError(t, err)

	var streamed string
	for _, ev := range ofKind(events, backend.KindTextDelta) {
		streamed += ev.TextDelta
	}
	require.Equal(t, "PONG", streamed)
	require.Equal(t, []string{"PONG"}, doneTexts(events))
	require.Equal(t, backend.KindDone, events[len(events)-1].Kind,
		"the stream must end at the turn's Done")
}

func TestClaudeCodeServesQueuedTurnsInOrder(t *testing.T) {
	cc := newFakeClaude(t, "multi_turn.ndjson", backend.ClaudeCodeConfig{})

	for _, want := range []string{"OK", "TURBOLIFT"} {
		events, err := collectTurn(t, cc, backend.Turn{SessionID: "study", Text: "next"})
		require.NoError(t, err)
		require.Equal(t, []string{want}, doneTexts(events))
	}
}

func TestClaudeCodeSessionsAreIsolated(t *testing.T) {
	cc := newFakeClaude(t, "simple_turn.ndjson", backend.ClaudeCodeConfig{})

	for _, sessionID := range []string{"kitchen", "bridge"} {
		events, err := collectTurn(t, cc, backend.Turn{SessionID: sessionID, Text: "ping"})
		require.NoError(t, err)
		require.Equal(t, []string{"PONG"}, doneTexts(events),
			"each session gets its own child process replaying from the top")
	}
}

func TestClaudeCodeChildDeathSurfacesError(t *testing.T) {
	t.Setenv("MENTAT_FAKE_DIE_ONCE", filepath.Join(t.TempDir(), "died"))
	cc := newFakeClaude(t, "simple_turn.ndjson", backend.ClaudeCodeConfig{})

	_, err := collectTurn(t, cc, backend.Turn{SessionID: "engine-room", Text: "ping"})
	require.Error(t, err, "a child dying mid-turn must surface as a stream error, not a silent end")
}

func TestClaudeCodeRespawnsWithResume(t *testing.T) {
	argsFile := filepath.Join(t.TempDir(), "args")
	t.Setenv("MENTAT_FAKE_ARGS_FILE", argsFile)
	t.Setenv("MENTAT_FAKE_DIE_ONCE", filepath.Join(t.TempDir(), "died"))
	cc := newFakeClaude(t, "simple_turn.ndjson", backend.ClaudeCodeConfig{})

	_, err := collectTurn(t, cc, backend.Turn{SessionID: "kitchen", Text: "ping"})
	require.Error(t, err, "first spawn dies mid-turn")

	events, err := collectTurn(t, cc, backend.Turn{SessionID: "kitchen", Text: "ping again"})
	require.NoError(t, err)
	require.Equal(t, []string{"PONG"}, doneTexts(events))

	raw, err := os.ReadFile(argsFile)
	require.NoError(t, err)
	spawns := strings.Split(strings.TrimSpace(string(raw)), "\n")
	require.Len(t, spawns, 2)

	sessionUUID := flagValue(t, spawns[0], "--session-id")
	require.Contains(t, spawns[1], "--resume "+sessionUUID,
		"the respawn must resume the same CLI session")
	require.NotContains(t, spawns[1], "--session-id")
}

func TestClaudeCodeRecordsParseableTranscript(t *testing.T) {
	recordDir := t.TempDir()
	cc := newFakeClaude(t, "simple_turn.ndjson", backend.ClaudeCodeConfig{RecordDir: recordDir})

	_, err := collectTurn(t, cc, backend.Turn{SessionID: "kitchen", Text: "ping"})
	require.NoError(t, err)
	require.NoError(t, cc.Close())

	recordings, err := filepath.Glob(filepath.Join(recordDir, "*.ndjson"))
	require.NoError(t, err)
	require.Len(t, recordings, 1)

	f, err := os.Open(recordings[0])
	require.NoError(t, err)
	defer f.Close()
	lines, err := streamjson.ReadAll(f)
	require.NoError(t, err, "a recording must round-trip through the parser")

	var sawResult bool
	for _, line := range lines {
		if line.Result != nil {
			sawResult = true
		}
	}
	require.True(t, sawResult)
}

func TestClaudeCodeIsolationArgsAlwaysPresent(t *testing.T) {
	argsFile := filepath.Join(t.TempDir(), "args")
	t.Setenv("MENTAT_FAKE_ARGS_FILE", argsFile)
	cc := newFakeClaude(t, "simple_turn.ndjson", backend.ClaudeCodeConfig{
		Model:           "claude-haiku-4-5",
		Effort:          "low",
		AllowedTools:    []string{"Read"},
		DisallowedTools: []string{"Bash", "Write"},
		MaxBudgetUSD:    0.50,
	})

	_, err := collectTurn(t, cc, backend.Turn{SessionID: "kitchen", Text: "ping"})
	require.NoError(t, err)

	raw, err := os.ReadFile(argsFile)
	require.NoError(t, err)
	args := string(raw)
	for _, required := range []string{
		"--input-format stream-json",
		"--output-format stream-json",
		"--include-partial-messages",
		"--setting-sources ",
		"--strict-mcp-config",
		"--model claude-haiku-4-5",
		"--effort low",
		"--allowedTools Read",
		"--disallowedTools Bash,Write",
		"--max-budget-usd 0.5",
	} {
		require.Contains(t, args, required)
	}
}

func TestClaudeCodeCloseSessionResumesOnNextTurn(t *testing.T) {
	argsFile := filepath.Join(t.TempDir(), "args")
	t.Setenv("MENTAT_FAKE_ARGS_FILE", argsFile)
	cc := newFakeClaude(t, "simple_turn.ndjson", backend.ClaudeCodeConfig{})

	events, err := collectTurn(t, cc, backend.Turn{SessionID: "kitchen", Text: "ping"})
	require.NoError(t, err)
	require.Equal(t, []string{"PONG"}, doneTexts(events))

	require.NoError(t, cc.CloseSession("kitchen"))
	require.NoError(t, cc.CloseSession("kitchen"), "closing twice must be harmless")
	require.NoError(t, cc.CloseSession("never-existed"), "closing an unknown session must be harmless")

	events, err = collectTurn(t, cc, backend.Turn{SessionID: "kitchen", Text: "ping again"})
	require.NoError(t, err)
	require.Equal(t, []string{"PONG"}, doneTexts(events))

	raw, err := os.ReadFile(argsFile)
	require.NoError(t, err)
	spawns := strings.Split(strings.TrimSpace(string(raw)), "\n")
	require.Len(t, spawns, 2, "the turn after CloseSession must respawn")
	sessionUUID := flagValue(t, spawns[0], "--session-id")
	require.Contains(t, spawns[1], "--resume "+sessionUUID,
		"an expired session resumes its CLI conversation rather than starting cold")
}

// readSpawns returns each recorded child invocation's argv line.
func readSpawns(t *testing.T, argsFile string) []string {
	t.Helper()
	raw, err := os.ReadFile(argsFile)
	require.NoError(t, err)
	return strings.Split(strings.TrimSpace(string(raw)), "\n")
}

// pickLine returns the first line of simple_turn.ndjson whose JSON "type"
// (and, when nonempty, the inner stream event delta type) matches. It sources
// synthetic fixtures from genuinely recorded wire bytes rather than invented
// shapes.
func pickLine(t *testing.T, eventType, deltaType string) string {
	t.Helper()
	data, err := os.ReadFile(cassettePath(t, "simple_turn.ndjson"))
	require.NoError(t, err)
	for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
		var probe struct {
			Type  string `json:"type"`
			Event struct {
				Delta struct {
					Type string `json:"type"`
				} `json:"delta"`
			} `json:"event"`
		}
		if json.Unmarshal([]byte(line), &probe) != nil || probe.Type != eventType {
			continue
		}
		if deltaType == "" || probe.Event.Delta.Type == deltaType {
			return line
		}
	}
	t.Fatalf("no %s/%s line in simple_turn.ndjson", eventType, deltaType)
	return ""
}

// writeBigCassette builds a one-turn transcript with eventLines content
// deltas — enough to overflow the supervisor's event buffer — bracketed by
// the real init and result lines from simple_turn.
func writeBigCassette(t *testing.T, eventLines int) string {
	t.Helper()
	initLine := pickLine(t, "system", "")
	deltaLine := pickLine(t, "stream_event", "text_delta")
	resultLine := pickLine(t, "result", "")

	lines := make([]string, 0, eventLines+2)
	lines = append(lines, initLine)
	for range eventLines {
		lines = append(lines, deltaLine)
	}
	lines = append(lines, resultLine)

	path := filepath.Join(t.TempDir(), "big.ndjson")
	require.NoError(t, os.WriteFile(path, []byte(strings.Join(lines, "\n")+"\n"), 0o600))
	return path
}

func TestClaudeCodeAbandonedTurnRespawnsSession(t *testing.T) {
	argsFile := filepath.Join(t.TempDir(), "args")
	t.Setenv("MENTAT_FAKE_ARGS_FILE", argsFile)
	cc := newFakeClaude(t, "multi_turn.ndjson", backend.ClaudeCodeConfig{})

	stream, err := cc.Converse(t.Context(), backend.Turn{SessionID: "study", Text: "first"})
	require.NoError(t, err)
	for _, streamErr := range stream {
		require.NoError(t, streamErr)
		break // abandon the turn after its first event
	}

	events, err := collectTurn(t, cc, backend.Turn{SessionID: "study", Text: "second"})
	require.NoError(t, err)
	require.NotEmpty(t, doneTexts(events))

	spawns := readSpawns(t, argsFile)
	require.Len(t, spawns, 2, "abandoning a turn must poison the session so the next turn respawns")
	sessionUUID := flagValue(t, spawns[0], "--session-id")
	require.Contains(t, spawns[1], "--resume "+sessionUUID,
		"the respawn after abandonment must resume the same CLI conversation")
}

func TestClaudeCodeAbandonedLargeTurnDoesNotDeadlock(t *testing.T) {
	t.Setenv("MENTAT_FAKE_CLAUDE", writeBigCassette(t, 300))
	cc, err := backend.NewClaudeCode(backend.ClaudeCodeConfig{Bin: os.Args[0]})
	require.NoError(t, err)
	t.Cleanup(func() { _ = cc.Close() })

	ctx, cancel := context.WithCancel(context.Background())
	stream, err := cc.Converse(ctx, backend.Turn{SessionID: "big", Text: "go"})
	require.NoError(t, err)
	for _, streamErr := range stream {
		require.NoError(t, streamErr)
		cancel() // abandon with hundreds of events still queued behind us
		break
	}

	done := make(chan error, 1)
	go func() { done <- cc.CloseSession("big") }()
	select {
	case closeErr := <-done:
		require.NoError(t, closeErr)
	case <-time.After(10 * time.Second):
		t.Fatal("CloseSession deadlocked: reader wedged on a full event buffer behind an abandoned turn")
	}
}

func TestClaudeCodeParseErrorDoesNotWedgeShutdown(t *testing.T) {
	initLine := pickLine(t, "system", "")
	rawFile := filepath.Join(t.TempDir(), "garbage.ndjson")
	require.NoError(t, os.WriteFile(rawFile, []byte(initLine+"\nthis is not json\n"), 0o600))
	t.Setenv("MENTAT_FAKE_RAW", rawFile)

	cc, err := backend.NewClaudeCode(backend.ClaudeCodeConfig{Bin: os.Args[0]})
	require.NoError(t, err)

	_, err = collectTurn(t, cc, backend.Turn{SessionID: "noise", Text: "go"})
	require.Error(t, err, "a malformed wire line must surface as a stream error")

	done := make(chan error, 1)
	go func() { done <- cc.Close() }()
	select {
	case closeErr := <-done:
		require.NoError(t, closeErr)
	case <-time.After(10 * time.Second):
		t.Fatal("Close deadlocked: parse-error path left the child unreaped on a blocking Wait")
	}
}

func flagValue(t *testing.T, args, flag string) string {
	t.Helper()
	fields := strings.Fields(args)
	for i, field := range fields {
		if field == flag && i+1 < len(fields) {
			return fields[i+1]
		}
	}
	t.Fatalf("flag %s not found in %q", flag, args)
	return ""
}
