package backend_test

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"

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
	os.Exit(m.Run())
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
