package backend_test

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/Veraticus/mentat/internal/backend"
	"github.com/Veraticus/mentat/internal/streamjson"
)

func translateCassette(t *testing.T, name string) []backend.Event {
	t.Helper()
	f, err := os.Open(filepath.Join("..", "..", "testdata", "cassettes", name))
	require.NoError(t, err)
	defer f.Close()

	lines, err := streamjson.ReadAll(f)
	require.NoError(t, err)

	translator := backend.NewTranslator()
	events := make([]backend.Event, 0, len(lines))
	for _, line := range lines {
		events = append(events, translator.Translate(line)...)
	}
	return events
}

func ofKind(events []backend.Event, kind backend.EventKind) []backend.Event {
	out := make([]backend.Event, 0, len(events))
	for _, ev := range events {
		if ev.Kind == kind {
			out = append(out, ev)
		}
	}
	return out
}

func doneTexts(events []backend.Event) []string {
	dones := ofKind(events, backend.KindDone)
	texts := make([]string, 0, len(dones))
	for _, ev := range dones {
		texts = append(texts, ev.Done.Text)
	}
	return texts
}

func TestTranslateSimpleTurn(t *testing.T) {
	t.Parallel()
	events := translateCassette(t, "simple_turn.ndjson")

	var streamed string
	for _, ev := range ofKind(events, backend.KindTextDelta) {
		streamed += ev.TextDelta
	}
	require.Equal(t, "PONG", streamed, "text deltas must reassemble the reply")

	require.NotEmpty(t, ofKind(events, backend.KindThinking),
		"haiku thought before answering; thinking progress must surface")
	require.NotEmpty(t, ofKind(events, backend.KindThinkingDelta),
		"thinking deltas were streamed and must surface")

	dones := ofKind(events, backend.KindDone)
	require.Len(t, dones, 1)
	done := dones[0].Done
	require.Equal(t, "PONG", done.Text)
	require.False(t, done.IsError)
	require.NotEmpty(t, done.SessionID)
	require.Positive(t, done.CostUSD)
	require.Positive(t, done.Usage.OutputTokens)
}

func TestTranslateMultiTurn(t *testing.T) {
	t.Parallel()
	events := translateCassette(t, "multi_turn.ndjson")
	require.Equal(t, []string{"OK", "TURBOLIFT"}, doneTexts(events),
		"each queued turn yields its own Done in order")
}

func TestTranslateToolUseCorrelatesNames(t *testing.T) {
	t.Parallel()
	events := translateCassette(t, "tool_use.ndjson")

	started := ofKind(events, backend.KindToolUseStarted)
	require.Len(t, started, 1)
	require.Equal(t, "Bash", started[0].Tool.Name)

	results := ofKind(events, backend.KindToolResult)
	require.Len(t, results, 1)
	require.Equal(t, "Bash", results[0].Tool.Name,
		"tool_result carries no name on the wire; the translator must correlate by tool_use_id")
	require.False(t, results[0].Tool.IsError)
	require.Equal(t, "mentat-spike-ok", results[0].Tool.Content)

	var sawStart bool
	for _, ev := range events {
		switch ev.Kind {
		case backend.KindToolUseStarted:
			sawStart = true
		case backend.KindToolResult:
			require.True(t, sawStart, "tool result must come after its tool start")
		default:
		}
	}
}

func TestTranslatePermissionDeny(t *testing.T) {
	t.Parallel()
	events := translateCassette(t, "permission_deny.ndjson")

	var denied bool
	for _, ev := range ofKind(events, backend.KindToolResult) {
		if ev.Tool.IsError {
			denied = true
			require.Contains(t, ev.Tool.Content, "permd policy",
				"the deny message must survive translation for the daemon's error reporting")
		}
	}
	require.True(t, denied, "expected a denied tool result")
}

func TestTranslateEffortLowHasNoThinking(t *testing.T) {
	t.Parallel()
	events := translateCassette(t, "effort_low_fable.ndjson")

	require.Empty(t, ofKind(events, backend.KindThinking))
	require.Empty(t, ofKind(events, backend.KindThinkingDelta))
	require.Equal(t, []string{"3^17"}, doneTexts(events))
}

func TestTranslateUnknownLineBecomesProtocolDrift(t *testing.T) {
	t.Parallel()
	line, err := streamjson.Parse([]byte(`{"type":"frobnicate","mystery":true}`))
	require.NoError(t, err)

	events := backend.NewTranslator().Translate(line)
	require.Len(t, events, 1)
	require.Equal(t, backend.KindProtocolDrift, events[0].Kind)
	require.JSONEq(t, `{"type":"frobnicate","mystery":true}`, string(events[0].Raw))
}

func TestCassetteReplaysOneTurn(t *testing.T) {
	t.Parallel()
	cassette, err := backend.NewCassette(
		filepath.Join("..", "..", "testdata", "cassettes", "simple_turn.ndjson"))
	require.NoError(t, err)

	var bk backend.Backend = cassette
	stream, err := bk.Converse(t.Context(), backend.Turn{SessionID: "s1", Text: "ping"})
	require.NoError(t, err)

	var events []backend.Event
	for ev, streamErr := range stream {
		require.NoError(t, streamErr)
		events = append(events, ev)
	}
	require.Equal(t, []string{"PONG"}, doneTexts(events))
	require.Equal(t, backend.KindDone, events[len(events)-1].Kind,
		"a turn's stream ends at its Done")
}

func TestCassetteReplaysTurnsSequentially(t *testing.T) {
	t.Parallel()
	cassette, err := backend.NewCassette(
		filepath.Join("..", "..", "testdata", "cassettes", "multi_turn.ndjson"))
	require.NoError(t, err)

	for _, want := range []string{"OK", "TURBOLIFT"} {
		stream, convErr := cassette.Converse(t.Context(), backend.Turn{Text: "next"})
		require.NoError(t, convErr)
		var events []backend.Event
		for ev, streamErr := range stream {
			require.NoError(t, streamErr)
			events = append(events, ev)
		}
		require.Equal(t, []string{want}, doneTexts(events))
	}

	_, err = cassette.Converse(t.Context(), backend.Turn{Text: "one too many"})
	require.Error(t, err, "an exhausted cassette must refuse further turns")
}

func TestCassetteHonorsContextCancellation(t *testing.T) {
	t.Parallel()
	cassette, err := backend.NewCassette(
		filepath.Join("..", "..", "testdata", "cassettes", "simple_turn.ndjson"))
	require.NoError(t, err)

	ctx, cancel := context.WithCancel(t.Context())
	stream, err := cassette.Converse(ctx, backend.Turn{Text: "ping"})
	require.NoError(t, err)

	cancel()
	var sawErr bool
	for _, streamErr := range stream {
		if streamErr != nil {
			sawErr = true
			require.ErrorIs(t, streamErr, context.Canceled)
			break
		}
	}
	require.True(t, sawErr, "cancellation must surface as a stream error")
}

func TestCassetteMissingFileIsAnError(t *testing.T) {
	t.Parallel()
	_, err := backend.NewCassette(filepath.Join(t.TempDir(), "absent.ndjson"))
	require.Error(t, err)
}

func TestCassetteCloseSessionIsHarmless(t *testing.T) {
	t.Parallel()
	cassette, err := backend.NewCassette(
		filepath.Join("..", "..", "testdata", "cassettes", "multi_turn.ndjson"))
	require.NoError(t, err)

	require.NoError(t, cassette.CloseSession("anything"))

	stream, err := cassette.Converse(t.Context(), backend.Turn{Text: "still works"})
	require.NoError(t, err)
	var events []backend.Event
	for ev, streamErr := range stream {
		require.NoError(t, streamErr)
		events = append(events, ev)
	}
	require.Equal(t, []string{"OK"}, doneTexts(events),
		"a cassette has no per-session state; CloseSession must not disturb replay")
}
