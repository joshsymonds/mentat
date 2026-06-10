package backend_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/Veraticus/mentat/internal/backend"
	"github.com/Veraticus/mentat/internal/streamjson"
)

// TestClaudeCodeLiveSmoke runs one real haiku turn against a real claude
// binary. It is the protocol-drift gate for claude version bumps: skipped
// unless MENTAT_CLAUDE_BIN is set, because it spends real tokens (~$0.01)
// and needs subscription auth.
func TestClaudeCodeLiveSmoke(t *testing.T) {
	bin := os.Getenv("MENTAT_CLAUDE_BIN")
	if bin == "" {
		t.Skip("set MENTAT_CLAUDE_BIN to a claude binary to run the live smoke test")
	}

	recordDir := t.TempDir()
	cc, err := backend.NewClaudeCode(backend.ClaudeCodeConfig{
		Bin:          bin,
		Model:        "claude-haiku-4-5",
		RecordDir:    recordDir,
		MaxBudgetUSD: 0.25,
	})
	require.NoError(t, err)
	defer func() { require.NoError(t, cc.Close()) }()

	events, err := collectTurn(t, cc, backend.Turn{
		SessionID: "live-smoke",
		Text:      "Reply with exactly the word PONG and nothing else.",
	})
	require.NoError(t, err)

	require.Equal(t, []string{"PONG"}, doneTexts(events))
	var streamed string
	for _, ev := range ofKind(events, backend.KindTextDelta) {
		streamed += ev.TextDelta
	}
	require.Equal(t, "PONG", streamed, "partial-message deltas must stream live")

	for _, ev := range events {
		require.NotEqual(t, backend.KindProtocolDrift, ev.Kind,
			"live claude emitted an event type this build does not recognize: %s", ev.Raw)
	}

	recordings, err := filepath.Glob(filepath.Join(recordDir, "*.ndjson"))
	require.NoError(t, err)
	require.Len(t, recordings, 1)
	recording, err := os.Open(recordings[0])
	require.NoError(t, err)
	defer recording.Close()
	lines, err := streamjson.ReadAll(recording)
	require.NoError(t, err, "the live recording must round-trip through the parser")
	require.NotEmpty(t, lines)
}
