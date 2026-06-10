package streamjson_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/Veraticus/mentat/internal/streamjson"
)

func loadCassette(t *testing.T, name string) []streamjson.Line {
	t.Helper()
	f, err := os.Open(filepath.Join("..", "..", "testdata", "cassettes", name))
	require.NoError(t, err)
	defer f.Close()

	lines, err := streamjson.ReadAll(f)
	require.NoError(t, err)
	require.NotEmpty(t, lines)
	return lines
}

// assistantTexts collects the text blocks of all assistant messages, in order.
func assistantTexts(lines []streamjson.Line) []string {
	var texts []string
	for _, ln := range lines {
		if ln.Assistant == nil {
			continue
		}
		for _, block := range ln.Assistant.Message.Content {
			if block.Type == "text" {
				texts = append(texts, block.Text)
			}
		}
	}
	return texts
}

func results(lines []streamjson.Line) []*streamjson.Result {
	var out []*streamjson.Result
	for _, ln := range lines {
		if ln.Result != nil {
			out = append(out, ln.Result)
		}
	}
	return out
}

func TestParseCassettesNoUnknownEvents(t *testing.T) {
	t.Parallel()
	cassettes, err := filepath.Glob(filepath.Join("..", "..", "testdata", "cassettes", "*.ndjson"))
	require.NoError(t, err)
	require.NotEmpty(t, cassettes)

	for _, path := range cassettes {
		f, err := os.Open(path)
		require.NoError(t, err)
		lines, err := streamjson.ReadAll(f)
		f.Close()
		require.NoError(t, err, path)
		for _, ln := range lines {
			// Exercise the production predicate directly, so the test and
			// Line.Unknown() can never disagree about what's recognized.
			require.False(t, ln.Unknown(), "unknown event type %q in %s", ln.Type, path)
			require.NotEmpty(t, ln.Raw)
		}
	}
}

func TestSimpleTurn(t *testing.T) {
	t.Parallel()
	lines := loadCassette(t, "simple_turn.ndjson")

	require.Equal(t, "system", lines[0].Type)
	require.Equal(t, "init", lines[0].Subtype)
	init := lines[0].Init
	require.NotNil(t, init)
	require.Equal(t, "claude-haiku-4-5", init.Model)
	require.Empty(t, init.Tools)
	require.Empty(t, init.MCPServers)
	require.Equal(t, "default", init.PermissionMode)
	require.NotEmpty(t, init.SessionID)

	res := results(lines)
	require.Len(t, res, 1)
	require.Equal(t, "success", res[0].Subtype)
	require.False(t, res[0].IsError)
	require.Equal(t, "PONG", res[0].Result)
	require.Equal(t, init.SessionID, res[0].SessionID)
	require.Positive(t, res[0].TotalCostUSD)
	require.Positive(t, res[0].Usage.OutputTokens)

	require.Equal(t, []string{"PONG"}, assistantTexts(lines))

	var deltas int
	for _, ln := range lines {
		if ln.Stream != nil && ln.Stream.Event.Type == "content_block_delta" {
			deltas++
		}
	}
	require.Positive(t, deltas, "expected streaming content_block_delta events")
}

func TestMultiTurnRetainsContext(t *testing.T) {
	t.Parallel()
	lines := loadCassette(t, "multi_turn.ndjson")

	res := results(lines)
	require.Len(t, res, 2, "one result event per queued user turn")
	for _, r := range res {
		require.Equal(t, "success", r.Subtype)
		require.Equal(t, "7f3e4d5c-1111-4222-8333-944455566677", r.SessionID)
	}
	require.Equal(t, []string{"OK", "TURBOLIFT"}, assistantTexts(lines))
}

func TestResumeRetainsContext(t *testing.T) {
	t.Parallel()
	lines := loadCassette(t, "resume.ndjson")

	res := results(lines)
	require.Len(t, res, 1)
	require.Equal(t, "7f3e4d5c-1111-4222-8333-944455566677", res[0].SessionID,
		"resumed session keeps its session ID")
	require.Equal(t, []string{"TURBOLIFT"}, assistantTexts(lines))
}

func TestToolUseEventsVisibleInStream(t *testing.T) {
	t.Parallel()
	lines := loadCassette(t, "tool_use.ndjson")

	var sawToolUse bool
	for _, ln := range lines {
		if ln.Assistant == nil {
			continue
		}
		for _, block := range ln.Assistant.Message.Content {
			if block.Type == "tool_use" {
				sawToolUse = true
				require.Equal(t, "Bash", block.Name)
				require.NotEmpty(t, block.ID)
			}
		}
	}
	require.True(t, sawToolUse, "expected an assistant tool_use block")

	var sawToolResult bool
	for _, ln := range lines {
		if ln.User == nil {
			continue
		}
		for _, block := range ln.User.Message.Content {
			if block.Type == "tool_result" {
				sawToolResult = true
				var content string
				require.NoError(t, json.Unmarshal(block.Content, &content))
				require.Equal(t, "mentat-spike-ok", content)
			}
		}
	}
	require.True(t, sawToolResult, "expected a user tool_result block")
}

func TestEffortLowSkipsThinking(t *testing.T) {
	t.Parallel()
	lines := loadCassette(t, "effort_low_fable.ndjson")

	for _, ln := range lines {
		require.Nil(t, ln.ThinkingTokens, "effort low on an easy prompt should not think")
	}
	res := results(lines)
	require.Len(t, res, 1)
	require.Equal(t, "success", res[0].Subtype)
	require.Equal(t, []string{"3^17"}, assistantTexts(lines))
}

func TestThinkingTokensParsed(t *testing.T) {
	t.Parallel()
	lines := loadCassette(t, "simple_turn.ndjson")

	var thinking int
	for _, ln := range lines {
		if ln.ThinkingTokens != nil {
			thinking++
			require.Positive(t, ln.ThinkingTokens.EstimatedTokens)
		}
	}
	require.Positive(t, thinking, "expected thinking_tokens progress events")
}

func TestPermissionAllowExecutesTool(t *testing.T) {
	t.Parallel()
	lines := loadCassette(t, "permission_allow.ndjson")

	var ranBash bool
	for _, ln := range lines {
		if ln.Assistant == nil {
			continue
		}
		for _, block := range ln.Assistant.Message.Content {
			if block.Type == "tool_use" && block.Name == "Bash" {
				ranBash = true
			}
		}
	}
	require.True(t, ranBash)

	for _, ln := range lines {
		if ln.User == nil {
			continue
		}
		for _, block := range ln.User.Message.Content {
			if block.Type == "tool_result" {
				require.False(t, block.IsError, "permd-approved tool call must succeed")
			}
		}
	}
	res := results(lines)
	require.Len(t, res, 1)
	require.False(t, res[0].IsError)
}

func TestPermissionDenyDeliversPolicyMessage(t *testing.T) {
	t.Parallel()
	lines := loadCassette(t, "permission_deny.ndjson")

	var sawDenial bool
	for _, ln := range lines {
		if ln.User == nil {
			continue
		}
		for _, block := range ln.User.Message.Content {
			if block.Type != "tool_result" || !block.IsError {
				continue
			}
			var content string
			require.NoError(t, json.Unmarshal(block.Content, &content))
			require.Contains(t, content, "permd policy: inputs containing 'deny-me'",
				"the deny message must reach the model verbatim")
			sawDenial = true
		}
	}
	require.True(t, sawDenial, "expected a denied tool_result")
}

func TestUnknownEventTypeIsPreservedNotFatal(t *testing.T) {
	t.Parallel()
	line, err := streamjson.Parse([]byte(`{"type":"frobnicate","mystery":true}`))
	require.NoError(t, err)
	require.Equal(t, "frobnicate", line.Type)
	require.True(t, line.Unknown(), "unrecognized types must be flagged, not dropped")
	require.JSONEq(t, `{"type":"frobnicate","mystery":true}`, string(line.Raw))
}

func TestInvalidJSONIsAnError(t *testing.T) {
	t.Parallel()
	_, err := streamjson.Parse([]byte(`{"type":`))
	require.Error(t, err)
}

func TestMissingTypeIsAnError(t *testing.T) {
	t.Parallel()
	_, err := streamjson.Parse([]byte(`{"subtype":"init"}`))
	require.Error(t, err)
}
