// Command permd is a spike-grade MCP permission server for Claude Code's
// --permission-prompt-tool. It approves every tool call except those whose
// input contains the marker string "deny-me", which it denies with an
// explanation. Decisions are logged to stderr (which Claude Code captures in
// its own MCP logs); in-stream evidence of consultation is the executed or
// denied tool_result in the session transcript.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"strings"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// approveInput is deliberately an open map: Claude Code's permission-prompt
// call carries tool_name and an object-valued input, plus fields that vary by
// CLI version. A struct schema with additionalProperties=false rejects them.
type approveInput map[string]any

// decision is the JSON contract Claude Code expects back from a
// permission-prompt tool, serialized as text content.
type decision struct {
	Behavior string `json:"behavior"`
	//nolint:tagliatelle // Claude Code's permission contract requires camelCase here.
	UpdatedInput json.RawMessage `json:"updatedInput,omitempty"`
	Message      string          `json:"message,omitempty"`
}

func main() {
	logger := slog.New(slog.NewTextHandler(os.Stderr, nil))

	server := mcp.NewServer(&mcp.Implementation{Name: "permd", Version: "0.0.1"}, nil)
	mcp.AddTool(server,
		&mcp.Tool{Name: "approve", Description: "Decide whether a tool call may proceed"},
		func(ctx context.Context, _ *mcp.CallToolRequest, input approveInput) (*mcp.CallToolResult, any, error) {
			toolName, _ := input["tool_name"].(string)
			toolInput, marshalErr := json.Marshal(input["input"])
			if marshalErr != nil {
				return nil, nil, fmt.Errorf("marshaling tool input: %w", marshalErr)
			}
			verdict := decide(toolName, toolInput)
			logger.LogAttrs(ctx, slog.LevelInfo, "permission decision",
				slog.String("tool_name", toolName),
				slog.String("behavior", verdict.Behavior),
				slog.String("input", string(toolInput)),
			)
			payload, err := json.Marshal(verdict)
			if err != nil {
				return nil, nil, fmt.Errorf("marshaling permission verdict: %w", err)
			}
			return &mcp.CallToolResult{
				Content: []mcp.Content{&mcp.TextContent{Text: string(payload)}},
			}, nil, nil
		})

	if err := server.Run(context.Background(), &mcp.StdioTransport{}); err != nil {
		logger.Error("permd exited", "error", err)
		os.Exit(1)
	}
}

func decide(_ string, toolInput json.RawMessage) decision {
	if strings.Contains(string(toolInput), "deny-me") {
		return decision{
			Behavior: "deny",
			Message:  "permd policy: inputs containing 'deny-me' are forbidden; do not retry this command",
		}
	}
	return decision{Behavior: "allow", UpdatedInput: toolInput}
}
