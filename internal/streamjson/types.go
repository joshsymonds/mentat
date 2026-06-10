// Package streamjson parses the Claude Code CLI's stream-json protocol:
// newline-delimited JSON events emitted on stdout when the CLI runs with
// --input-format stream-json --output-format stream-json. Field names mirror
// the wire format exactly; the protocol is owned by the CLI, not by us.
package streamjson

import "encoding/json"

// Line is one parsed NDJSON line. Type and Subtype are always populated;
// exactly one payload pointer is non-nil for recognized payload-bearing
// events. Raw always holds the original bytes so callers can log or
// re-record lines we do not yet model.
type Line struct {
	Type           string
	Subtype        string
	SessionID      string
	Init           *Init
	ThinkingTokens *ThinkingTokens
	Assistant      *MessageEvent
	User           *MessageEvent
	Stream         *StreamEvent
	Result         *Result
	RateLimit      *RateLimit
	Raw            []byte
}

// Unknown reports whether the line's type is outside the event families this
// package recognizes. Unknown lines signal protocol drift after a claude
// binary upgrade and should be surfaced, never silently dropped.
func (l Line) Unknown() bool {
	switch l.Type {
	case "system", "assistant", "user", "stream_event", "result",
		"rate_limit_event", "control_request", "control_response", "control_cancel_request":
		return false
	default:
		return true
	}
}

// Init is the system/init event: the first event of every session,
// describing the session's effective configuration.
type Init struct {
	Model          string      `json:"model"`
	PermissionMode string      `json:"permissionMode"`
	Tools          []string    `json:"tools"`
	MCPServers     []MCPServer `json:"mcp_servers"`
	SessionID      string      `json:"session_id"`
	CWD            string      `json:"cwd"`
}

// MCPServer is one entry of Init.MCPServers.
type MCPServer struct {
	Name   string `json:"name"`
	Status string `json:"status"`
}

// ThinkingTokens is the system/thinking_tokens event: a progress signal
// emitted while the model is thinking, with a running token estimate.
type ThinkingTokens struct {
	EstimatedTokens      int    `json:"estimated_tokens"`
	EstimatedTokensDelta int    `json:"estimated_tokens_delta"`
	SessionID            string `json:"session_id"`
}

// MessageEvent is an assistant or user event carrying a complete API message.
// Assistant events hold model output (text, thinking, tool_use blocks); user
// events hold tool results fed back into the loop.
type MessageEvent struct {
	Message   APIMessage `json:"message"`
	SessionID string     `json:"session_id"`
}

// APIMessage is the message body inside a MessageEvent.
type APIMessage struct {
	Role    string         `json:"role"`
	Model   string         `json:"model"`
	Content []ContentBlock `json:"content"`
}

// ContentBlock is one block of an APIMessage's content. Which fields are
// populated depends on Type: text, thinking, tool_use, or tool_result.
type ContentBlock struct {
	Type      string          `json:"type"`
	Text      string          `json:"text"`
	Thinking  string          `json:"thinking"`
	ID        string          `json:"id"`
	Name      string          `json:"name"`
	Input     json.RawMessage `json:"input"`
	ToolUseID string          `json:"tool_use_id"`
	Content   json.RawMessage `json:"content"`
	IsError   bool            `json:"is_error"`
}

// StreamEvent wraps an incremental API streaming event; present only when
// the CLI runs with --include-partial-messages. These carry the live text
// and thinking deltas a voice pipeline streams into TTS.
type StreamEvent struct {
	Event     APIStreamEvent `json:"event"`
	SessionID string         `json:"session_id"`
}

// APIStreamEvent is the inner Anthropic streaming event.
type APIStreamEvent struct {
	Type         string        `json:"type"`
	Index        *int          `json:"index"`
	Delta        *Delta        `json:"delta"`
	ContentBlock *ContentBlock `json:"content_block"`
}

// Delta is the incremental payload of a content_block_delta or message_delta.
type Delta struct {
	Type       string `json:"type"`
	Text       string `json:"text"`
	Thinking   string `json:"thinking"`
	StopReason string `json:"stop_reason"`
}

// Result is the per-turn completion event: outcome, final text, cost, and
// usage. A session emits one Result per completed user turn.
type Result struct {
	Subtype      string  `json:"subtype"`
	IsError      bool    `json:"is_error"`
	NumTurns     int     `json:"num_turns"`
	Result       string  `json:"result"`
	StopReason   string  `json:"stop_reason"`
	SessionID    string  `json:"session_id"`
	DurationMS   int64   `json:"duration_ms"`
	TotalCostUSD float64 `json:"total_cost_usd"`
	Usage        Usage   `json:"usage"`
}

// Usage is the token accounting attached to a Result.
type Usage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
}

// RateLimit is the rate_limit_event: subscription window status.
type RateLimit struct {
	Info      RateLimitInfo `json:"rate_limit_info"`
	SessionID string        `json:"session_id"`
}

// RateLimitInfo describes the current rate-limit window.
type RateLimitInfo struct {
	Status        string `json:"status"`
	ResetsAt      int64  `json:"resetsAt"`
	RateLimitType string `json:"rateLimitType"`
}
