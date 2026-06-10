// Package backend defines the daemon's harness-altitude conversation
// abstraction. A Backend accepts user turns and streams typed events back;
// the agentic loop, tool execution, and MCP connections live inside the
// implementation, invisible to callers. Implementations: cassette replay
// (this package) and the live Claude Code CLI supervisor.
package backend

import (
	"context"
	"iter"
)

// Turn is one user utterance entering a conversation session.
type Turn struct {
	// SessionID groups turns into a conversation. Implementations decide
	// what continuity it buys (the live backend maps it to a CLI session).
	SessionID string
	// Text is the user's utterance.
	Text string
	// Meta carries surface context (area, user identity). Opaque to the
	// backend contract; implementations may inject it into the session.
	Meta map[string]string
}

// Backend streams conversation events for user turns. Converse returns an
// error for failures to start the turn; failures mid-stream arrive as the
// iterator's error values.
type Backend interface {
	Converse(ctx context.Context, turn Turn) (iter.Seq2[Event, error], error)
}

// EventKind discriminates Event payloads.
type EventKind int

// Event kinds, in rough lifecycle order of a turn.
const (
	// KindTextDelta is an incremental chunk of the assistant's reply text.
	KindTextDelta EventKind = iota + 1
	// KindThinkingDelta is an incremental chunk of visible thinking text.
	KindThinkingDelta
	// KindThinking is a thinking-in-progress signal with a token estimate.
	KindThinking
	// KindToolUseStarted reports the model invoking a tool.
	KindToolUseStarted
	// KindToolResult reports a tool call completing.
	KindToolResult
	// KindDone reports the turn completing, carrying the final Result.
	KindDone
	// KindProtocolDrift reports a wire event this build does not recognize.
	// It signals the claude binary moved ahead of the parser: log it loudly.
	KindProtocolDrift
)

// Event is one occurrence in a turn's stream. Kind is always set; which
// other fields are populated depends on Kind.
type Event struct {
	Kind EventKind
	// TextDelta is set for KindTextDelta.
	TextDelta string
	// ThinkingDelta is set for KindThinkingDelta.
	ThinkingDelta string
	// ThinkingTokens is set for KindThinking: estimated tokens so far.
	ThinkingTokens int
	// Tool is set for KindToolUseStarted and KindToolResult.
	Tool *ToolEvent
	// Done is set for KindDone.
	Done *Result
	// Raw is set for KindProtocolDrift: the unrecognized wire bytes.
	Raw []byte
}

// ToolEvent describes a tool invocation or its outcome.
type ToolEvent struct {
	// Name is the tool's name. For results it is correlated from the
	// originating tool_use, since the wire result carries only an ID.
	Name string
	// IsError marks a failed or denied tool call (results only).
	IsError bool
	// Content is the tool's textual output or denial message (results only).
	Content string
}

// Result is a turn's final outcome.
//
// IsError reports protocol-level failure only. A turn whose tool calls were
// all denied still completes with IsError=false; whether the user's intent
// succeeded lives in Text, not here.
type Result struct {
	Text       string
	IsError    bool
	StopReason string
	SessionID  string
	CostUSD    float64
	Usage      Usage
}

// Usage is the token accounting for a completed turn.
type Usage struct {
	InputTokens              int
	OutputTokens             int
	CacheReadInputTokens     int
	CacheCreationInputTokens int
}
