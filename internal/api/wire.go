package api

import (
	"encoding/json"
	"fmt"
	"iter"
	"net/http"

	"github.com/Veraticus/mentat/internal/backend"
)

// wireEvent is the NDJSON line format of the conversation stream. Exactly
// one of the payload fields beyond Kind is meaningful per kind; clients
// switch on kind.
type wireEvent struct {
	Kind    string    `json:"kind"`
	Text    string    `json:"text,omitempty"`
	Tokens  int       `json:"tokens,omitempty"`
	Tool    string    `json:"tool,omitempty"`
	IsError bool      `json:"is_error,omitempty"`
	Content string    `json:"content,omitempty"`
	Raw     string    `json:"raw,omitempty"`
	Message string    `json:"message,omitempty"`
	Done    *wireDone `json:"done,omitempty"`
}

// wireDone is the terminal event's payload.
type wireDone struct {
	Text         string  `json:"text"`
	IsError      bool    `json:"is_error"`
	StopReason   string  `json:"stop_reason,omitempty"`
	SessionID    string  `json:"session_id"`
	CostUSD      float64 `json:"cost_usd"`
	InputTokens  int     `json:"input_tokens"`
	OutputTokens int     `json:"output_tokens"`
}

// streamEvents writes a turn's events as flushed NDJSON lines. A mid-stream
// failure becomes a terminal {"kind":"error"} line: by then the 200 header
// has shipped, so in-band error delivery is the only honest option.
func streamEvents(w http.ResponseWriter, stream iter.Seq2[backend.Event, error]) {
	w.Header().Set("Content-Type", "application/x-ndjson")
	flusher, canFlush := w.(http.Flusher)

	writeLine := func(line wireEvent) bool {
		payload, err := marshalLine(line)
		if err != nil {
			return false
		}
		if _, writeErr := w.Write(payload); writeErr != nil {
			return false
		}
		if canFlush {
			flusher.Flush()
		}
		return true
	}

	for event, err := range stream {
		if err != nil {
			writeLine(wireEvent{Kind: "error", Message: err.Error()})
			return
		}
		if !writeLine(toWire(event)) {
			return
		}
	}
}

func marshalLine(line wireEvent) ([]byte, error) {
	payload, err := json.Marshal(line)
	if err != nil {
		return nil, fmt.Errorf("encoding wire event: %w", err)
	}
	return append(payload, '\n'), nil
}

func toWire(event backend.Event) wireEvent {
	switch event.Kind {
	case backend.KindTextDelta:
		return wireEvent{Kind: "text_delta", Text: event.TextDelta}
	case backend.KindThinkingDelta:
		return wireEvent{Kind: "thinking_delta", Text: event.ThinkingDelta}
	case backend.KindThinking:
		return wireEvent{Kind: "thinking", Tokens: event.ThinkingTokens}
	case backend.KindToolUseStarted:
		return wireEvent{Kind: "tool_start", Tool: event.Tool.Name}
	case backend.KindToolResult:
		return wireEvent{
			Kind:    "tool_result",
			Tool:    event.Tool.Name,
			IsError: event.Tool.IsError,
			Content: event.Tool.Content,
		}
	case backend.KindDone:
		return wireEvent{Kind: "done", Done: &wireDone{
			Text:         event.Done.Text,
			IsError:      event.Done.IsError,
			StopReason:   event.Done.StopReason,
			SessionID:    event.Done.SessionID,
			CostUSD:      event.Done.CostUSD,
			InputTokens:  event.Done.Usage.InputTokens,
			OutputTokens: event.Done.Usage.OutputTokens,
		}}
	case backend.KindProtocolDrift:
		return wireEvent{Kind: "protocol_drift", Raw: string(event.Raw)}
	default:
		return wireEvent{Kind: "unknown"}
	}
}
