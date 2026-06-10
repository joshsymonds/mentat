package api

import (
	"encoding/json"
	"fmt"
	"iter"
	"log/slog"
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
	Text                     string  `json:"text"`
	IsError                  bool    `json:"is_error"`
	StopReason               string  `json:"stop_reason,omitempty"`
	SessionID                string  `json:"session_id"`
	CostUSD                  float64 `json:"cost_usd"`
	InputTokens              int     `json:"input_tokens"`
	OutputTokens             int     `json:"output_tokens"`
	CacheReadInputTokens     int     `json:"cache_read_input_tokens"`
	CacheCreationInputTokens int     `json:"cache_creation_input_tokens"`
}

// fallbackErrorLine is a fixed payload used when even the dynamic error line
// can't be encoded. As a constant of only literal text it can never fail to
// marshal, so the client always gets a terminal error rather than a silent EOF.
const fallbackErrorLine = `{"kind":"error","message":"internal encoding failure"}` + "\n"

// streamEvents writes a turn's events as flushed NDJSON lines. A mid-stream
// failure becomes a terminal {"kind":"error"} line: by then the 200 header
// has shipped, so in-band error delivery is the only honest option.
func streamEvents(w http.ResponseWriter, logger *slog.Logger, stream iter.Seq2[backend.Event, error]) {
	w.Header().Set("Content-Type", "application/x-ndjson")
	flusher, canFlush := w.(http.Flusher)

	writeRaw := func(payload []byte) bool {
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
			writeRaw(errorLineBytes(logger, err.Error()))
			return
		}
		payload, marshalErr := marshalLine(toWire(event))
		if marshalErr != nil {
			// A wireEvent should always encode, but a NaN/Inf cost would
			// fail — never let that silently truncate the stream.
			logger.Error("encoding conversation event", slog.Any("error", marshalErr))
			writeRaw([]byte(fallbackErrorLine))
			return
		}
		if !writeRaw(payload) {
			return
		}
	}
}

// errorLineBytes encodes a terminal error line, falling back to the fixed
// literal if the message itself somehow won't encode.
func errorLineBytes(logger *slog.Logger, message string) []byte {
	payload, err := marshalLine(wireEvent{Kind: "error", Message: message})
	if err != nil {
		logger.Error("encoding error line", slog.Any("error", err))
		return []byte(fallbackErrorLine)
	}
	return payload
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
			Text:                     event.Done.Text,
			IsError:                  event.Done.IsError,
			StopReason:               event.Done.StopReason,
			SessionID:                event.Done.SessionID,
			CostUSD:                  event.Done.CostUSD,
			InputTokens:              event.Done.Usage.InputTokens,
			OutputTokens:             event.Done.Usage.OutputTokens,
			CacheReadInputTokens:     event.Done.Usage.CacheReadInputTokens,
			CacheCreationInputTokens: event.Done.Usage.CacheCreationInputTokens,
		}}
	case backend.KindProtocolDrift:
		return wireEvent{Kind: "protocol_drift", Raw: string(event.Raw)}
	default:
		return wireEvent{Kind: "unknown"}
	}
}
