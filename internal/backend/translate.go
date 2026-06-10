package backend

import (
	"encoding/json"

	"github.com/Veraticus/mentat/internal/streamjson"
)

// Translator converts wire-level streamjson lines into daemon-altitude
// events. It is stateful: tool results carry only a tool_use_id on the wire,
// so the translator remembers each tool_use's name to label its result.
// One Translator serves one session's line stream; not safe for concurrent use.
type Translator struct {
	toolNames map[string]string
}

// NewTranslator returns a Translator ready to consume a session's lines.
func NewTranslator() *Translator {
	return &Translator{toolNames: make(map[string]string)}
}

// Translate converts one wire line into zero or more events. Lines that
// carry no daemon-relevant signal (init, status, rate limits, hooks)
// translate to nothing. Unknown line types translate to KindProtocolDrift.
func (tr *Translator) Translate(line streamjson.Line) []Event {
	if line.Unknown() {
		return []Event{{Kind: KindProtocolDrift, Raw: line.Raw}}
	}
	switch {
	case line.ThinkingTokens != nil:
		return []Event{{Kind: KindThinking, ThinkingTokens: line.ThinkingTokens.EstimatedTokens}}
	case line.Stream != nil:
		return translateStream(line.Stream)
	case line.Assistant != nil:
		return tr.translateAssistant(line.Assistant)
	case line.User != nil:
		return tr.translateUser(line.User)
	case line.Result != nil:
		return []Event{{Kind: KindDone, Done: &Result{
			Text:       line.Result.Result,
			IsError:    line.Result.IsError,
			StopReason: line.Result.StopReason,
			SessionID:  line.Result.SessionID,
			CostUSD:    line.Result.TotalCostUSD,
			Usage: Usage{
				InputTokens:              line.Result.Usage.InputTokens,
				OutputTokens:             line.Result.Usage.OutputTokens,
				CacheReadInputTokens:     line.Result.Usage.CacheReadInputTokens,
				CacheCreationInputTokens: line.Result.Usage.CacheCreationInputTokens,
			},
		}}}
	default:
		return nil
	}
}

// translateAssistant surfaces tool invocations. Assistant text blocks are
// deliberately not translated: streamed deltas carry live text, and the
// turn's Done carries the authoritative final text.
func (tr *Translator) translateAssistant(msg *streamjson.MessageEvent) []Event {
	var events []Event
	for _, block := range msg.Message.Content {
		if block.Type != "tool_use" {
			continue
		}
		tr.toolNames[block.ID] = block.Name
		events = append(events, Event{
			Kind: KindToolUseStarted,
			Tool: &ToolEvent{Name: block.Name},
		})
	}
	return events
}

func (tr *Translator) translateUser(msg *streamjson.MessageEvent) []Event {
	var events []Event
	for _, block := range msg.Message.Content {
		if block.Type != "tool_result" {
			continue
		}
		events = append(events, Event{
			Kind: KindToolResult,
			Tool: &ToolEvent{
				Name:    tr.toolNames[block.ToolUseID],
				IsError: block.IsError,
				Content: toolContentText(block.Content),
			},
		})
	}
	return events
}

func translateStream(stream *streamjson.StreamEvent) []Event {
	if stream.Event.Type != "content_block_delta" || stream.Event.Delta == nil {
		return nil
	}
	switch stream.Event.Delta.Type {
	case "text_delta":
		return []Event{{Kind: KindTextDelta, TextDelta: stream.Event.Delta.Text}}
	case "thinking_delta":
		return []Event{{Kind: KindThinkingDelta, ThinkingDelta: stream.Event.Delta.Thinking}}
	default:
		return nil
	}
}

// toolContentText extracts a human-readable string from a tool_result's
// content, which the wire encodes either as a JSON string or as a block
// array. Unrecognized shapes fall back to the raw JSON.
func toolContentText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return text
	}
	var blocks []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(raw, &blocks); err == nil {
		var combined string
		for _, block := range blocks {
			if block.Type == "text" {
				combined += block.Text
			}
		}
		return combined
	}
	return string(raw)
}
