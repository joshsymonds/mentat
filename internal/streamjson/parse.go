package streamjson

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
)

// maxLineBytes bounds a single NDJSON line. Assistant messages carrying large
// tool results routinely exceed bufio.Scanner's 64KB default.
const maxLineBytes = 16 * 1024 * 1024

// Parse decodes one NDJSON line into a Line. Unrecognized event types are not
// an error: they return a Line with only Type/Subtype/Raw populated and
// Unknown() == true, so callers can surface protocol drift without dying on it.
func Parse(raw []byte) (Line, error) {
	var envelope struct {
		Type      string `json:"type"`
		Subtype   string `json:"subtype"`
		SessionID string `json:"session_id"`
	}
	if err := json.Unmarshal(raw, &envelope); err != nil {
		return Line{}, fmt.Errorf("parsing stream-json envelope: %w", err)
	}
	if envelope.Type == "" {
		return Line{}, errors.New("stream-json line has no type field")
	}

	line := Line{
		Type:      envelope.Type,
		Subtype:   envelope.Subtype,
		SessionID: envelope.SessionID,
		Raw:       append([]byte(nil), raw...),
	}
	if err := decodePayload(&line, raw); err != nil {
		return Line{}, err
	}
	return line, nil
}

// ReadAll parses every line from r until EOF. It is the cassette-replay
// primitive: a recorded transcript in, the typed event sequence out.
func ReadAll(r io.Reader) ([]Line, error) {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, bufio.MaxScanTokenSize), maxLineBytes)

	var lines []Line
	for scanner.Scan() {
		raw := scanner.Bytes()
		if len(raw) == 0 {
			continue
		}
		line, err := Parse(raw)
		if err != nil {
			return nil, fmt.Errorf("line %d: %w", len(lines)+1, err)
		}
		lines = append(lines, line)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("reading stream-json: %w", err)
	}
	return lines, nil
}

func decodePayload(line *Line, raw []byte) error {
	switch line.Type {
	case "system":
		return decodeSystemPayload(line, raw)
	case "assistant":
		event, err := decodeAs[MessageEvent](raw, "assistant")
		line.Assistant = event
		return err
	case "user":
		event, err := decodeAs[MessageEvent](raw, "user")
		line.User = event
		return err
	case "stream_event":
		event, err := decodeAs[StreamEvent](raw, "stream_event")
		line.Stream = event
		return err
	case "result":
		event, err := decodeAs[Result](raw, "result")
		line.Result = event
		return err
	case "rate_limit_event":
		event, err := decodeAs[RateLimit](raw, "rate_limit_event")
		line.RateLimit = event
		return err
	default:
		// Recognized control_* families and unknown types alike carry no
		// typed payload yet; Raw preserves them for the caller.
		return nil
	}
}

func decodeSystemPayload(line *Line, raw []byte) error {
	switch line.Subtype {
	case "init":
		event, err := decodeAs[Init](raw, "system/init")
		line.Init = event
		return err
	case "thinking_tokens":
		event, err := decodeAs[ThinkingTokens](raw, "system/thinking_tokens")
		line.ThinkingTokens = event
		return err
	default:
		// status, hook_started, hook_response, and future system subtypes
		// are recognized but carry no payload we model yet.
		return nil
	}
}

func decodeAs[T any](raw []byte, kind string) (*T, error) {
	var value T
	if err := json.Unmarshal(raw, &value); err != nil {
		return nil, fmt.Errorf("decoding %s event: %w", kind, err)
	}
	return &value, nil
}
