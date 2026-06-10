package backend

import (
	"context"
	"fmt"
	"iter"
	"os"
	"sync"

	"github.com/Veraticus/mentat/internal/streamjson"
)

// Cassette is a Backend that replays a recorded stream-json transcript.
// Each Converse call replays the transcript's next turn (the events up to
// and including its Done), so a multi-turn recording serves sequential
// Converse calls exactly as the live session did. Tests run against
// cassettes with no network and no claude binary.
type Cassette struct {
	mu    sync.Mutex
	turns [][]Event
	next  int
}

// NewCassette loads and translates a recorded transcript from path.
func NewCassette(path string) (*Cassette, error) {
	recording, err := os.Open(path) //nolint:gosec // Path is operator config, not user input.
	if err != nil {
		return nil, fmt.Errorf("opening cassette: %w", err)
	}
	defer recording.Close()

	lines, err := streamjson.ReadAll(recording)
	if err != nil {
		return nil, fmt.Errorf("reading cassette %s: %w", path, err)
	}

	translator := NewTranslator()
	var events []Event
	for _, line := range lines {
		events = append(events, translator.Translate(line)...)
	}
	return &Cassette{turns: splitTurns(events)}, nil
}

// Converse replays the next recorded turn. The recorded Turn argument is
// ignored: a cassette answers with what was recorded, not with what was
// asked. It errs when the recording is exhausted.
func (c *Cassette) Converse(ctx context.Context, _ Turn) (iter.Seq2[Event, error], error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.next >= len(c.turns) {
		return nil, fmt.Errorf("cassette exhausted after %d turns", len(c.turns))
	}
	events := c.turns[c.next]
	c.next++

	return func(yield func(Event, error) bool) {
		for _, event := range events {
			if err := ctx.Err(); err != nil {
				yield(Event{}, fmt.Errorf("cassette replay interrupted: %w", err))
				return
			}
			if !yield(event, nil) {
				return
			}
		}
	}, nil
}

// CloseSession is a no-op: a cassette holds no per-session resources.
func (c *Cassette) CloseSession(string) error {
	return nil
}

// splitTurns groups a transcript's events into per-turn segments, each
// ending at its Done. Trailing events with no Done (a truncated recording)
// form a final partial turn rather than being dropped.
func splitTurns(events []Event) [][]Event {
	var turns [][]Event
	start := 0
	for i, event := range events {
		if event.Kind == KindDone {
			turns = append(turns, events[start:i+1])
			start = i + 1
		}
	}
	if start < len(events) {
		turns = append(turns, events[start:])
	}
	return turns
}
