// Command record captures a real claude turn into a cassette file. It drives
// the live backend's RecordDir mechanism — the same recording wrapper the
// daemon uses — so every cassette originates from genuine CLI output rather
// than hand-authored JSON. Requires MENTAT_CLAUDE_BIN; spends real tokens.
//
//	MENTAT_CLAUDE_BIN=$(command -v claude) \
//	  go run ./cmd/record -out testdata/cassettes/foo.ndjson "your prompt"
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/Veraticus/mentat/internal/backend"
)

func main() {
	out := flag.String("out", "", "destination cassette path (required)")
	model := flag.String("model", "claude-haiku-4-5", "model for the recorded turn")
	effort := flag.String("effort", "", "effort level for the recorded turn")
	flag.Parse()

	if err := run(*out, *model, *effort, strings.Join(flag.Args(), " ")); err != nil {
		fmt.Fprintln(os.Stderr, "record:", err)
		os.Exit(1)
	}
}

func run(out, model, effort, prompt string) error {
	if out == "" {
		return errors.New("-out is required")
	}
	if strings.TrimSpace(prompt) == "" {
		return errors.New("a prompt argument is required")
	}
	bin := os.Getenv("MENTAT_CLAUDE_BIN")
	if bin == "" {
		return errors.New("MENTAT_CLAUDE_BIN must point at a claude binary")
	}

	recordDir, err := os.MkdirTemp("", "mentat-record-")
	if err != nil {
		return fmt.Errorf("creating record dir: %w", err)
	}
	defer os.RemoveAll(recordDir)

	bk, err := backend.NewClaudeCode(backend.ClaudeCodeConfig{
		Bin:       bin,
		Model:     model,
		Effort:    effort,
		RecordDir: recordDir,
	})
	if err != nil {
		return fmt.Errorf("starting backend: %w", err)
	}

	if turnErr := recordTurn(bk, prompt); turnErr != nil {
		_ = bk.Close()
		return turnErr
	}
	// Close flushes and reaps the child, finalizing the recording file.
	if closeErr := bk.Close(); closeErr != nil {
		return fmt.Errorf("closing backend: %w", closeErr)
	}

	return moveRecording(recordDir, out)
}

func recordTurn(bk *backend.ClaudeCode, prompt string) error {
	stream, err := bk.Converse(context.Background(), backend.Turn{
		SessionID: "record",
		Text:      prompt,
	})
	if err != nil {
		return fmt.Errorf("starting turn: %w", err)
	}
	var done bool
	for event, streamErr := range stream {
		if streamErr != nil {
			return fmt.Errorf("during turn: %w", streamErr)
		}
		if event.Kind == backend.KindDone {
			done = true
		}
	}
	if !done {
		return errors.New("turn ended without a completion event; recording would be partial")
	}
	return nil
}

// moveRecording locates the single transcript the backend wrote and moves it
// to the destination path.
func moveRecording(recordDir, out string) error {
	entries, err := filepath.Glob(filepath.Join(recordDir, "*.ndjson"))
	if err != nil {
		return fmt.Errorf("locating recording: %w", err)
	}
	if len(entries) != 1 {
		return fmt.Errorf("expected exactly one recording, found %d", len(entries))
	}
	data, err := os.ReadFile(entries[0])
	if err != nil {
		return fmt.Errorf("reading recording: %w", err)
	}
	if err := os.WriteFile(out, data, 0o600); err != nil {
		return fmt.Errorf("writing %s: %w", out, err)
	}
	fmt.Fprintf(os.Stderr, "recorded %d bytes to %s\n", len(data), out)
	return nil
}
