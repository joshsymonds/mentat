import { describe, expect, it } from 'vitest';

import type { Event, Result } from '../src/backend.ts';
import { errorLine, toWireLine } from '../src/wire.ts';

// Golden lines: byte-for-byte the Go daemon's wire format (go-v2
// internal/api/wire.go). Surfaces depend on this exact encoding.

function doneResult(overrides: Partial<Result> = {}): Event {
  return {
    kind: 'done',
    result: {
      text: 'hi',
      isError: false,
      stopReason: 'end_turn',
      sessionId: 'abc',
      costUsd: 0.0262,
      usage: {
        inputTokens: 30,
        outputTokens: 1253,
        cacheReadInputTokens: 26270,
        cacheCreationInputTokens: 13396,
      },
      ...overrides,
    },
  };
}

describe('toWireLine golden format', () => {
  it('encodes text deltas', () => {
    expect(toWireLine({ kind: 'textDelta', text: 'Hello' })).toBe(
      '{"kind":"text_delta","text":"Hello"}',
    );
  });

  it('encodes thinking deltas', () => {
    expect(toWireLine({ kind: 'thinkingDelta', text: 'hmm' })).toBe(
      '{"kind":"thinking_delta","text":"hmm"}',
    );
  });

  it('encodes thinking progress', () => {
    expect(toWireLine({ kind: 'thinking', tokens: 128 })).toBe(
      '{"kind":"thinking","tokens":128}',
    );
  });

  it('omits zero tokens like Go omitempty', () => {
    expect(toWireLine({ kind: 'thinking', tokens: 0 })).toBe('{"kind":"thinking"}');
  });

  it('encodes tool starts', () => {
    expect(toWireLine({ kind: 'toolStart', tool: 'mcp__memory__memory_save' })).toBe(
      '{"kind":"tool_start","tool":"mcp__memory__memory_save"}',
    );
  });

  it('encodes failed tool results', () => {
    expect(
      toWireLine({ kind: 'toolResult', tool: 'X', isError: true, content: 'denied' }),
    ).toBe('{"kind":"tool_result","tool":"X","is_error":true,"content":"denied"}');
  });

  it('omits false/empty tool result fields like Go omitempty', () => {
    expect(toWireLine({ kind: 'toolResult', tool: 'X', isError: false, content: '' })).toBe(
      '{"kind":"tool_result","tool":"X"}',
    );
  });

  it('encodes done with the full payload, no omissions except stop_reason', () => {
    expect(toWireLine(doneResult())).toBe(
      '{"kind":"done","done":{"text":"hi","is_error":false,"stop_reason":"end_turn",' +
        '"session_id":"abc","cost_usd":0.0262,"input_tokens":30,"output_tokens":1253,' +
        '"cache_read_input_tokens":26270,"cache_creation_input_tokens":13396}}',
    );
  });

  it('omits an empty stop_reason', () => {
    expect(toWireLine(doneResult({ stopReason: '' }))).toBe(
      '{"kind":"done","done":{"text":"hi","is_error":false,' +
        '"session_id":"abc","cost_usd":0.0262,"input_tokens":30,"output_tokens":1253,' +
        '"cache_read_input_tokens":26270,"cache_creation_input_tokens":13396}}',
    );
  });

  it('never emits malformed JSON for non-finite cost', () => {
    expect(toWireLine(doneResult({ costUsd: Number.NaN }))).toBe(
      '{"kind":"error","message":"internal encoding failure"}',
    );
    expect(toWireLine(doneResult({ costUsd: Number.POSITIVE_INFINITY }))).toBe(
      '{"kind":"error","message":"internal encoding failure"}',
    );
  });

  it('returns null for unknown events (logged, never streamed)', () => {
    expect(toWireLine({ kind: 'unknown', raw: { type: 'novel' } })).toBeNull();
  });
});

describe('errorLine', () => {
  it('encodes terminal error lines', () => {
    expect(errorLine('boom')).toBe('{"kind":"error","message":"boom"}');
  });
});
