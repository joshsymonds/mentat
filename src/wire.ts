// The NDJSON line format of the conversation stream — byte-compatible with
// the Go daemon's wire encoding (go-v2 internal/api/wire.go), including its
// omitempty semantics and key order. Surfaces depend on this exact format.

import type { Event } from './backend.ts';

/**
 * Fixed payload used when a value cannot be encoded honestly (e.g. a
 * non-finite cost, which JSON.stringify would silently corrupt to null).
 * As a constant of only literal text it always parses, so the client gets a
 * terminal error rather than a malformed line.
 */
const FALLBACK_ERROR_LINE = '{"kind":"error","message":"internal encoding failure"}';

/** Encodes a terminal error line. */
export function errorLine(message: string): string {
  return JSON.stringify({ kind: 'error', message });
}

/**
 * Encodes one event as a wire line, or null for events that are not part of
 * the client stream (`unknown` SDK messages are logged by the caller instead;
 * the Go version's protocol_drift wire event has no v3 successor).
 *
 * Property insertion order is load-bearing: JSON.stringify preserves it, and
 * the golden tests pin it to the Go field order.
 */
export function toWireLine(event: Event): string | null {
  switch (event.kind) {
    case 'textDelta':
      return JSON.stringify({ kind: 'text_delta', ...omitEmpty('text', event.text) });
    case 'thinkingDelta':
      return JSON.stringify({ kind: 'thinking_delta', ...omitEmpty('text', event.text) });
    case 'thinking':
      return JSON.stringify({ kind: 'thinking', ...omitZero('tokens', event.tokens) });
    case 'toolStart':
      return JSON.stringify({ kind: 'tool_start', ...omitEmpty('tool', event.tool) });
    case 'toolResult':
      return JSON.stringify({
        kind: 'tool_result',
        ...omitEmpty('tool', event.tool),
        ...(event.isError ? { is_error: true } : {}),
        ...omitEmpty('content', event.content),
      });
    case 'done': {
      const { result } = event;
      if (!Number.isFinite(result.costUsd)) {
        return FALLBACK_ERROR_LINE;
      }
      return JSON.stringify({
        kind: 'done',
        done: {
          text: result.text,
          is_error: result.isError,
          ...omitEmpty('stop_reason', result.stopReason),
          session_id: result.sessionId,
          cost_usd: result.costUsd,
          input_tokens: result.usage.inputTokens,
          output_tokens: result.usage.outputTokens,
          cache_read_input_tokens: result.usage.cacheReadInputTokens,
          cache_creation_input_tokens: result.usage.cacheCreationInputTokens,
        },
      });
    }
    case 'unknown':
      return null;
  }
}

function omitEmpty(key: string, value: string): Record<string, string> {
  return value === '' ? {} : { [key]: value };
}

function omitZero(key: string, value: number): Record<string, number> {
  return value === 0 ? {} : { [key]: value };
}
