import { readFileSync } from 'node:fs';

import { describe, expect, it } from 'vitest';

import type { Event } from '../src/backend.ts';
import { Translator } from '../src/translate.ts';

// Replays a fixture recorded from a real SDK turn (scripts/record-fixture.ts).
// The fixture contains thinking, tool_use/tool_result pairs, text deltas, and
// the terminal result — the full shape of a tool-using turn.

function replayFixture(name: string): Event[] {
  const translator = new Translator();
  const lines = readFileSync(`test/fixtures/${name}.jsonl`, 'utf8').trimEnd().split('\n');
  return lines.flatMap((line) => translator.translate(JSON.parse(line)));
}

describe('Translator on a recorded turn', () => {
  const events = replayFixture('turn-with-tool');

  it('produces no unknown events from a current recording', () => {
    expect(events.filter((e) => e.kind === 'unknown')).toEqual([]);
  });

  it('streams text deltas', () => {
    const deltas = events.filter((e) => e.kind === 'textDelta');
    expect(deltas.length).toBeGreaterThan(0);
    expect(deltas.every((e) => e.text.length > 0)).toBe(true);
  });

  it('streams thinking deltas and thinking progress', () => {
    expect(events.some((e) => e.kind === 'thinkingDelta')).toBe(true);
    const progress = events.filter((e) => e.kind === 'thinking');
    expect(progress.length).toBeGreaterThan(0);
    expect(progress.at(-1)?.tokens).toBeGreaterThan(0);
  });

  it('emits a toolStart per tool_use', () => {
    const starts = events.filter((e) => e.kind === 'toolStart');
    // The recorded turn lazy-loads the MCP tool schema via ToolSearch before
    // calling it, so the stream holds more than one tool call.
    expect(starts.length).toBeGreaterThan(0);
    expect(starts.every((e) => e.tool.length > 0)).toBe(true);
    expect(starts.map((e) => e.tool)).toContain('mcp__memory__memory_save');
  });

  it('correlates tool results to the originating tool name', () => {
    const results = events.filter((e) => e.kind === 'toolResult');
    const starts = events.filter((e) => e.kind === 'toolStart');
    expect(results.length).toBe(starts.length);
    expect(new Set(results.map((e) => e.tool))).toEqual(new Set(starts.map((e) => e.tool)));
    expect(results.every((e) => !e.isError)).toBe(true);
    // Tool result content may be text blocks or non-text blocks (e.g.
    // tool_reference); at least one carries the tool's text output.
    expect(results.some((e) => e.content.includes('saved'))).toBe(true);
  });

  it('emits exactly one done with real cost and usage', () => {
    const dones = events.filter((e) => e.kind === 'done');
    expect(dones).toHaveLength(1);
    const result = dones[0]?.result;
    expect(result?.isError).toBe(false);
    expect(result?.text.length).toBeGreaterThan(0);
    expect(result?.stopReason).toBe('end_turn');
    expect(result?.sessionId.length).toBeGreaterThan(0);
    expect(result?.costUsd).toBeGreaterThan(0);
    expect(Number.isFinite(result?.costUsd)).toBe(true);
    expect(result?.usage.outputTokens).toBeGreaterThan(0);
    expect(result?.usage.inputTokens).toBeGreaterThan(0);
  });

  it('ends with done as the final event', () => {
    expect(events.at(-1)?.kind).toBe('done');
  });
});

describe('Translator drift detection', () => {
  it('flags novel top-level message types as unknown', () => {
    const translator = new Translator();
    const novel = { type: 'definitely-novel', payload: 1 };
    expect(translator.translate(novel)).toEqual([{ kind: 'unknown', raw: novel }]);
  });

  it('flags novel system subtypes as unknown', () => {
    const translator = new Translator();
    const novel = { type: 'system', subtype: 'brand-new-subtype' };
    expect(translator.translate(novel)).toEqual([{ kind: 'unknown', raw: novel }]);
  });

  it('flags novel stream_event types as unknown', () => {
    const translator = new Translator();
    const novel = { type: 'stream_event', event: { type: 'novel_block' } };
    expect(translator.translate(novel)).toEqual([{ kind: 'unknown', raw: novel }]);
  });
});
