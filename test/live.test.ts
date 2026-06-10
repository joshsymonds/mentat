import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import process from 'node:process';

import { describe, expect, it } from 'vitest';

import type { Event } from '../src/backend.ts';
import { ClaudeCode, type ClaudeCodeConfig } from '../src/claudecode.ts';
import { nullLogger } from '../src/log.ts';
import { allowAllPolicy } from '../src/policy.ts';

// Live smoke: one real conversation through the supervisor, resumed across a
// simulated daemon restart. Needs a claude binary and spends a few cents at
// haiku; skipped unless MENTAT_CLAUDE_BIN is set. Run it when bumping the
// pinned SDK or claude binary.

const bin = process.env.MENTAT_CLAUDE_BIN;
const TURN_TIMEOUT_MS = 120_000;

function liveConfig(statePath: string): ClaudeCodeConfig {
  if (bin === undefined) throw new Error('unreachable: suite is skipped');
  return {
    bin,
    model: 'claude-haiku-4-5',
    maxBudgetUsd: 0.5,
    statePath,
    policy: allowAllPolicy(nullLogger),
    logger: nullLogger,
  };
}

async function collect(events: AsyncIterable<Event>): Promise<Event[]> {
  const out: Event[] = [];
  for await (const event of events) {
    out.push(event);
  }
  return out;
}

describe.skipIf(bin === undefined || bin === '')('live smoke', () => {
  const statePath = join(mkdtempSync(join(tmpdir(), 'mentat-live-')), 'state.json');

  it(
    'streams a real turn with deltas, cost, and no drift',
    async () => {
      const backend = new ClaudeCode(liveConfig(statePath));
      const events = await collect(
        await backend.converse({
          sessionId: 'live-smoke',
          text: 'Remember the number 4519. Reply with one short sentence.',
        }),
      );
      await backend.close();

      expect(events.filter((e) => e.kind === 'textDelta').length).toBeGreaterThan(0);
      expect(events.filter((e) => e.kind === 'unknown')).toEqual([]);
      const done = events.at(-1);
      if (done?.kind !== 'done') throw new Error('turn did not end in done');
      expect(done.result.isError).toBe(false);
      expect(done.result.costUsd).toBeGreaterThan(0);
    },
    TURN_TIMEOUT_MS,
  );

  it(
    'resumes the conversation in a fresh backend and stays isolated',
    async () => {
      const backend = new ClaudeCode(liveConfig(statePath));
      const events = await collect(
        await backend.converse({
          sessionId: 'live-smoke',
          text:
            'Two questions. 1: What number did I ask you to remember earlier? ' +
            '2: How many skills are listed as available to you? ' +
            'Answer like: "number; skill count". Be terse.',
        }),
      );
      await backend.close();

      const done = events.at(-1);
      if (done?.kind !== 'done') throw new Error('turn did not end in done');
      expect(done.result.text).toContain('4519');
      // Isolation: the interactive user's skills must not leak into the child.
      expect(done.result.text).not.toContain('deep-research');
    },
    TURN_TIMEOUT_MS,
  );
});
