import { readFileSync, mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import type { Options } from '@anthropic-ai/claude-agent-sdk';
import { describe, expect, it } from 'vitest';

import type { Event } from '../src/backend.ts';
import {
  AtCapacityError,
  ClaudeCode,
  DEFAULT_DISALLOWED_TOOLS,
  buildChildEnv,
  buildOptions,
  type ClaudeCodeConfig,
  type QueryFn,
} from '../src/claudecode.ts';
import { nullLogger } from '../src/log.ts';
import { allowAllPolicy, type PolicyFn, type TurnContext } from '../src/policy.ts';

// Minimal valid message shapes for orchestration tests (serialization,
// capacity, abandonment). Protocol-shape fidelity is covered by the recorded
// fixture; these only exercise the session machinery around it.

function textDeltaMsg(text: string): unknown {
  return {
    type: 'stream_event',
    event: { type: 'content_block_delta', delta: { type: 'text_delta', text } },
  };
}

function resultMsg(sessionUuid: string, text: string): unknown {
  return {
    type: 'result',
    subtype: 'success',
    is_error: false,
    result: text,
    stop_reason: 'end_turn',
    session_id: sessionUuid,
    total_cost_usd: 0.01,
    usage: {
      input_tokens: 1,
      output_tokens: 2,
      cache_read_input_tokens: 0,
      cache_creation_input_tokens: 0,
    },
  };
}

interface FakeQuery {
  fn: QueryFn;
  optionsSeen: Options[];
  interrupts: number;
  calls: number;
}

/**
 * A QueryFn that, for each user message read from the prompt, emits the next
 * scripted message batch. `endAfter` ends the stream (child death) after that
 * many batches instead of waiting for more input.
 */
function fakeQuery(script: (turnIndex: number) => unknown[], endAfter?: number): FakeQuery {
  const fake: FakeQuery = { fn: undefined as unknown as QueryFn, optionsSeen: [], interrupts: 0, calls: 0 };
  fake.fn = ({ prompt, options }) => {
    fake.optionsSeen.push(options);
    fake.calls += 1;
    async function* messages(): AsyncGenerator {
      let turn = 0;
      for await (const _user of prompt) {
        yield* script(turn);
        turn += 1;
        if (endAfter !== undefined && turn >= endAfter) {
          return;
        }
      }
    }
    const iterable = messages();
    return {
      [Symbol.asyncIterator]: () => iterable[Symbol.asyncIterator](),
      interrupt: () => {
        fake.interrupts += 1;
        return Promise.resolve();
      },
    };
  };
  return fake;
}

function makeConfig(overrides: Partial<ClaudeCodeConfig> = {}): ClaudeCodeConfig {
  return {
    bin: '/pinned/claude',
    policy: allowAllPolicy(nullLogger),
    logger: nullLogger,
    ...overrides,
  };
}

async function collect(events: AsyncIterable<Event>): Promise<Event[]> {
  const out: Event[] = [];
  for await (const event of events) {
    out.push(event);
  }
  return out;
}

describe('buildOptions isolation invariants', () => {
  const context: TurnContext = { sessionId: 's', meta: {} };
  const options = buildOptions(
    makeConfig({
      model: 'claude-haiku-4-5',
      effort: 'low',
      systemPrompt: 'be helpful',
      addDirs: ['/memory'],
      allowedTools: ['Read'],
      maxBudgetUsd: 1.5,
    }),
    () => context,
  );

  it('never loads user settings or skills', () => {
    expect(options.settingSources).toEqual([]);
    expect(options.skills).toEqual([]);
    expect(options.strictMcpConfig).toBe(true);
  });

  it('pins the executable with no PATH fallback', () => {
    expect(options.pathToClaudeCodeExecutable).toBe('/pinned/claude');
  });

  it('streams partial messages', () => {
    expect(options.includePartialMessages).toBe(true);
  });

  it('passes session configuration through', () => {
    expect(options.model).toBe('claude-haiku-4-5');
    expect(options.effort).toBe('low');
    expect(options.systemPrompt).toBe('be helpful');
    expect(options.additionalDirectories).toEqual(['/memory']);
    expect(options.allowedTools).toEqual(['Read']);
    expect(options.maxBudgetUsd).toBe(1.5);
  });

  it('defaults to disallowing dangerous built-ins when no policy is set', () => {
    expect(options.disallowedTools).toEqual(DEFAULT_DISALLOWED_TOOLS);
    const custom = buildOptions(makeConfig({ disallowedTools: ['Bash'] }), () => context);
    expect(custom.disallowedTools).toEqual(['Bash']);
  });

  it('sets resume only when respawning', () => {
    expect(options.resume).toBeUndefined();
    const respawn = buildOptions(makeConfig(), () => context, 'old-uuid');
    expect(respawn.resume).toBe('old-uuid');
  });
});

describe('buildChildEnv', () => {
  it('allowlists, never inherits', () => {
    const env = buildChildEnv(
      {
        HOME: '/home/u',
        PATH: '/bin',
        ANTHROPIC_BASE_URL: 'https://x',
        XDG_CONFIG_HOME: '/cfg',
        CLAUDECODE: '1',
        CLAUDE_CODE_ENTRYPOINT: 'cli',
        RANDOM_SECRET: 'hunter2',
      },
      [],
    );
    expect(env.HOME).toBe('/home/u');
    expect(env.ANTHROPIC_BASE_URL).toBe('https://x');
    expect(env.XDG_CONFIG_HOME).toBe('/cfg');
    expect(env).not.toHaveProperty('CLAUDECODE');
    expect(env).not.toHaveProperty('CLAUDE_CODE_ENTRYPOINT');
    expect(env).not.toHaveProperty('RANDOM_SECRET');
  });

  it('extends the allowlist with extraEnv only', () => {
    const env = buildChildEnv({ RANDOM_SECRET: 'x', OTHER: 'y' }, ['RANDOM_SECRET']);
    expect(env.RANDOM_SECRET).toBe('x');
    expect(env).not.toHaveProperty('OTHER');
  });
});

describe('ClaudeCode turns', () => {
  it('streams a recorded turn end to end', async () => {
    const lines = readFileSync('test/fixtures/turn-with-tool.jsonl', 'utf8')
      .trimEnd()
      .split('\n')
      .map((line) => JSON.parse(line) as unknown);
    const fake = fakeQuery(() => lines);
    const backend = new ClaudeCode(makeConfig({ queryFn: fake.fn }));
    const events = await collect(await backend.converse({ sessionId: 's1', text: 'hi' }));
    expect(events.some((e) => e.kind === 'textDelta')).toBe(true);
    expect(events.some((e) => e.kind === 'toolStart')).toBe(true);
    expect(events.at(-1)?.kind).toBe('done');
    expect(fake.calls).toBe(1);
  });

  it('requires a sessionId', async () => {
    const backend = new ClaudeCode(makeConfig({ queryFn: fakeQuery(() => []).fn }));
    await expect(backend.converse({ sessionId: '', text: 'hi' })).rejects.toThrow(/sessionId/);
  });

  it('serializes turns within a session', async () => {
    const order: string[] = [];
    const fake = fakeQuery((turn) => [textDeltaMsg(`t${String(turn)}`), resultMsg('u1', 'ok')]);
    const backend = new ClaudeCode(makeConfig({ queryFn: fake.fn }));

    const first = await backend.converse({ sessionId: 's1', text: 'one' });
    const secondPromise = backend.converse({ sessionId: 's1', text: 'two' });
    for await (const event of first) {
      if (event.kind === 'textDelta') order.push(event.text);
      if (event.kind === 'done') order.push('done-1');
    }
    for await (const event of await secondPromise) {
      if (event.kind === 'textDelta') order.push(event.text);
      if (event.kind === 'done') order.push('done-2');
    }
    expect(order).toEqual(['t0', 'done-1', 't1', 'done-2']);
    expect(fake.calls).toBe(1);
  });

  it('reuses one child across turns and respawns with resume after death', async () => {
    // Child ends its stream after the first turn (death); the second turn
    // must respawn a new child carrying resume=<uuid from turn one>.
    const fake = fakeQuery((turn) => [resultMsg('cli-uuid-9', `r${String(turn)}`)], 1);
    const backend = new ClaudeCode(makeConfig({ queryFn: fake.fn }));
    const first = await collect(await backend.converse({ sessionId: 's1', text: 'one' }));
    expect(first.at(-1)?.kind).toBe('done');

    const second = await collect(await backend.converse({ sessionId: 's1', text: 'two' }));
    expect(second.at(-1)?.kind).toBe('done');
    expect(fake.calls).toBe(2);
    expect(fake.optionsSeen[0]?.resume).toBeUndefined();
    expect(fake.optionsSeen[1]?.resume).toBe('cli-uuid-9');
  });

  it('throws mid-stream when the child dies mid-turn', async () => {
    const fake = fakeQuery(() => [textDeltaMsg('partial')], 1);
    // endAfter=1 ends the stream after the batch, but the batch has no result:
    // the turn sees stream end before done.
    const backend = new ClaudeCode(makeConfig({ queryFn: fake.fn }));
    await expect(
      collect(await backend.converse({ sessionId: 's1', text: 'hi' })),
    ).rejects.toThrow(/ended mid-turn/);
  });

  it('interrupts abandoned turns and keeps the session usable', async () => {
    const fake = fakeQuery((turn) =>
      turn === 0
        ? [textDeltaMsg('a'), textDeltaMsg('b'), resultMsg('u1', 'first')]
        : [resultMsg('u1', 'second')],
    );
    const backend = new ClaudeCode(makeConfig({ queryFn: fake.fn }));
    const stream = await backend.converse({ sessionId: 's1', text: 'one' });
    for await (const event of stream) {
      if (event.kind === 'textDelta') break; // abandon mid-turn
    }
    expect(fake.interrupts).toBe(1);
    const second = await collect(await backend.converse({ sessionId: 's1', text: 'two' }));
    expect(second.at(-1)?.kind).toBe('done');
    expect(fake.calls).toBe(1); // same child, not respawned
  });

  it('refuses new sessions at capacity but serves existing ones', async () => {
    const fake = fakeQuery(() => [resultMsg('u1', 'ok')]);
    const backend = new ClaudeCode(makeConfig({ queryFn: fake.fn, maxSessions: 1 }));
    await collect(await backend.converse({ sessionId: 's1', text: 'one' }));
    await expect(backend.converse({ sessionId: 's2', text: 'hi' })).rejects.toThrow(
      AtCapacityError,
    );
    const again = await collect(await backend.converse({ sessionId: 's1', text: 'two' }));
    expect(again.at(-1)?.kind).toBe('done');

    await backend.closeSession('s1');
    const other = await collect(await backend.converse({ sessionId: 's2', text: 'now' }));
    expect(other.at(-1)?.kind).toBe('done');
  });
});

describe('policy seam', () => {
  it('binds the active turn context per turn, never cached', async () => {
    const seen: TurnContext[] = [];
    const recordingPolicy: PolicyFn = (_tool, input, context) => {
      seen.push(context);
      return { behavior: 'allow', updatedInput: input };
    };
    const fake = fakeQuery(() => [resultMsg('u1', 'ok')]);
    const backend = new ClaudeCode(makeConfig({ queryFn: fake.fn, policy: recordingPolicy }));

    const callPolicy = async (): Promise<void> => {
      const canUseTool = fake.optionsSeen[0]?.canUseTool;
      if (canUseTool === undefined) throw new Error('canUseTool not wired');
      await canUseTool('tool_x', {}, { signal: new AbortController().signal, toolUseID: "t1" });
    };

    const first = await backend.converse({
      sessionId: 's1',
      text: 'one',
      meta: { surface: 'voice', user: 'josh' },
    });
    const iterator = first[Symbol.asyncIterator]();
    await callPolicy(); // mid-turn: context is bound
    while (!(await iterator.next()).done) {
      // drain
    }

    const second = await backend.converse({
      sessionId: 's1',
      text: 'two',
      meta: { surface: 'signal', user: 'guest' },
    });
    const iterator2 = second[Symbol.asyncIterator]();
    await callPolicy();
    while (!(await iterator2.next()).done) {
      // drain
    }

    expect(seen).toHaveLength(2);
    expect(seen[0]?.meta).toEqual({ surface: 'voice', user: 'josh' });
    expect(seen[1]?.meta).toEqual({ surface: 'signal', user: 'guest' });
    expect(seen.every((c) => c.sessionId === 's1')).toBe(true);
  });

  it('returns the policy denial to the SDK', async () => {
    const denyPolicy: PolicyFn = () => ({ behavior: 'deny', message: 'not on voice' });
    const fake = fakeQuery(() => [resultMsg('u1', 'ok')]);
    const backend = new ClaudeCode(makeConfig({ queryFn: fake.fn, policy: denyPolicy }));
    const stream = await backend.converse({ sessionId: 's1', text: 'one' });
    const canUseTool = fake.optionsSeen[0]?.canUseTool;
    if (canUseTool === undefined) throw new Error('canUseTool not wired');
    const decision = await canUseTool('tool_x', {}, { signal: new AbortController().signal, toolUseID: "t1" });
    expect(decision).toEqual({ behavior: 'deny', message: 'not on voice' });
    await collect(stream);
  });
});

describe('record mode', () => {
  it('captures the exact message stream, neutralizing path traversal', async () => {
    const lines = readFileSync('test/fixtures/turn-with-tool.jsonl', 'utf8')
      .trimEnd()
      .split('\n')
      .map((line) => JSON.parse(line) as unknown);
    const recordDir = mkdtempSync(join(tmpdir(), 'mentat-rec-'));
    const fake = fakeQuery(() => lines);
    const backend = new ClaudeCode(makeConfig({ queryFn: fake.fn, recordDir }));
    await collect(await backend.converse({ sessionId: 's/../x', text: 'hi' }));

    const recorded = readFileSync(join(recordDir, 's%2F..%2Fx.jsonl'), 'utf8')
      .trimEnd()
      .split('\n')
      .map((line) => JSON.parse(line) as unknown);
    expect(recorded).toEqual(lines);
  });
});

describe('resume state persistence', () => {
  it('persists the session map and loads it back', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'mentat-test-'));
    const statePath = join(dir, 'state.json');
    const fake = fakeQuery(() => [resultMsg('cli-uuid-1', 'ok')]);
    const backend = new ClaudeCode(makeConfig({ queryFn: fake.fn, statePath }));
    await collect(await backend.converse({ sessionId: 's1', text: 'hi' }));
    expect(JSON.parse(readFileSync(statePath, 'utf8'))).toEqual({ s1: 'cli-uuid-1' });

    // A fresh daemon resumes from the persisted map.
    const fake2 = fakeQuery(() => [resultMsg('cli-uuid-1', 'ok')]);
    const backend2 = new ClaudeCode(makeConfig({ queryFn: fake2.fn, statePath }));
    await collect(await backend2.converse({ sessionId: 's1', text: 'again' }));
    expect(fake2.optionsSeen[0]?.resume).toBe('cli-uuid-1');
  });

  it('refuses to start on a corrupt state file', () => {
    const dir = mkdtempSync(join(tmpdir(), 'mentat-test-'));
    const statePath = join(dir, 'state.json');
    writeFileSync(statePath, 'not json');
    expect(() => new ClaudeCode(makeConfig({ statePath }))).toThrow(/state/);
  });
});
