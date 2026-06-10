import { createServer, type Server } from 'node:http';
import type { AddressInfo } from 'node:net';

import { afterEach, describe, expect, it } from 'vitest';

import type { Backend, Event, Turn } from '../src/backend.ts';
import { AtCapacityError } from '../src/claudecode.ts';
import { nullLogger } from '../src/log.ts';
import { SessionTracker, createHandler } from '../src/server.ts';

function doneEvent(text: string): Event {
  return {
    kind: 'done',
    result: {
      text,
      isError: false,
      stopReason: 'end_turn',
      sessionId: 'cli-uuid',
      costUsd: 0.01,
      usage: {
        inputTokens: 1,
        outputTokens: 2,
        cacheReadInputTokens: 0,
        cacheCreationInputTokens: 0,
      },
    },
  };
}

type Script = (turn: Turn) => Event[] | Error;

/** Scripted Backend: each converse yields the scripted events; an Error entry
 * in the events array becomes a mid-stream iterator failure. */
class FakeBackend implements Backend {
  readonly turns: Turn[] = [];
  readonly closed: string[] = [];
  private readonly script: Script;
  midStreamError: Error | undefined;

  constructor(script: Script) {
    this.script = script;
  }

  converse(turn: Turn): Promise<AsyncIterable<Event>> {
    this.turns.push(turn);
    const scripted = this.script(turn);
    if (scripted instanceof Error) {
      return Promise.reject(scripted);
    }
    const eventList = scripted;
    const midStreamError = this.midStreamError;
    async function* events(): AsyncGenerator<Event> {
      for (const event of eventList) {
        await Promise.resolve(); // events arrive asynchronously, as live ones do
        yield event;
      }
      if (midStreamError !== undefined) {
        throw midStreamError;
      }
    }
    return Promise.resolve(events());
  }

  closeSession(sessionId: string): Promise<void> {
    this.closed.push(sessionId);
    return Promise.resolve();
  }
}

const servers: Server[] = [];

async function serve(backend: Backend, tracker = new SessionTracker()): Promise<string> {
  const server = createServer(createHandler(backend, tracker, nullLogger));
  servers.push(server);
  await new Promise<void>((resolve) => {
    server.listen(0, '127.0.0.1', resolve);
  });
  const { port } = server.address() as AddressInfo;
  return `http://127.0.0.1:${String(port)}`;
}

afterEach(async () => {
  for (const server of servers.splice(0)) {
    await new Promise((resolve) => server.close(resolve));
  }
});

async function post(base: string, body: unknown): Promise<Response> {
  return fetch(`${base}/v1/conversation`, {
    method: 'POST',
    body: typeof body === 'string' ? body : JSON.stringify(body),
  });
}

describe('POST /v1/conversation', () => {
  it('streams a turn as NDJSON lines', async () => {
    const backend = new FakeBackend(() => [
      { kind: 'textDelta', text: 'Hel' },
      { kind: 'textDelta', text: 'lo' },
      doneEvent('Hello'),
    ]);
    const base = await serve(backend);
    const res = await post(base, { session_id: 's1', text: 'hi', meta: { surface: 'voice' } });
    expect(res.status).toBe(200);
    expect(res.headers.get('content-type')).toBe('application/x-ndjson');
    const lines = (await res.text()).trimEnd().split('\n');
    expect(lines).toEqual([
      '{"kind":"text_delta","text":"Hel"}',
      '{"kind":"text_delta","text":"lo"}',
      '{"kind":"done","done":{"text":"Hello","is_error":false,"stop_reason":"end_turn",' +
        '"session_id":"cli-uuid","cost_usd":0.01,"input_tokens":1,"output_tokens":2,' +
        '"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}',
    ]);
    expect(backend.turns[0]?.meta).toEqual({ surface: 'voice' });
  });

  it('rejects bad JSON and missing fields', async () => {
    const base = await serve(new FakeBackend(() => []));
    expect((await post(base, 'not json')).status).toBe(400);
    expect((await post(base, { text: 'no session' })).status).toBe(400);
    expect((await post(base, { session_id: 's1' })).status).toBe(400);
  });

  it('rejects oversized bodies', async () => {
    const base = await serve(new FakeBackend(() => []));
    const res = await post(base, { session_id: 's1', text: 'x'.repeat(1 << 21) });
    expect(res.status).toBe(413);
  });

  it('maps capacity refusal to 503 and other failures to 502', async () => {
    const base503 = await serve(new FakeBackend(() => new AtCapacityError()));
    const res503 = await post(base503, { session_id: 's1', text: 'hi' });
    expect(res503.status).toBe(503);
    const base502 = await serve(new FakeBackend(() => new Error('spawn failed')));
    const res502 = await post(base502, { session_id: 's1', text: 'hi' });
    expect(res502.status).toBe(502);
  });

  it('turns a mid-stream failure into a terminal error line on a 200', async () => {
    const backend = new FakeBackend(() => [{ kind: 'textDelta', text: 'par' }]);
    backend.midStreamError = new Error('child died');
    const base = await serve(backend);
    const res = await post(base, { session_id: 's1', text: 'hi' });
    expect(res.status).toBe(200);
    const lines = (await res.text()).trimEnd().split('\n');
    expect(lines[0]).toBe('{"kind":"text_delta","text":"par"}');
    expect(lines.at(-1)).toBe('{"kind":"error","message":"child died"}');
  });

  it('never writes unknown events to the stream', async () => {
    const backend = new FakeBackend(() => [
      { kind: 'unknown', raw: { type: 'novel' } },
      doneEvent('ok'),
    ]);
    const base = await serve(backend);
    const lines = (await (await post(base, { session_id: 's1', text: 'hi' })).text())
      .trimEnd()
      .split('\n');
    expect(lines).toHaveLength(1);
    expect(lines[0]).toContain('"kind":"done"');
  });

  it('404s unknown routes and methods', async () => {
    const base = await serve(new FakeBackend(() => []));
    expect((await fetch(`${base}/v1/conversation`)).status).toBe(404);
    expect((await fetch(`${base}/nope`)).status).toBe(404);
  });
});

describe('GET /healthz', () => {
  it('reports ok', async () => {
    const base = await serve(new FakeBackend(() => []));
    const res = await fetch(`${base}/healthz`);
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ status: 'ok' });
  });
});

describe('SessionTracker idle expiry', () => {
  it('expires only idle sessions and never in-flight turns', () => {
    let now = 1_000_000;
    const tracker = new SessionTracker(() => now);
    tracker.beginTurn('idle');
    tracker.endTurn('idle');
    tracker.beginTurn('active'); // never ended: turn in flight

    now += 60_000;
    expect(tracker.expireIdle(30_000)).toEqual(['idle']);
    expect(tracker.expireIdle(30_000)).toEqual([]); // already claimed
    expect(tracker.expireIdle(0)).toEqual([]); // active never expires
  });

  it('spares a session that turns active again between scan and claim', () => {
    let now = 1_000_000;
    const tracker = new SessionTracker(() => now);
    tracker.beginTurn('s');
    tracker.endTurn('s');
    now += 60_000;
    tracker.beginTurn('s'); // reactivated: lastActive is fresh and a turn is in flight
    expect(tracker.expireIdle(30_000)).toEqual([]);
  });
});
