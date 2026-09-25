import { createServer, type Server } from 'node:http';
import { connect, type AddressInfo } from 'node:net';
import type { ReadableStreamDefaultReader } from 'node:stream/web';
import { setTimeout as delay } from 'node:timers/promises';

import { afterEach, describe, expect, it, vi } from 'vitest';

import { AtCapacityError, type Backend, type Event, type Turn } from '../src/backend.ts';
import { nullLogger, type Logger } from '../src/log.ts';
import type { McpDependencies } from '../src/mcp.ts';
import { PhoneBridge } from '../src/phone.ts';
import { SessionTracker, createHandler } from '../src/server.ts';
import type { TokenIssuer } from '../src/voicetoken.ts';

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

async function serve(
  backend: Backend,
  tracker = new SessionTracker(),
  issuer?: TokenIssuer,
  logger: Logger = nullLogger,
  bridge?: PhoneBridge,
): Promise<string> {
  const mcp: McpDependencies | undefined = bridge === undefined ? undefined : { bridge, places: {} };
  const server = createServer(createHandler(backend, tracker, logger, issuer, mcp));
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

  it('passes a valid effort through to the backend turn', async () => {
    const backend = new FakeBackend(() => [doneEvent('ok')]);
    const base = await serve(backend);
    const res = await post(base, { session_id: 's1', text: 'hi', effort: 'low' });
    expect(res.status).toBe(200);
    expect(backend.turns[0]?.effort).toBe('low');
  });

  it('omits effort from the turn when the request has none', async () => {
    const backend = new FakeBackend(() => [doneEvent('ok')]);
    const base = await serve(backend);
    await post(base, { session_id: 's1', text: 'hi' });
    expect(backend.turns[0]?.effort).toBeUndefined();
  });

  it('rejects an invalid effort with 400', async () => {
    const backend = new FakeBackend(() => [doneEvent('ok')]);
    const base = await serve(backend);
    expect((await post(base, { session_id: 's1', text: 'hi', effort: 'turbo' })).status).toBe(400);
    expect((await post(base, { session_id: 's1', text: 'hi', effort: 7 })).status).toBe(400);
    expect(backend.turns).toHaveLength(0);
  });

  it('passes a valid model through to the backend turn', async () => {
    const backend = new FakeBackend(() => [doneEvent('ok')]);
    const base = await serve(backend);
    const res = await post(base, { session_id: 's1', text: 'hi', model: 'sonnet' });
    expect(res.status).toBe(200);
    expect(backend.turns[0]?.model).toBe('sonnet');
  });

  it('accepts a full dashed model id', async () => {
    const backend = new FakeBackend(() => [doneEvent('ok')]);
    const base = await serve(backend);
    await post(base, { session_id: 's1', text: 'hi', model: 'claude-sonnet-4-6' });
    expect(backend.turns[0]?.model).toBe('claude-sonnet-4-6');
  });

  it('omits model from the turn when the request has none', async () => {
    const backend = new FakeBackend(() => [doneEvent('ok')]);
    const base = await serve(backend);
    await post(base, { session_id: 's1', text: 'hi' });
    expect(backend.turns[0]?.model).toBeUndefined();
  });

  it('rejects an invalid model with 400', async () => {
    const backend = new FakeBackend(() => [doneEvent('ok')]);
    const base = await serve(backend);
    expect((await post(base, { session_id: 's1', text: 'hi', model: '../evil' })).status).toBe(400);
    expect((await post(base, { session_id: 's1', text: 'hi', model: '' })).status).toBe(400);
    expect((await post(base, { session_id: 's1', text: 'hi', model: 'a'.repeat(65) })).status).toBe(
      400,
    );
    expect((await post(base, { session_id: 's1', text: 'hi', model: 7 })).status).toBe(400);
    expect((await post(base, { session_id: 's1', text: 'hi', model: 'so net' })).status).toBe(400);
    expect(backend.turns).toHaveLength(0);
  });

  it('aborts the turn signal when the client disconnects mid-stream', async () => {
    // Pins the voice-surface interruption contract: the moment the consumer
    // goes away, the backend's turn signal fires (which the live backend
    // turns into an interrupt-and-drain of the child).
    let seenTurn: Turn | undefined;
    const backend: Backend = {
      converse(turn: Turn): Promise<AsyncIterable<Event>> {
        seenTurn = turn;
        async function* events(): AsyncGenerator<Event> {
          yield { kind: 'textDelta', text: 'started' };
          // Block until the turn is aborted — a turn mid-generation.
          await new Promise<void>((resolve) => {
            turn.signal?.addEventListener('abort', () => {
              resolve();
            });
          });
        }
        return Promise.resolve(events());
      },
      closeSession: () => Promise.resolve(),
    };
    const base = await serve(backend);
    const controller = new AbortController();
    const res = await fetch(`${base}/v1/conversation`, {
      method: 'POST',
      body: JSON.stringify({ session_id: 's1', text: 'hi' }),
      signal: controller.signal,
    });
    const reader = res.body?.getReader();
    await reader?.read(); // first delta arrived; the stream is live
    controller.abort(); // client hangs up mid-stream
    await vi.waitFor(() => {
      expect(seenTurn?.signal?.aborted).toBe(true);
    });
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

  it('survives a client aborting mid-request-body', async () => {
    const backend = new FakeBackend(() => [doneEvent('ok')]);
    const base = await serve(backend);
    const port = Number(new URL(base).port);
    await new Promise<void>((resolve) => {
      const socket = connect(port, '127.0.0.1', () => {
        socket.write(
          'POST /v1/conversation HTTP/1.1\r\nHost: x\r\nContent-Length: 1000\r\n\r\n{"partial',
          () => {
            socket.destroy(); // abort mid-body: must not crash the daemon
            resolve();
          },
        );
      });
    });
    await delay(100);
    const res = await fetch(`${base}/healthz`);
    expect(res.status).toBe(200);
  });

  it('404s unknown routes and methods', async () => {
    const base = await serve(new FakeBackend(() => []));
    expect((await fetch(`${base}/v1/conversation`)).status).toBe(404);
    expect((await fetch(`${base}/nope`)).status).toBe(404);
  });
});

describe('POST /v1/voice/token', () => {
  const grant = {
    token: 'signed-token',
    room: 'android-device-id',
    url: 'wss://voice.example.test',
    expires_at: '2026-08-19T13:34:56.000Z',
  };
  const issuer: TokenIssuer = { issue: () => grant };

  it('returns the issued grant as single-line JSON and issues without context for a non-JSON body', async () => {
    const issue = vi.fn<TokenIssuer['issue']>(() => grant);
    const base = await serve(new FakeBackend(() => []), new SessionTracker(), { issue });

    const res = await fetch(`${base}/v1/voice/token`, {
      method: 'POST',
      body: 'not json \x01',
    });

    expect(res.status).toBe(200);
    expect(res.headers.get('content-type')).toBe('application/json');
    expect(await res.text()).toBe(`${JSON.stringify(grant)}\n`);
    expect(issue).toHaveBeenCalledWith(undefined);
  });

  it('passes the call context in the body to the issuer', async () => {
    const issue = vi.fn<TokenIssuer['issue']>(() => grant);
    const base = await serve(new FakeBackend(() => []), new SessionTracker(), { issue });

    const res = await fetch(`${base}/v1/voice/token`, {
      method: 'POST',
      body: JSON.stringify({ context: { time_zone: 'America/Chicago', driving: true } }),
    });

    expect(res.status).toBe(200);
    expect(issue).toHaveBeenCalledWith({ timeZone: 'America/Chicago', driving: true });
  });

  it('returns 500 when issuance throws and continues serving requests', async () => {
    const issue = vi
      .fn<TokenIssuer['issue']>()
      .mockImplementationOnce(() => {
        throw new Error('signing failed');
      })
      .mockReturnValue(grant);
    const logger = { ...nullLogger, error: vi.fn<Logger['error']>() };
    const base = await serve(
      new FakeBackend(() => []),
      new SessionTracker(),
      { issue },
      logger,
    );

    const failed = await fetch(`${base}/v1/voice/token`, {
      method: 'POST',
      signal: AbortSignal.timeout(100),
    });

    expect(failed.status).toBe(500);
    expect(await failed.text()).toBe('{"error":"internal"}\n');
    expect(logger.error).toHaveBeenCalledWith('voice token issuance failed', {
      error: 'Error: signing failed',
    });

    const succeeded = await fetch(`${base}/v1/voice/token`, { method: 'POST' });
    expect(succeeded.status).toBe(200);
    expect(await succeeded.text()).toBe(`${JSON.stringify(grant)}\n`);
  });

  it('returns the existing not-found response when no issuer is configured', async () => {
    const base = await serve(new FakeBackend(() => []));

    const res = await fetch(`${base}/v1/voice/token`, { method: 'POST' });

    expect(res.status).toBe(404);
    expect(await res.text()).toBe('{"error":"not found"}\n');
  });

  it('does not expose the token route over GET', async () => {
    const base = await serve(new FakeBackend(() => []), new SessionTracker(), issuer);

    const res = await fetch(`${base}/v1/voice/token`);

    expect(res.status).toBe(404);
    expect(await res.text()).toBe('{"error":"not found"}\n');
  });
});

describe('Phone routes', () => {
  it('opens the phone command stream with headers before body data', async () => {
    const bridge = new PhoneBridge(nullLogger, { heartbeatMs: 20_000 });
    const base = await serve(new FakeBackend(() => []), new SessionTracker(), undefined, nullLogger, bridge);
    const res = await fetch(`${base}/v1/phone/commands`, { headers: { 'X-Mentat-Phone': '1' } });
    expect(res.status).toBe(200);
    expect(res.headers.get('content-type')).toBe('application/x-ndjson');
    bridge.close();
    await res.body?.cancel();
  });

  it('rejects a command stream without the phone header without evicting the attached phone', async () => {
    const bridge = new PhoneBridge(nullLogger, {
      uuid: () => 'header-required',
      now: () => new Date('2026-09-09T12:00:00.000Z'),
    });
    const base = await serve(new FakeBackend(() => []), new SessionTracker(), undefined, nullLogger, bridge);
    const first = await fetch(`${base}/v1/phone/commands`, { headers: { 'X-Mentat-Phone': '1' } });
    if (first.body === null) throw new Error('missing first phone body');
    const firstReader = first.body.getReader() as unknown as ReadableStreamDefaultReader<Uint8Array>;
    let pending: ReturnType<PhoneBridge['dispatch']> | undefined;
    try {
      const rejected = await fetch(`${base}/v1/phone/commands`);
      expect(rejected.status).toBe(403);
      expect(await rejected.text()).toBe('{"error":"phone header required"}\n');

      pending = bridge.dispatch({ kind: 'sms', to: 'Sarah', body: 'still attached' });
      const chunk = await firstReader.read();
      if (chunk.done) throw new Error('missing command');
      expect(JSON.parse(new TextDecoder().decode(chunk.value))).toMatchObject({
        id: 'header-required',
        kind: 'sms',
        to: 'Sarah',
        body: 'still attached',
      });
    } finally {
      bridge.close();
      if (pending !== undefined) await expect(pending).rejects.toThrow('phone offline');
      await firstReader.cancel();
    }
  });

  it('writes heartbeat pings to the phone command stream', async () => {
    const bridge = new PhoneBridge(nullLogger, { heartbeatMs: 50 });
    const base = await serve(new FakeBackend(() => []), new SessionTracker(), undefined, nullLogger, bridge);
    const response = await fetch(`${base}/v1/phone/commands`, { headers: { 'X-Mentat-Phone': '1' } });
    if (response.body === null) throw new Error('missing phone body');
    const reader = response.body.getReader() as unknown as ReadableStreamDefaultReader<Uint8Array>;
    try {
      const ping = (async () => {
        let text = '';
        for (;;) {
          const chunk = await reader.read();
          if (chunk.done) throw new Error('phone stream ended before ping');
          text += new TextDecoder().decode(chunk.value);
          const newline = text.indexOf('\n');
          if (newline >= 0) {
            const line = text.slice(0, newline);
            if (line === '{"kind":"ping"}') return;
            text = text.slice(newline + 1);
          }
        }
      })();
      await Promise.race([
        ping,
        delay(1_000).then(() => { throw new Error('timed out waiting for phone ping'); }),
      ]);
    } finally {
      bridge.close();
      await reader.cancel();
    }
  });

  it('ends the first phone stream when a second phone connects', async () => {
    const bridge = new PhoneBridge(nullLogger, { heartbeatMs: 20_000 });
    const base = await serve(new FakeBackend(() => []), new SessionTracker(), undefined, nullLogger, bridge);
    const first = await fetch(`${base}/v1/phone/commands`, { headers: { 'X-Mentat-Phone': '1' } });
    const second = await fetch(`${base}/v1/phone/commands`, { headers: { 'X-Mentat-Phone': '1' } });
    if (first.body === null || second.body === null) throw new Error('missing phone body');
    const firstReader = first.body.getReader() as unknown as ReadableStreamDefaultReader<Uint8Array>;
    const secondReader = second.body.getReader();
    try {
      await expect(firstReader.read()).resolves.toMatchObject({ done: true });
      const secondRead = await Promise.race([
        secondReader.read().then((result) => result.done),
        delay(100).then(() => false),
      ]);
      expect(secondRead).toBe(false);
    } finally {
      bridge.close();
      await firstReader.cancel();
      await secondReader.cancel();
    }
  });

  it('resolves a pending command from the results route', async () => {
    const bridge = new PhoneBridge(nullLogger, {
      uuid: () => 'server-command',
      now: () => new Date('2026-09-09T12:00:00.000Z'),
    });
    const base = await serve(new FakeBackend(() => []), new SessionTracker(), undefined, nullLogger, bridge);
    const stream = await fetch(`${base}/v1/phone/commands`, { headers: { 'X-Mentat-Phone': '1' } });
    const pending = bridge.dispatch({ kind: 'sms', to: 'Sarah', body: 'late' });
    const reader = stream.body?.getReader();
    const chunk = await reader?.read();
    if (chunk === undefined || chunk.done || chunk.value === undefined) throw new Error('missing command');
    expect(new TextDecoder().decode(chunk.value as Uint8Array)).toBe(
      '{"id":"server-command","kind":"sms","to":"Sarah","body":"late","expires_at":"' +
        '2026-09-09T12:00:15.000Z' +
        '"}\n',
    );
    const result = await fetch(`${base}/v1/phone/results`, {
      method: 'POST',
      body: JSON.stringify({ id: 'server-command', status: 'ok', detail: 'sent' }),
    });
    expect(result.status).toBe(204);
    await expect(pending).resolves.toEqual({ detail: 'sent' });
    bridge.close();
    await reader?.cancel();
  });

  it('accepts object payloads and preserves UTF-8 and escaped text', async () => {
    const bridge = new PhoneBridge(nullLogger, { uuid: () => 'payload-command' });
    const base = await serve(new FakeBackend(() => []), new SessionTracker(), undefined, nullLogger, bridge);
    const stream = await fetch(`${base}/v1/phone/commands`, { headers: { 'X-Mentat-Phone': '1' } });
    const pending = bridge.dispatch({ kind: 'conversations', limit: 1 });
    const reader = stream.body?.getReader();
    const chunk = await reader?.read();
    if (chunk === undefined || chunk.done || chunk.value === undefined) throw new Error('missing command');
    const result = await fetch(`${base}/v1/phone/results`, {
      method: 'POST',
      body: JSON.stringify({
        id: 'payload-command',
        status: 'ok',
        detail: 'héllo ☃',
        payload: { text: 'héllo ☃ "quoted"\nline two' },
      }),
    });
    expect(result.status).toBe(204);
    await expect(pending).resolves.toEqual({
      detail: 'héllo ☃',
      payload: { text: 'héllo ☃ "quoted"\nline two' },
    });
    bridge.close();
    await reader?.cancel();
  });

  it('rejects non-object phone result payloads', async () => {
    const bridge = new PhoneBridge(nullLogger);
    const base = await serve(new FakeBackend(() => []), new SessionTracker(), undefined, nullLogger, bridge);
    for (const payload of [[], null, 'x', 1]) {
      const res = await fetch(`${base}/v1/phone/results`, {
        method: 'POST',
        body: JSON.stringify({ id: 'payload', status: 'ok', detail: 'sent', payload }),
      });
      expect(res.status).toBe(400);
      expect(await res.text()).toBe('{"error":"id, status, and detail are required"}\n');
    }
  });

  it('rejects malformed phone results and method mismatches', async () => {
    const bridge = new PhoneBridge(nullLogger);
    const base = await serve(new FakeBackend(() => []), new SessionTracker(), undefined, nullLogger, bridge);
    for (const body of ['not json', JSON.stringify({ status: 'ok', detail: 'sent' }), JSON.stringify({ id: 'x', status: 'bad', detail: 'sent' })]) {
      const res = await fetch(`${base}/v1/phone/results`, { method: 'POST', body });
      expect(res.status).toBe(400);
    }
    expect((await fetch(`${base}/mcp`)).status).toBe(405);
    expect((await fetch(`${base}/mcp`, { method: 'DELETE' })).status).toBe(405);
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
