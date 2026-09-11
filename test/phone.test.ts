import { EventEmitter } from 'node:events';
import type { ServerResponse } from 'node:http';

import { afterEach, describe, expect, it, vi } from 'vitest';

import { nullLogger } from '../src/log.ts';
import { PhoneBridge } from '../src/phone.ts';

type FakeResponse = EventEmitter & {
  destroyed: boolean;
  ended: boolean;
  chunks: string[];
  writeHead: (status: number, headers?: Record<string, string>) => void;
  flushHeaders: () => void;
  write: (chunk: string) => boolean;
  end: () => void;
};

function response(): FakeResponse {
  const emitter = new EventEmitter() as FakeResponse;
  emitter.destroyed = false;
  emitter.ended = false;
  emitter.chunks = [];
  emitter.writeHead = () => undefined;
  emitter.flushHeaders = () => undefined;
  emitter.write = (chunk: string): boolean => {
    emitter.chunks.push(chunk);
    return true;
  };
  emitter.end = (): void => {
    emitter.ended = true;
    emitter.destroyed = true;
    emitter.emit('close');
  };
  return emitter;
}

function bridgeAt(
  now: Date = new Date('2026-09-09T12:00:00.000Z'),
  timeoutMs = 15_000,
  heartbeatMs = 20_000,
): { bridge: PhoneBridge; setNow: (value: Date) => void } {
  let current = now;
  const bridge = new PhoneBridge(nullLogger, {
    now: () => current,
    uuid: () => 'command-id',
    timeoutMs,
    heartbeatMs,
  });
  return { bridge, setNow: (value) => (current = value) };
}

afterEach(() => {
  vi.useRealTimers();
});

describe('PhoneBridge', () => {
  it('rejects dispatch immediately while the phone is offline', async () => {
    const { bridge } = bridgeAt();
    await expect(bridge.dispatch({ kind: 'sms', to: 'Sarah', body: 'late' })).rejects.toThrow(
      'phone offline',
    );
  });

  it('writes SMS and open commands with an expiration timestamp', async () => {
    const { bridge } = bridgeAt();
    const phone = response();
    bridge.attach(phone as unknown as ServerResponse);

    const sms = bridge.dispatch({ kind: 'sms', to: 'Sarah', body: 'late' });
    expect(phone.chunks).toEqual([
      '{"id":"command-id","kind":"sms","to":"Sarah","body":"late","expires_at":"2026-09-09T12:00:15.000Z"}\n',
    ]);
    expect(bridge.complete({ id: 'command-id', status: 'ok', detail: 'sent' })).toBe(true);
    await expect(sms).resolves.toEqual({ detail: 'sent' });

    const messages = bridge.dispatch({
      kind: 'messages',
      conversation: 'sms:12',
      limit: 7,
      before: '1789001656600:sms:s12',
    });
    expect(phone.chunks.at(-1)).toBe(
      '{"id":"command-id","kind":"messages","conversation":"sms:12","limit":7,' +
        '"before":"1789001656600:sms:s12","expires_at":"2026-09-09T12:00:15.000Z"}\n',
    );
    expect(bridge.complete({
      id: 'command-id',
      status: 'ok',
      detail: 'read',
      payload: { messages: [{ id: 'sms:s12' }] },
    })).toBe(true);
    await expect(messages).resolves.toEqual({
      detail: 'read',
      payload: { messages: [{ id: 'sms:s12' }] },
    });

    const conversations = bridge.dispatch({ kind: 'conversations', limit: 3 });
    expect(phone.chunks.at(-1)).toBe(
      '{"id":"command-id","kind":"conversations","limit":3,' +
        '"expires_at":"2026-09-09T12:00:15.000Z"}\n',
    );
    expect(bridge.complete({ id: 'command-id', status: 'ok', detail: 'listed' })).toBe(true);
    await expect(conversations).resolves.toEqual({ detail: 'listed' });

    const open = bridge.dispatch({ kind: 'open', uri: 'google.navigation:q=Union+Station' });
    expect(phone.chunks.at(-1)).toBe(
      '{"id":"command-id","kind":"open","uri":"google.navigation:q=Union+Station","expires_at":"2026-09-09T12:00:15.000Z"}\n',
    );
    expect(bridge.complete({ id: 'command-id', status: 'ok', detail: 'launched' })).toBe(true);
    await expect(open).resolves.toEqual({ detail: 'launched' });
  });

  it('writes navigation, dial, alarm, timer, and location commands with expiration timestamps', async () => {
    const { bridge } = bridgeAt();
    const phone = response();
    bridge.attach(phone as unknown as ServerResponse);

    const navigate = bridge.dispatch({
      kind: 'navigate',
      name: 'Union Station',
      address: '800 N 6th Ave, Portland, OR',
      place_id: 'ChIJplace',
      lat: 45.528,
      lng: -122.676,
    });
    expect(phone.chunks.at(-1)).toBe(
      '{"id":"command-id","kind":"navigate","name":"Union Station",' +
        '"address":"800 N 6th Ave, Portland, OR","place_id":"ChIJplace",' +
        '"lat":45.528,"lng":-122.676,"expires_at":"2026-09-09T12:00:15.000Z"}\n',
    );
    expect(bridge.complete({ id: 'command-id', status: 'ok', detail: 'started' })).toBe(true);
    await expect(navigate).resolves.toEqual({ detail: 'started' });

    const dial = bridge.dispatch({ kind: 'dial', number: '+15555550123' });
    expect(phone.chunks.at(-1)).toBe(
      '{"id":"command-id","kind":"dial","number":"+15555550123",' +
        '"expires_at":"2026-09-09T12:00:15.000Z"}\n',
    );
    expect(bridge.complete({ id: 'command-id', status: 'ok', detail: 'ready' })).toBe(true);
    await expect(dial).resolves.toEqual({ detail: 'ready' });

    const alarm = bridge.dispatch({ kind: 'alarm', hour: 7, minute: 30, label: 'Wake up' });
    expect(phone.chunks.at(-1)).toBe(
      '{"id":"command-id","kind":"alarm","hour":7,"minute":30,"label":"Wake up",' +
        '"expires_at":"2026-09-09T12:00:15.000Z"}\n',
    );
    expect(bridge.complete({ id: 'command-id', status: 'ok', detail: 'set' })).toBe(true);
    await expect(alarm).resolves.toEqual({ detail: 'set' });

    const timer = bridge.dispatch({ kind: 'timer', seconds: 90 });
    expect(phone.chunks.at(-1)).toBe(
      '{"id":"command-id","kind":"timer","seconds":90,' +
        '"expires_at":"2026-09-09T12:00:15.000Z"}\n',
    );
    expect(bridge.complete({ id: 'command-id', status: 'ok', detail: 'started' })).toBe(true);
    await expect(timer).resolves.toEqual({ detail: 'started' });

    const location = bridge.dispatch({ kind: 'location' });
    expect(phone.chunks.at(-1)).toBe(
      '{"id":"command-id","kind":"location",' +
        '"expires_at":"2026-09-09T12:00:15.000Z"}\n',
    );
    const payload = { lat: 45.52, lng: -122.67, accuracy_m: 12, age_s: 3 };
    expect(bridge.complete({ id: 'command-id', status: 'ok', detail: 'located', payload })).toBe(true);
    await expect(location).resolves.toEqual({ detail: 'located', payload });
  });

  it('resolves successful results and rejects error results with the detail', async () => {
    const { bridge } = bridgeAt();
    const phone = response();
    bridge.attach(phone as unknown as ServerResponse);

    const success = bridge.dispatch({ kind: 'sms', to: 'Sarah', body: 'late' });
    expect(bridge.complete({ id: 'command-id', status: 'ok', detail: 'sent to +15555550123' })).toBe(
      true,
    );
    await expect(success).resolves.toEqual({ detail: 'sent to +15555550123' });

    const failure = bridge.dispatch({ kind: 'open', uri: 'geo:0,0?q=Union+Station' });
    expect(bridge.complete({ id: 'command-id', status: 'error', detail: 'not launched' })).toBe(true);
    await expect(failure).rejects.toThrow('not launched');
  });

  it('times out once and ignores late or unknown results', async () => {
    vi.useFakeTimers();
    const { bridge } = bridgeAt();
    const phone = response();
    bridge.attach(phone as unknown as ServerResponse);
    const pending = bridge.dispatch({ kind: 'sms', to: 'Sarah', body: 'late' });

    const timedOut = expect(pending).rejects.toThrow('timed out: outcome unknown');
    await vi.advanceTimersByTimeAsync(15_000);
    await timedOut;
    expect(bridge.complete({ id: 'command-id', status: 'ok', detail: 'late' })).toBe(false);
    expect(bridge.complete({ id: 'unknown', status: 'ok', detail: 'unknown' })).toBe(false);
    expect(phone.chunks).toHaveLength(1);
  });

  it('sends heartbeat pings while attached and stops when detached', async () => {
    vi.useFakeTimers();
    const { bridge } = bridgeAt(undefined, 15_000, 20_000);
    const phone = response();
    bridge.attach(phone as unknown as ServerResponse);

    await vi.advanceTimersByTimeAsync(20_000);
    expect(phone.chunks).toEqual(['{"kind":"ping"}\n']);
    bridge.close();
    await vi.advanceTimersByTimeAsync(40_000);
    expect(phone.chunks).toEqual(['{"kind":"ping"}\n']);
  });

  it('replaces an attached stream and closes the old one', () => {
    const { bridge } = bridgeAt();
    const first = response();
    const second = response();
    bridge.attach(first as unknown as ServerResponse);
    bridge.attach(second as unknown as ServerResponse);
    expect(first.ended).toBe(true);
    expect(second.ended).toBe(false);
  });

  it('closes the stream and rejects every pending command', async () => {
    let nextId = 0;
    const bridge = new PhoneBridge(nullLogger, { uuid: () => `close-${String(nextId++)}` });
    const phone = response();
    bridge.attach(phone as unknown as ServerResponse);
    const first = bridge.dispatch({ kind: 'sms', to: 'Sarah', body: 'late' });
    const second = bridge.dispatch({ kind: 'open', uri: 'https://example.test' });

    const firstRejection = expect(first).rejects.toThrow('phone offline');
    const secondRejection = expect(second).rejects.toThrow('phone offline');
    bridge.close();
    expect(phone.ended).toBe(true);
    await firstRejection;
    await secondRejection;
  });
});
