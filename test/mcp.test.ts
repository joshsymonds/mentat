import type { ReadableStreamDefaultReader } from 'node:stream/web';

import { afterEach, describe, expect, it, vi } from 'vitest';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';
import type { Transport } from '@modelcontextprotocol/sdk/shared/transport.js';

import { nullLogger } from '../src/log.ts';
import { PhoneBridge } from '../src/phone.ts';
import { SessionTracker, createHandler } from '../src/server.ts';
import type { PlacesDeps } from '../src/places.ts';
import type { Backend } from '../src/backend.ts';
import { createServer, type Server } from 'node:http';

const servers: Server[] = [];
const backend: Backend = {
  converse: () => Promise.resolve((async function* () { await Promise.resolve(); return; })()),
  closeSession: () => Promise.resolve(),
};

async function serve(bridge: PhoneBridge, places: PlacesDeps = {}): Promise<string> {
  const server = createServer(createHandler(backend, new SessionTracker(), nullLogger, undefined, { bridge, places }));
  servers.push(server);
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('server did not bind');
  return `http://127.0.0.1:${String(address.port)}`;
}

async function readLine(reader: ReadableStreamDefaultReader<Uint8Array>): Promise<string> {
  const decoder = new TextDecoder();
  let text = '';
  for (;;) {
    const chunk = await reader.read();
    if (chunk.done) throw new Error('phone stream ended');
    text += decoder.decode(chunk.value, { stream: true });
    const newline = text.indexOf('\n');
    if (newline >= 0) return text.slice(0, newline);
  }
}

function textContent(value: unknown): string {
  if (value === null || typeof value !== 'object') throw new Error('missing MCP result');
  const content = (value as { content?: unknown }).content;
  if (!Array.isArray(content)) throw new Error('missing MCP content');
  const first = (content as unknown[])[0];
  if (first === null || typeof first !== 'object') throw new Error('missing MCP text');
  const type = (first as { type?: unknown }).type;
  const text = (first as { text?: unknown }).text;
  if (type !== 'text' || typeof text !== 'string') throw new Error('missing MCP text');
  return text;
}

afterEach(async () => {
  for (const server of servers.splice(0)) {
    await new Promise((resolve) => server.close(resolve));
  }
});

describe('POST /mcp', () => {
  it('lists the phone tools with required string schemas', async () => {
    const bridge = new PhoneBridge(nullLogger);
    const base = await serve(bridge);
    const transport = new StreamableHTTPClientTransport(new URL(`${base}/mcp`));
    const client = new Client({ name: 'test-client', version: '1.0.0' });
    await client.connect(transport as Transport);
    const listed = await client.listTools();
    expect(listed.tools.map((tool) => tool.name)).toEqual([
      'send_sms',
      'open_on_phone',
      'list_conversations',
      'read_conversation',
      'search_messages',
      'find_places',
      'navigate_to',
      'dial',
      'set_alarm',
      'set_timer',
      'end_conversation',
    ]);
    const send = listed.tools.find((tool) => tool.name === 'send_sms');
    const open = listed.tools.find((tool) => tool.name === 'open_on_phone');
    expect(send).toBeDefined();
    expect(open).toBeDefined();
    expect(send?.inputSchema.type).toBe('object');
    expect(Object.keys(send?.inputSchema.properties ?? {})).toEqual(['to', 'body', 'send']);
    expect(send?.inputSchema.required).toEqual(['to', 'body']);
    expect(send?.inputSchema.properties?.to).toEqual({ type: 'string' });
    expect(send?.inputSchema.properties?.body).toEqual({ type: 'string' });
    expect(send?.inputSchema.properties?.send).toMatchObject({ type: 'boolean' });
    expect(open?.inputSchema.type).toBe('object');
    expect(Object.keys(open?.inputSchema.properties ?? {})).toEqual(['uri']);
    expect(open?.inputSchema.required).toEqual(['uri']);
    expect(open?.inputSchema.properties?.uri).toEqual({ type: 'string' });
    await client.close();
    bridge.close();
  });

  it('dispatches send_sms and returns the phone detail text', async () => {
    const bridge = new PhoneBridge(nullLogger, {
      uuid: () => 'mcp-sms',
      now: () => new Date('2026-09-09T12:00:00.000Z'),
    });
    const base = await serve(bridge);
    const phone = await fetch(`${base}/v1/phone/commands`, { headers: { 'X-Mentat-Phone': '1' } });
    if (phone.body === null) throw new Error('missing phone body');
    const reader = phone.body.getReader() as unknown as ReadableStreamDefaultReader<Uint8Array>;
    const transport = new StreamableHTTPClientTransport(new URL(`${base}/mcp`));
    const client = new Client({ name: 'test-client', version: '1.0.0' });
    await client.connect(transport as Transport);

    const call = client.callTool({ name: 'send_sms', arguments: { to: 'Sarah', body: 'late', send: true } });
    const commandLine = await readLine(reader);
    expect(commandLine).toBe(
      '{"id":"mcp-sms","kind":"sms","to":"Sarah","body":"late","expires_at":"2026-09-09T12:00:15.000Z"}',
    );
    const command = JSON.parse(commandLine) as unknown as Record<string, string>;
    await fetch(`${base}/v1/phone/results`, {
      method: 'POST',
      body: JSON.stringify({ id: command.id, status: 'ok', detail: 'sent to +15555550123' }),
    });
    const result = await call;
    expect(result).toMatchObject({ content: [{ type: 'text', text: 'sent to +15555550123' }] });
    await client.close();
    bridge.close();
    await reader.cancel();
  });

  it('requires explicit send confirmation before dispatching SMS', async () => {
    const bridge = new PhoneBridge(nullLogger);
    const dispatch = vi.spyOn(bridge, 'dispatch');
    const base = await serve(bridge);
    const transport = new StreamableHTTPClientTransport(new URL(`${base}/mcp`));
    const client = new Client({ name: 'test-client', version: '1.0.0' });
    await client.connect(transport as Transport);

    const result = await client.callTool({
      name: 'send_sms',
      arguments: { to: 'Sarah', body: 'late' },
    });
    expect(result).toEqual({
      isError: true,
      content: [{
        type: 'text',
        text: 'Message was not sent. Call again with send=true after confirming the recipient and message with the user.',
      }],
    });
    expect(dispatch).not.toHaveBeenCalled();
    await client.close();
    bridge.close();
  });

  it('dispatches open_on_phone and returns the phone detail text', async () => {
    const bridge = new PhoneBridge(nullLogger, {
      uuid: () => 'mcp-open',
      now: () => new Date('2026-09-09T12:00:00.000Z'),
    });
    const base = await serve(bridge);
    const phone = await fetch(`${base}/v1/phone/commands`, { headers: { 'X-Mentat-Phone': '1' } });
    if (phone.body === null) throw new Error('missing phone body');
    const reader = phone.body.getReader() as unknown as ReadableStreamDefaultReader<Uint8Array>;
    const transport = new StreamableHTTPClientTransport(new URL(`${base}/mcp`));
    const client = new Client({ name: 'test-client', version: '1.0.0' });
    await client.connect(transport as Transport);

    const call = client.callTool({
      name: 'open_on_phone',
      arguments: { uri: 'google.navigation:q=Union+Station' },
    });
    const commandLine = await readLine(reader);
    expect(commandLine).toBe(
      '{"id":"mcp-open","kind":"open","uri":"google.navigation:q=Union+Station","expires_at":"2026-09-09T12:00:15.000Z"}',
    );
    const command = JSON.parse(commandLine) as unknown as Record<string, string>;
    await fetch(`${base}/v1/phone/results`, {
      method: 'POST',
      body: JSON.stringify({ id: command.id, status: 'ok', detail: 'launched' }),
    });
    const result = await call;
    expect(result).toMatchObject({ content: [{ type: 'text', text: 'launched' }] });
    await client.close();
    bridge.close();
    await reader.cancel();
  });

  it('dispatches read tools and returns structured payloads as JSON text', async () => {
    const ids = ['mcp-list', 'mcp-read', 'mcp-search'];
    const bridge = new PhoneBridge(nullLogger, {
      uuid: () => ids.shift() ?? 'unexpected',
      now: () => new Date('2026-09-09T12:00:00.000Z'),
    });
    const base = await serve(bridge);
    const phone = await fetch(`${base}/v1/phone/commands`, { headers: { 'X-Mentat-Phone': '1' } });
    if (phone.body === null) throw new Error('missing phone body');
    const reader = phone.body.getReader() as unknown as ReadableStreamDefaultReader<Uint8Array>;
    const transport = new StreamableHTTPClientTransport(new URL(`${base}/mcp`));
    const client = new Client({ name: 'test-client', version: '1.0.0' });
    await client.connect(transport as Transport);

    const cases = [
      {
        name: 'list_conversations',
        args: { channel: 'sms', limit: 3 },
        command: { kind: 'conversations', limit: 3 },
        payload: { conversations: [] },
      },
      {
        name: 'read_conversation',
        args: { conversation: 'sms:12', limit: 4, before: '1789001656600:sms:s12' },
        command: {
          kind: 'messages',
          conversation: 'sms:12',
          limit: 4,
          before: '1789001656600:sms:s12',
        },
        payload: { messages: [] },
      },
      {
        name: 'search_messages',
        args: { query: 'hello', channel: 'sms', limit: 5, before: '2026-09-09T00:00:00Z' },
        command: {
          kind: 'search',
          query: 'hello',
          limit: 5,
          before: '2026-09-09T00:00:00Z',
        },
        payload: { messages: [] },
      },
    ] as const;
    for (const testCase of cases) {
      const call = client.callTool({ name: testCase.name, arguments: testCase.args });
      const commandLine = await readLine(reader);
      const command = JSON.parse(commandLine) as Record<string, unknown>;
      expect(command).toMatchObject(testCase.command);
      await fetch(`${base}/v1/phone/results`, {
        method: 'POST',
        body: JSON.stringify({ id: command.id, status: 'ok', detail: 'read', payload: testCase.payload }),
      });
      const result = await call;
      expect(result).toMatchObject({
        content: [{ type: 'text', text: JSON.stringify(testCase.payload) }],
      });
    }

    await client.close();
    bridge.close();
    await reader.cancel();
  });

  it('maps read phone errors and malformed successful results', async () => {
    const bridge = new PhoneBridge(nullLogger, { uuid: () => 'mcp-read-error' });
    const base = await serve(bridge);
    const phone = await fetch(`${base}/v1/phone/commands`, { headers: { 'X-Mentat-Phone': '1' } });
    if (phone.body === null) throw new Error('missing phone body');
    const reader = phone.body.getReader() as unknown as ReadableStreamDefaultReader<Uint8Array>;
    const transport = new StreamableHTTPClientTransport(new URL(`${base}/mcp`));
    const client = new Client({ name: 'test-client', version: '1.0.0' });
    await client.connect(transport as Transport);

    const failedCall = client.callTool({
      name: 'read_conversation',
      arguments: { conversation: 'sms:12' },
    });
    const failedCommand = JSON.parse(await readLine(reader)) as Record<string, unknown>;
    await fetch(`${base}/v1/phone/results`, {
      method: 'POST',
      body: JSON.stringify({ id: failedCommand.id, status: 'error', detail: 'not found' }),
    });
    await expect(failedCall).resolves.toMatchObject({
      isError: true,
      content: [{ type: 'text', text: 'not found' }],
    });

    const malformedCall = client.callTool({
      name: 'list_conversations',
      arguments: {},
    });
    const malformedCommand = JSON.parse(await readLine(reader)) as Record<string, unknown>;
    await fetch(`${base}/v1/phone/results`, {
      method: 'POST',
      body: JSON.stringify({ id: malformedCommand.id, status: 'ok', detail: 'listed' }),
    });
    await expect(malformedCall).resolves.toMatchObject({
      isError: true,
      content: [{ type: 'text', text: 'malformed result' }],
    });

    await client.close();
    bridge.close();
    await reader.cancel();
  });

  it('rejects unsupported channels and prefixed conversation ids before dispatch', async () => {
    const offlineBridge = new PhoneBridge(nullLogger);
    const offlineBase = await serve(offlineBridge);
    const offlineClient = new Client({ name: 'test-client', version: '1.0.0' });
    const offlineTransport = new StreamableHTTPClientTransport(new URL(`${offlineBase}/mcp`));
    await offlineClient.connect(offlineTransport as Transport);

    for (const args of [
      { name: 'list_conversations', arguments: { channel: 'signal' } },
      { name: 'search_messages', arguments: { query: 'hello', channel: 'signal' } },
      { name: 'read_conversation', arguments: { conversation: 'signal:12' } },
    ]) {
      const call = await offlineClient.callTool(args);
      expect(call).toMatchObject({ isError: true });
      expect((call.content as { text: string }[])[0]?.text).toContain('sms');
    }
    await offlineClient.close();
    offlineBridge.close();

    const bridge = new PhoneBridge(nullLogger, { uuid: () => 'mcp-channel' });
    const base = await serve(bridge);
    const phone = await fetch(`${base}/v1/phone/commands`, { headers: { 'X-Mentat-Phone': '1' } });
    if (phone.body === null) throw new Error('missing phone body');
    const reader = phone.body.getReader() as unknown as ReadableStreamDefaultReader<Uint8Array>;
    const client = new Client({ name: 'test-client', version: '1.0.0' });
    const transport = new StreamableHTTPClientTransport(new URL(`${base}/mcp`));
    await client.connect(transport as Transport);

    for (const conversation of ['sms:12', 'Justin']) {
      const call = client.callTool({ name: 'read_conversation', arguments: { conversation } });
      const command = JSON.parse(await readLine(reader)) as Record<string, unknown>;
      expect(command).toMatchObject({ kind: 'messages', conversation });
      await fetch(`${base}/v1/phone/results`, {
        method: 'POST',
        body: JSON.stringify({ id: command.id, status: 'ok', detail: 'read', payload: { messages: [] } }),
      });
      await expect(call).resolves.toMatchObject({ content: [{ type: 'text', text: '{"messages":[]}' }] });
    }

    await client.close();
    bridge.close();
    await reader.cancel();
  });

  it('requires strict ISO datetime before values and dispatches valid forms', async () => {
    const invalidBridge = new PhoneBridge(nullLogger);
    const dispatch = vi.spyOn(invalidBridge, 'dispatch');
    const invalidBase = await serve(invalidBridge);
    const invalidClient = new Client({ name: 'test-client', version: '1.0.0' });
    const invalidTransport = new StreamableHTTPClientTransport(new URL(`${invalidBase}/mcp`));
    await invalidClient.connect(invalidTransport as Transport);

    for (const args of [
      { name: 'read_conversation', arguments: { conversation: 'sms:12', before: 'September 9, 2026' } },
      { name: 'search_messages', arguments: { query: 'hello', before: '9/9/2026' } },
    ]) {
      const result = await invalidClient.callTool(args);
      expect(result).toMatchObject({ isError: true });
      expect((result.content as { text: string }[])[0]?.text).not.toBe('phone offline');
    }
    expect(dispatch).not.toHaveBeenCalled();
    await invalidClient.close();
    invalidBridge.close();

    const bridge = new PhoneBridge(nullLogger, { uuid: () => 'mcp-valid-before' });
    const base = await serve(bridge);
    const phone = await fetch(`${base}/v1/phone/commands`, { headers: { 'X-Mentat-Phone': '1' } });
    if (phone.body === null) throw new Error('missing phone body');
    const reader = phone.body.getReader() as unknown as ReadableStreamDefaultReader<Uint8Array>;
    const client = new Client({ name: 'test-client', version: '1.0.0' });
    const transport = new StreamableHTTPClientTransport(new URL(`${base}/mcp`));
    await client.connect(transport as Transport);

    const cases = [
      {
        name: 'read_conversation',
        args: { conversation: 'sms:12', before: '2026-09-09T00:00:00Z' },
        command: { kind: 'messages', conversation: 'sms:12', before: '2026-09-09T00:00:00Z' },
      },
      {
        name: 'read_conversation',
        args: { conversation: 'sms:12', before: '2026-09-09T00:00:00.000+02:00' },
        command: { kind: 'messages', conversation: 'sms:12', before: '2026-09-09T00:00:00.000+02:00' },
      },
      {
        name: 'search_messages',
        args: { query: 'hello', before: '1789001656600:sms:s12' },
        command: { kind: 'search', query: 'hello', before: '1789001656600:sms:s12' },
      },
    ] as const;
    for (const testCase of cases) {
      const call = client.callTool({ name: testCase.name, arguments: testCase.args });
      const command = JSON.parse(await readLine(reader)) as Record<string, unknown>;
      expect(command).toMatchObject(testCase.command);
      await fetch(`${base}/v1/phone/results`, {
        method: 'POST',
        body: JSON.stringify({ id: command.id, status: 'ok', detail: 'ok', payload: { messages: [] } }),
      });
      await expect(call).resolves.toMatchObject({ content: [{ type: 'text', text: '{"messages":[]}' }] });
    }

    await client.close();
    bridge.close();
    await reader.cancel();
  });

  it('applies read defaults and clamps before dispatch', async () => {
    let nextId = 0;
    const bridge = new PhoneBridge(nullLogger, {
      uuid: () => `mcp-limits-${String(nextId++)}`,
      now: () => new Date('2026-09-09T12:00:00.000Z'),
    });
    const base = await serve(bridge);
    const phone = await fetch(`${base}/v1/phone/commands`, { headers: { 'X-Mentat-Phone': '1' } });
    if (phone.body === null) throw new Error('missing phone body');
    const reader = phone.body.getReader() as unknown as ReadableStreamDefaultReader<Uint8Array>;
    const client = new Client({ name: 'test-client', version: '1.0.0' });
    const transport = new StreamableHTTPClientTransport(new URL(`${base}/mcp`));
    await client.connect(transport as Transport);

    const cases = [
      { name: 'list_conversations', args: {}, command: { kind: 'conversations', limit: 20 } },
      { name: 'list_conversations', args: { limit: 500 }, command: { kind: 'conversations', limit: 100 } },
      { name: 'read_conversation', args: { conversation: 'sms:12' }, command: { kind: 'messages', conversation: 'sms:12', limit: 50 } },
      { name: 'read_conversation', args: { conversation: 'sms:12', limit: 500 }, command: { kind: 'messages', conversation: 'sms:12', limit: 200 } },
      { name: 'search_messages', args: { query: 'hello' }, command: { kind: 'search', query: 'hello', limit: 30 } },
      { name: 'search_messages', args: { query: 'hello', limit: 500 }, command: { kind: 'search', query: 'hello', limit: 100 } },
      { name: 'read_conversation', args: { conversation: 'sms:12', before: '2026-09-09T00:00:00Z' }, command: { kind: 'messages', conversation: 'sms:12', limit: 50, before: '2026-09-09T00:00:00Z' } },
      { name: 'search_messages', args: { query: 'hello', before: '1789001656600:sms:s12' }, command: { kind: 'search', query: 'hello', limit: 30, before: '1789001656600:sms:s12' } },
    ] as const;
    for (const testCase of cases) {
      const call = client.callTool({ name: testCase.name, arguments: testCase.args });
      const command = JSON.parse(await readLine(reader)) as Record<string, unknown>;
      expect(command).toMatchObject(testCase.command);
      await fetch(`${base}/v1/phone/results`, {
        method: 'POST',
        body: JSON.stringify({ id: command.id, status: 'ok', detail: 'ok', payload: { messages: [] } }),
      });
      await call;
    }

    await client.close();
    bridge.close();
    await reader.cancel();

    const invalidBridge = new PhoneBridge(nullLogger);
    const invalidBase = await serve(invalidBridge);
    const invalidClient = new Client({ name: 'test-client', version: '1.0.0' });
    const invalidTransport = new StreamableHTTPClientTransport(new URL(`${invalidBase}/mcp`));
    await invalidClient.connect(invalidTransport as Transport);
    for (const args of [
      { name: 'list_conversations', arguments: { limit: 0 } },
      { name: 'read_conversation', arguments: { conversation: 'sms:12', limit: 1.5 } },
      { name: 'search_messages', arguments: { query: 'hello', before: 'yesterday' } },
    ]) {
      const call = await invalidClient.callTool(args);
      expect(call).toMatchObject({ isError: true });
      expect((call.content as { text: string }[])[0]?.text).not.toBe('phone offline');
    }
    await invalidClient.close();
    invalidBridge.close();
  });

  it('returns an MCP error immediately when the phone is offline', async () => {
    const bridge = new PhoneBridge(nullLogger);
    const base = await serve(bridge);
    const transport = new StreamableHTTPClientTransport(new URL(`${base}/mcp`));
    const client = new Client({ name: 'test-client', version: '1.0.0' });
    await client.connect(transport as Transport);
    const result = await client.callTool({
      name: 'send_sms',
      arguments: { to: 'Sarah', body: 'late', send: true },
    });
    expect(result.isError).toBe(true);
    expect(result).toMatchObject({ isError: true, content: [{ type: 'text', text: 'phone offline' }] });

    for (const args of [
      { name: 'list_conversations', arguments: {} },
      { name: 'read_conversation', arguments: { conversation: 'sms:12' } },
      { name: 'search_messages', arguments: { query: 'x' } },
    ]) {
      const readResult = await client.callTool(args);
      expect(readResult).toMatchObject({
        isError: true,
        content: [{ type: 'text', text: 'phone offline' }],
      });
    }

    await client.close();
    bridge.close();
  });
  it('finds nearby places from phone location and returns navigable candidates', async () => {
    const placesFetch = vi.fn((input: string | URL | Request, init?: RequestInit) => {
      const inputUrl = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url;
      expect(inputUrl).toBe('https://places.googleapis.com/v1/places:searchText');
      expect(init?.headers).toMatchObject({
        'X-Goog-Api-Key': 'places-key',
        'X-Goog-FieldMask': 'places.id,places.displayName,places.formattedAddress,places.location',
      });
      if (typeof init?.body !== 'string') throw new Error('missing request body');
      expect(JSON.parse(init.body)).toEqual({
        textQuery: 'coffee',
        pageSize: 5,
        locationBias: {
          circle: { center: { latitude: 45.52, longitude: -122.67 }, radius: 20000 },
        },
      });
      return Promise.resolve(new Response(JSON.stringify({
        places: [
          { id: 'place-1', displayName: { text: 'Coffee One' }, formattedAddress: '1 Main St, Portland, OR', location: { latitude: 45.521, longitude: -122.671 } },
          { id: 'place-2', displayName: { text: 'Coffee Two' }, formattedAddress: '2 Main St, Portland, OR', location: { latitude: 45.522, longitude: -122.672 } },
          { id: 'place-3', displayName: { text: 'Coffee Three' }, formattedAddress: '3 Main St, Portland, OR', location: { latitude: 45.523, longitude: -122.673 } },
          { id: 'place-4', displayName: { text: 'Coffee Four' }, formattedAddress: '4 Main St, Portland, OR', location: { latitude: 45.524, longitude: -122.674 } },
        ],
      })));
    });
    const bridge = new PhoneBridge(nullLogger, {
      uuid: () => 'mcp-location',
    });
    const base = await serve(bridge, { apiKey: 'places-key', fetch: placesFetch });
    const phone = await fetch(`${base}/v1/phone/commands`, { headers: { 'X-Mentat-Phone': '1' } });
    if (phone.body === null) throw new Error('missing phone body');
    const reader = phone.body.getReader() as unknown as ReadableStreamDefaultReader<Uint8Array>;
    const transport = new StreamableHTTPClientTransport(new URL(`${base}/mcp`));
    const client = new Client({ name: 'test-client', version: '1.0.0' });
    await client.connect(transport as Transport);

    const call = client.callTool({ name: 'find_places', arguments: { query: 'coffee' } });
    const command = JSON.parse(await readLine(reader)) as Record<string, unknown>;
    expect(command).toMatchObject({ kind: 'location' });
    await fetch(`${base}/v1/phone/results`, {
      method: 'POST',
      body: JSON.stringify({ id: command.id, status: 'ok', detail: 'located', payload: { lat: 45.52, lng: -122.67, accuracy_m: 12, age_s: 3 } }),
    });
    const result = await call;
    expect(result).toMatchObject({
      content: [{
        type: 'text',
        text: JSON.stringify([
          { name: 'Coffee One', address: '1 Main St, Portland, OR', place_id: 'place-1', lat: 45.521, lng: -122.671 },
          { name: 'Coffee Two', address: '2 Main St, Portland, OR', place_id: 'place-2', lat: 45.522, lng: -122.672 },
          { name: 'Coffee Three', address: '3 Main St, Portland, OR', place_id: 'place-3', lat: 45.523, lng: -122.673 },
        ]),
      }],
    });
    expect(placesFetch).toHaveBeenCalledOnce();
    await client.close();
    bridge.close();
    await reader.cancel();
  });

  it('requires locality when phone location is unavailable and searches without bias after locality', async () => {
    const placesFetch = vi.fn((_input: string | URL | Request, init?: RequestInit) => {
      if (typeof init?.body !== 'string') throw new Error('missing request body');
      expect(JSON.parse(init.body)).toEqual({ textQuery: 'coffee in Beaverton', pageSize: 5 });
      return Promise.resolve(new Response(JSON.stringify({ places: [] })));
    });
    const bridge = new PhoneBridge(nullLogger, {
      uuid: () => 'mcp-location-error',
    });
    const base = await serve(bridge, { apiKey: 'places-key', fetch: placesFetch });
    const phone = await fetch(`${base}/v1/phone/commands`, { headers: { 'X-Mentat-Phone': '1' } });
    if (phone.body === null) throw new Error('missing phone body');
    const reader = phone.body.getReader() as unknown as ReadableStreamDefaultReader<Uint8Array>;
    const transport = new StreamableHTTPClientTransport(new URL(`${base}/mcp`));
    const client = new Client({ name: 'test-client', version: '1.0.0' });
    await client.connect(transport as Transport);

    const unavailable = client.callTool({ name: 'find_places', arguments: { query: 'coffee' } });
    const firstCommand = JSON.parse(await readLine(reader)) as unknown as Record<string, unknown>;
    await fetch(`${base}/v1/phone/results`, {
      method: 'POST',
      body: JSON.stringify({ id: firstCommand.id, status: 'error', detail: 'location unavailable' }),
    });
    expect(textContent((await unavailable) as unknown)).toContain('ask roughly where Josh is');
    expect(placesFetch).not.toHaveBeenCalled();

    const withLocality = client.callTool({ name: 'find_places', arguments: { query: 'coffee', locality: 'Beaverton' } });
    const secondCommand = JSON.parse(await readLine(reader)) as unknown as Record<string, unknown>;
    await fetch(`${base}/v1/phone/results`, {
      method: 'POST',
      body: JSON.stringify({ id: secondCommand.id, status: 'error', detail: 'location unavailable' }),
    });
    expect(textContent((await withLocality) as unknown)).toContain('NO_RESULTS');
    expect(placesFetch).toHaveBeenCalledOnce();
    await client.close();
    bridge.close();
    await reader.cancel();
  });

  it('reports unconfigured place search without dispatching to the phone', async () => {
    const bridge = new PhoneBridge(nullLogger);
    const dispatch = vi.spyOn(bridge, 'dispatch');
    const base = await serve(bridge);
    const transport = new StreamableHTTPClientTransport(new URL(`${base}/mcp`));
    const client = new Client({ name: 'test-client', version: '1.0.0' });
    await client.connect(transport as Transport);
    const result = await client.callTool({ name: 'find_places', arguments: { query: 'coffee' } });
    expect(textContent(result as unknown)).toContain('place search is not configured');
    expect(dispatch).not.toHaveBeenCalled();
    await client.close();
    bridge.close();
  });

  it('dispatches validated phone action tools and ends a conversation locally', async () => {
    const ids = ['mcp-navigate', 'mcp-dial', 'mcp-alarm', 'mcp-timer'];
    const bridge = new PhoneBridge(nullLogger, {
      uuid: () => ids.shift() ?? 'unexpected',
      now: () => new Date('2026-09-09T12:00:00.000Z'),
    });
    const base = await serve(bridge);
    const phone = await fetch(`${base}/v1/phone/commands`, { headers: { 'X-Mentat-Phone': '1' } });
    if (phone.body === null) throw new Error('missing phone body');
    const reader = phone.body.getReader() as unknown as ReadableStreamDefaultReader<Uint8Array>;
    const transport = new StreamableHTTPClientTransport(new URL(`${base}/mcp`));
    const client = new Client({ name: 'test-client', version: '1.0.0' });
    await client.connect(transport as Transport);

    const cases = [
      { name: 'navigate_to', arguments: { name: 'Union Station', address: '800 N 6th Ave', place_id: 'place-1', lat: 45.528, lng: -122.676 }, kind: 'navigate' },
      { name: 'dial', arguments: { number: '+15555550123' }, kind: 'dial' },
      { name: 'set_alarm', arguments: { hour: 7, minute: 30, label: 'Wake up' }, kind: 'alarm' },
      { name: 'set_timer', arguments: { seconds: 90, label: 'Tea' }, kind: 'timer' },
    ] as const;
    for (const testCase of cases) {
      const call = client.callTool({ name: testCase.name, arguments: testCase.arguments });
      const command = JSON.parse(await readLine(reader)) as Record<string, unknown>;
      expect(command).toMatchObject({ kind: testCase.kind, ...testCase.arguments });
      await fetch(`${base}/v1/phone/results`, { method: 'POST', body: JSON.stringify({ id: command.id, status: 'ok', detail: 'ok' }) });
      await expect(call).resolves.toMatchObject({ content: [{ type: 'text', text: 'ok' }] });
    }

    const ended = await client.callTool({ name: 'end_conversation', arguments: { reason: 'done' } });
    expect(textContent(ended as unknown)).toContain('ended');

    for (const invalid of [
      { name: 'dial', arguments: { number: '' } },
      { name: 'set_alarm', arguments: { hour: 24, minute: 0 } },
      { name: 'set_alarm', arguments: { hour: 0, minute: 60 } },
      { name: 'set_timer', arguments: { seconds: 0 } },
    ]) {
      const result = await client.callTool(invalid);
      expect(result).toMatchObject({ isError: true });
    }
    await client.close();
    bridge.close();
    await reader.cancel();
  });

});
