import type { IncomingMessage, ServerResponse } from 'node:http';

import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/streamableHttp.js';
import type { Transport } from '@modelcontextprotocol/sdk/shared/transport.js';
import { z } from 'zod';

import { searchPlaces, type PlacesDeps, type PlacesLocation } from './places.ts';
import type { PhoneBridge, PhoneLocationPayload, PhoneOutcome } from './phone.ts';

const SUPPORTED_CHANNELS = ['sms'] as const;
const SEND_CONFIRMATION_ERROR =
  'Message was not sent. Call again with send=true after confirming the recipient and message with the user.';
const LOCATION_UNAVAILABLE =
  'I could not get a fresh phone location, so ask roughly where Josh is and call find_places again with locality.';
const PLACES_UNCONFIGURED = 'place search is not configured on the server.';
const NO_PLACES = 'NO_RESULTS: no places matched.';

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function errorResult(error: unknown) {
  return {
    isError: true as const,
    content: [{ type: 'text' as const, text: errorText(error) }],
  };
}

function textResult(text: string) {
  return { content: [{ type: 'text' as const, text }] };
}

function channelError(value: string) {
  return {
    isError: true as const,
    content: [{ type: 'text' as const, text: `unsupported channel ${value}; supported channels: sms` }],
  };
}

function conversationChannel(id: string): string | undefined {
  return /^([^:]+):/.exec(id)?.[1];
}

function supportedChannel(value: string): boolean {
  return SUPPORTED_CHANNELS.some((channel) => channel === value);
}

function validBefore(value: string): boolean {
  return (
    (/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2}(\.\d{1,9})?)?(Z|[+-]\d{2}:\d{2})$/.test(value) &&
      !Number.isNaN(Date.parse(value))) ||
    /^\d+:[a-z]+:[A-Za-z0-9]+$/.test(value)
  );
}

function malformedResult() {
  return {
    isError: true as const,
    content: [{ type: 'text' as const, text: 'malformed result' }],
  };
}

function payloadResult(outcome: PhoneOutcome) {
  if (outcome.payload === undefined) {
    return malformedResult();
  }
  const text = JSON.stringify(outcome.payload);
  return { content: [{ type: 'text' as const, text }] };
}

function locationFromPayload(payload: Record<string, unknown> | undefined): PlacesLocation | undefined {
  if (payload === undefined) {
    return undefined;
  }
  const { lat, lng } = payload as Partial<PhoneLocationPayload>;
  return typeof lat === 'number' && Number.isFinite(lat) && typeof lng === 'number' && Number.isFinite(lng)
    ? { lat, lng }
    : undefined;
}

export interface McpDependencies {
  bridge: PhoneBridge;
  places: PlacesDeps;
}

export async function handleMcp(
  deps: McpDependencies,
  req: IncomingMessage,
  res: ServerResponse,
  body: unknown,
): Promise<void> {
  const { bridge, places } = deps;
  const server = new McpServer({ name: 'mentat', version: '3.0.0' });
  server.registerTool(
    'send_sms',
    {
      description:
        'Send a text message only after the recipient and message were confirmed with the user. Set send=true only after that explicit confirmation.',
      inputSchema: {
        to: z.string(),
        body: z.string(),
        send: z.boolean().optional().describe('true only after the recipient and message were confirmed with the user'),
      },
    },
    async ({ to, body: message, send }) => {
      if (send !== true) {
        return {
          isError: true as const,
          content: [{ type: 'text' as const, text: SEND_CONFIRMATION_ERROR }],
        };
      }
      try {
        const outcome = await bridge.dispatch({ kind: 'sms', to, body: message });
        return textResult(outcome.detail);
      } catch (error) {
        return errorResult(error);
      }
    },
  );
  server.registerTool(
    'open_on_phone',
    {
      description:
        'Open a URI on the phone. Examples include google.navigation:q=…, geo:0,0?q=…, and https://….',
      inputSchema: { uri: z.string() },
    },
    async ({ uri }) => {
      try {
        const outcome = await bridge.dispatch({ kind: 'open', uri });
        return textResult(outcome.detail);
      } catch (error) {
        return errorResult(error);
      }
    },
  );
  server.registerTool(
    'list_conversations',
    {
      description: 'List SMS conversations newest first. Defaults to 20, caps at 100, and ids use sms:<thread>.',
      inputSchema: { channel: z.string().optional(), limit: z.number().int().min(1).optional() },
    },
    async ({ channel, limit }) => {
      if (channel !== undefined && !supportedChannel(channel)) {
        return channelError(channel);
      }
      try {
        const outcome = await bridge.dispatch({
          kind: 'conversations',
          limit: Math.min(limit ?? 20, 100),
        });
        return payloadResult(outcome);
      } catch (error) {
        return errorResult(error);
      }
    },
  );
  server.registerTool(
    'read_conversation',
    {
      description:
        'Read messages by sms:<thread> id, phone number, or contact name. Defaults to 50, caps at 200, and before accepts an ISO time or previous next.',
      inputSchema: {
        conversation: z.string(),
        limit: z.number().int().min(1).optional(),
        before: z.string().refine(validBefore).optional(),
      },
    },
    async ({ conversation, limit, before }) => {
      const channel = conversationChannel(conversation);
      if (channel !== undefined && !supportedChannel(channel)) {
        return channelError(channel);
      }
      try {
        const outcome = await bridge.dispatch({
          kind: 'messages',
          conversation,
          limit: Math.min(limit ?? 50, 200),
          ...(before !== undefined && { before }),
        });
        return payloadResult(outcome);
      } catch (error) {
        return errorResult(error);
      }
    },
  );
  server.registerTool(
    'search_messages',
    {
      description:
        'Search SMS message bodies. Defaults to 30, caps at 100, and before accepts an ISO time or previous next; ids use sms:<thread>.',
      inputSchema: {
        query: z.string(),
        channel: z.string().optional(),
        limit: z.number().int().min(1).optional(),
        before: z.string().refine(validBefore).optional(),
      },
    },
    async ({ query, channel, limit, before }) => {
      if (channel !== undefined && !supportedChannel(channel)) {
        return channelError(channel);
      }
      try {
        const outcome = await bridge.dispatch({
          kind: 'search',
          query,
          limit: Math.min(limit ?? 30, 100),
          ...(before !== undefined && { before }),
        });
        return payloadResult(outcome);
      } catch (error) {
        return errorResult(error);
      }
    },
  );
  server.registerTool(
    'find_places',
    {
      description:
        'Search for a named place near the phone. If the phone cannot provide a fresh location, ask roughly where Josh is and call again with locality.',
      inputSchema: { query: z.string().min(1), locality: z.string().min(1).optional() },
    },
    async ({ query, locality }) => {
      if (places.apiKey === undefined || places.apiKey.trim() === '') {
        return textResult(PLACES_UNCONFIGURED);
      }
      let location: PlacesLocation | undefined;
      try {
        const outcome = await bridge.dispatch({ kind: 'location' });
        location = locationFromPayload(outcome.payload);
      } catch {
        // A locality allows the search to continue without a phone location.
      }
      if (location === undefined && (locality === undefined || locality.trim() === '')) {
        return textResult(LOCATION_UNAVAILABLE);
      }
      try {
        const candidates = await searchPlaces({
          query,
          ...(locality !== undefined && { locality }),
          ...(location !== undefined && { location }),
          apiKey: places.apiKey,
          ...(places.fetch !== undefined && { fetch: places.fetch }),
        });
        return textResult(candidates.length === 0 ? NO_PLACES : JSON.stringify(candidates));
      } catch (error) {
        return errorResult(error);
      }
    },
  );
  server.registerTool(
    'navigate_to',
    {
      description: 'Start phone navigation to the selected place returned by find_places.',
      inputSchema: {
        name: z.string().min(1),
        address: z.string().min(1),
        place_id: z.string().min(1),
        lat: z.number(),
        lng: z.number(),
      },
    },
    async ({ name, address, place_id, lat, lng }) => {
      try {
        const outcome = await bridge.dispatch({ kind: 'navigate', name, address, place_id, lat, lng });
        return textResult(outcome.detail);
      } catch (error) {
        return errorResult(error);
      }
    },
  );
  server.registerTool(
    'dial',
    {
      description: 'Open the phone dialer with a non-empty number prefilled without placing the call.',
      inputSchema: { number: z.string().min(1) },
    },
    async ({ number }) => {
      try {
        const outcome = await bridge.dispatch({ kind: 'dial', number });
        return textResult(outcome.detail);
      } catch (error) {
        return errorResult(error);
      }
    },
  );
  server.registerTool(
    'set_alarm',
    {
      description: 'Set an alarm on the phone for the given 24-hour clock hour and minute.',
      inputSchema: {
        hour: z.number().int().min(0).max(23),
        minute: z.number().int().min(0).max(59),
        label: z.string().optional(),
      },
    },
    async ({ hour, minute, label }) => {
      try {
        const outcome = await bridge.dispatch({
          kind: 'alarm',
          hour,
          minute,
          ...(label !== undefined && { label }),
        });
        return textResult(outcome.detail);
      } catch (error) {
        return errorResult(error);
      }
    },
  );
  server.registerTool(
    'set_timer',
    {
      description: 'Start a phone timer for at least one second, with an optional label.',
      inputSchema: {
        seconds: z.number().int().min(1),
        label: z.string().optional(),
      },
    },
    async ({ seconds, label }) => {
      try {
        const outcome = await bridge.dispatch({
          kind: 'timer',
          seconds,
          ...(label !== undefined && { label }),
        });
        return textResult(outcome.detail);
      } catch (error) {
        return errorResult(error);
      }
    },
  );
  server.registerTool(
    'end_conversation',
    {
      description:
        'End the current voice conversation: after a sign-off, or as soon as the request is complete and nothing is left open.',
      inputSchema: { reason: z.enum(['signoff', 'done']) },
    },
    () => textResult('Conversation ended.'),
  );

  const transport = new StreamableHTTPServerTransport({});
  res.once('close', () => {
    void transport.close().catch(() => undefined);
    void server.close().catch(() => undefined);
  });
  try {
    await server.connect(transport as Transport);
    await transport.handleRequest(req, res, body);
  } catch (error) {
    if (!res.headersSent) {
      res.writeHead(500, { 'content-type': 'application/json' });
      res.end(
        JSON.stringify({
          jsonrpc: '2.0',
          error: { code: -32603, message: errorText(error) },
          id: null,
        }) + '\n',
      );
    } else if (!res.destroyed) {
      res.destroy();
    }
  }
}
