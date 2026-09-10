import type { IncomingMessage, ServerResponse } from 'node:http';

import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/streamableHttp.js';
import type { Transport } from '@modelcontextprotocol/sdk/shared/transport.js';
import { z } from 'zod';

import type { PhoneBridge, PhoneOutcome } from './phone.ts';

const SUPPORTED_CHANNELS = ['sms'] as const;

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
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

export async function handleMcp(
  bridge: PhoneBridge,
  req: IncomingMessage,
  res: ServerResponse,
  body: unknown,
): Promise<void> {
  const server = new McpServer({ name: 'mentat', version: '3.0.0' });
  server.registerTool(
    'send_sms',
    {
      description: 'Send a text message to a phone number or contact name.',
      inputSchema: { to: z.string(), body: z.string() },
    },
    async ({ to, body: message }) => {
      try {
        const outcome = await bridge.dispatch({ kind: 'sms', to, body: message });
        return { content: [{ type: 'text' as const, text: outcome.detail }] };
      } catch (error) {
        return {
          isError: true,
          content: [{ type: 'text' as const, text: errorText(error) }],
        };
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
        return { content: [{ type: 'text' as const, text: outcome.detail }] };
      } catch (error) {
        return {
          isError: true,
          content: [{ type: 'text' as const, text: errorText(error) }],
        };
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
        return {
          isError: true,
          content: [{ type: 'text' as const, text: errorText(error) }],
        };
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
        return {
          isError: true,
          content: [{ type: 'text' as const, text: errorText(error) }],
        };
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
        return {
          isError: true,
          content: [{ type: 'text' as const, text: errorText(error) }],
        };
      }
    },
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
