import type { IncomingMessage, ServerResponse } from 'node:http';

import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/streamableHttp.js';
import type { Transport } from '@modelcontextprotocol/sdk/shared/transport.js';
import { z } from 'zod';

import type { PhoneBridge } from './phone.ts';

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
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
        const text = await bridge.dispatch({ kind: 'sms', to, body: message });
        return { content: [{ type: 'text' as const, text }] };
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
        const text = await bridge.dispatch({ kind: 'open', uri });
        return { content: [{ type: 'text' as const, text }] };
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
