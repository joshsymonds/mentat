// The HTTP surface: one POST per turn, the response streaming the turn's
// events as NDJSON lines. Authentication is deliberately absent: the daemon
// binds localhost and trusts the deploy's tailnet ingress.

import { once } from 'node:events';
import type { IncomingMessage, RequestListener, ServerResponse } from 'node:http';

import {
  AtCapacityError,
  EFFORT_LEVELS,
  MODEL_PATTERN,
  type Backend,
  type Effort,
  type Event,
} from './backend.ts';
import type { Logger } from './log.ts';
import { handleMcp, type McpDependencies } from './mcp.ts';
import { parseCallContext, type TokenIssuer } from './voicetoken.ts';
import { errorLine, toWireLine } from './wire.ts';

/** An utterance is tiny; this only stops unbounded request bodies. */
const MAX_REQUEST_BYTES = 1 << 20;

/**
 * Per-session activity for idle expiry. A session with a turn in flight is
 * never expired: it is skipped in the scan and re-checked at claim time, so a
 * turn that arrives after the scan still spares its session.
 */
export class SessionTracker {
  private readonly sessions = new Map<string, { lastActive: number; activeTurns: number }>();
  private readonly now: () => number;

  constructor(now: () => number = Date.now) {
    this.now = now;
  }

  beginTurn(sessionId: string): void {
    const activity = this.sessions.get(sessionId) ?? { lastActive: 0, activeTurns: 0 };
    activity.activeTurns += 1;
    activity.lastActive = this.now();
    this.sessions.set(sessionId, activity);
  }

  endTurn(sessionId: string): void {
    const activity = this.sessions.get(sessionId);
    if (activity !== undefined) {
      activity.activeTurns -= 1;
      activity.lastActive = this.now();
    }
  }

  /** Removes and returns sessions idle longer than maxIdleMs. */
  expireIdle(maxIdleMs: number): string[] {
    const cutoff = this.now() - maxIdleMs;
    const expired: string[] = [];
    for (const [sessionId, activity] of this.sessions) {
      if (activity.activeTurns === 0 && activity.lastActive <= cutoff) {
        this.sessions.delete(sessionId);
        expired.push(sessionId);
      }
    }
    return expired;
  }
}

interface TurnRequest {
  sessionId: string;
  text: string;
  meta?: Record<string, string>;
  effort?: Effort;
  model?: string;
}

export function createHandler(
  backend: Backend,
  tracker: SessionTracker,
  logger: Logger,
  issuer?: TokenIssuer,
  mcp?: McpDependencies,
): RequestListener {
  return (req, res) => {
    if (req.method === 'POST' && req.url === '/v1/voice/token' && issuer !== undefined) {
      handleVoiceToken(issuer, logger, req, res).catch((error: unknown) => {
        logger.error('voice token handler failed', { error: String(error) });
        if (!res.destroyed) {
          res.destroy();
        }
      });
      return;
    }
    if (req.method === 'POST' && req.url === '/v1/conversation') {
      // The last-resort catch: a rejection escaping the handler must never
      // become an unhandledRejection — on Node that exits the process,
      // killing every live session over one bad request.
      handleConversation(backend, tracker, logger, req, res).catch((error: unknown) => {
        logger.error('conversation handler failed', { error: String(error) });
        if (!res.destroyed) {
          res.destroy();
        }
      });
      return;
    }
    if (req.url === '/v1/phone/commands') {
      if (req.method !== 'GET' || mcp === undefined) {
        fail(res, 404, 'not found');
        return;
      }
      const phoneHeader = req.headers['x-mentat-phone'];
      const hasPhoneHeader = typeof phoneHeader === 'string'
        ? phoneHeader.length > 0
        : Array.isArray(phoneHeader)
          ? phoneHeader.some((value) => value.length > 0)
          : false;
      if (!hasPhoneHeader) {
        fail(res, 403, 'phone header required');
        return;
      }
      mcp.bridge.attach(res);
      return;
    }
    if (req.url === '/v1/phone/results') {
      if (req.method !== 'POST' || mcp === undefined) {
        fail(res, 404, 'not found');
        return;
      }
      readJsonBody(req, res).then((body) => {
        if (body === undefined) {
          return;
        }
        if (!isPhoneResult(body)) {
          fail(res, 400, 'id, status, and detail are required');
          return;
        }
        mcp.bridge.complete(body);
        res.writeHead(204);
        res.end();
      }).catch((error: unknown) => {
        logger.error('phone result handler failed', { error: String(error) });
        if (!res.destroyed) {
          res.destroy();
        }
      });
      return;
    }
    if (req.url === '/mcp') {
      if (req.method === 'GET' || req.method === 'DELETE') {
        res.writeHead(405, { 'content-type': 'application/json' });
        res.end(JSON.stringify({
          jsonrpc: '2.0',
          error: { code: -32000, message: 'Method not allowed.' },
          id: null,
        }) + '\n');
        return;
      }
      if (req.method !== 'POST' || mcp === undefined) {
        fail(res, 404, 'not found');
        return;
      }
      readJsonBody(req, res).then((body) => {
        if (body !== undefined) {
          return handleMcp(mcp, req, res, body);
        }
        return undefined;
      }).catch((error: unknown) => {
        logger.error('MCP handler failed', { error: String(error) });
        if (!res.headersSent && !res.destroyed) {
          fail(res, 500, 'internal');
        } else if (!res.destroyed) {
          res.destroy();
        }
      });
      return;
    }
    if (req.method === 'GET' && req.url === '/healthz') {
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end('{"status":"ok"}\n');
      return;
    }
    res.writeHead(404, { 'content-type': 'application/json' });
    res.end('{"error":"not found"}\n');
  };
}

async function handleConversation(
  backend: Backend,
  tracker: SessionTracker,
  logger: Logger,
  req: IncomingMessage,
  res: ServerResponse,
): Promise<void> {
  const parsed = await readTurnRequest(req, res);
  if (parsed === undefined) {
    return; // response already written (or the client is gone)
  }

  const tailscaleUser = req.headers['tailscale-user-login'];
  if (typeof tailscaleUser === 'string' && tailscaleUser !== '') {
    logger.info('turn received', { session_id: parsed.sessionId, tailscale_user: tailscaleUser });
  }

  // Aborts the turn the moment the client disconnects — even while the
  // backend is silently waiting on the model — so the child is interrupted
  // promptly instead of at the next event.
  const abort = new AbortController();
  res.on('close', () => {
    abort.abort();
  });

  tracker.beginTurn(parsed.sessionId);
  try {
    let stream: AsyncIterable<Event>;
    try {
      stream = await backend.converse({
        sessionId: parsed.sessionId,
        text: parsed.text,
        signal: abort.signal,
        ...(parsed.meta !== undefined && { meta: parsed.meta }),
        ...(parsed.effort !== undefined && { effort: parsed.effort }),
        ...(parsed.model !== undefined && { model: parsed.model }),
      });
    } catch (error) {
      logger.error('backend refused turn', {
        session_id: parsed.sessionId,
        error: String(error),
      });
      if (error instanceof AtCapacityError) {
        fail(res, 503, 'at capacity, retry shortly');
      } else {
        fail(res, 502, 'backend refused the turn');
      }
      return;
    }
    await streamEvents(res, stream);
  } finally {
    tracker.endTurn(parsed.sessionId);
  }
}

/** Reads and validates the turn request, writing the failure response itself
 * (and returning undefined) when the body is unusable. */
async function readTurnRequest(
  req: IncomingMessage,
  res: ServerResponse,
): Promise<TurnRequest | undefined> {
  const body = await readJsonBody(req, res);
  if (body === undefined) {
    return undefined;
  }
  if (body === null || typeof body !== 'object') {
    fail(res, 400, 'invalid JSON body');
    return undefined;
  }
  const record = body as Record<string, unknown>;
  const sessionId = record.session_id;
  const text = record.text;
  if (typeof sessionId !== 'string' || sessionId === '' || typeof text !== 'string' || text === '') {
    fail(res, 400, 'session_id and text are required');
    return undefined;
  }
  const effort = record.effort;
  if (effort !== undefined && (typeof effort !== 'string' || !EFFORT_LEVELS.has(effort))) {
    fail(res, 400, 'effort must be one of low|medium|high|xhigh|max');
    return undefined;
  }
  const model = record.model;
  if (model !== undefined && (typeof model !== 'string' || !MODEL_PATTERN.test(model))) {
    fail(res, 400, 'model must be a short model alias or id');
    return undefined;
  }
  const meta = parseMeta(record.meta);
  return {
    sessionId,
    text,
    ...(meta !== undefined && { meta }),
    ...(effort !== undefined && { effort: effort as Effort }),
    ...(model !== undefined && { model }),
  };
}

// The call context in the body only colours the greeting, so a body that is
// empty or does not parse still gets a token — never a failed call.
async function handleVoiceToken(
  issuer: TokenIssuer,
  logger: Logger,
  req: IncomingMessage,
  res: ServerResponse,
): Promise<void> {
  const chunks: Buffer[] = [];
  let size = 0;
  try {
    for await (const chunk of req) {
      const buffer = chunk as Buffer;
      size += buffer.length;
      if (size > MAX_REQUEST_BYTES) {
        fail(res, 413, 'request body too large');
        return;
      }
      chunks.push(buffer);
    }
  } catch {
    res.destroy();
    return;
  }
  let body: unknown;
  try {
    body = size > 0 ? (JSON.parse(Buffer.concat(chunks).toString('utf8')) as unknown) : undefined;
  } catch {
    logger.warn('voice token body is not JSON; issuing without call context');
  }
  try {
    const grant = JSON.stringify(issuer.issue(parseCallContext(body))) + '\n';
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end(grant);
  } catch (error) {
    logger.error('voice token issuance failed', { error: String(error) });
    fail(res, 500, 'internal');
  }
}

async function readJsonBody(
  req: IncomingMessage,
  res: ServerResponse,
): Promise<unknown> {
  const chunks: Buffer[] = [];
  let size = 0;
  try {
    for await (const chunk of req) {
      const buffer = chunk as Buffer;
      size += buffer.length;
      if (size > MAX_REQUEST_BYTES) {
        fail(res, 413, 'request body too large');
        return undefined;
      }
      chunks.push(buffer);
    }
  } catch {
    res.destroy();
    return undefined;
  }

  try {
    return JSON.parse(Buffer.concat(chunks).toString('utf8')) as unknown;
  } catch {
    fail(res, 400, 'invalid JSON body');
    return undefined;
  }
}

function isPhoneResult(
  value: unknown,
): value is { id: string; status: 'ok' | 'error'; detail: string; payload?: Record<string, unknown> } {
  if (value === null || typeof value !== 'object') {
    return false;
  }
  const record = value as Record<string, unknown>;
  return (
    typeof record.id === 'string' &&
    record.id !== '' &&
    (record.status === 'ok' || record.status === 'error') &&
    typeof record.detail === 'string' &&
    (record.payload === undefined ||
      (record.payload !== null && typeof record.payload === 'object' && !Array.isArray(record.payload)))
  );
}

function parseMeta(value: unknown): Record<string, string> | undefined {
  if (value === null || value === undefined || typeof value !== 'object') {
    return undefined;
  }
  const meta: Record<string, string> = {};
  for (const [key, entry] of Object.entries(value)) {
    if (typeof entry === 'string') {
      meta[key] = entry;
    }
  }
  return meta;
}

/**
 * Writes a turn's events as NDJSON lines. A mid-stream failure becomes a
 * terminal {"kind":"error"} line: by then the 200 header has shipped, so
 * in-band delivery is the only honest option. A closed response stops
 * consumption (the turn's abort signal has already interrupted the backend);
 * a full write buffer pauses consumption until the socket drains.
 */
async function streamEvents(res: ServerResponse, stream: AsyncIterable<Event>): Promise<void> {
  res.writeHead(200, { 'content-type': 'application/x-ndjson' });
  res.flushHeaders();
  try {
    for await (const event of stream) {
      if (res.destroyed) {
        return;
      }
      const line = toWireLine(event);
      if (line !== null && !res.write(line + '\n')) {
        await Promise.race([once(res, 'drain'), once(res, 'close')]);
      }
    }
  } catch (error) {
    if (!res.destroyed) {
      res.write(errorLine(error instanceof Error ? error.message : String(error)) + '\n');
    }
  } finally {
    res.end();
  }
}

function fail(res: ServerResponse, status: number, message: string): void {
  res.writeHead(status, { 'content-type': 'application/json' });
  res.end(JSON.stringify({ error: message }) + '\n');
}
