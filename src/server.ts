// The HTTP surface: one POST per turn, the response streaming the turn's
// events as NDJSON lines. Authentication is deliberately absent: the daemon
// binds localhost and trusts the deploy's tailnet ingress.

import type { IncomingMessage, RequestListener, ServerResponse } from 'node:http';

import type { Backend } from './backend.ts';
import { AtCapacityError } from './claudecode.ts';
import type { Logger } from './log.ts';
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
}

export function createHandler(
  backend: Backend,
  tracker: SessionTracker,
  logger: Logger,
): RequestListener {
  return (req, res) => {
    if (req.method === 'POST' && req.url === '/v1/conversation') {
      void handleConversation(backend, tracker, logger, req, res);
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
    return; // response already written
  }

  const tailscaleUser = req.headers['tailscale-user-login'];
  if (typeof tailscaleUser === 'string' && tailscaleUser !== '') {
    logger.info('turn received', { session_id: parsed.sessionId, tailscale_user: tailscaleUser });
  }

  tracker.beginTurn(parsed.sessionId);
  try {
    let stream: AsyncIterable<import('./backend.ts').Event>;
    try {
      stream = await backend.converse({
        sessionId: parsed.sessionId,
        text: parsed.text,
        ...(parsed.meta !== undefined && { meta: parsed.meta }),
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
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = chunk as Buffer;
    size += buffer.length;
    if (size > MAX_REQUEST_BYTES) {
      fail(res, 413, 'request body too large');
      return undefined;
    }
    chunks.push(buffer);
  }

  let body: unknown;
  try {
    body = JSON.parse(Buffer.concat(chunks).toString('utf8'));
  } catch {
    fail(res, 400, 'invalid JSON body');
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
  const meta = parseMeta(record.meta);
  return { sessionId, text, ...(meta !== undefined && { meta }) };
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
 * consumption, which is the backend's signal to interrupt the turn.
 */
async function streamEvents(
  res: ServerResponse,
  stream: AsyncIterable<import('./backend.ts').Event>,
): Promise<void> {
  res.writeHead(200, { 'content-type': 'application/x-ndjson' });
  res.flushHeaders();
  try {
    for await (const event of stream) {
      if (res.destroyed) {
        return; // client left; stop consuming so the backend interrupts
      }
      const line = toWireLine(event);
      if (line !== null) {
        res.write(line + '\n');
      }
    }
  } catch (error) {
    res.write(errorLine(error instanceof Error ? error.message : String(error)) + '\n');
  } finally {
    res.end();
  }
}

function fail(res: ServerResponse, status: number, message: string): void {
  res.writeHead(status, { 'content-type': 'application/json' });
  res.end(JSON.stringify({ error: message }) + '\n');
}
