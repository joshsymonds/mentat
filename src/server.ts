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
): RequestListener {
  return (req, res) => {
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
    // The client aborted mid-body; there is nobody to answer.
    res.destroy();
    return undefined;
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
