// The live Backend: one persistent SDK session (one claude child) per
// sessionId, speaking streaming input. Turns within a session are serialized;
// conversations resume across child death and daemon restarts via a persisted
// sessionId→CLI-UUID map. The SDK call is injectable so every behavior here
// tests offline.

import { existsSync, readFileSync, renameSync, writeFileSync } from 'node:fs';
import process from 'node:process';

import { query, type Options, type SDKUserMessage } from '@anthropic-ai/claude-agent-sdk';

import type { Backend, Event, Turn } from './backend.ts';
import type { Logger } from './log.ts';
import type { PolicyFn, TurnContext } from './policy.ts';
import { Translator } from './translate.ts';

/** A turn for a new session beyond maxSessions is refused with this. */
export class AtCapacityError extends Error {
  constructor() {
    super('claudecode: at session capacity');
    this.name = 'AtCapacityError';
  }
}

/**
 * With no operator tool policy at all, disallow the dangerous built-ins: the
 * isolated child still carries the full toolset, and a voice surface must not
 * drive Bash/Write/etc. by default. Opt into danger explicitly.
 */
export const DEFAULT_DISALLOWED_TOOLS = [
  'Bash',
  'Write',
  'Edit',
  'NotebookEdit',
  'WebFetch',
  'WebSearch',
  'Task',
];

/**
 * After interrupting an abandoned turn, how many leftover messages to drain
 * looking for the interrupted turn's terminal result before declaring the
 * session unsalvageable and respawning it instead.
 */
const ABANDON_DRAIN_LIMIT = 1000;

/** The slice of the SDK's query() the backend consumes — injectable in tests. */
export interface QueryHandle extends AsyncIterable<unknown> {
  interrupt(): Promise<void>;
}

export type QueryFn = (args: {
  prompt: AsyncIterable<SDKUserMessage>;
  options: Options;
}) => QueryHandle;

export interface ClaudeCodeConfig {
  /** Absolute path to the claude binary. Required; no PATH fallback, no SDK
   * auto-download — the deploy pins the binary. */
  bin: string;
  model?: string;
  effort?: Options['effort'];
  /** Replaces the CLI's system prompt when set. */
  systemPrompt?: string;
  /** Extra directories granted to sessions (the memory dir rides here). */
  addDirs?: string[];
  /** Explicit MCP server map; nothing else is reachable (strictMcpConfig). */
  mcpServers?: Options['mcpServers'];
  allowedTools?: string[];
  disallowedTools?: string[];
  maxBudgetUsd?: number;
  /** Cap on concurrent live children; 0/absent disables the cap. */
  maxSessions?: number;
  /** Env var names passed through beyond the standard allowlist. */
  extraEnv?: string[];
  /** Persists the sessionId→CLI-UUID map across daemon restarts. */
  statePath?: string;
  policy: PolicyFn;
  logger: Logger;
  /** Test seam; production uses the real SDK. */
  queryFn?: QueryFn;
}

/**
 * Child env allowlist, ported from go-v2 childEnv: the child is a
 * tool-bearing agent processing untrusted text, so it gets least privilege —
 * shell/locale/proxy basics and the Anthropic/Claude auth surface, nothing
 * else. The deny entries override the prefixes: those nesting markers would
 * make the child believe it runs inside an interactive Claude Code session.
 */
export function buildChildEnv(
  source: Record<string, string | undefined>,
  extraEnv: readonly string[] = [],
): Record<string, string> {
  const allowExact = new Set([
    'HOME',
    'PATH',
    'USER',
    'LOGNAME',
    'SHELL',
    'TERM',
    'LANG',
    'TMPDIR',
    'TZ',
    'HTTP_PROXY',
    'HTTPS_PROXY',
    'NO_PROXY',
    'http_proxy',
    'https_proxy',
    'no_proxy',
    ...extraEnv,
  ]);
  const allowPrefixes = ['LC_', 'XDG_', 'ANTHROPIC_', 'CLAUDE_CODE_', 'AWS_'];
  const denyExact = new Set(['CLAUDECODE', 'CLAUDE_CODE_ENTRYPOINT']);

  const out: Record<string, string> = {};
  for (const [key, value] of Object.entries(source)) {
    if (value === undefined || denyExact.has(key)) {
      continue;
    }
    if (allowExact.has(key) || allowPrefixes.some((prefix) => key.startsWith(prefix))) {
      out[key] = value;
    }
  }
  return out;
}

/**
 * Assembles the per-session SDK options. The isolation flags are
 * unconditional: a bare child inherits the operator's interactive Claude Code
 * configuration (settings, skills, MCP servers), which must never drive a
 * daemon. Exported pure so tests pin every invariant.
 */
export function buildOptions(
  config: ClaudeCodeConfig,
  getContext: () => TurnContext,
  resumeUuid?: string,
): Options {
  return {
    settingSources: [],
    skills: [],
    strictMcpConfig: true,
    includePartialMessages: true,
    pathToClaudeCodeExecutable: config.bin,
    env: buildChildEnv(process.env, config.extraEnv ?? []),
    disallowedTools: config.disallowedTools ?? DEFAULT_DISALLOWED_TOOLS,
    ...(config.model !== undefined && { model: config.model }),
    ...(config.effort !== undefined && { effort: config.effort }),
    ...(config.systemPrompt !== undefined && { systemPrompt: config.systemPrompt }),
    ...(config.addDirs !== undefined && { additionalDirectories: config.addDirs }),
    ...(config.mcpServers !== undefined && { mcpServers: config.mcpServers }),
    ...(config.allowedTools !== undefined && { allowedTools: config.allowedTools }),
    ...(config.maxBudgetUsd !== undefined && { maxBudgetUsd: config.maxBudgetUsd }),
    ...(resumeUuid !== undefined && { resume: resumeUuid }),
    canUseTool: async (toolName, input) => {
      const decision = await config.policy(toolName, input, getContext());
      return decision.behavior === 'allow'
        ? { behavior: 'allow', updatedInput: decision.updatedInput }
        : { behavior: 'deny', message: decision.message };
    },
  };
}

/** Unbounded push-queue bridging turns into the SDK's prompt iterable. */
class AsyncQueue<T> implements AsyncIterable<T> {
  private readonly values: T[] = [];
  private readonly waiters: ((result: IteratorResult<T>) => void)[] = [];
  private ended = false;

  push(value: T): void {
    const waiter = this.waiters.shift();
    if (waiter !== undefined) {
      waiter({ value, done: false });
    } else {
      this.values.push(value);
    }
  }

  end(): void {
    this.ended = true;
    for (const waiter of this.waiters.splice(0)) {
      waiter({ value: undefined, done: true });
    }
  }

  [Symbol.asyncIterator](): AsyncIterator<T> {
    return {
      next: (): Promise<IteratorResult<T>> => {
        const value = this.values.shift();
        if (value !== undefined) {
          return Promise.resolve({ value, done: false });
        }
        if (this.ended) {
          return Promise.resolve({ value: undefined, done: true });
        }
        return new Promise((resolve) => {
          this.waiters.push(resolve);
        });
      },
    };
  }
}

/** Serializes turns within a session: at most one active turn. */
class Mutex {
  private tail: Promise<void> = Promise.resolve();

  acquire(): Promise<() => void> {
    const prev = this.tail;
    let release!: () => void;
    this.tail = new Promise((resolve) => {
      release = resolve;
    });
    return prev.then(() => release);
  }
}

interface Session {
  queue: AsyncQueue<SDKUserMessage>;
  iterator: AsyncIterator<unknown>;
  handle: QueryHandle;
  translator: Translator;
  mutex: Mutex;
  activeContext: TurnContext | null;
  dead: boolean;
}

function userMessage(text: string): SDKUserMessage {
  return {
    type: 'user',
    message: { role: 'user', content: [{ type: 'text', text }] },
    parent_tool_use_id: null,
    session_id: '',
  };
}

export class ClaudeCode implements Backend {
  private readonly config: ClaudeCodeConfig;
  private readonly logger: Logger;
  private readonly queryFn: QueryFn;
  private readonly sessions = new Map<string, Session>();
  /**
   * sessionId→CLI-UUID for every session this backend has started. It
   * outlives the live Session entries (and a daemon restart, when statePath
   * is set), so a turn can resume a conversation whose child is gone.
   */
  private readonly resumable: Map<string, string>;

  constructor(config: ClaudeCodeConfig) {
    if (config.bin === '') {
      throw new Error('claudecode: bin is required (no PATH fallback)');
    }
    this.config = config;
    this.logger = config.logger;
    this.queryFn = config.queryFn ?? (({ prompt, options }) => query({ prompt, options }));
    this.resumable = loadResumable(config.statePath);
  }

  async converse(turn: Turn): Promise<AsyncIterable<Event>> {
    if (turn.sessionId === '') {
      throw new Error('claudecode: turn requires a sessionId');
    }
    const { session, release } = await this.startTurn(turn);
    return this.streamTurn(turn, session, release, false);
  }

  /** Acquires the session's turn slot and sends the turn into the child. */
  private async startTurn(turn: Turn): Promise<{ session: Session; release: () => void }> {
    const session = this.sessionFor(turn.sessionId);
    const release = await session.mutex.acquire();
    // The session may have died while this turn waited on the previous one;
    // respawn rather than reading a dead iterator.
    if (session.dead) {
      release();
      return this.startTurn(turn);
    }
    session.activeContext = { sessionId: turn.sessionId, meta: turn.meta ?? {} };
    session.queue.push(userMessage(turn.text));
    return { session, release };
  }

  private async *streamTurn(
    turn: Turn,
    session: Session,
    release: () => void,
    retried: boolean,
  ): AsyncGenerator<Event> {
    const sessionId = turn.sessionId;
    let sawDone = false;
    let messagesRead = 0;
    try {
      while (!sawDone) {
        const next = await session.iterator.next();
        if (next.done === true) {
          session.dead = true;
          this.sessions.delete(sessionId);
          if (messagesRead === 0 && !retried) {
            // The child died between turns: nothing of this turn was
            // processed, so respawn with resume and replay it — the Go
            // version's dead-session path. One retry only; a child that
            // dies instantly on respawn is a real error.
            this.logger.warn('claudecode: child died between turns, respawning', {
              session_id: sessionId,
            });
            release();
            const restarted = await this.startTurn(turn);
            yield* this.streamTurn(turn, restarted.session, restarted.release, true);
            return;
          }
          throw new Error('claudecode: session ended mid-turn');
        }
        messagesRead += 1;
        for (const event of session.translator.translate(next.value)) {
          if (event.kind === 'unknown') {
            this.logger.error('claudecode: unknown SDK message', {
              session_id: sessionId,
              raw: JSON.stringify(event.raw).slice(0, 2000),
            });
          }
          if (event.kind === 'done') {
            sawDone = true;
            this.recordResume(sessionId, event.result.sessionId);
          }
          yield event;
        }
      }
    } finally {
      session.activeContext = null;
      if (!sawDone && !session.dead) {
        await this.abandonTurn(sessionId, session);
      }
      release();
    }
  }

  /**
   * The consumer left before the turn's done, so the session's stream is
   * stranded mid-turn. Interrupt the child and drain to the interrupted
   * turn's terminal result; if that fails, drop the session so the next turn
   * respawns with resume.
   */
  private async abandonTurn(sessionId: string, session: Session): Promise<void> {
    try {
      await session.handle.interrupt();
      for (let drained = 0; drained < ABANDON_DRAIN_LIMIT; drained += 1) {
        const next = await session.iterator.next();
        if (next.done === true) {
          break;
        }
        if (session.translator.translate(next.value).some((event) => event.kind === 'done')) {
          this.logger.warn('claudecode: turn abandoned, session interrupted', {
            session_id: sessionId,
          });
          return;
        }
      }
    } catch {
      // fall through to respawn
    }
    session.dead = true;
    this.sessions.delete(sessionId);
    this.logger.warn('claudecode: turn abandoned, session dropped for respawn', {
      session_id: sessionId,
    });
  }

  closeSession(sessionId: string): Promise<void> {
    const session = this.sessions.get(sessionId);
    this.sessions.delete(sessionId);
    // Ending the input queue is the CLI's exit signal; the resume uuid is
    // retained so the next turn restores the conversation.
    session?.queue.end();
    return Promise.resolve();
  }

  async close(): Promise<void> {
    for (const sessionId of [...this.sessions.keys()]) {
      await this.closeSession(sessionId);
    }
  }

  /** Number of sessions with a live child. */
  liveSessions(): number {
    return [...this.sessions.values()].filter((session) => !session.dead).length;
  }

  private sessionFor(sessionId: string): Session {
    const existing = this.sessions.get(sessionId);
    if (existing !== undefined && !existing.dead) {
      return existing;
    }
    const maxSessions = this.config.maxSessions ?? 0;
    if (maxSessions > 0 && this.liveCountExcluding(sessionId) >= maxSessions) {
      throw new AtCapacityError();
    }
    const queue = new AsyncQueue<SDKUserMessage>();
    const session: Session = {
      queue,
      iterator: undefined as unknown as AsyncIterator<unknown>,
      handle: undefined as unknown as QueryHandle,
      translator: new Translator(),
      mutex: new Mutex(),
      activeContext: null,
      dead: false,
    };
    const options = buildOptions(
      this.config,
      () => session.activeContext ?? { sessionId, meta: {} },
      this.resumable.get(sessionId),
    );
    session.handle = this.queryFn({ prompt: queue, options });
    session.iterator = session.handle[Symbol.asyncIterator]();
    this.sessions.set(sessionId, session);
    return session;
  }

  /** Live children, ignoring the session being replaced/respawned. */
  private liveCountExcluding(sessionId: string): number {
    let live = 0;
    for (const [id, session] of this.sessions) {
      if (id !== sessionId && !session.dead) {
        live += 1;
      }
    }
    return live;
  }

  private recordResume(sessionId: string, cliUuid: string): void {
    if (cliUuid === '' || this.resumable.get(sessionId) === cliUuid) {
      return;
    }
    this.resumable.set(sessionId, cliUuid);
    this.persistResumable();
  }

  /**
   * Atomic write (temp + rename). Failures are logged, not fatal: a write
   * error degrades resume-across-restart but must not fail the turn.
   */
  private persistResumable(): void {
    const statePath = this.config.statePath;
    if (statePath === undefined || statePath === '') {
      return;
    }
    try {
      const tmp = statePath + '.tmp';
      writeFileSync(tmp, JSON.stringify(Object.fromEntries(this.resumable)), { mode: 0o600 });
      renameSync(tmp, statePath);
    } catch (error) {
      this.logger.error('claudecode: persisting resume state failed', {
        path: statePath,
        error: String(error),
      });
    }
  }
}

/**
 * A missing file or unset path yields an empty map; a corrupt file is an
 * error so the operator notices rather than silently losing every
 * conversation.
 */
function loadResumable(statePath: string | undefined): Map<string, string> {
  if (statePath === undefined || statePath === '' || !existsSync(statePath)) {
    return new Map();
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(readFileSync(statePath, 'utf8'));
  } catch (error) {
    throw new Error(`claudecode: parsing state ${statePath}: ${String(error)}`);
  }
  if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error(`claudecode: parsing state ${statePath}: not an object`);
  }
  const map = new Map<string, string>();
  for (const [key, value] of Object.entries(parsed)) {
    if (typeof value !== 'string') {
      throw new Error(`claudecode: parsing state ${statePath}: non-string uuid for ${key}`);
    }
    map.set(key, value);
  }
  return map;
}
