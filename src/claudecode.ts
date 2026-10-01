// The live Backend: one persistent SDK session (one claude child) per
// sessionId, speaking streaming input. Turns within a session are serialized;
// conversations resume across child death and daemon restarts via a persisted
// sessionId→CLI-UUID map. The SDK call is injectable so every behavior here
// tests offline.

import { closeSync, existsSync, openSync, readFileSync, renameSync, writeFileSync, writeSync } from 'node:fs';
import { join } from 'node:path';
import process from 'node:process';

import { query, type Options, type SDKUserMessage } from '@anthropic-ai/claude-agent-sdk';
import { z } from 'zod';

import { AtCapacityError, type Backend, type Event, type Turn } from './backend.ts';
import { shouldDenyFileTool } from './file-guard.ts';
import type { Logger } from './log.ts';
import type { CallContext, PolicyFn, TurnContext } from './policy.ts';
import { Translator } from './translate.ts';

/**
 * With no operator tool policy at all (neither allow nor disallow lists),
 * disallow the dangerous built-ins: the isolated child still carries the full
 * toolset, and a voice surface must not drive Bash/Write/etc. by default.
 * Opt into danger explicitly.
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

/**
 * Time bound on the abandon path's interrupt and per-message drain reads. A
 * wedged child that never honors the interrupt would otherwise hold the
 * session's turn slot forever — and the janitor can't expire a session whose
 * turn is still counted active.
 */
const ABANDON_TIMEOUT_MS = 10_000;

/**
 * How long a voice end_conversation permission check waits for the session's
 * stream to reach that tool_use block before allowing the call anyway.
 */
const END_CONVERSATION_OBSERVE_TIMEOUT_MS = 2_000;

/** The slice of the SDK's query() the backend consumes — injectable in tests. */
interface QueryHandle extends AsyncIterable<unknown> {
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
  /** Model fixed for voice children; defaults to the standard-speed Sol route. */
  voiceModel?: string;
  effort?: Options['effort'];
  /** Replaces the CLI's system prompt when set. */
  systemPrompt?: string;
  /** Everyday memory index source; records are read only when a child starts. */
  memory?: { index(): Promise<string> };
  /** Configured memory directory, retained for startup composition. */
  memoryDir?: string;
  /** Explicit MCP server map; nothing else is reachable (strictMcpConfig). */
  mcpServers?: Options['mcpServers'];
  /** Per-voice-session gateway and caller key loaded from its configured file. */
  voiceGateway?: { url: string; callerKey: string };
  allowedTools?: string[];
  disallowedTools?: string[];
  maxBudgetUsd?: number;
  /** Cap on concurrent live children; 0/absent disables the cap. */
  maxSessions?: number;
  /** Env var names passed through beyond the standard allowlist. */
  extraEnv?: string[];
  /** Persists the sessionId→CLI-UUID map across daemon restarts. */
  statePath?: string;
  /**
   * When set, appends each session's raw SDK message stream to
   * <recordDir>/<sessionId>.jsonl — recordings are future test fixtures.
   * Grows without bound; the operator owns retention.
   */
  recordDir?: string;
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
  voiceGateway?: { url: string; callerKey: string },
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
  if (voiceGateway !== undefined) {
    delete out.ANTHROPIC_API_KEY;
    delete out.ANTHROPIC_AUTH_TOKEN;
    out.ANTHROPIC_BASE_URL = voiceGateway.url;
    out.ANTHROPIC_CUSTOM_HEADERS = `X-Patchbay-Key: ${voiceGateway.callerKey}`;
    out.CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC = '1';
  }
  return out;
}

/**
 * The effective disallow list, ported from go-v2 toolPolicy: the dangerous
 * built-ins default applies only when the operator set no tool policy at all.
 * An allow-only policy (e.g. MENTAT_ALLOWED_TOOLS=Bash) must not be silently
 * overridden by default deny rules, which take precedence in the CLI.
 */
function effectiveDisallowedTools(config: ClaudeCodeConfig): string[] {
  const allow = config.allowedTools ?? [];
  const disallow = config.disallowedTools ?? [];
  if (allow.length === 0 && disallow.length === 0) {
    return DEFAULT_DISALLOWED_TOOLS;
  }
  return disallow;
}

/**
 * Simple-name shapes for the spawn-time surface context line. The system
 * prompt is a privileged channel; meta values that don't look like plain
 * surface/user names stay out of it entirely.
 */
const SURFACE_SHAPE = /^[a-z0-9][a-z0-9_-]{0,31}$/;
const USER_SHAPE = /^[a-zA-Z0-9][a-zA-Z0-9_@.-]{0,63}$/;
const DEFAULT_VOICE_MODEL = 'chatgpt/sol-fast';
const OPUS_MODEL = 'claude-opus-5-5';

function preloadMcpServers(servers: Options['mcpServers']): Options['mcpServers'] {
  if (servers === undefined) {
    return undefined;
  }
  return Object.fromEntries(
    Object.entries(servers).map(([name, server]) => [name, { ...server, alwaysLoad: true }]),
  );
}

/**
 * The session's surface context as a system-prompt line, or undefined when
 * the spawning turn carried no (well-formed) surface. The daemon stays
 * semantics-free: what a surface means lives in the operator's prompt.
 */
function surfaceContextLine(meta: Record<string, string> | undefined): string | undefined {
  if (meta === undefined) {
    return undefined;
  }
  const surface = meta.surface;
  if (surface === undefined || !SURFACE_SHAPE.test(surface)) {
    return undefined;
  }
  const user = meta.user;
  return user !== undefined && USER_SHAPE.test(user)
    ? `Session surface: ${surface} (user: ${user})`
    : `Session surface: ${surface}`;
}

/**
 * Assembles the per-session SDK options. The isolation flags are
 * unconditional: a bare child inherits the operator's interactive Claude Code
 * configuration (settings, skills, MCP servers), which must never drive a
 * daemon. Exported pure so tests pin every invariant.
 *
 * spawnMeta is the session-creating turn's meta; its surface/user become a
 * trailing system-prompt line so the model knows which surface it serves.
 * Only an operator-configured prompt is extended — supplying systemPrompt to
 * the SDK replaces its default, so a daemon without one must not gain a
 * prompt consisting solely of the surface line.
 *
 * callContextFor resolves what the session's stream showed before a tool_use
 * id; voice end_conversation checks wait on it so the policy can refuse a
 * hang-up that would skip speaking an earlier tool's result.
 */
export function buildOptions(
  config: ClaudeCodeConfig,
  getContext: () => TurnContext,
  resumeUuid?: string,
  spawnMeta?: Record<string, string>,
  callContextFor?: (toolUseID: string) => Promise<CallContext | undefined>,
  memoryIndex?: string,
): Options {
  const voiceGateway = spawnMeta?.surface === 'voice' ? config.voiceGateway : undefined;
  const mcpServers =
    voiceGateway !== undefined ? preloadMcpServers(config.mcpServers) : config.mcpServers;
  const surfaceLine = surfaceContextLine(spawnMeta);
  const env = buildChildEnv(process.env, config.extraEnv ?? [], voiceGateway);
  if (spawnMeta?.surface === 'voice' && config.model === OPUS_MODEL) {
    env.CLAUDE_CODE_DISABLE_FAST_MODE = '1';
  }
  const systemPrompt = config.systemPrompt === undefined
    ? undefined
    : [
        config.systemPrompt,
        surfaceLine,
        ...(memoryIndex !== undefined
          ? [`Everyday memory index:\n${memoryIndex === '' ? 'No everyday memories yet.' : memoryIndex}`]
          : []),
      ].filter((line): line is string => line !== undefined).join('\n\n');
  return {
    settingSources: [],
    skills: [],
    strictMcpConfig: true,
    includePartialMessages: true,
    pathToClaudeCodeExecutable: config.bin,
    env,
    disallowedTools: effectiveDisallowedTools(config),
    ...(config.model !== undefined && { model: config.model }),
    ...(config.effort !== undefined && { effort: config.effort }),
    ...(systemPrompt !== undefined && { systemPrompt }),
    ...(mcpServers !== undefined && { mcpServers }),
    ...(config.allowedTools !== undefined && { allowedTools: config.allowedTools }),
    ...(config.maxBudgetUsd !== undefined && { maxBudgetUsd: config.maxBudgetUsd }),
    ...(resumeUuid !== undefined && { resume: resumeUuid }),
    hooks: {
      PreToolUse: [
        {
          hooks: [(input) => {
            if (input.hook_event_name !== 'PreToolUse') return Promise.resolve({});
            const toolInput =
              typeof input.tool_input === 'object' &&
              input.tool_input !== null &&
              !Array.isArray(input.tool_input)
                ? input.tool_input as Record<string, unknown>
                : {};
            const denied = shouldDenyFileTool(input.tool_name, toolInput, {
              ...(config.memoryDir !== undefined && { memoryDir: config.memoryDir }),
              ...(config.recordDir !== undefined && { recordDir: config.recordDir }),
              ...(env.HOME !== undefined && { home: env.HOME }),
              cwd: input.cwd,
              transcriptPath: input.transcript_path,
              sessionId: input.session_id,
            });
            return Promise.resolve(
              denied
                ? {
                    hookSpecificOutput: {
                      hookEventName: 'PreToolUse',
                      permissionDecision: 'deny',
                      permissionDecisionReason: 'Access to protected Mentat files is denied.',
                    },
                  }
                : {},
            );
          }],
        },
      ],
      ...(spawnMeta?.surface === 'voice'
        ? {
            PostToolUse: [
              {
                matcher: 'mcp__mentat__end_conversation',
                hooks: [(input) => {
                  const isToolError =
                    input.hook_event_name === 'PostToolUse' &&
                    typeof input.tool_response === 'object' &&
                    input.tool_response !== null &&
                    'isError' in input.tool_response &&
                    input.tool_response.isError === true;
                  return Promise.resolve(
                    input.hook_event_name === 'PostToolUse' &&
                    input.tool_name === 'mcp__mentat__end_conversation' &&
                    !isToolError
                      ? { continue: false, stopReason: 'Conversation ended.' }
                      : { continue: true },
                  );
                }],
              },
            ],
          }
        : {}),
    },
    canUseTool: async (toolName, input, { toolUseID }) => {
      // Read before any wait: the decision belongs to the turn that asked.
      const context = getContext();
      const call =
        toolName === 'mcp__mentat__end_conversation' && context.meta.surface === 'voice'
          ? await callContextFor?.(toolUseID)
          : undefined;
      const decision = await config.policy(toolName, input, context, call);
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

const contentBlockStartSchema = z.looseObject({
  type: z.literal('stream_event'),
  event: z.looseObject({
    type: z.literal('content_block_start'),
    content_block: z.looseObject({ type: z.string(), id: z.string().optional() }),
  }),
});

/**
 * The turn's content blocks in stream order, reduced to one fact per tool_use:
 * whether another tool_use started after the model's last text block. Each
 * fact is fixed when its block starts, so a permission check asked before or
 * after the consumer reaches that block gets the same answer.
 */
class ToolUseOrder {
  private toolSinceText = false;
  /** tool_use id → whether another tool_use preceded it since the last text. */
  private readonly followsTool = new Map<string, boolean>();
  private readonly waiters = new Map<string, (followsTool: boolean) => void>();

  /** Turn start: nothing from an earlier turn counts against this one. */
  reset(): void {
    this.toolSinceText = false;
    this.followsTool.clear();
  }

  observe(message: unknown): void {
    const parsed = contentBlockStartSchema.safeParse(message);
    if (!parsed.success) {
      return;
    }
    const block = parsed.data.event.content_block;
    if (block.type === 'text') {
      this.toolSinceText = false;
      return;
    }
    if (block.type !== 'tool_use' || block.id === undefined) {
      return;
    }
    const followsTool = this.toolSinceText;
    this.toolSinceText = true;
    this.followsTool.set(block.id, followsTool);
    this.waiters.get(block.id)?.(followsTool);
  }

  /** The fact for toolUseID once its block is observed; undefined if it
   * isn't observed within timeoutMs. */
  wait(toolUseID: string, timeoutMs: number): Promise<boolean | undefined> {
    const seen = this.followsTool.get(toolUseID);
    if (seen !== undefined) {
      return Promise.resolve(seen);
    }
    return new Promise((resolve) => {
      const timer = setTimeout(() => {
        this.waiters.delete(toolUseID);
        resolve(undefined);
      }, timeoutMs);
      this.waiters.set(toolUseID, (followsTool) => {
        clearTimeout(timer);
        this.waiters.delete(toolUseID);
        resolve(followsTool);
      });
    });
  }
}

interface Recorder {
  write(message: unknown): void;
  close(): void;
}

const NULL_RECORDER: Recorder = { write: () => undefined, close: () => undefined };

/**
 * Per-session message recorder. The client-chosen sessionId is URI-encoded so
 * it cannot traverse out of recordDir. The file descriptor opens lazily and
 * stays open for the session's lifetime — recording runs per message,
 * including per-token partials, so a sync open/close per write would stall
 * the event loop. Failures are logged once and recording stops — a full disk
 * must not fail turns.
 */
function makeRecorder(
  recordDir: string | undefined,
  sessionId: string,
  logger: Logger,
): Recorder {
  if (recordDir === undefined || recordDir === '') {
    return NULL_RECORDER;
  }
  const path = join(recordDir, encodeURIComponent(sessionId) + '.jsonl');
  let fd: number | undefined;
  let broken = false;
  return {
    write: (message) => {
      if (broken) {
        return;
      }
      try {
        fd ??= openSync(path, 'a');
        writeSync(fd, JSON.stringify(message) + '\n');
      } catch (error) {
        broken = true;
        logger.error('claudecode: session recording failed', {
          path,
          error: String(error),
        });
      }
    },
    close: () => {
      if (fd !== undefined) {
        try {
          closeSync(fd);
        } catch {
          // already closed or invalid; nothing to release
        }
        fd = undefined;
      }
    },
  };
}

interface Session {
  queue: AsyncQueue<SDKUserMessage>;
  iterator: AsyncIterator<unknown>;
  handle: QueryHandle;
  translator: Translator;
  mutex: Mutex;
  /** The ACTIVE turn's identity context — set at turn start, cleared at turn
   * end, never carried across turns (authority is per-turn). */
  context: { current: TurnContext | null };
  /** Reset at each turn start with the context; fed from the stream. */
  toolUses: ToolUseOrder;
  recorder: Recorder;
  dead: boolean;
  closed: boolean;
}

function userMessage(text: string): SDKUserMessage {
  return {
    type: 'user',
    message: { role: 'user', content: [{ type: 'text', text }] },
    parent_tool_use_id: null,
    session_id: '',
  };
}

/** Sentinel resolved by a turn's abort signal, raced against iterator reads. */
const ABORTED = Symbol('aborted');

function abortPromise(signal: AbortSignal | undefined): Promise<typeof ABORTED> {
  if (signal === undefined) {
    return new Promise(() => undefined); // never resolves; generator teardown handles it
  }
  return new Promise((resolve) => {
    if (signal.aborted) {
      resolve(ABORTED);
      return;
    }
    signal.addEventListener(
      'abort',
      () => {
        resolve(ABORTED);
      },
      { once: true },
    );
  });
}

function withTimeout<T>(promise: Promise<T>, ms: number, what: string): Promise<T> {
  let timer: NodeJS.Timeout | undefined;
  const timeout = new Promise<never>((_, reject) => {
    timer = setTimeout(() => {
      reject(new Error(`claudecode: ${what} timed out after ${String(ms)}ms`));
    }, ms);
  });
  return Promise.race([promise, timeout]).finally(() => {
    clearTimeout(timer);
  });
}

export class ClaudeCode implements Backend {
  private readonly config: ClaudeCodeConfig;
  private readonly logger: Logger;
  private readonly queryFn: QueryFn;
  private readonly sessions = new Map<string, Session>();
  private readonly starting = new Map<string, Promise<Session>>();
  private readonly closing = new Set<string>();
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

  /** Starts the voice child at token time without queuing a synthetic turn. */
  async prestartVoiceSession(sessionId: string): Promise<boolean> {
    if (this.config.voiceGateway === undefined) {
      return false;
    }
    if (sessionId === '') {
      throw new Error('claudecode: pre-start requires a sessionId');
    }
    const turn: Turn = {
      sessionId,
      text: '',
      meta: { surface: 'voice', user: 'josh' },
      effort: 'low',
      model: this.config.voiceModel ?? DEFAULT_VOICE_MODEL,
    };
    for (;;) {
      const session = await this.sessionFor(turn);
      const release = await session.mutex.acquire();
      if (session.dead) {
        release();
        if (session.closed) {
          throw new Error('claudecode: session closed during pre-start');
        }
        continue;
      }
      release();
      return true;
    }
  }

  /** Acquires the session's turn slot and sends the turn into the child. */
  private async startTurn(turn: Turn): Promise<{ session: Session; release: () => void }> {
    const session = await this.sessionFor(turn);
    const release = await session.mutex.acquire();
    // The session may have died while this turn waited on the previous one;
    // respawn rather than reading a dead iterator.
    if (session.dead) {
      release();
      if (session.closed) {
        throw new Error('claudecode: session closed during turn start');
      }
      return this.startTurn(turn);
    }
    session.context.current = { sessionId: turn.sessionId, meta: turn.meta ?? {} };
    session.toolUses.reset();
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
    const aborted = abortPromise(turn.signal);
    let sawDone = false;
    let messagesRead = 0;
    // The in-flight iterator read. It survives this generator's teardown
    // (promises are multi-consumer), so the abandon path can hand it to the
    // drain instead of losing whatever message it resolves to.
    let pending: Promise<IteratorResult<unknown>> | null = null;
    try {
      while (!sawDone) {
        pending = session.iterator.next();
        const next = await Promise.race([pending, aborted]);
        if (next === ABORTED) {
          return; // finally interrupts the turn, draining from `pending`
        }
        pending = null;
        if (next.done === true) {
          this.dropSession(sessionId, session);
          if (session.closed) {
            throw new Error('claudecode: session closed during turn');
          }
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
        session.recorder.write(next.value);
        session.toolUses.observe(next.value);
        for (const event of this.translateAndLog(sessionId, session, next.value)) {
          if (event.kind === 'done') {
            sawDone = true;
            this.recordResume(sessionId, event.result.sessionId);
          }
          yield event;
        }
      }
    } finally {
      if (!sawDone && !session.dead) {
        await this.abandonTurn(sessionId, session, pending);
      }
      // Cleared only after the abandon settles: permission decisions made
      // while the interrupt is in flight still belong to this turn's identity.
      session.context.current = null;
      release();
    }
  }

  private translateAndLog(sessionId: string, session: Session, message: unknown): Event[] {
    const events = session.translator.translate(message);
    for (const event of events) {
      if (event.kind === 'unknown') {
        this.logger.error('claudecode: unknown SDK message', {
          session_id: sessionId,
          raw: JSON.stringify(event.raw).slice(0, 2000),
        });
      }
    }
    return events;
  }

  /**
   * The consumer left before the turn's done, so the session's stream is
   * stranded mid-turn. Interrupt the child and drain to the interrupted
   * turn's terminal result; if that fails or times out, drop the session so
   * the next turn respawns with resume. `pending` is the turn's in-flight
   * read, drained first so its message isn't lost.
   */
  private async abandonTurn(
    sessionId: string,
    session: Session,
    pending: Promise<IteratorResult<unknown>> | null,
  ): Promise<void> {
    try {
      await withTimeout(session.handle.interrupt(), ABANDON_TIMEOUT_MS, 'interrupt');
      let read = pending ?? session.iterator.next();
      for (let drained = 0; drained < ABANDON_DRAIN_LIMIT; drained += 1) {
        const next = await withTimeout(read, ABANDON_TIMEOUT_MS, 'abandon drain');
        if (next.done === true) {
          break;
        }
        session.recorder.write(next.value);
        session.toolUses.observe(next.value);
        const events = this.translateAndLog(sessionId, session, next.value);
        if (events.some((event) => event.kind === 'done')) {
          this.logger.warn('claudecode: turn abandoned, session interrupted', {
            session_id: sessionId,
          });
          return;
        }
        read = session.iterator.next();
      }
      this.logger.warn('claudecode: turn abandoned, session dropped for respawn', {
        session_id: sessionId,
        error: 'drain exhausted without a terminal result',
      });
    } catch (error) {
      this.logger.warn('claudecode: turn abandoned, session dropped for respawn', {
        session_id: sessionId,
        error: String(error),
      });
    }
    this.dropSession(sessionId, session);
  }

  /** Marks a session dead and releases its resources. The resume uuid is
   * retained, so the conversation survives into the next spawn. */
  private dropSession(sessionId: string, session: Session, closed = false): void {
    session.dead = true;
    session.closed ||= closed;
    this.sessions.delete(sessionId);
    session.queue.end(); // input end is the child's exit signal
    session.recorder.close();
  }

  async closeSession(sessionId: string): Promise<void> {
    this.closing.add(sessionId);
    try {
      await this.starting.get(sessionId)?.catch(() => undefined);
      const session = this.sessions.get(sessionId);
      if (session !== undefined) {
        this.dropSession(sessionId, session, true);
      }
    } finally {
      this.closing.delete(sessionId);
    }
  }

  async close(): Promise<void> {
    await Promise.allSettled([...this.starting.values()]);
    for (const sessionId of [...this.sessions.keys()]) {
      await this.closeSession(sessionId);
    }
  }

  private sessionFor(turn: Turn): Promise<Session> {
    const sessionId = turn.sessionId;
    if (this.closing.has(sessionId)) {
      return Promise.reject(new Error('claudecode: session is closing'));
    }
    const existing = this.sessions.get(sessionId);
    if (existing !== undefined && !existing.dead) {
      return Promise.resolve(existing);
    }
    const starting = this.starting.get(sessionId);
    if (starting !== undefined) {
      return starting;
    }
    const maxSessions = this.config.maxSessions ?? 0;
    if (maxSessions > 0 && this.liveCountExcluding(sessionId) >= maxSessions) {
      return Promise.reject(new AtCapacityError());
    }

    const pending = this.createSession(turn);
    this.starting.set(sessionId, pending);
    void pending.then(
      () => {
        if (this.starting.get(sessionId) === pending) this.starting.delete(sessionId);
      },
      () => {
        if (this.starting.get(sessionId) === pending) this.starting.delete(sessionId);
      },
    );
    return pending;
  }

  private async createSession(turn: Turn): Promise<Session> {
    const sessionId = turn.sessionId;
    const queue = new AsyncQueue<SDKUserMessage>();
    const context: Session['context'] = { current: null };
    // The creating turn's effort/model win over daemon defaults for non-voice
    // sessions; voice children use their configured model or the voice default.
    // These SDK options are fixed at spawn. The turn's meta also supplies the
    // spawn-time surface context line.
    const isVoice = turn.meta?.surface === 'voice';
    const config = {
      ...this.config,
      ...(turn.effort !== undefined && { effort: turn.effort }),
      ...(turn.model !== undefined && { model: turn.model }),
      ...(isVoice && { model: this.config.voiceModel ?? DEFAULT_VOICE_MODEL }),
    };
    let memoryIndex: string | undefined;
    if (this.config.memory !== undefined && this.config.systemPrompt !== undefined) {
      try {
        memoryIndex = await this.config.memory.index();
      } catch (error) {
        this.logger.error('claudecode: memory index unavailable', {
          session_id: sessionId,
          error: String(error),
        });
      }
    }
    const toolUses = new ToolUseOrder();
    const options = buildOptions(
      config,
      () => context.current ?? { sessionId, meta: {} },
      this.resumable.get(sessionId),
      turn.meta,
      async (toolUseID) => {
        const followsUnspokenTool = await toolUses.wait(
          toolUseID,
          END_CONVERSATION_OBSERVE_TIMEOUT_MS,
        );
        if (followsUnspokenTool === undefined) {
          this.logger.warn('claudecode: end_conversation tool_use not observed, allowing', {
            session_id: sessionId,
            tool_use_id: toolUseID,
            timeout_ms: END_CONVERSATION_OBSERVE_TIMEOUT_MS,
          });
          return undefined;
        }
        return { followsUnspokenTool };
      },
      memoryIndex,
    );
    const handle = this.queryFn({ prompt: queue, options });
    const session: Session = {
      queue,
      iterator: handle[Symbol.asyncIterator](),
      handle,
      translator: new Translator(),
      mutex: new Mutex(),
      context,
      toolUses,
      recorder: makeRecorder(this.config.recordDir, sessionId, this.logger),
      dead: false,
      closed: false,
    };
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
    for (const id of this.starting.keys()) {
      const session = this.sessions.get(id);
      if (id !== sessionId && (session === undefined || session.dead)) {
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
