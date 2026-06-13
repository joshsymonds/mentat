// The daemon's harness-altitude conversation abstraction. A Backend accepts
// user turns and streams typed events back; the agentic loop, tool execution,
// and MCP connections live inside the implementation, invisible to callers.

/** Reasoning-effort levels a turn may request (mirrors the SDK's enum). */
export type Effort = 'low' | 'medium' | 'high' | 'xhigh' | 'max';

export const EFFORT_LEVELS: ReadonlySet<string> = new Set<Effort>([
  'low',
  'medium',
  'high',
  'xhigh',
  'max',
]);

/**
 * Shape of an acceptable per-turn model name: a CLI alias ("sonnet") or a
 * full model id ("claude-sonnet-4-6"). Deliberately tighter than what the
 * CLI would accept — the value lands on a child's argv.
 */
export const MODEL_PATTERN = /^[a-zA-Z0-9][a-zA-Z0-9.-]{0,63}$/;

/** One user utterance entering a conversation session. */
export interface Turn {
  /**
   * Groups turns into a conversation. Implementations decide what continuity
   * it buys (the live backend maps it to a CLI session).
   */
  sessionId: string;
  /** The user's utterance. */
  text: string;
  /**
   * Surface context (surface, area, user identity, auth metadata). Bound to
   * this turn only — never cached on the session: the conversation is memory,
   * authority is per-turn.
   */
  meta?: Record<string, string>;
  /**
   * Aborting cancels the turn promptly (the caller went away), even while the
   * backend is waiting silently on the model. Without it, cancellation is
   * only noticed when the consumer stops iterating, which the backend can't
   * see until the next event arrives.
   */
  signal?: AbortSignal;
  /**
   * Reasoning effort for the session this turn belongs to. Effort is fixed
   * when the backend spawns the session, so it only takes effect on the turn
   * that creates (or respawns) one; later turns on a live session keep the
   * creation effort. Surfaces with latency budgets (voice) send "low".
   */
  effort?: Effort;
  /**
   * Model for the session this turn belongs to (alias or full id). Like
   * effort, fixed at session spawn: only the creating turn's value applies.
   * Surfaces with latency budgets (voice) send a fast model.
   */
  model?: string;
}

/** A turn could not start because the backend is at its session capacity. */
export class AtCapacityError extends Error {
  constructor() {
    super('backend: at session capacity');
    this.name = 'AtCapacityError';
  }
}

/** Token accounting for a completed turn. */
interface Usage {
  inputTokens: number;
  outputTokens: number;
  cacheReadInputTokens: number;
  cacheCreationInputTokens: number;
}

/**
 * A turn's final outcome.
 *
 * isError reports protocol-level failure only. A turn whose tool calls were
 * all denied still completes with isError=false; whether the user's intent
 * succeeded lives in text, not here.
 */
export interface Result {
  text: string;
  isError: boolean;
  stopReason: string;
  sessionId: string;
  costUsd: number;
  usage: Usage;
}

/**
 * One occurrence in a turn's stream, in rough lifecycle order of a turn.
 * `unknown` carries an SDK message this build does not recognize — it signals
 * the SDK moved ahead of this daemon and is logged loudly, never dropped.
 */
export type Event =
  | { kind: 'textDelta'; text: string }
  | { kind: 'thinkingDelta'; text: string }
  | { kind: 'thinking'; tokens: number }
  | { kind: 'toolStart'; tool: string }
  | { kind: 'toolResult'; tool: string; isError: boolean; content: string }
  | { kind: 'done'; result: Result }
  | { kind: 'unknown'; raw: unknown };

/**
 * Streams conversation events for user turns. converse rejects for failures
 * to start the turn (AtCapacityError when the session cap is hit); failures
 * mid-stream are thrown by the iterator. The returned iterable MUST be
 * consumed (or explicitly closed by breaking out of iteration): the turn has
 * already been sent to the session, and implementations may hold the
 * session's turn slot until the iterable finishes. closeSession releases a
 * session's resources (idle expiry); it is harmless for unknown or
 * already-closed sessions, and a later turn with the same sessionId may
 * transparently restore context.
 */
export interface Backend {
  converse(turn: Turn): Promise<AsyncIterable<Event>>;
  closeSession(sessionId: string): Promise<void>;
}
