// The daemon's harness-altitude conversation abstraction. A Backend accepts
// user turns and streams typed events back; the agentic loop, tool execution,
// and MCP connections live inside the implementation, invisible to callers.

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
}

/** Token accounting for a completed turn. */
export interface Usage {
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
 * to start the turn; failures mid-stream are thrown by the iterator.
 * closeSession releases a session's resources (idle expiry); it is harmless
 * for unknown or already-closed sessions, and a later turn with the same
 * sessionId may transparently restore context.
 */
export interface Backend {
  converse(turn: Turn): Promise<AsyncIterable<Event>>;
  closeSession(sessionId: string): Promise<void>;
}
