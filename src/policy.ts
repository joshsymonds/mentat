// The permission seam: every tool call the model wants to make passes through
// one PolicyFn before it executes. The context is the ACTIVE turn's identity
// (surface, user, auth metadata from Turn.meta) — bound per turn and never
// cached on the session, because the conversation is memory and authority is
// per-turn. Future tiers/step-up live here without touching the architecture.

import type { Logger } from './log.ts';

/** The identity context of the turn that initiated a tool call. */
export interface TurnContext {
  sessionId: string;
  meta: Record<string, string>;
}

/**
 * What the session's own stream showed before this call. Supplied only where
 * a check needs it (voice end_conversation); undefined everywhere else.
 */
export interface CallContext {
  /** Another tool_use started after the model's last text block this turn,
   * so its result has not been spoken to the caller yet. */
  followsUnspokenTool: boolean;
}

type PolicyDecision =
  | { behavior: 'allow'; updatedInput: Record<string, unknown> }
  | { behavior: 'deny'; message: string };

export type PolicyFn = (
  toolName: string,
  input: Record<string, unknown>,
  context: TurnContext,
  call?: CallContext,
) => PolicyDecision | Promise<PolicyDecision>;

/**
 * Allows every tool call, logging one structured decision line each — the
 * shipped default for a single-user daemon whose tool surface is read-mostly.
 * Tighten by replacing the PolicyFn, not by editing call sites.
 */
export function allowAllPolicy(logger: Logger): PolicyFn {
  return (toolName, input, context, call) => {
    const surface = context.meta.surface ?? '';
    const user = context.meta.user ?? '';
    if (
      (toolName === 'mcp__mentat__end_conversation' || toolName === 'mcp__mentat__set_voice_mode') &&
      surface !== 'voice'
    ) {
      logger.info('permission decision', {
        tool: toolName,
        decision: 'deny',
        reason: 'non-voice surface',
        session_id: context.sessionId,
        surface,
        user,
      });
      return {
        behavior: 'deny',
        message: `${toolName} is only allowed on the voice surface; received ${surface || 'unknown'}`,
      };
    }
    if (toolName === 'mcp__mentat__end_conversation' && call?.followsUnspokenTool === true) {
      logger.info('permission decision', {
        tool: toolName,
        decision: 'deny',
        reason: 'unspoken tool result',
        session_id: context.sessionId,
        surface,
        user,
      });
      return {
        behavior: 'deny',
        message:
          'Speak your answer to the caller first. Call end_conversation by itself only after your final spoken answer; other tool results from this step have not been spoken yet.',
      };
    }
    logger.info('permission decision', {
      tool: toolName,
      decision: 'allow',
      session_id: context.sessionId,
      surface,
      user,
    });
    return { behavior: 'allow', updatedInput: input };
  };
}
