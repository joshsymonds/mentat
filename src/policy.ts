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

type PolicyDecision =
  | { behavior: 'allow'; updatedInput: Record<string, unknown> }
  | { behavior: 'deny'; message: string };

export type PolicyFn = (
  toolName: string,
  input: Record<string, unknown>,
  context: TurnContext,
) => PolicyDecision | Promise<PolicyDecision>;

/**
 * Allows every tool call, logging one structured decision line each — the
 * shipped default for a single-user daemon whose tool surface is read-mostly.
 * Tighten by replacing the PolicyFn, not by editing call sites.
 */
export function allowAllPolicy(logger: Logger): PolicyFn {
  return (toolName, input, context) => {
    const surface = context.meta.surface ?? '';
    const user = context.meta.user ?? '';
    if (toolName === 'mcp__mentat__end_conversation' && surface !== 'voice') {
      logger.info('permission decision', {
        tool: toolName,
        decision: 'deny',
        session_id: context.sessionId,
        surface,
        user,
      });
      return {
        behavior: 'deny',
        message: `mcp__mentat__end_conversation is only allowed on the voice surface; received ${surface || 'unknown'}`,
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
