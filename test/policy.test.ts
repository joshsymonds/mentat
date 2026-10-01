import { describe, expect, it } from 'vitest';

import { nullLogger, type Logger } from '../src/log.ts';
import { allowAllPolicy } from '../src/policy.ts';

function capturingLogger(lines: { message: string; fields?: Record<string, unknown> }[]): Logger {
  const push = (message: string, fields?: Record<string, unknown>): void => {
    lines.push({ message, ...(fields !== undefined && { fields }) });
  };
  return { info: push, warn: push, error: push };
}

describe('allowAllPolicy', () => {
  it('denies end_conversation outside the voice surface and names the surface', async () => {
    const decision = await allowAllPolicy(nullLogger)(
      'mcp__mentat__end_conversation',
      { reason: 'done' },
      { sessionId: 'session', meta: { surface: 'chat' } },
    );
    expect(decision.behavior).toBe('deny');
    if (decision.behavior === 'allow') throw new Error('expected a denial');
    expect(decision.message).toContain('chat');
  });

  it('allows end_conversation on the voice surface', () => {
    expect(
      allowAllPolicy(nullLogger)(
        'mcp__mentat__end_conversation',
        { reason: 'done' },
        { sessionId: 'session', meta: { surface: 'voice' } },
      ),
    ).toEqual({ behavior: 'allow', updatedInput: { reason: 'done' } });
  });

  it('allows set_voice_mode on the voice surface and preserves its input', async () => {
    const input = { mode: 'conversation', language: 'es' };
    const decision = await allowAllPolicy(nullLogger)(
      'mcp__mentat__set_voice_mode',
      input,
      { sessionId: 'session', meta: { surface: 'voice' } },
    );
    expect(decision).toEqual({ behavior: 'allow', updatedInput: input });
    if (decision.behavior === 'deny') throw new Error('expected an allow decision');
    expect(decision.updatedInput).toBe(input);
  });

  it('denies set_voice_mode outside the voice surface and names the surface', async () => {
    const decision = await allowAllPolicy(nullLogger)(
      'mcp__mentat__set_voice_mode',
      { mode: 'conversation', language: 'es' },
      { sessionId: 'session', meta: { surface: 'chat' } },
    );
    expect(decision.behavior).toBe('deny');
    if (decision.behavior === 'allow') throw new Error('expected a denial');
    expect(decision.message).toContain('chat');
  });

  it('denies set_voice_mode when the surface is missing', async () => {
    const decision = await allowAllPolicy(nullLogger)(
      'mcp__mentat__set_voice_mode',
      { mode: 'conversation', language: 'es' },
      { sessionId: 'session', meta: {} },
    );
    expect(decision.behavior).toBe('deny');
    if (decision.behavior === 'allow') throw new Error('expected a denial');
    expect(decision.message).toContain('unknown');
  });

  it('logs the reason for a non-voice set_voice_mode denial', async () => {
    const lines: { message: string; fields?: Record<string, unknown> }[] = [];
    await allowAllPolicy(capturingLogger(lines))(
      'mcp__mentat__set_voice_mode',
      { mode: 'conversation', language: 'es' },
      { sessionId: 'session', meta: { surface: 'chat', user: 'josh' } },
    );
    expect(lines).toEqual([
      {
        message: 'permission decision',
        fields: {
          tool: 'mcp__mentat__set_voice_mode',
          decision: 'deny',
          reason: 'non-voice surface',
          session_id: 'session',
          surface: 'chat',
          user: 'josh',
        },
      },
    ]);
  });

  it('allows other tools on every surface', () => {
    expect(
      allowAllPolicy(nullLogger)(
        'mcp__mentat__dial',
        { number: '+15555550123' },
        { sessionId: 'session', meta: { surface: 'chat' } },
      ),
    ).toEqual({ behavior: 'allow', updatedInput: { number: '+15555550123' } });
  });

  it('denies voice end_conversation that follows an unspoken tool and logs the reason', async () => {
    const lines: { message: string; fields?: Record<string, unknown> }[] = [];
    const decision = await allowAllPolicy(capturingLogger(lines))(
      'mcp__mentat__end_conversation',
      { reason: 'done' },
      { sessionId: 'session', meta: { surface: 'voice', user: 'josh' } },
      { followsUnspokenTool: true },
    );
    expect(decision).toEqual({
      behavior: 'deny',
      message:
        'Speak your answer to the caller first. Call end_conversation by itself only after your final spoken answer; other tool results from this step have not been spoken yet.',
    });
    expect(lines).toEqual([
      {
        message: 'permission decision',
        fields: {
          tool: 'mcp__mentat__end_conversation',
          decision: 'deny',
          reason: 'unspoken tool result',
          session_id: 'session',
          surface: 'voice',
          user: 'josh',
        },
      },
    ]);
  });

  it('allows voice end_conversation when no tool followed the last text', () => {
    expect(
      allowAllPolicy(nullLogger)(
        'mcp__mentat__end_conversation',
        { reason: 'done' },
        { sessionId: 'session', meta: { surface: 'voice' } },
        { followsUnspokenTool: false },
      ),
    ).toEqual({ behavior: 'allow', updatedInput: { reason: 'done' } });
  });

  it('logs the reason for a non-voice end_conversation denial', async () => {
    const lines: { message: string; fields?: Record<string, unknown> }[] = [];
    await allowAllPolicy(capturingLogger(lines))(
      'mcp__mentat__end_conversation',
      { reason: 'done' },
      { sessionId: 'session', meta: { surface: 'chat' } },
    );
    expect(lines).toEqual([
      {
        message: 'permission decision',
        fields: {
          tool: 'mcp__mentat__end_conversation',
          decision: 'deny',
          reason: 'non-voice surface',
          session_id: 'session',
          surface: 'chat',
          user: '',
        },
      },
    ]);
  });
});
