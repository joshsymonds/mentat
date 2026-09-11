import { describe, expect, it } from 'vitest';

import { nullLogger } from '../src/log.ts';
import { allowAllPolicy } from '../src/policy.ts';

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

  it('allows other tools on every surface', () => {
    expect(
      allowAllPolicy(nullLogger)(
        'mcp__mentat__dial',
        { number: '+15555550123' },
        { sessionId: 'session', meta: { surface: 'chat' } },
      ),
    ).toEqual({ behavior: 'allow', updatedInput: { number: '+15555550123' } });
  });
});
