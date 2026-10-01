import { afterEach, describe, expect, it, vi } from 'vitest';

afterEach(() => vi.unstubAllEnvs());

import { formatMemoryRecord, parseMemoryRecord, type MemoryHeader } from '../src/memory-format.ts';

const header: MemoryHeader = {
  name: 'Example Person',
  relation: 'colleague',
  aliases: ['Example', 'E. Person'],
  contact: 'example@example.test',
  group: 'work',
  summary: 'Works on a synthetic project.',
};

describe('memory record format', () => {
  it('requires name, group, and summary when parsing', () => {
    expect(() => parseMemoryRecord('---\ngroup: work\nsummary: Example\n---\n- fact')).toThrow();
    expect(() => parseMemoryRecord('---\nname: Example\nsummary: Example\n---\n- fact')).toThrow();
    expect(() => parseMemoryRecord('---\nname: Example\ngroup: work\n---\n- fact')).toThrow();
  });

  it('formats and parses valid headers and facts without changing dated facts', () => {
    vi.stubEnv('TZ', 'America/Los_Angeles');
    const fact = '[2024-02-29, third-party] Synthetic dated fact.';
    const text = formatMemoryRecord(header, [fact, 'Synthetic undated fact.'], 'josh', new Date('2026-09-30T23:59:00-07:00'));

    expect(text).toContain('- [2024-02-29, third-party] Synthetic dated fact.\n');
    expect(parseMemoryRecord(text)).toEqual({
      header,
      facts: [fact, '[2026-09-30, josh] Synthetic undated fact.'],
    });
  });

  it('refuses multiline header values and comma-containing aliases', () => {
    expect(() => formatMemoryRecord({ ...header, name: 'Example\nname: forged' }, [], 'josh', new Date())).toThrow();
    expect(() => formatMemoryRecord({ ...header, aliases: ['Example, Inc.'] }, [], 'josh', new Date())).toThrow();
    expect(() => parseMemoryRecord('---\nname: Example\ngroup: work\nsummary: first\nsecond\n---\n- fact')).toThrow();
    expect(parseMemoryRecord('---\nname: Example\naliases: Example, Other\ngroup: work\nsummary: Example\n---\n- fact').header.aliases).toEqual(['Example', 'Other']);
  });

  it('rejects invalid record and fact grammar', () => {
    expect(() => parseMemoryRecord('name: Example\n- fact')).toThrow();
    expect(() => parseMemoryRecord('---\nname: Example\ngroup: work\nsummary: Example\n---\nfact')).toThrow();
    expect(() => parseMemoryRecord('---\nname: Example\ngroup: work\nsummary: Example\n---\n- [2024-02-30, josh] Invalid date')).toThrow();
    expect(() => formatMemoryRecord(header, ['[2024-02-30, josh] Invalid date'], 'josh', new Date())).toThrow();
  });

  it('round-trips a formatted record deterministically', () => {
    const now = new Date('2026-09-30T12:34:56.000Z');
    const text = formatMemoryRecord(header, ['Synthetic fact.'], 'inferred', now);

    const parsed = parseMemoryRecord(text);

    expect(formatMemoryRecord(parsed.header, parsed.facts, 'josh', now)).toBe(text);
  });
});
