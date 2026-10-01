import { describe, expect, it } from 'vitest';

import { formatMemoryIndex } from '../src/memory-index.ts';

describe('formatMemoryIndex', () => {
  it('groups by topic and sorts groups and identities deterministically', () => {
    expect(
      formatMemoryIndex([
        {
          id: 'zeta',
          header: {
            name: 'Zora',
            relation: 'neighbor',
            aliases: ['Z'],
            group: 'people',
            summary: 'Keeps bees',
          },
        },
        {
          id: 'topic-b',
          header: {
            name: 'Beta topic',
            aliases: [],
            group: 'topics',
            summary: 'Synthetic topic record',
          },
        },
        {
          id: 'alpha',
          header: {
            name: 'Ari',
            relation: 'friend',
            aliases: ['Ari-alt', 'A'],
            group: 'people',
            summary: 'Likes tea',
          },
        },
      ]),
    ).toBe(
      '## people\n' +
        '- Ari (friend; also "Ari-alt", "A") — Likes tea [alpha]\n' +
        '- Zora (neighbor; also "Z") — Keeps bees [zeta]\n' +
        '\n' +
        '## topics\n' +
        '- Beta topic — Synthetic topic record [topic-b]\n',
    );
  });

  it('omits empty relation and aliases without leaving punctuation', () => {
    expect(
      formatMemoryIndex([
        {
          id: 'plain',
          header: {
            name: 'Cedar',
            relation: '',
            aliases: [],
            group: 'places',
            summary: 'Synthetic place',
          },
        },
      ]),
    ).toBe('## places\n- Cedar — Synthetic place [plain]\n');
  });

  it('renders only the everyday aliases present in the supplied header', () => {
    const everyday = {
      id: 'mira',
      header: {
        name: 'Mira',
        relation: 'friend',
        aliases: ['M'],
        group: 'people',
        summary: 'Synthetic person',
      },
    };

    expect(formatMemoryIndex([everyday])).toBe(
      '## people\n- Mira (friend; also "M") — Synthetic person [mira]\n',
    );
  });
});
