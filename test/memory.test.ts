import { execFileSync } from 'node:child_process';
import { accessSync, mkdirSync, mkdtempSync, readFileSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { delimiter, join } from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';

import { nullLogger } from '../src/log.ts';
import {
  MEMORY_INDEX_CAP_BYTES,
  MemoryStore,
  type FactSource,
  type MemoryHeader,
  type MemoryStoreOptions,
} from '../src/memory.ts';

const directories: string[] = [];
const fixedDate = new Date(2026, 8, 30, 12);

function makeStore(overrides: Partial<MemoryStoreOptions> = {}): MemoryStore {
  const dir = overrides.dir ?? mkdtempSync(join(tmpdir(), 'mentat-memory-'));
  if (!directories.includes(dir)) directories.push(dir);
  return new MemoryStore({ dir, logger: nullLogger, now: () => fixedDate, ...overrides });
}

function header(overrides: Partial<MemoryHeader> = {}): MemoryHeader {
  return {
    name: 'Synthetic Rowan',
    relation: 'friend',
    aliases: ['Rowan alias'],
    group: 'people',
    summary: 'Synthetic summary',
    ...overrides,
  };
}

function git(dir: string, ...args: string[]): string {
  return execFileSync('git', ['-C', dir, ...args], { encoding: 'utf8' }).trim();
}

function firstDirectory(): string {
  const dir = directories[0];
  if (dir === undefined) throw new Error('test memory directory was not created');
  return dir;
}

function gitExecutable(): string {
  const searchPath = process.env.PATH;
  if (searchPath === undefined) throw new Error('test PATH is missing');
  for (const directory of searchPath.split(delimiter)) {
    const candidate = join(directory, 'git');
    try {
      accessSync(candidate);
      return candidate;
    } catch {
      continue;
    }
  }
  throw new Error('git executable was not found');
}

afterEach(() => {
  for (const dir of directories.splice(0)) rmSync(dir, { recursive: true, force: true });
});

describe('MemoryStore', () => {
  it('publishes the fixed default index cap', () => {
    expect(MEMORY_INDEX_CAP_BYTES).toBe(16384);
  });

  it('stamps undated facts locally, preserves dated facts, and returns content revisions', async () => {
    const store = makeStore();
    const source: FactSource = 'inferred';
    const saved = await store.save({
      id: 'rowan',
      tier: 'everyday',
      header: header(),
      facts: ['likes tea', '[2025-02-03, josh] older fact'],
      source,
    });
    const record = readFileSync(join(firstDirectory(), 'rowan.md'), 'utf8');

    expect(record).toContain('- [2026-09-30, inferred] likes tea');
    expect(record).toContain('- [2025-02-03, josh] older fact');
    expect(saved.revision).toMatch(/^[a-f0-9]{64}$/);
    await expect(store.read('rowan')).resolves.toEqual({ text: record, revision: saved.revision });
  });

  it('rejects a stale revision without overwriting the committed record', async () => {
    const store = makeStore();
    const first = await store.save({
      id: 'rowan', tier: 'everyday', header: header(), facts: ['first fact'], source: 'josh',
    });
    const second = await store.save({
      id: 'rowan', tier: 'everyday', header: header({ summary: 'Updated summary' }),
      facts: ['second fact'], source: 'josh', revision: first.revision,
    });

    await expect(store.save({
      id: 'rowan', tier: 'everyday', header: header(), facts: ['stale write'],
      source: 'josh', revision: first.revision,
    })).rejects.toThrow(/revision/i);
    await expect(store.read('rowan')).resolves.toMatchObject({ revision: second.revision });
    expect(readFileSync(join(firstDirectory(), 'rowan.md'), 'utf8')).toContain('second fact');
  });

  it('serializes concurrent saves across store instances', async () => {
    const first = makeStore();
    const second = makeStore({ dir: firstDirectory() });
    const input = {
      id: 'rowan', tier: 'everyday' as const, header: header(), facts: ['one winner'], source: 'josh' as const,
    };
    const results = await Promise.allSettled([first.save(input), second.save(input)]);

    expect(results.filter((result) => result.status === 'fulfilled')).toHaveLength(1);
    expect(results.filter((result) => result.status === 'rejected')).toHaveLength(1);
    expect(readFileSync(join(firstDirectory(), 'rowan.md'), 'utf8')).toContain('one winner');
  });

  it('refuses aliases held by another identity across tiers', async () => {
    const store = makeStore();
    await store.save({
      id: 'rowan', tier: 'private',
      header: header({ aliases: ['Hidden Rowan alias'] }), facts: ['private fact'], source: 'josh',
    });

    await expect(store.save({
      id: 'sage', tier: 'everyday', header: header({ name: 'Synthetic Sage', aliases: ['hidden rowan alias'] }),
      facts: ['everyday fact'], source: 'josh',
    })).rejects.toThrow(/rowan/i);
  });

  it('confirms a save only after its commit is visible in git', async () => {
    const store = makeStore();
    await store.save({
      id: 'rowan', tier: 'everyday', header: header(), facts: ['committed fact'], source: 'josh',
    });

    expect(git(firstDirectory(), 'show', 'HEAD:rowan.md')).toContain('committed fact');
  });

  it('restores the previous file when git commit fails', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'mentat-memory-fail-'));
    directories.push(dir);
    const realGit = gitExecutable();
    const shim = join(dir, 'git-shim');
    writeFileSync(shim, `#!/bin/sh\nif [ -f "${join(dir, 'fail-commit')}" ] && [ "$1" = "-C" ] && [ "$3" = "commit" ]; then exit 23; fi\nexec "${realGit}" "$@"\n`, { mode: 0o755 });
    const memoryDir = join(dir, 'memory');
    const store = makeStore({ dir: memoryDir, git: shim });
    const first = await store.save({
      id: 'rowan', tier: 'everyday', header: header(), facts: ['original fact'], source: 'josh',
    });
    writeFileSync(join(dir, 'fail-commit'), 'fail');

    await expect(store.save({
      id: 'rowan', tier: 'everyday', header: header({ summary: 'Updated summary' }),
      facts: ['replacement fact'], source: 'josh', revision: first.revision,
    })).rejects.toThrow(/commit/i);
    expect(readFileSync(join(memoryDir, 'rowan.md'), 'utf8')).toContain('original fact');
    expect(readFileSync(join(memoryDir, 'rowan.md'), 'utf8')).not.toContain('replacement fact');
  });

  it('looks up private records by case-insensitive keywords without matching everyday records', async () => {
    const store = makeStore();
    await store.save({
      id: 'rowan', tier: 'everyday',
      header: header({ name: 'Synthetic Rowan', aliases: ['Ordinary alias'], summary: 'Everyday summary' }),
      facts: ['everyday-only phrase'], source: 'josh',
    });
    await store.save({
      id: 'sage', tier: 'private',
      header: header({ name: 'Synthetic Sage', aliases: ['Private alias'], summary: 'Private summary' }),
      facts: ['private fact with cobalt'], source: 'josh',
    });
    await store.save({
      id: 'wren', tier: 'private',
      header: header({ name: 'Synthetic Wren', aliases: ['Second private alias'], summary: 'Another private summary' }),
      facts: ['private fact with cobalt'], source: 'josh',
    });

    const result = await store.lookup('COBALT');
    expect(result).toContain('sage');
    expect(result).toContain('private fact with cobalt');
    expect(result).toContain('wren');
    expect(result.indexOf('sage')).toBeLessThan(result.indexOf('wren'));
    await expect(store.lookup('EVERYDAY-ONLY')).resolves.toBe('NO_MATCH');
    await expect(store.lookup('absent keyword')).resolves.toBe('NO_MATCH');
    await expect(store.lookup('')).resolves.toBe('NO_MATCH');
    await expect(store.lookup('  \t ')).resolves.toBe('NO_MATCH');
    await expect(store.lookup('PRIVATE ALIAS')).resolves.toContain('sage');
    await expect(store.lookup('synthetic sage')).resolves.toContain('## sage');
  });

  it('refuses reads of private and missing identities', async () => {
    const store = makeStore();
    await store.save({
      id: 'rowan', tier: 'private', header: header(), facts: ['private fact'], source: 'josh',
    });

    await expect(store.read('rowan')).rejects.toThrow(/private/i);
    await expect(store.read('missing')).rejects.toThrow(/missing|not found/i);
  });

  it('excludes private records and aliases from the index for a paired identity', async () => {
    const store = makeStore();
    const everyday = await store.save({
      id: 'rowan', tier: 'everyday', header: header(), facts: ['everyday fact'], source: 'josh',
    });
    await store.save({
      id: 'rowan', tier: 'private',
      header: header({ aliases: ['Private Rowan alias'], summary: 'Private synthetic summary' }),
      facts: ['private fact'], source: 'josh',
    });
    const text = await store.index();

    expect(text).toContain('Synthetic Rowan');
    expect(text).toContain('Rowan alias');
    expect(text).not.toContain('Private Rowan alias');
    expect(text).not.toContain('Private synthetic summary');
    expect((await store.read('rowan')).revision).toBe(everyday.revision);
  });

  it('refuses index growth past the cap and warns at eighty percent', async () => {
    const warnings: string[] = [];
    const logger = { ...nullLogger, warn: (message: string) => warnings.push(message) };
    const store = makeStore({ logger, capBytes: 200 });
    await store.save({
      id: 'rowan', tier: 'everyday', header: header({ summary: 'x'.repeat(110) }),
      facts: ['fact'], source: 'josh',
    });
    expect(warnings.length).toBeGreaterThan(0);
    const before = await store.index();

    await expect(store.save({
      id: 'sage', tier: 'everyday', header: header({ name: 'Synthetic Sage', aliases: ['Sage alias'], summary: 'y'.repeat(180) }),
      facts: ['fact'], source: 'josh',
    })).rejects.toThrow(/exceed.*200.*cap/i);
    expect(await store.index()).toBe(before);
  });

  it('returns and logs an over-cap hand-edited index, allowing a shrinking save', async () => {
    const errors: string[] = [];
    const logger = { ...nullLogger, error: (message: string) => errors.push(message) };
    const store = makeStore({ logger, capBytes: 2000 });
    const saved = await store.save({
      id: 'rowan', tier: 'everyday', header: header({ summary: 'x'.repeat(300) }),
      facts: ['fact'], source: 'josh',
    });
    const dir = firstDirectory();
    const longIndex = await store.index();
    const overCap = makeStore({ dir, logger, capBytes: Buffer.byteLength(longIndex) - 1 });

    expect(await overCap.index()).toBe(longIndex);
    expect(errors.length).toBeGreaterThan(0);
    await expect(overCap.save({
      id: 'rowan', tier: 'private', header: header({ aliases: ['Private Rowan alias'] }),
      facts: ['private fact'], source: 'josh',
    })).rejects.toThrow(/exceed.*byte cap/i);
    const shrunk = await overCap.save({
      id: 'rowan', tier: 'everyday', header: header({ summary: 'short' }),
      facts: ['fact'], source: 'josh', revision: saved.revision,
    });
    expect(shrunk.revision).not.toBe(saved.revision);
    expect(Buffer.byteLength(await overCap.index())).toBeLessThan(Buffer.byteLength(longIndex));
  });

  it('forgets one fact as one commit and records a dated tombstone', async () => {
    const store = makeStore();
    await store.save({
      id: 'rowan', tier: 'everyday', header: header(),
      facts: ['keep this fact', '[2025-02-03, josh] forget this fact'], source: 'josh',
    });
    const before = git(firstDirectory(), 'rev-list', '--count', 'HEAD');

    await store.forget('rowan', 'everyday', 'forget this fact');

    const record = readFileSync(join(firstDirectory(), 'rowan.md'), 'utf8');
    const tombstones = await store.tombstones();
    expect(record).toContain('keep this fact');
    expect(record).not.toContain('forget this fact');
    expect(tombstones).toContain('[2026-09-30] rowan: [2025-02-03, josh] forget this fact');
    expect(git(firstDirectory(), 'rev-list', '--count', 'HEAD')).toBe(String(Number(before) + 1));
    expect(git(firstDirectory(), 'show', 'HEAD:_tombstones.md')).toBe(tombstones.trimEnd());
  });

  it('forgets a whole record, tombstones every fact, and permits index shrinkage over cap', async () => {
    const store = makeStore({ capBytes: 1000 });
    await store.save({
      id: 'rowan', tier: 'everyday', header: header({ summary: 'x'.repeat(120) }),
      facts: ['first removed fact', '[2025-02-03, inferred] second removed fact'], source: 'josh',
    });
    const dir = firstDirectory();
    const record = readFileSync(join(dir, 'rowan.md'), 'utf8');
    const indexLength = Buffer.byteLength(await store.index());
    const capped = makeStore({ dir, capBytes: indexLength - 1 });
    const before = git(dir, 'rev-list', '--count', 'HEAD');

    await capped.forget('rowan', 'everyday');

    expect(() => readFileSync(join(dir, 'rowan.md'), 'utf8')).toThrow();
    expect(await capped.tombstones()).toContain('[2026-09-30] rowan: [2026-09-30, josh] first removed fact');
    expect(await capped.tombstones()).toContain('[2026-09-30] rowan: [2025-02-03, inferred] second removed fact');
    expect(await capped.tombstones()).toContain('[2026-09-30] rowan (everyday): record forgotten — Synthetic Rowan: ');
    expect(await capped.tombstones()).toContain('x'.repeat(120));
    expect(git(dir, 'rev-list', '--count', 'HEAD')).toBe(String(Number(before) + 1));
    expect(git(dir, 'show', 'HEAD:_tombstones.md')).toBe((await capped.tombstones()).trimEnd());
    expect(record).toContain('first removed fact');
    expect(await capped.index()).toBe('');
  });

  it('writes a record-level tombstone for a whole-record forget with no facts', async () => {
    const store = makeStore();
    await store.save({
      id: 'rowan', tier: 'private', header: header({ summary: 'Empty record summary' }),
      facts: [], source: 'josh',
    });

    await store.forget('rowan', 'private');

    await expect(store.tombstones()).resolves.toBe(
      '- [2026-09-30] rowan (private): record forgotten — Synthetic Rowan: Empty record summary\n',
    );
    expect(git(firstDirectory(), 'show', 'HEAD:_tombstones.md')).toContain(
      'rowan (private): record forgotten — Synthetic Rowan: Empty record summary',
    );
  });

  it('restores the record and tombstones when a forget commit fails', async () => {
    const parent = mkdtempSync(join(tmpdir(), 'mentat-memory-forget-fail-'));
    directories.push(parent);
    const realGit = gitExecutable();
    const shim = join(parent, 'git-shim');
    writeFileSync(shim, `#!/bin/sh\nif [ -f "${join(parent, 'fail-commit')}" ] && [ "$1" = "-C" ] && [ "$3" = "commit" ]; then exit 23; fi\nexec "${realGit}" "$@"\n`, { mode: 0o755 });
    const dir = join(parent, 'memory');
    const store = makeStore({ dir, git: shim });
    await store.save({
      id: 'rowan', tier: 'everyday', header: header(), facts: ['first fact', 'second fact'], source: 'josh',
    });
    await store.forget('rowan', 'everyday', 'first fact');
    const record = readFileSync(join(dir, 'rowan.md'), 'utf8');
    const tombstones = await store.tombstones();
    writeFileSync(join(parent, 'fail-commit'), 'fail');

    await expect(store.forget('rowan', 'everyday', 'second fact')).rejects.toThrow(/commit/i);

    expect(readFileSync(join(dir, 'rowan.md'), 'utf8')).toBe(record);
    await expect(store.tombstones()).resolves.toBe(tombstones);
  });

  it('refuses invalid forget ids and symlink record paths', async () => {
    const parent = mkdtempSync(join(tmpdir(), 'mentat-memory-forget-path-'));
    directories.push(parent);
    const dir = join(parent, 'memory');
    mkdirSync(dir);
    writeFileSync(join(parent, 'outside.md'), 'outside');
    symlinkSync(join(parent, 'outside.md'), join(dir, 'rowan.md'));
    const store = makeStore({ dir });

    await expect(store.forget('../outside', 'everyday')).rejects.toThrow(/id/i);
    await expect(store.forget('rowan', 'everyday')).rejects.toThrow(/symbolic link|symlink/i);
  });

  it('rejects invalid ids and symlinks that escape the memory directory', async () => {
    const parent = mkdtempSync(join(tmpdir(), 'mentat-memory-path-'));
    directories.push(parent);
    const dir = join(parent, 'memory');
    mkdirSync(dir);
    writeFileSync(join(parent, 'outside.md'), 'outside');
    symlinkSync(join(parent, 'outside.md'), join(dir, 'rowan.md'));
    const store = makeStore({ dir });

    await expect(store.read('../outside')).rejects.toThrow(/id/i);
    await expect(store.read('rowan')).rejects.toThrow(/symbolic link|symlink/i);
  });
});
