import { createHash, randomUUID } from 'node:crypto';
import { execFile } from 'node:child_process';
import { lstat, mkdir, readFile, readdir, realpath, rename, rm, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { promisify } from 'node:util';

import { formatMemoryRecord, parseMemoryRecord, type FactSource, type MemoryHeader } from './memory-format.ts';
import { formatMemoryIndex } from './memory-index.ts';
import type { Logger } from './log.ts';

const execFileAsync = promisify(execFile);
const locks = new Map<string, Promise<void>>();
const ID_PATTERN = /^[a-z0-9][a-z0-9-]{0,63}$/;
const RECORD_PATTERN = /^([a-z0-9][a-z0-9-]{0,63})(\.private)?\.md$/;

export type Tier = 'everyday' | 'private';
export type { FactSource, MemoryHeader };

export interface SaveInput {
  id: string;
  tier: Tier;
  header: MemoryHeader;
  facts: string[];
  source: FactSource;
  revision?: string;
}

export interface MemoryStoreOptions {
  dir: string;
  logger: Logger;
  now?: () => Date;
  capBytes?: number;
  git?: string;
}

export const MEMORY_INDEX_CAP_BYTES = 16384;

interface StoredRecord {
  id: string;
  tier: Tier;
  text: string;
  revision: string;
  header: MemoryHeader;
  facts: string[];
}

export class MemoryStore {
  private readonly dir: string;
  private readonly logger: Logger;
  private readonly now: () => Date;
  private readonly capBytes: number;
  private readonly git: string;

  constructor(options: MemoryStoreOptions) {
    this.dir = resolve(options.dir);
    this.logger = options.logger;
    this.now = options.now ?? (() => new Date());
    this.capBytes = options.capBytes ?? MEMORY_INDEX_CAP_BYTES;
    this.git = options.git ?? 'git';
  }

  async index(): Promise<string> {
    const records = await this.everydayRecords();
    const text = this.formatIndex(records);
    this.reportIndexSize(Buffer.byteLength(text));
    return text;
  }

  async read(id: string): Promise<{ text: string; revision: string }> {
    this.validateId(id);
    const everydayPath = this.recordPath(id, 'everyday');
    const everyday = await this.readRecordPath(everydayPath);
    if (everyday !== undefined) {
      return { text: everyday.text, revision: everyday.revision };
    }

    const privateRecord = await this.readRecordPath(this.recordPath(id, 'private'));
    if (privateRecord !== undefined) throw new Error(`memory record ${id} is private`);
    throw new Error(`memory record ${id} is missing`);
  }

  async lookup(query: string): Promise<string> {
    const keyword = query.trim().toLowerCase();
    if (keyword === '') return 'NO_MATCH';
    const matches = (await this.allRecords())
      .filter((record) => record.tier === 'private' && record.text.toLowerCase().includes(keyword))
      .sort((left, right) => left.id < right.id ? -1 : left.id > right.id ? 1 : 0);
    if (matches.length === 0) return 'NO_MATCH';
    return matches.map((record) => `## ${record.id}\n${record.text}`).join('\n');
  }

  async forget(id: string, tier: Tier, fact?: string): Promise<void> {
    this.validateId(id);
    return this.withLock(async () => {
      await this.ensureDirectory();
      const path = this.recordPath(id, tier);
      const existing = await this.readRecordPath(path);
      if (existing === undefined) throw new Error(`memory record ${id} is missing`);

      const factToRemove = fact === undefined ? '' : this.findFact(existing.facts, fact) ?? '';
      if (fact !== undefined && factToRemove === '') {
        throw new Error(`memory fact is missing from record ${id}`);
      }
      const removedFacts = factToRemove === '' ? existing.facts : [factToRemove];

      const before = await this.everydayRecords();
      const after = tier === 'everyday' && fact === undefined
        ? before.filter((record) => record.id !== id)
        : tier === 'everyday'
          ? [...before.filter((record) => record.id !== id), this.toRecord(
            id,
            tier,
            existing.text.split('\n').filter((line) => line !== `- ${factToRemove}`).join('\n'),
          )]
          : before;
      const currentSize = Buffer.byteLength(this.formatIndex(before));
      const nextSize = Buffer.byteLength(this.formatIndex(after));
      if (nextSize > this.capBytes && nextSize > currentSize) {
        throw new Error(`memory index would exceed ${String(this.capBytes)} byte cap`);
      }

      const tombstonesPath = join(this.dir, '_tombstones.md');
      const previousTombstones = await this.readTombstonesFile();
      const date = this.localDate();
      const entries = [
        ...(fact === undefined
          ? [`- [${date}] ${id} (${tier}): record forgotten — ${existing.header.name}: ${existing.header.summary}`]
          : []),
        ...removedFacts.map((removedFact) => `- [${date}] ${id}: ${removedFact}`),
      ];
      const nextTombstones = entries.length === 0
        ? previousTombstones
        : `${previousTombstones}${previousTombstones !== '' && !previousTombstones.endsWith('\n') ? '\n' : ''}${entries.join('\n')}\n`;
      await this.ensureRepository();
      try {
        if (fact === undefined) {
          await rm(path);
        } else {
          const lines = existing.text.split('\n');
          const factLineIndex = lines.findIndex((line) => line === `- ${factToRemove}`);
          if (factLineIndex < 0) throw new Error(`memory fact is missing from record ${id}`);
          lines.splice(factLineIndex, 1);
          await this.writeRecord(path, lines.join('\n'));
        }
        if (entries.length > 0) await this.writeRecord(tombstonesPath, nextTombstones);
        await this.runGit('add', '-A');
        await this.runGit('commit', '-m', `memory: forget ${id} (${tier})`);
      } catch (error) {
        await this.restoreRecord(path, existing.text);
        await this.restoreRecord(tombstonesPath, previousTombstones === '' ? undefined : previousTombstones);
        for (const stagedPath of [path, tombstonesPath]) {
          try {
            await this.runGit('reset', '--', stagedPath.slice(this.dir.length + 1));
          } catch (resetError) {
            this.logger.error('could not reset staged memory after failed forget', {
              id,
              error: this.errorMessage(resetError),
            });
          }
        }
        throw new Error(`memory commit failed for ${id}: ${this.errorMessage(error)}`, { cause: error });
      }

      this.reportIndexSize(nextSize);
    });
  }

  async tombstones(): Promise<string> {
    await this.ensureDirectory();
    return this.readTombstonesFile();
  }

  async save(input: SaveInput): Promise<{ revision: string }> {
    this.validateId(input.id);
    return this.withLock(async () => {
      await this.ensureDirectory();
      const path = this.recordPath(input.id, input.tier);
      const existing = await this.readRecordPath(path);
      if (existing === undefined && input.revision !== undefined) {
        throw new Error(`revision supplied for new memory record ${input.id}`);
      }
      if (existing !== undefined && input.revision !== existing.revision) {
        throw new Error(`revision mismatch for memory record ${input.id}`);
      }

      const before = await this.everydayRecords();
      this.checkIdentityClashes(input, await this.allRecords());
      const text = formatMemoryRecord(input.header, input.facts, input.source, this.now());
      const revision = this.revision(text);
      const after = input.tier === 'everyday'
        ? [...before.filter((record) => record.id !== input.id), this.toRecord(input.id, input.tier, text)]
        : before;
      const currentSize = Buffer.byteLength(this.formatIndex(before));
      const nextIndex = this.formatIndex(after);
      const nextSize = Buffer.byteLength(nextIndex);
      const overCapWithoutShrink = currentSize > this.capBytes && nextSize >= currentSize;
      const growthPastCap = currentSize <= this.capBytes && nextSize > this.capBytes && nextSize > currentSize;
      if (overCapWithoutShrink || growthPastCap) {
        throw new Error(`memory index would exceed ${String(this.capBytes)} byte cap`);
      }

      await this.ensureRepository();
      const previousText = existing?.text;
      await this.writeRecord(path, text);
      try {
        await this.runGit('add', '-A');
        await this.runGit('commit', '-m', `memory: save ${input.id} (${input.tier})`);
      } catch (error) {
        await this.restoreRecord(path, previousText);
        try {
          await this.runGit('reset', '--', path.slice(this.dir.length + 1));
        } catch (resetError) {
          this.logger.error('could not reset staged memory after failed commit', {
            id: input.id,
            error: this.errorMessage(resetError),
          });
        }
        throw new Error(`memory commit failed for ${input.id}: ${this.errorMessage(error)}`, { cause: error });
      }

      this.reportIndexSize(nextSize);
      return { revision };
    });
  }

  private async withLock<T>(operation: () => Promise<T>): Promise<T> {
    const key = this.dir;
    const previous = locks.get(key) ?? Promise.resolve();
    let release: () => void = () => undefined;
    const gate = new Promise<void>((resolveGate) => {
      release = resolveGate;
    });
    const queued = previous.then(() => gate);
    locks.set(key, queued);
    await previous;
    try {
      return await operation();
    } finally {
      release();
      if (locks.get(key) === queued) locks.delete(key);
    }
  }

  private validateId(id: string): void {
    if (typeof id !== 'string' || !ID_PATTERN.test(id)) {
      throw new Error('invalid memory id');
    }
  }

  private recordPath(id: string, tier: Tier): string {
    return join(this.dir, `${id}${tier === 'private' ? '.private' : ''}.md`);
  }

  private revision(text: string): string {
    return createHash('sha256').update(text).digest('hex');
  }

  private async ensureDirectory(): Promise<void> {
    await mkdir(this.dir, { recursive: true });
    await realpath(this.dir);
  }

  private async ensureRepository(): Promise<void> {
    let topLevel: string | undefined;
    try {
      const result = await this.runGit('rev-parse', '--show-toplevel');
      topLevel = resolve(result);
    } catch {
      topLevel = undefined;
    }
    if (topLevel !== this.dir) {
      await this.runGit('init', '-q', this.dir);
    }
    await this.runGit('config', 'user.name', 'Mentat Memory');
    await this.runGit('config', 'user.email', 'mentat-memory@localhost');
  }

  private async runGit(...args: string[]): Promise<string> {
    const { stdout } = await execFileAsync(this.git, ['-C', this.dir, ...args], {
      encoding: 'utf8',
      maxBuffer: 1024 * 1024,
    });
    return stdout.trim();
  }

  private async writeRecord(path: string, text: string): Promise<void> {
    await this.assertRecordPath(path);
    const temporary = join(this.dir, `.memory-${String(process.pid)}-${randomUUID()}.tmp`);
    try {
      await writeFile(temporary, text, { encoding: 'utf8', flag: 'wx' });
      await rename(temporary, path);
    } finally {
      await rm(temporary, { force: true });
    }
  }

  private async restoreRecord(path: string, text: string | undefined): Promise<void> {
    if (text === undefined) {
      await rm(path, { force: true });
      return;
    }
    await this.writeRecord(path, text);
  }

  private async readTombstonesFile(): Promise<string> {
    const path = join(this.dir, '_tombstones.md');
    try {
      await this.assertRecordPath(path);
      return await readFile(path, 'utf8');
    } catch (error) {
      if (this.isMissing(error)) return '';
      throw error;
    }
  }

  private findFact(facts: string[], fact: string): string | undefined {
    return facts.find((storedFact) => storedFact === fact
      || storedFact.replace(/^\[\d{4}-\d{2}-\d{2}, (?:josh|inferred|third-party)\] /, '') === fact);
  }

  private localDate(): string {
    const now = this.now();
    return [
      now.getFullYear(),
      String(now.getMonth() + 1).padStart(2, '0'),
      String(now.getDate()).padStart(2, '0'),
    ].join('-');
  }

  private async readRecordPath(path: string): Promise<StoredRecord | undefined> {
    try {
      await this.assertRecordPath(path);
      const text = await readFile(path, 'utf8');
      const parsed = parseMemoryRecord(text);
      const name = path.slice(this.dir.length + 1);
      const match = RECORD_PATTERN.exec(name);
      if (!match) throw new Error('invalid memory record path');
      return {
        id: match[1] ?? '',
        tier: match[2] === '.private' ? 'private' : 'everyday',
        text,
        revision: this.revision(text),
        header: parsed.header,
        facts: parsed.facts,
      };
    } catch (error) {
      if (this.isMissing(error)) return undefined;
      throw error;
    }
  }

  private async assertRecordPath(path: string): Promise<void> {
    let stat;
    try {
      stat = await lstat(path);
    } catch (error) {
      if (this.isMissing(error)) return;
      throw error;
    }
    if (stat.isSymbolicLink()) throw new Error(`symbolic link memory path refused: ${path}`);
    if (!stat.isFile()) throw new Error(`memory record path is not a file: ${path}`);
    const root = await realpath(this.dir);
    const target = await realpath(path);
    if (dirname(target) !== root) throw new Error(`memory record escapes its directory: ${path}`);
  }

  private async allRecords(): Promise<StoredRecord[]> {
    let names: string[];
    try {
      names = await readdir(this.dir);
    } catch (error) {
      if (this.isMissing(error)) return [];
      throw error;
    }
    const records: StoredRecord[] = [];
    for (const name of names) {
      if (!RECORD_PATTERN.test(name)) continue;
      const record = await this.readRecordPath(join(this.dir, name));
      if (record !== undefined) records.push(record);
    }
    return records;
  }

  private async everydayRecords(): Promise<StoredRecord[]> {
    return (await this.allRecords()).filter((record) => record.tier === 'everyday');
  }

  private formatIndex(records: StoredRecord[]): string {
    return formatMemoryIndex(records.map(({ id, header }) => ({ id, header })));
  }

  private toRecord(id: string, tier: Tier, text: string): StoredRecord {
    const parsed = parseMemoryRecord(text);
    return {
      id,
      tier,
      text,
      revision: this.revision(text),
      header: parsed.header,
      facts: parsed.facts,
    };
  }

  private checkIdentityClashes(input: SaveInput, records: StoredRecord[]): void {
    const proposed = [input.header.name, ...input.header.aliases].map((value) => value.toLowerCase());
    for (const record of records) {
      if (record.id === input.id) continue;
      const current = [record.header.name, ...record.header.aliases].map((value) => value.toLowerCase());
      if (proposed.some((value) => current.includes(value))) {
        throw new Error(`memory identity clashes with ${record.id}`);
      }
    }
  }

  private reportIndexSize(size: number): void {
    if (size > this.capBytes) {
      this.logger.error('memory index exceeds size cap', { bytes: size, capBytes: this.capBytes });
    } else if (size >= this.capBytes * 0.8) {
      this.logger.warn('memory index approaching size cap', { bytes: size, capBytes: this.capBytes });
    }
  }

  private isMissing(error: unknown): boolean {
    return typeof error === 'object' && error !== null && 'code' in error && error.code === 'ENOENT';
  }

  private errorMessage(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
  }
}

