import { mkdtempSync, mkdirSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, describe, expect, it } from 'vitest';

import { shouldDenyFileTool } from '../src/file-guard.ts';

const temporaryDirectories: string[] = [];

function temporaryDirectory(): string {
  const directory = mkdtempSync(join(tmpdir(), 'mentat-file-guard-'));
  temporaryDirectories.push(directory);
  return directory;
}

afterEach(() => {
  for (const directory of temporaryDirectories.splice(0)) rmSync(directory, { recursive: true, force: true });
});

describe('shouldDenyFileTool', () => {
  it('denies paths overlapping protected roots except this session’s tool results', () => {
    const root = temporaryDirectory();
    const home = join(root, 'child-home');
    const memoryDir = join(root, 'memory');
    const recordDir = join(root, 'recordings');
    const transcriptDir = join(root, 'session-transcripts');
    const transcriptPath = join(transcriptDir, 'session.jsonl');
    const toolResults = join(transcriptDir, 'session-id', 'tool-results');
    const alias = join(root, 'memory-alias');
    const safeDirectory = join(root, 'safe');
    mkdirSync(join(memoryDir, 'nested'), { recursive: true });
    mkdirSync(safeDirectory, { recursive: true });
    mkdirSync(memoryDir, { recursive: true });
    mkdirSync(recordDir, { recursive: true });
    mkdirSync(toolResults, { recursive: true });
    writeFileSync(join(memoryDir, 'private.txt'), 'synthetic');
    symlinkSync(memoryDir, alias);
    symlinkSync(join(memoryDir, 'nested'), join(safeDirectory, 'link'));

    const context = {
      memoryDir,
      recordDir,
      home,
      cwd: root,
      transcriptPath,
      sessionId: 'session-id',
    };
    const cases: {
      name: string;
      toolName: string;
      input: Record<string, unknown>;
      denied: boolean;
    }[] = [
      { name: 'allows unrelated read', toolName: 'Read', input: { file_path: join(root, 'public.txt') }, denied: false },
      { name: 'denies memory root', toolName: 'Read', input: { file_path: memoryDir }, denied: true },
      { name: 'denies descendants of memory root', toolName: 'Read', input: { file_path: join(memoryDir, 'private.txt') }, denied: true },
      { name: 'denies ancestors of memory root', toolName: 'Glob', input: { path: root }, denied: true },
      { name: 'denies a symlink alias to memory', toolName: 'Read', input: { file_path: join(alias, 'private.txt') }, denied: true },
      { name: 'denies traversal through a symlink before parent resolution', toolName: 'Read', input: { file_path: `${safeDirectory}/link/../private.txt` }, denied: true },
      { name: 'denies recording root', toolName: 'Grep', input: { path: recordDir }, denied: true },
      { name: 'denies child Claude state', toolName: 'Read', input: { file_path: join(home, '.claude', 'projects', 'session.jsonl') }, denied: true },
      { name: 'denies the child transcript root wherever it lives', toolName: 'Read', input: { file_path: transcriptPath }, denied: true },
      { name: 'uses cwd when Glob path is absent', toolName: 'Glob', input: {}, denied: true },
      { name: 'uses cwd when Grep path is absent', toolName: 'Grep', input: {}, denied: true },
      { name: 'allows this session’s Read tool result', toolName: 'Read', input: { file_path: join(toolResults, 'result.txt') }, denied: false },
      { name: 'denies Glob in this session’s tool results', toolName: 'Glob', input: { path: toolResults }, denied: true },
      { name: 'allows other tools without examining paths', toolName: 'OtherTool', input: { file_path: memoryDir }, denied: false },
    ];

    for (const testCase of cases) {
      expect(
        shouldDenyFileTool(testCase.toolName, testCase.input, context),
        testCase.name,
      ).toBe(testCase.denied);
    }  });
});
