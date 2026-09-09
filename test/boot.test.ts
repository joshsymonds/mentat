// The daemon runs its sources directly under Node's type stripping, which
// vitest never exercises: it transpiles. Booting main.ts with no config
// proves every module strips and loads; the only acceptable failure is the
// config error that follows loading.
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

describe('node src/main.ts', () => {
  it('loads every module under strip-only TypeScript and fails only on config', () => {
    const main = fileURLToPath(new URL('../src/main.ts', import.meta.url));
    const result = spawnSync(process.execPath, [main], {
      env: { PATH: process.env.PATH ?? '' },
      encoding: 'utf8',
      timeout: 30_000,
    });
    const output = result.stdout + result.stderr;
    expect(output).not.toContain('SyntaxError');
    expect(output).toContain('MENTAT_CLAUDE_BIN is required');
    expect(result.status).toBe(1);
  });
});
