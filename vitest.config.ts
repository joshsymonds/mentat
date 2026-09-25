import { configDefaults, defineConfig } from 'vitest/config';

// nested checkouts (.worktrees/, .claude/worktrees/) carry their own copies of test/
export default defineConfig({
  test: { exclude: [...configDefaults.exclude, '.worktrees/**', '.claude/**'] },
});
