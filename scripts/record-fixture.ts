// Records one real SDK turn as a test fixture: every SDK message, one JSON
// line each, appended to test/fixtures/<name>.jsonl. Fixtures must be
// recorded, never hand-authored — hand-authored streams test our imagination.
//
//   MENTAT_CLAUDE_BIN=$(command -v claude) node scripts/record-fixture.ts turn-with-tool
//
// Needs a claude binary and spends real tokens (~a few cents at haiku).

import { appendFileSync, mkdirSync } from 'node:fs';
import process from 'node:process';

import {
  createSdkMcpServer,
  query,
  tool,
  type SDKUserMessage,
} from '@anthropic-ai/claude-agent-sdk';
import { z } from 'zod';

const name = process.argv[2];
if (name === undefined || name === '') {
  console.error('usage: MENTAT_CLAUDE_BIN=... node scripts/record-fixture.ts <name>');
  process.exit(2);
}
const bin = process.env.MENTAT_CLAUDE_BIN;
if (bin === undefined || bin === '') {
  console.error('MENTAT_CLAUDE_BIN is required (no PATH fallback)');
  process.exit(2);
}

// Same allowlist shape the daemon will use: never the full daemon env.
const childEnv: Record<string, string> = {};
for (const key of ['HOME', 'PATH', 'USER', 'LOGNAME', 'SHELL', 'TERM', 'LANG', 'TMPDIR', 'TZ']) {
  const value = process.env[key];
  if (value !== undefined) childEnv[key] = value;
}

const memoryServer = createSdkMcpServer({
  name: 'memory',
  version: '0.0.1',
  tools: [
    tool(
      'memory_save',
      'Persist a fact under a key.',
      { key: z.string(), value: z.string() },
      ({ key, value }) => {
        console.error(`[memory_save] ${key}=${value}`);
        return Promise.resolve({ content: [{ type: 'text' as const, text: `saved ${key}` }] });
      },
    ),
  ],
});

// eslint-disable-next-line @typescript-eslint/require-await -- one fixed turn; the SDK requires an AsyncIterable prompt
async function* turns(): AsyncGenerator<SDKUserMessage> {
  yield {
    type: 'user',
    message: {
      role: 'user',
      content: [
        {
          type: 'text',
          text:
            'Remember this number: 7341. Save it by calling memory_save with key ' +
            "'fixture' and value '7341', then reply with one short sentence.",
        },
      ],
    },
    parent_tool_use_id: null,
    session_id: '',
  };
}

const stream = query({
  prompt: turns(),
  options: {
    model: 'claude-haiku-4-5',
    pathToClaudeCodeExecutable: bin,
    env: childEnv,
    settingSources: [],
    skills: [],
    strictMcpConfig: true,
    mcpServers: { memory: memoryServer },
    disallowedTools: ['Bash', 'Write', 'Edit', 'NotebookEdit', 'WebFetch', 'WebSearch', 'Task'],
    includePartialMessages: true,
    maxBudgetUsd: 0.5,
    canUseTool: (toolName, input) => {
      console.error(`[canUseTool] ${toolName}`);
      return Promise.resolve({ behavior: 'allow' as const, updatedInput: input });
    },
  },
});

mkdirSync('test/fixtures', { recursive: true });
const path = `test/fixtures/${name}.jsonl`;
let lines = 0;
for await (const message of stream) {
  appendFileSync(path, JSON.stringify(message) + '\n');
  lines += 1;
}
console.error(`recorded ${String(lines)} messages to ${path}`);
