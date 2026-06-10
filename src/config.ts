// Daemon configuration, env-first: every knob is a MENTAT_* variable (the
// deploy is a systemd unit with an environment file, not a flag parser).

import { readFileSync } from 'node:fs';
import { isIP } from 'node:net';

import type { Options } from '@anthropic-ai/claude-agent-sdk';

export interface ListenAddress {
  host: string;
  port: number;
}

export interface Config {
  listen: ListenAddress;
  bin: string;
  model?: string;
  effort?: Options['effort'];
  systemPrompt?: string;
  memoryDir?: string;
  mcpServers?: Options['mcpServers'];
  recordDir?: string;
  statePath?: string;
  allowedTools?: string[];
  disallowedTools?: string[];
  extraEnv?: string[];
  maxSessions: number;
  sessionTtlMs: number;
  maxBudgetUsd?: number;
}

const DEFAULT_LISTEN = '127.0.0.1:8484';
const DEFAULT_MAX_SESSIONS = 16;
const DEFAULT_SESSION_TTL_MS = 15 * 60 * 1000;

export function loadConfig(env: Record<string, string | undefined>): Config {
  const bin = env.MENTAT_CLAUDE_BIN;
  if (bin === undefined || bin === '') {
    throw new Error('MENTAT_CLAUDE_BIN is required (no PATH fallback; the deploy pins the binary)');
  }

  const listen = parseListen(env.MENTAT_LISTEN ?? DEFAULT_LISTEN);
  validateListen(listen.host, env.MENTAT_ALLOW_NON_LOOPBACK !== undefined);

  return {
    listen,
    bin,
    maxSessions: intOr(env.MENTAT_MAX_SESSIONS, DEFAULT_MAX_SESSIONS),
    sessionTtlMs:
      env.MENTAT_SESSION_TTL !== undefined
        ? parseDuration(env.MENTAT_SESSION_TTL)
        : DEFAULT_SESSION_TTL_MS,
    ...(env.MENTAT_MODEL !== undefined && { model: env.MENTAT_MODEL }),
    ...(env.MENTAT_EFFORT !== undefined && { effort: env.MENTAT_EFFORT as Options['effort'] }),
    ...(env.MENTAT_SYSTEM_PROMPT !== undefined && { systemPrompt: env.MENTAT_SYSTEM_PROMPT }),
    ...(env.MENTAT_MEMORY_DIR !== undefined && { memoryDir: env.MENTAT_MEMORY_DIR }),
    ...(env.MENTAT_MCP_CONFIG !== undefined && {
      mcpServers: resolveMcpServers(env.MENTAT_MCP_CONFIG),
    }),
    ...(env.MENTAT_RECORD_DIR !== undefined && { recordDir: env.MENTAT_RECORD_DIR }),
    ...(env.MENTAT_STATE_PATH !== undefined && { statePath: env.MENTAT_STATE_PATH }),
    ...(env.MENTAT_ALLOWED_TOOLS !== undefined && {
      allowedTools: splitCsv(env.MENTAT_ALLOWED_TOOLS),
    }),
    ...(env.MENTAT_DISALLOWED_TOOLS !== undefined && {
      disallowedTools: splitCsv(env.MENTAT_DISALLOWED_TOOLS),
    }),
    ...(env.MENTAT_EXTRA_ENV !== undefined && { extraEnv: splitCsv(env.MENTAT_EXTRA_ENV) }),
    ...(env.MENTAT_MAX_BUDGET_USD !== undefined && {
      maxBudgetUsd: floatOrThrow(env.MENTAT_MAX_BUDGET_USD, 'MENTAT_MAX_BUDGET_USD'),
    }),
  };
}

function parseListen(addr: string): ListenAddress {
  const colon = addr.lastIndexOf(':');
  if (colon === -1) {
    throw new Error(`invalid listen address ${addr}: want host:port`);
  }
  const host = addr.slice(0, colon).replace(/^\[|\]$/g, '');
  const port = Number(addr.slice(colon + 1));
  if (host === '' || !Number.isInteger(port) || port < 1 || port > 65535) {
    throw new Error(`invalid listen address ${addr}`);
  }
  return { host, port };
}

/**
 * Refuses a non-loopback bind unless explicitly allowed: the conversation API
 * has no authentication of its own and relies on the deploy's tailnet
 * ingress, so binding the open internet is a footgun.
 */
export function validateListen(host: string, allowNonLoopback: boolean): void {
  if (allowNonLoopback || host === 'localhost') {
    return;
  }
  if (isIP(host) === 0) {
    throw new Error(`listen host ${host} is neither an IP nor localhost`);
  }
  const isLoopback = host === '::1' || host.startsWith('127.');
  if (!isLoopback) {
    throw new Error(
      `refusing to bind non-loopback ${host}; set MENTAT_ALLOW_NON_LOOPBACK to override`,
    );
  }
}

/** Go-style duration strings: "15m", "30s", "500ms", "1h30m". */
export function parseDuration(value: string): number {
  const pattern = /(\d+)(ms|s|m|h)/g;
  let total = 0;
  let matchedLength = 0;
  const unitMs = { ms: 1, s: 1000, m: 60_000, h: 3_600_000 };
  for (const match of value.matchAll(pattern)) {
    total += Number(match[1]) * unitMs[match[2] as keyof typeof unitMs];
    matchedLength += match[0].length;
  }
  if (value === '' || matchedLength !== value.length) {
    throw new Error(`invalid duration ${value}: want forms like 15m, 30s, 1h30m`);
  }
  return total;
}

/**
 * MENTAT_MCP_CONFIG is inline JSON (starts with "{") or a path to a JSON
 * file; either way the shape is {"mcpServers": {...}} (the claude CLI's
 * --mcp-config format).
 */
function resolveMcpServers(value: string): Options['mcpServers'] {
  const raw = value.trimStart().startsWith('{') ? value : readFileSync(value, 'utf8');
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch (error) {
    throw new Error(`invalid MCP config: ${String(error)}`);
  }
  if (
    parsed === null ||
    typeof parsed !== 'object' ||
    !('mcpServers' in parsed) ||
    typeof (parsed).mcpServers !== 'object'
  ) {
    throw new Error('invalid MCP config: want {"mcpServers": {...}}');
  }
  return (parsed as { mcpServers: Options['mcpServers'] }).mcpServers;
}

function splitCsv(value: string): string[] {
  return value
    .split(',')
    .map((part) => part.trim())
    .filter((part) => part !== '');
}

function intOr(value: string | undefined, fallback: number): number {
  if (value === undefined || value === '') {
    return fallback;
  }
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 0) {
    throw new Error(`invalid integer ${value}`);
  }
  return parsed;
}

function floatOrThrow(value: string, name: string): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < 0) {
    throw new Error(`invalid ${name}: ${value}`);
  }
  return parsed;
}
