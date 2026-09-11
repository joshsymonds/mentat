import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import { loadConfig, parseDuration, validateListen } from '../src/config.ts';

const baseEnv = { MENTAT_CLAUDE_BIN: '/pinned/claude' };

describe('loadConfig', () => {
  it('requires the claude binary path (no PATH fallback)', () => {
    expect(() => loadConfig({})).toThrow(/MENTAT_CLAUDE_BIN/);
  });

  it('applies defaults', () => {
    const config = loadConfig(baseEnv);
    expect(config.listen).toEqual({ host: '127.0.0.1', port: 8484 });
    expect(config.maxSessions).toBe(16);
    expect(config.sessionTtlMs).toBe(15 * 60 * 1000);
    expect(config.allowedTools).toBeUndefined();
    expect(config.mcpServers).toBeUndefined();
  });

  it('loads the optional Places API key', () => {
    expect(loadConfig({ ...baseEnv, MENTAT_PLACES_API_KEY: 'places-key' }).placesApiKey).toBe('places-key');
    expect(loadConfig({ ...baseEnv, MENTAT_PLACES_API_KEY: '' }).placesApiKey).toBeUndefined();
  });

  it('loads LiveKit voice-token configuration when all variables are set', () => {
    const config = loadConfig({
      ...baseEnv,
      LIVEKIT_API_KEY: 'voice-key',
      LIVEKIT_API_SECRET: 'voice-secret',
      MENTAT_VOICE_PUBLIC_LIVEKIT_URL: 'wss://voice.example.test',
    });

    expect(config.voiceToken).toEqual({
      apiKey: 'voice-key',
      apiSecret: 'voice-secret',
      url: 'wss://voice.example.test',
    });
  });

  it('accepts an unencrypted WebSocket URL for loopback LiveKit deployments', () => {
    for (const url of ['ws://127.0.0.1:7880', 'ws://localhost:7880', 'ws://[::1]:7880']) {
      const config = loadConfig({
        ...baseEnv,
        LIVEKIT_API_KEY: 'voice-key',
        LIVEKIT_API_SECRET: 'voice-secret',
        MENTAT_VOICE_PUBLIC_LIVEKIT_URL: url,
      });

      expect(config.voiceToken?.url).toBe(url);
    }
  });

  it('rejects an unencrypted WebSocket URL to a non-loopback host', () => {
    const configuredEnv = {
      ...baseEnv,
      LIVEKIT_API_KEY: 'voice-key',
      LIVEKIT_API_SECRET: 'voice-secret',
    };

    for (const url of ['ws://remote.example', 'ws://192.168.1.10:7880']) {
      expect(() =>
        loadConfig({ ...configuredEnv, MENTAT_VOICE_PUBLIC_LIVEKIT_URL: url }),
      ).toThrow(/MENTAT_VOICE_PUBLIC_LIVEKIT_URL.*loopback.*wss:/);
    }

    expect(
      loadConfig({
        ...configuredEnv,
        MENTAT_VOICE_PUBLIC_LIVEKIT_URL: 'wss://remote.example',
      }).voiceToken?.url,
    ).toBe('wss://remote.example');
  });

  it('rejects malformed and non-WebSocket public LiveKit URLs', () => {
    const configuredEnv = {
      ...baseEnv,
      LIVEKIT_API_KEY: 'voice-key',
      LIVEKIT_API_SECRET: 'voice-secret',
    };

    expect(() =>
      loadConfig({
        ...configuredEnv,
        MENTAT_VOICE_PUBLIC_LIVEKIT_URL: 'http://voice.example.test',
      }),
    ).toThrow(/MENTAT_VOICE_PUBLIC_LIVEKIT_URL.*ws:.*wss:/);
    expect(() =>
      loadConfig({
        ...configuredEnv,
        MENTAT_VOICE_PUBLIC_LIVEKIT_URL: 'not a URL',
      }),
    ).toThrow(/MENTAT_VOICE_PUBLIC_LIVEKIT_URL.*ws:.*wss:/);
  });

  it('omits voice-token configuration when all variables are unset or empty', () => {
    expect(loadConfig(baseEnv).voiceToken).toBeUndefined();
    expect(
      loadConfig({
        ...baseEnv,
        LIVEKIT_API_KEY: '',
        LIVEKIT_API_SECRET: '',
        MENTAT_VOICE_PUBLIC_LIVEKIT_URL: '',
      }).voiceToken,
    ).toBeUndefined();
  });

  it('rejects every partial voice-token configuration and names missing variables', () => {
    const cases: { env: Record<string, string>; missing: string[] }[] = [
      {
        env: { LIVEKIT_API_KEY: 'voice-key' },
        missing: ['LIVEKIT_API_SECRET', 'MENTAT_VOICE_PUBLIC_LIVEKIT_URL'],
      },
      {
        env: { LIVEKIT_API_SECRET: 'voice-secret' },
        missing: ['LIVEKIT_API_KEY', 'MENTAT_VOICE_PUBLIC_LIVEKIT_URL'],
      },
      {
        env: { MENTAT_VOICE_PUBLIC_LIVEKIT_URL: 'wss://voice.example.test' },
        missing: ['LIVEKIT_API_KEY', 'LIVEKIT_API_SECRET'],
      },
      {
        env: { LIVEKIT_API_KEY: 'voice-key', LIVEKIT_API_SECRET: 'voice-secret' },
        missing: ['MENTAT_VOICE_PUBLIC_LIVEKIT_URL'],
      },
      {
        env: {
          LIVEKIT_API_KEY: 'voice-key',
          MENTAT_VOICE_PUBLIC_LIVEKIT_URL: 'wss://voice.example.test',
        },
        missing: ['LIVEKIT_API_SECRET'],
      },
      {
        env: {
          LIVEKIT_API_SECRET: 'voice-secret',
          MENTAT_VOICE_PUBLIC_LIVEKIT_URL: 'wss://voice.example.test',
        },
        missing: ['LIVEKIT_API_KEY'],
      },
    ];

    for (const testCase of cases) {
      let message = '';
      try {
        loadConfig({ ...baseEnv, ...testCase.env });
      } catch (error) {
        message = String(error);
      }
      expect(message).not.toBe('');
      for (const variable of testCase.missing) {
        expect(message).toContain(variable);
      }
    }
  });

  it('parses CSV tool lists and extra env', () => {
    const config = loadConfig({
      ...baseEnv,
      MENTAT_ALLOWED_TOOLS: 'Read, Grep ,Glob',
      MENTAT_DISALLOWED_TOOLS: 'Bash',
      MENTAT_EXTRA_ENV: 'FOO,BAR',
    });
    expect(config.allowedTools).toEqual(['Read', 'Grep', 'Glob']);
    expect(config.disallowedTools).toEqual(['Bash']);
    expect(config.extraEnv).toEqual(['FOO', 'BAR']);
  });

  it('accepts inline JSON MCP config', () => {
    const config = loadConfig({
      ...baseEnv,
      MENTAT_MCP_CONFIG: '{"mcpServers":{"shimmer":{"type":"http","url":"https://x/mcp"}}}',
    });
    expect(config.mcpServers).toEqual({ shimmer: { type: 'http', url: 'https://x/mcp' } });
  });

  it('accepts a path to an MCP config file', () => {
    const dir = mkdtempSync(join(tmpdir(), 'mentat-test-'));
    const path = join(dir, 'mcp.json');
    writeFileSync(path, '{"mcpServers":{"memory":{"type":"stdio","command":"m"}}}');
    const config = loadConfig({ ...baseEnv, MENTAT_MCP_CONFIG: path });
    expect(config.mcpServers).toEqual({ memory: { type: 'stdio', command: 'm' } });
  });

  it('rejects malformed MCP config', () => {
    expect(() => loadConfig({ ...baseEnv, MENTAT_MCP_CONFIG: '{"oops"' })).toThrow(/MCP/i);
    expect(() => loadConfig({ ...baseEnv, MENTAT_MCP_CONFIG: '{"mcpServers":null}' })).toThrow(
      /MCP/i,
    );
  });

  it('validates MENTAT_EFFORT instead of passing it through', () => {
    expect(loadConfig({ ...baseEnv, MENTAT_EFFORT: 'low' }).effort).toBe('low');
    expect(() => loadConfig({ ...baseEnv, MENTAT_EFFORT: 'hgih' })).toThrow(/MENTAT_EFFORT/);
  });

  it('refuses a non-loopback listen address without the override', () => {
    expect(() => loadConfig({ ...baseEnv, MENTAT_LISTEN: '0.0.0.0:8484' })).toThrow(
      /non-loopback/,
    );
    const allowed = loadConfig({
      ...baseEnv,
      MENTAT_LISTEN: '0.0.0.0:8484',
      MENTAT_ALLOW_NON_LOOPBACK: '1',
    });
    expect(allowed.listen).toEqual({ host: '0.0.0.0', port: 8484 });
  });

  it('treats an empty override as unset (a blank env-file line must not disable the guard)', () => {
    expect(() =>
      loadConfig({ ...baseEnv, MENTAT_LISTEN: '0.0.0.0:8484', MENTAT_ALLOW_NON_LOOPBACK: '' }),
    ).toThrow(/non-loopback/);
  });
});

describe('validateListen', () => {
  const check = (host: string, allow: boolean) => () => {
    validateListen(host, allow);
  };

  it('accepts loopback forms', () => {
    expect(check('127.0.0.1', false)).not.toThrow();
    expect(check('localhost', false)).not.toThrow();
    expect(check('::1', false)).not.toThrow();
  });

  it('refuses everything else without the override', () => {
    expect(check('0.0.0.0', false)).toThrow(/non-loopback/);
    expect(check('192.168.1.10', false)).toThrow(/non-loopback/);
    expect(check('example.com', false)).toThrow();
    expect(check('0.0.0.0', true)).not.toThrow();
  });
});

describe('parseDuration', () => {
  it('parses Go-style duration strings', () => {
    expect(parseDuration('15m')).toBe(900_000);
    expect(parseDuration('30s')).toBe(30_000);
    expect(parseDuration('500ms')).toBe(500);
    expect(parseDuration('2h')).toBe(7_200_000);
    expect(parseDuration('1h30m')).toBe(5_400_000);
  });

  it('rejects garbage', () => {
    expect(() => parseDuration('soon')).toThrow();
    expect(() => parseDuration('15')).toThrow();
    expect(() => parseDuration('')).toThrow();
  });
});
