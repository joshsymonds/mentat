// mentatd: a Backend behind the streaming conversation API, run as a
// systemd-style foreground process. It binds localhost; tailnet exposure
// happens at deploy via tailscale serve.

import { createServer } from 'node:http';
import process from 'node:process';

import { ClaudeCode } from './claudecode.ts';
import { loadConfig } from './config.ts';
import { startJanitor } from './janitor.ts';
import { jsonLogger } from './log.ts';
import { allowAllPolicy } from './policy.ts';
import { SessionTracker, createHandler } from './server.ts';

const logger = jsonLogger();

let exitCode = 0;
try {
  const config = loadConfig(process.env);
  const backend = new ClaudeCode({
    bin: config.bin,
    policy: allowAllPolicy(logger),
    logger,
    maxSessions: config.maxSessions,
    ...(config.model !== undefined && { model: config.model }),
    ...(config.effort !== undefined && { effort: config.effort }),
    ...(config.systemPrompt !== undefined && { systemPrompt: config.systemPrompt }),
    ...(config.memoryDir !== undefined && { addDirs: [config.memoryDir] }),
    ...(config.mcpServers !== undefined && { mcpServers: config.mcpServers }),
    ...(config.allowedTools !== undefined && { allowedTools: config.allowedTools }),
    ...(config.disallowedTools !== undefined && { disallowedTools: config.disallowedTools }),
    ...(config.extraEnv !== undefined && { extraEnv: config.extraEnv }),
    ...(config.statePath !== undefined && { statePath: config.statePath }),
    ...(config.maxBudgetUsd !== undefined && { maxBudgetUsd: config.maxBudgetUsd }),
  });

  const tracker = new SessionTracker();
  const server = createServer(createHandler(backend, tracker, logger));
  const stopJanitor = startJanitor(tracker, backend, config.sessionTtlMs, logger);

  server.listen(config.listen.port, config.listen.host, () => {
    logger.info('mentatd listening', {
      addr: `${config.listen.host}:${String(config.listen.port)}`,
    });
  });

  const shutdown = (): void => {
    logger.info('shutting down');
    stopJanitor();
    server.close(() => {
      void backend.close().finally(() => {
        process.exit(exitCode);
      });
    });
    // A drain that outlives the grace period gets cut off.
    setTimeout(() => {
      process.exit(exitCode);
    }, 15_000).unref();
  };
  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);
} catch (error) {
  logger.error('mentatd failed', { error: String(error) });
  exitCode = 1;
  process.exit(exitCode);
}
