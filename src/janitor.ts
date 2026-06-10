// Periodically releases idle sessions' child processes. Conversation identity
// survives expiry: the next turn with the same sessionId resumes.

import type { Backend } from './backend.ts';
import type { Logger } from './log.ts';
import type { SessionTracker } from './server.ts';

const MIN_TICK_MS = 30_000;

export function startJanitor(
  tracker: SessionTracker,
  backend: Backend,
  ttlMs: number,
  logger: Logger,
): () => void {
  const interval = Math.max(ttlMs / 4, MIN_TICK_MS);
  const timer = setInterval(() => {
    const expired = tracker.expireIdle(ttlMs);
    for (const sessionId of expired) {
      backend.closeSession(sessionId).catch((error: unknown) => {
        logger.error('closing idle session failed', {
          session_id: sessionId,
          error: String(error),
        });
      });
    }
    if (expired.length > 0) {
      logger.info('expired idle sessions', { count: expired.length });
    }
  }, interval);
  timer.unref();
  return () => {
    clearInterval(timer);
  };
}
