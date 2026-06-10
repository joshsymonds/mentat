// Minimal structured JSON-lines logging (parity with the Go daemon's slog
// JSON handler) — the daemon's only log surface, no logging dependency.

import process from 'node:process';

export interface Logger {
  info(message: string, fields?: Record<string, unknown>): void;
  warn(message: string, fields?: Record<string, unknown>): void;
  error(message: string, fields?: Record<string, unknown>): void;
}

/** Logger writing one JSON object per line to the given stream. */
export function jsonLogger(out: NodeJS.WritableStream = process.stdout): Logger {
  const write = (level: string, message: string, fields?: Record<string, unknown>): void => {
    out.write(
      JSON.stringify({ time: new Date().toISOString(), level, msg: message, ...fields }) + '\n',
    );
  };
  return {
    info: (message, fields) => {
      write('info', message, fields);
    },
    warn: (message, fields) => {
      write('warn', message, fields);
    },
    error: (message, fields) => {
      write('error', message, fields);
    },
  };
}

/** Discards everything; the test default. */
export const nullLogger: Logger = {
  info: () => undefined,
  warn: () => undefined,
  error: () => undefined,
};
