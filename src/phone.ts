import { randomUUID } from 'node:crypto';
import type { ServerResponse } from 'node:http';

import type { Logger } from './log.ts';

export type PhoneCommand =
  | { kind: 'sms'; to: string; body: string }
  | { kind: 'open'; uri: string }
  | { kind: 'conversations'; limit: number }
  | { kind: 'messages'; conversation: string; limit: number; before?: string }
  | { kind: 'search'; query: string; limit: number; before?: string }
  | { kind: 'navigate'; name: string; address: string; place_id: string; lat: number; lng: number }
  | { kind: 'dial'; number: string }
  | { kind: 'alarm'; hour: number; minute: number; label?: string }
  | { kind: 'timer'; seconds: number; label?: string }
  | { kind: 'location' };

export interface PhoneLocationPayload {
  lat: number;
  lng: number;
  accuracy_m: number;
  age_s: number;
}

export interface PhoneResult {
  id: string;
  status: 'ok' | 'error';
  detail: string;
  payload?: Record<string, unknown>;
}

export interface PhoneOutcome {
  detail: string;
  payload?: Record<string, unknown>;
}

export interface PhoneBridgeDeps {
  now?: () => Date;
  uuid?: () => string;
  timeoutMs?: number;
  heartbeatMs?: number;
}

interface PendingCommand {
  resolve: (outcome: PhoneOutcome) => void;
  reject: (error: Error) => void;
  timer: ReturnType<typeof setTimeout>;
}

export class PhoneBridge {
  private readonly now: () => Date;
  private readonly uuid: () => string;
  private readonly timeoutMs: number;
  private readonly heartbeatMs: number;
  private attached: ServerResponse | undefined;
  private heartbeatTimer: ReturnType<typeof setInterval> | undefined;
  private readonly pending = new Map<string, PendingCommand>();
  private readonly logger: Logger;

  constructor(logger: Logger, deps: PhoneBridgeDeps = {}) {
    this.logger = logger;
    this.now = deps.now ?? (() => new Date());
    this.uuid = deps.uuid ?? randomUUID;
    this.timeoutMs = deps.timeoutMs ?? 15_000;
    this.heartbeatMs = deps.heartbeatMs ?? 20_000;
  }

  attach(res: ServerResponse): void {
    if (this.attached !== undefined) {
      this.detach(this.attached);
    }

    this.attached = res;
    res.once('close', () => {
      this.detach(res);
    });
    this.logger.info('phone attached');
    try {
      res.writeHead(200, { 'content-type': 'application/x-ndjson' });
      res.flushHeaders();
    } catch {
      this.detach(res);
      return;
    }
    this.heartbeatTimer = setInterval(() => {
      if (this.attached !== res || res.destroyed) {
        this.detach(res);
        return;
      }
      try {
        res.write('{"kind":"ping"}\n');
      } catch {
        this.detach(res);
      }
    }, this.heartbeatMs);
  }

  dispatch(command: PhoneCommand): Promise<PhoneOutcome> {
    const res = this.attached;
    if (res === undefined || res.destroyed) {
      if (res?.destroyed) {
        this.detach(res);
      }
      return Promise.reject(new Error('phone offline'));
    }

    const id = this.uuid();
    const expiresAt = new Date(this.now().getTime() + this.timeoutMs).toISOString();
    const wireCommand = { id, ...command, expires_at: expiresAt };
    return new Promise<PhoneOutcome>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error('timed out: outcome unknown'));
      }, this.timeoutMs);
      this.pending.set(id, { resolve, reject, timer });
      try {
        res.write(JSON.stringify(wireCommand) + '\n');
      } catch {
        clearTimeout(timer);
        this.pending.delete(id);
        reject(new Error('phone offline'));
        this.detach(res);
      }
    });
  }

  complete(result: PhoneResult): boolean {
    const command = this.pending.get(result.id);
    if (command === undefined) {
      return false;
    }
    this.pending.delete(result.id);
    clearTimeout(command.timer);
    if (result.status === 'ok') {
      command.resolve({
        detail: result.detail,
        ...(result.payload !== undefined && { payload: result.payload }),
      });
    } else {
      command.reject(new Error(result.detail));
    }
    return true;
  }

  close(): void {
    if (this.attached !== undefined) {
      this.detach(this.attached);
    }
    for (const [id, command] of this.pending) {
      this.pending.delete(id);
      clearTimeout(command.timer);
      command.reject(new Error('phone offline'));
    }
  }

  private detach(res: ServerResponse): void {
    if (this.attached !== res) {
      return;
    }
    this.attached = undefined;
    if (this.heartbeatTimer !== undefined) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = undefined;
    }
    this.logger.info('phone detached');
    if (!res.destroyed) {
      res.end();
    }
  }
}
