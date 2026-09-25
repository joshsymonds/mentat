import { createHmac, randomUUID } from 'node:crypto';

interface TokenGrant {
  token: string;
  room: string;
  url: string;
  expires_at: string;
}

/** What the phone knows at call start, carried to the voice worker as participant attributes. */
export interface CallContext {
  timeZone?: string;
  location?: { lat: number; lng: number; accuracyM: number; ageS: number };
  driving?: boolean;
}

export interface TokenIssuer {
  issue(context?: CallContext): TokenGrant;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function finiteIn(value: unknown, min: number, max: number): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value >= min && value <= max;
}

function isTimeZone(value: unknown): value is string {
  if (typeof value !== 'string' || !/^[A-Za-z][A-Za-z0-9_+\-/]{0,63}$/.test(value)) {
    return false;
  }
  try {
    new Intl.DateTimeFormat('en-US', { timeZone: value });
    return true;
  } catch {
    return false;
  }
}

/**
 * The call context from a token request body, keeping only the fields that
 * validate. The context is best-effort: a malformed field is dropped, never an error.
 */
export function parseCallContext(body: unknown): CallContext | undefined {
  if (!isRecord(body) || !isRecord(body.context)) {
    return undefined;
  }
  const raw = body.context;
  const context: CallContext = {};
  if (isTimeZone(raw.time_zone)) {
    context.timeZone = raw.time_zone;
  }
  const location = raw.location;
  if (
    isRecord(location) &&
    finiteIn(location.lat, -90, 90) &&
    finiteIn(location.lng, -180, 180) &&
    finiteIn(location.accuracy_m, 0, 1_000_000) &&
    finiteIn(location.age_s, 0, 86_400)
  ) {
    context.location = {
      lat: location.lat,
      lng: location.lng,
      accuracyM: location.accuracy_m,
      ageS: location.age_s,
    };
  }
  if (typeof raw.driving === 'boolean') {
    context.driving = raw.driving;
  }
  return Object.keys(context).length > 0 ? context : undefined;
}

function contextAttributes(context: CallContext): Record<string, string> {
  return {
    ...(context.timeZone !== undefined && { 'mentat.time_zone': context.timeZone }),
    ...(context.location !== undefined && {
      'mentat.location': JSON.stringify({
        lat: context.location.lat,
        lng: context.location.lng,
        accuracy_m: context.location.accuracyM,
        age_s: context.location.ageS,
      }),
    }),
    ...(context.driving !== undefined && { 'mentat.driving': String(context.driving) }),
  };
}

interface TokenIssuerConfig {
  apiKey: string;
  apiSecret: string;
  url: string;
}

interface TokenIssuerDeps {
  now?: () => Date;
  uuid?: () => string;
  sign?: (input: string, secret: string) => string;
}

function hmacSha256(input: string, secret: string): string {
  return createHmac('sha256', secret).update(input).digest('base64url');
}

export function createTokenIssuer(
  config: TokenIssuerConfig,
  deps: TokenIssuerDeps = {},
): TokenIssuer {
  const now = deps.now ?? (() => new Date());
  const uuid = deps.uuid ?? randomUUID;
  const sign = deps.sign ?? hmacSha256;

  return {
    issue(context?: CallContext): TokenGrant {
      const id = uuid();
      const room = `android-${id}`;
      const issuedAt = Math.floor(now().getTime() / 1000);
      const expiresAt = issuedAt + 3600;
      const header = Buffer.from(JSON.stringify({ alg: 'HS256', typ: 'JWT' })).toString(
        'base64url',
      );
      const payload = Buffer.from(
        JSON.stringify({
          iss: config.apiKey,
          sub: `pixel-${id}`,
          iat: issuedAt,
          nbf: issuedAt,
          exp: expiresAt,
          video: {
            room,
            roomJoin: true,
            canPublish: true,
            canPublishSources: ['microphone'],
            canSubscribe: true,
            canPublishData: true,
          },
          ...(context !== undefined && { attributes: contextAttributes(context) }),
        }),
      ).toString('base64url');
      const signingInput = `${header}.${payload}`;
      const signature = sign(signingInput, config.apiSecret);

      return {
        token: `${signingInput}.${signature}`,
        room,
        url: config.url,
        expires_at: new Date(expiresAt * 1000).toISOString(),
      };
    },
  };
}
