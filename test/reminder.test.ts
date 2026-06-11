// The daily-reminder pipeline: Morgen events → one mentat turn → ntfy push.
// Everything network-shaped takes an injected fetch, so the whole script
// tests offline.

import { describe, expect, it } from 'vitest';

import {
  buildTurnText,
  converse,
  fetchTodayEvents,
  isMirror,
  pushNtfy,
  requireEnv,
  todayWindow,
  type MorgenEvent,
} from '../scripts/daily-reminder.ts';

type FetchFn = typeof fetch;

function urlOf(input: Parameters<FetchFn>[0]): string {
  if (typeof input === 'string') {
    return input;
  }
  return input instanceof URL ? input.href : input.url;
}

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'content-type': 'application/json' },
  });
}

describe('todayWindow', () => {
  it('returns host-local midnight to midnight as UTC ISO Z stamps', () => {
    const now = new Date(2026, 5, 11, 9, 30, 0); // local 2026-06-11 09:30
    const { startIso, endIso } = todayWindow(now);
    expect(startIso).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$/);
    expect(endIso).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$/);
    const start = new Date(startIso);
    const end = new Date(endIso);
    expect(end.getTime() - start.getTime()).toBe(24 * 60 * 60 * 1000);
    expect(start.getTime()).toBeLessThanOrEqual(now.getTime());
    expect(end.getTime()).toBeGreaterThan(now.getTime());
  });
});

describe('isMirror', () => {
  it('detects the Calendar Propagation marker anywhere in the description', () => {
    expect(isMirror({ title: '[Busy]', start: 'x', description: 'note\nCalendar Propagation: abc' })).toBe(true);
    expect(isMirror({ title: 'Dad birthday', start: 'x', description: '' })).toBe(false);
    expect(isMirror({ title: 'No description', start: 'x' })).toBe(false);
  });
});

describe('fetchTodayEvents', () => {
  it('groups calendars by account, queries each, merges and filters mirrors', async () => {
    const urls: string[] = [];
    const fakeFetch: FetchFn = (input, init) => {
      const url = urlOf(input);
      urls.push(url);
      expect(init?.headers).toMatchObject({ Authorization: 'ApiKey k3y' });
      if (url.includes('/calendars/list')) {
        return Promise.resolve(
          jsonResponse({
            data: {
              calendars: [
                { id: 'c1', accountId: 'a1', name: 'primary' },
                { id: 'c2', accountId: 'a1', name: 'holidays' },
                { id: 'c3', accountId: 'a2', name: 'work' },
              ],
            },
          }),
        );
      }
      const u = new URL(url);
      const account = u.searchParams.get('accountId');
      expect(u.searchParams.get('start')).toBeTruthy();
      expect(u.searchParams.get('end')).toBeTruthy();
      if (account === 'a1') {
        expect(u.searchParams.get('calendarIds')).toBe('c1,c2');
        return Promise.resolve(
          jsonResponse({
            data: {
              events: [
                { title: "Dad's birthday", start: '2026-06-11T00:00:00', showWithoutTime: true },
                { title: '[Busy]', start: '2026-06-11T10:00:00', description: 'Calendar Propagation: xyz' },
              ],
            },
          }),
        );
      }
      expect(u.searchParams.get('calendarIds')).toBe('c3');
      return Promise.resolve(
        jsonResponse({ data: { events: [{ title: 'Standup', start: '2026-06-11T09:15:00' }] } }),
      );
    };

    const events = await fetchTodayEvents(fakeFetch, 'k3y', new Date(2026, 5, 11, 7, 0, 0));
    expect(events.map((e) => e.title).sort()).toEqual(["Dad's birthday", 'Standup']);
    expect(urls.filter((u) => u.includes('/events/list'))).toHaveLength(2);
  });

  it('throws on a non-200 Morgen response', async () => {
    const fakeFetch: FetchFn = () => Promise.resolve(new Response('nope', { status: 401 }));
    await expect(fetchTodayEvents(fakeFetch, 'bad', new Date())).rejects.toThrow(/401/);
  });
});

describe('buildTurnText', () => {
  it('names the date and every event, marking all-day ones', () => {
    const events: MorgenEvent[] = [
      { title: "Dad's birthday", start: '2026-06-11T00:00:00', showWithoutTime: true },
      { title: 'Dentist', start: '2026-06-11T14:30:00' },
    ];
    const text = buildTurnText(events, new Date(2026, 5, 11, 9, 0, 0));
    expect(text).toContain("Dad's birthday");
    expect(text).toContain('all-day');
    expect(text).toContain('Dentist');
    expect(text).toContain('2026-06-11');
  });

  it('says the calendar is empty rather than listing nothing', () => {
    const text = buildTurnText([], new Date(2026, 5, 11, 9, 0, 0));
    expect(text.toLowerCase()).toContain('no events');
  });
});

describe('converse', () => {
  it('POSTs the turn and returns the done text', async () => {
    let body: unknown;
    const fakeFetch: FetchFn = (input, init) => {
      expect(urlOf(input)).toBe('http://127.0.0.1:8484/v1/conversation');
      expect(init?.method).toBe('POST');
      body = JSON.parse(init?.body as string);
      const ndjson =
        '{"kind":"text_delta","text":"It"}\n' +
        '{"kind":"done","text":"It is Dad\'s birthday today. Call him.","is_error":false,"session_id":"reminder-2026-06-11"}\n';
      return Promise.resolve(new Response(ndjson, { status: 200 }));
    };
    const text = await converse(fakeFetch, 'http://127.0.0.1:8484', 'turn text', new Date(2026, 5, 11));
    expect(text).toBe("It is Dad's birthday today. Call him.");
    expect(body).toMatchObject({
      session_id: 'reminder-2026-06-11',
      text: 'turn text',
      meta: { surface: 'reminder', user: 'josh' },
    });
  });

  it('throws on a terminal error line', async () => {
    const fakeFetch: FetchFn = () =>
      Promise.resolve(new Response('{"kind":"error","message":"backend exploded"}\n', { status: 200 }));
    await expect(converse(fakeFetch, 'http://x', 't', new Date())).rejects.toThrow(/backend exploded/);
  });

  it('throws when the stream ends without a done line', async () => {
    const fakeFetch: FetchFn = () =>
      Promise.resolve(new Response('{"kind":"text_delta","text":"hi"}\n', { status: 200 }));
    await expect(converse(fakeFetch, 'http://x', 't', new Date())).rejects.toThrow(/done/);
  });

  it('throws on a non-200 status', async () => {
    const fakeFetch: FetchFn = () => Promise.resolve(new Response('busy', { status: 503 }));
    await expect(converse(fakeFetch, 'http://x', 't', new Date())).rejects.toThrow(/503/);
  });
});

describe('pushNtfy', () => {
  it('POSTs the message with title and bearer auth', async () => {
    let url = '';
    let init: RequestInit | undefined;
    const fakeFetch: FetchFn = (input, requestInit) => {
      url = urlOf(input);
      init = requestInit;
      return Promise.resolve(new Response('', { status: 200 }));
    };
    await pushNtfy(fakeFetch, 'https://ntfy.example/topic', 'tok', 'Call your dad.');
    expect(url).toBe('https://ntfy.example/topic');
    expect(init?.method).toBe('POST');
    expect(init?.headers).toMatchObject({
      Title: 'mentat',
      Authorization: 'Bearer tok',
    });
    expect(init?.body).toBe('Call your dad.');
  });

  it('omits the auth header when the token is empty', async () => {
    let headers: Record<string, string> = {};
    const fakeFetch: FetchFn = (_input, init) => {
      headers = (init?.headers ?? {}) as Record<string, string>;
      return Promise.resolve(new Response('', { status: 200 }));
    };
    await pushNtfy(fakeFetch, 'https://ntfy.example/topic', '', 'msg');
    expect(Object.keys(headers)).not.toContain('Authorization');
  });

  it('throws on a failed push', async () => {
    const fakeFetch: FetchFn = () => Promise.resolve(new Response('', { status: 500 }));
    await expect(pushNtfy(fakeFetch, 'https://x/t', '', 'm')).rejects.toThrow(/500/);
  });
});

describe('requireEnv', () => {
  it('returns the value when present and throws naming the variable when missing', () => {
    expect(requireEnv({ FOO: 'bar' }, 'FOO')).toBe('bar');
    expect(() => requireEnv({}, 'MORGEN_API_KEY')).toThrow(/MORGEN_API_KEY/);
    expect(() => requireEnv({ X: '' }, 'X')).toThrow(/X/);
  });
});
