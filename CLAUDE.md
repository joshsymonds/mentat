# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Mentat is a personal assistant daemon: `mentatd` supervises persistent Claude Code
sessions through the Claude Agent SDK and exposes them as a streaming NDJSON
conversation API (`POST /v1/conversation`). Surfaces (Home Assistant voice, chat,
Signal) are HTTP clients; tools arrive via MCP; the agentic harness is Claude Code
itself. v3 is deliberately small — don't rebuild queueing/validation/reliability the
SDK provides (v1 did, 53k lines, archived on `go-2025`; the v2 Go supervisor that
parsed stream-json itself is archived on `go-v2`).

## Commands

```sh
just            # lint + test (default)
just lint       # eslint (strict-type-checked) + tsc --strict + knip
just test       # vitest, offline (no claude binary, no network)
npx vitest run test/claudecode.test.ts -t "name"   # single test
MENTAT_CLAUDE_BIN=$(command -v claude) node scripts/record-fixture.ts <name>
                # record a real SDK turn → test/fixtures/<name>.jsonl (spends tokens)
```

The live smoke (`test/live.test.ts`) skips unless `MENTAT_CLAUDE_BIN` is set;
everything else runs offline against fixtures. CI runs lint + tests on every push;
the live smoke is a manual `workflow_dispatch` workflow.

## Architecture

Data flows through three altitudes, the SDK owning everything below them:

1. **`src/translate.ts`** — converts SDK messages into backend events. Stateful per
   session (correlates tool_use ids to names); deliberately exhaustive: every known
   message type is mapped or explicitly ignored, anything novel becomes an `unknown`
   event that must be logged loudly, never dropped (it means the SDK moved ahead of
   this build).

2. **`src/claudecode.ts`** — the `Backend`: one persistent SDK streaming-input session
   (one claude child) per sessionId. Turns serialize on a per-session mutex; abandoned
   turns interrupt-and-drain (falling back to drop-and-resume); a child death between
   turns respawns with `resume` and replays the turn. The sessionId→CLI-UUID map
   persists to `MENTAT_STATE_PATH` so conversations survive daemon restarts. The SDK
   call is injectable (`queryFn`) — all lifecycle behavior tests offline.
   `src/policy.ts` is the permission seam: every tool call passes one `PolicyFn`
   carrying the active turn's `meta` (surface, user) — bound per turn, never cached
   on the session.

3. **`src/server.ts` + `src/wire.ts`** — the HTTP surface. One POST per turn; the
   response streams events as NDJSON lines **byte-compatible with the v2 Go wire
   format** (key order, omitempty semantics — pinned by golden tests in
   `test/wire.test.ts`). Mid-stream failures become a terminal `{"kind":"error"}`
   line because the 200 already shipped. `SessionTracker` + `src/janitor.ts` expire
   idle children (in-flight turns never expire); `src/config.ts` is env-first
   (`MENTAT_*`); `src/main.ts` assembles the daemon.

## Invariants to preserve

- **Child isolation is mandatory.** Every session sets `settingSources: []` AND
  `skills: []` (skills leak independently of settings), `strictMcpConfig: true`, and
  an explicit env allowlist (`buildChildEnv`) — never the daemon's env. The default
  tool policy disallows dangerous built-ins (Bash, Write, Edit, …). `buildOptions`
  is exported pure precisely so tests pin all of this.
- **No PATH fallback and no SDK auto-download for the claude binary** — the deploy
  pins it; `MENTAT_CLAUDE_BIN` is required. The SDK npm version is pinned exact in
  package.json; bumps are deliberate commits validated by the live smoke.
- **The API has no authentication by design.** `mentatd` refuses non-loopback bind
  unless `MENTAT_ALLOW_NON_LOOPBACK` is set; exposure happens via tailnet at deploy.
- **Authority is per-turn.** Permission context comes from the turn's `meta` and is
  never cached on the session — the conversation is memory, authority is per-turn.
- **Fixtures are recorded, never hand-authored** (`scripts/record-fixture.ts`, or
  `MENTAT_RECORD_DIR` on a live daemon). Minimal hand-built message shapes are
  acceptable only for orchestration unit tests, not protocol-shape coverage.
- **The wire format is a contract**: golden tests pin the exact NDJSON encoding
  surfaces consume. Changing it breaks clients.
