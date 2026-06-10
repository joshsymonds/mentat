# Mentat

A personal assistant daemon: Claude as the brain, your infrastructure as the body.

Mentat supervises persistent [Claude Code](https://code.claude.com) sessions through
the [Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk/overview) and exposes
them as a streaming conversation API. Surfaces (Home Assistant voice, chat, Signal)
are HTTP clients; tools arrive via MCP; the agentic harness is Claude Code itself.

## History

- **v1** (2025, archived on [`go-2025`](https://github.com/Veraticus/mentat/tree/go-2025)):
  53,000 lines of Go building queueing, validation, and session reliability the
  platform didn't yet provide.
- **v2** (2026, archived on [`go-v2`](https://github.com/Veraticus/mentat/tree/go-v2)):
  ~1,900 lines of Go supervising claude CLI children over the stream-json protocol.
  Small because the platform caught up; retired because the Agent SDK absorbed the
  protocol layer, the permission server, and the respawn machinery.
- **v3** (current): TypeScript on the Agent SDK. Permission policy is an in-process
  callback, protocol drift is a version bump, and upcoming in-process tools (memory)
  are plain functions.

## Development

```sh
just lint   # eslint (strict-type-checked) + tsc --strict + knip
just test   # vitest, offline — no claude binary, no network
```

Testing is fixture-based: the backend takes an injectable query function, and tests
replay SDK message streams recorded from real turns (`test/fixtures/*.jsonl`).
To record a new fixture:

```sh
MENTAT_CLAUDE_BIN=$(command -v claude) node scripts/record-fixture.ts <name>
```

It spends real tokens (a few cents at haiku). Never hand-author fixtures.

The **Live smoke** workflow is `workflow_dispatch`-only: run it when bumping the
pinned SDK version or claude binary — it drives one real conversation (streaming,
resume-across-restart, isolation probe) and needs a `CLAUDE_CODE_OAUTH_TOKEN`
repository secret (`claude setup-token`).

## Running

Configuration is environment-first; every knob is a `MENTAT_*` variable:

```sh
MENTAT_CLAUDE_BIN=/path/to/claude \
MENTAT_MODEL=claude-haiku-4-5 \
MENTAT_STATE_PATH=/var/lib/mentat/state.json \
npm start
```

`POST /v1/conversation` with `{"session_id","text","meta"}` streams the turn as
NDJSON; `GET /healthz` reports liveness. See `src/config.ts` for the full surface.

## Operational notes

- The conversation API has no authentication; mentatd refuses a non-loopback
  bind unless `MENTAT_ALLOW_NON_LOOPBACK` is set. Expose it over a tailnet
  (`tailscale serve`), never the open internet.
- Permission policy is the in-process `canUseTool` callback (`src/policy.ts`);
  the shipped default allows everything and logs one structured line per
  decision. Tool-call identity context comes from each turn's `meta` — bound
  per turn, never cached on the session.
- Child sessions are isolated: no user settings, no skills, explicit MCP config
  only, allowlisted env. `MENTAT_CLAUDE_BIN` is required — there is deliberately
  no PATH fallback and no SDK auto-download; the deploy pins both the npm SDK
  version (exact in package.json) and the claude binary.
- Session child processes expire after `MENTAT_SESSION_TTL` idle (default 15m);
  conversations survive expiry and daemon restarts via the persisted resume map.
- Session recordings (`MENTAT_RECORD_DIR`) grow without bound — the daemon never
  prunes them. The operator owns retention (tmpfiles.d age rule at deploy).

## CI

Every push runs lint and the offline test suite. Live smoke is manual (above).
