# Mentat

A personal assistant daemon: Claude as the brain, your infrastructure as the body.

Mentat supervises persistent [Claude Code](https://code.claude.com) sessions over
the stream-json protocol and exposes them as a streaming conversation API. Surfaces
(Home Assistant voice, chat, Signal) are clients; tools arrive via MCP; the agentic
harness is Claude Code itself.

## History

v1 (2025, archived on the [`go-2025`](https://github.com/Veraticus/mentat/tree/go-2025)
branch) was 53,000 lines of Go building queueing, validation, and session reliability
the platform didn't yet provide. v2 is small because the platform caught up: subscription
auth for daemons, a documented duplex protocol, and a harness worth renting.

## Development

```sh
just lint   # golangci-lint, full ruleset
just test   # go test -race ./...
```

Cassette-based testing: integration tests replay recorded stream-json transcripts
from `testdata/cassettes/` — no network, no claude binary required. To record a new
cassette, `just record-cassette <name> "<prompt>"` drives a real turn through the
live backend's recording wrapper (`cmd/record`) and writes
`testdata/cassettes/<name>.ndjson`. It needs a claude binary (`MENTAT_CLAUDE_BIN`,
or `claude` on `PATH`) and spends real tokens.

## Operational notes

- The conversation API has no authentication; `mentatd` refuses a non-loopback
  `-listen` unless `--allow-non-loopback` is set. Expose it over a tailnet
  (`tailscale serve`), never the open internet.
- Session recordings (`-record-dir`) grow without bound — the daemon never
  prunes them. The operator owns retention (the Nix deploy uses a `tmpfiles.d`
  age rule).
- Child tool policy defaults to disallowing dangerous built-ins (Bash, Write,
  Edit, …). Override with `-allowed-tools` / `-disallowed-tools`, and gate tool
  calls through an MCP permission tool via `-permission-prompt-tool`.

## CI

Every push runs lint, `go test -race` (cassettes only, no secrets), and a build.
The **Live smoke** workflow is `workflow_dispatch`-only: run it when bumping the
pinned claude binary — it executes one real haiku turn through the supervisor and
fails on unrecognized protocol events. It needs a `CLAUDE_CODE_OAUTH_TOKEN`
repository secret (`claude setup-token`) and spends about a cent per run.
