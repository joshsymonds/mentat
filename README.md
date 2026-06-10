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
from `testdata/cassettes/` — no network, no claude binary required. Re-recording
cassettes (`just record-cassettes`) requires a real claude binary and spends real
tokens.
