# Claude Code stream-json protocol — spike findings

Validated 2026-06-10 against `claude` 2.1.170 (Go 1.26.2, MCP go-sdk v1.6.1).
Every claim below is backed by a recorded transcript in `testdata/cassettes/`.

## Verdict

**GO on all six scenarios.** The headless CLI surface expresses everything the
daemon architecture needs. The TypeScript-SDK revisit clause in the epic was
not triggered.

| # | Scenario | Verdict | Cassette |
|---|----------|---------|----------|
| a | Simple turn, streamed deltas | GO | `simple_turn.ndjson` |
| b | Multi-turn persistent session | GO | `multi_turn.ndjson` |
| c | In-stream tool events | GO | `tool_use.ndjson` |
| d | Session resume across processes | GO | `resume.ndjson` |
| e | Permission prompting via MCP | GO | `permission_allow.ndjson`, `permission_deny.ndjson` |
| f | Effort control + thinking events | GO | `effort_low_fable.ndjson` (+ thinking in `simple_turn.ndjson`) |

## Invocation contract

```sh
claude -p \
  --input-format stream-json --output-format stream-json \
  --verbose --include-partial-messages \
  --setting-sources "" --strict-mcp-config \
  --session-id <uuid> \
  --model <model> --effort <low|medium|high|xhigh|max> \
  --mcp-config '<json>' --permission-prompt-tool mcp__<server>__<tool>
```

- One persistent child process per session; user turns are NDJSON lines on
  stdin (`{"type":"user","message":{"role":"user","content":[{"type":"text","text":...}]}}`).
  Queued turns process in order; one `result` event per turn.
- `--resume <session-id>` in a fresh process restores full context
  (proven: codeword survived a process restart).
- `--session-id <uuid>` lets the daemon pick session identity up front —
  no need to parse it out of init before correlating.

## Isolation is mandatory

A bare child process inherits the interactive user's full Claude Code
configuration: in testing that meant **141 tools, 3 MCP servers, gambit
SessionStart hooks, and permissionMode=bypassPermissions**. A daemon
controlling a house must never run in that config.

`--setting-sources "" --strict-mcp-config` yields: 0 MCP servers, no hooks,
`permissionMode: default`. Tool surface is then explicitly constructed via
`--mcp-config` / `--allowedTools` / `--disallowedTools`. Note: without further
flags the default -p toolset (32 built-in tools incl. Bash/Write) is still
present — the daemon must restrict this explicitly.

## Event taxonomy (observed)

| type | subtype / inner | Payload notes |
|------|-----------------|---------------|
| `system` | `init` | model, permissionMode, tools[], mcp_servers[], session_id, cwd |
| `system` | `thinking_tokens` | estimated_tokens (+delta) — live "the model is thinking" signal |
| `system` | `status` | e.g. "requesting" |
| `system` | `hook_started` / `hook_response` | only when hooks configured (never, for the daemon) |
| `assistant` | — | complete API message: thinking/text/tool_use blocks |
| `user` | — | tool_result blocks fed back to the loop (`is_error` flag) |
| `stream_event` | message_start/content_block_delta/... | true streaming; `text_delta` AND `thinking_delta` carry live text |
| `result` | success/... | per-turn: is_error, result text, stop_reason, session_id, duration_ms, ttft_ms, total_cost_usd, usage |
| `rate_limit_event` | — | subscription window status (five_hour, resetsAt, overage) |
| `control_request` / `control_response` | — | reserved; not yet observed in these flows |

Parser policy (`internal/streamjson`): unknown event types parse into
`Line{Type, Raw}` with `Unknown() == true` — surfaced, never fatal, never
dropped. This is the protocol-drift tripwire for claude-binary bumps.

## Permission prompting (the canUseTool equivalent)

`--permission-prompt-tool mcp__<server>__<tool>` is **hidden from `--help` in
2.1.170 but fully functional**. Flow proven end-to-end with `cmd/permd`:

1. Model emits a tool_use not covered by allow rules (e.g. `touch /tmp/...`;
   note `echo` was auto-allowed in default mode without consulting the tool).
2. CLI calls the MCP permission tool with an object input including
   `tool_name` and object-valued `input`.
3. Tool returns text content containing JSON:
   `{"behavior":"allow","updatedInput":{...}}` or
   `{"behavior":"deny","message":"..."}`.
4. Allow → tool executes. Deny → tool_result `is_error:true` carrying the
   deny message verbatim to the model (which then explains and stops).

**Schema gotcha:** the permission tool's input schema must be an open object.
A typed struct schema (additionalProperties=false / wrong type for `input`)
makes every gated call fail validation — the model sees repeated
`tool_use_error` and gives up, while `result` still reports `success`. Use a
`map[string]any` input type.

**Daemon design note:** `result.is_error=false` does NOT mean the turn
achieved the user's intent (see above failure mode). Intent-level success is
the model's text, not the protocol status.

## Effort and thinking

- `--effort <low|medium|high|xhigh|max>` is a first-class session flag.
- Fable 5 at `--effort low` answered an easy comparison correctly with **zero
  thinking events** — the conversation-lane behavior the design wants.
- Haiku at default effort emitted `system/thinking_tokens` progress events and
  `thinking_delta` stream events — both available as live "computer is
  thinking" signals for the latency multiplexer.

## Economics observed

- Haiku turn: ~$0.003–0.009. Fable 5 turn with cold cache: **$0.27** (the
  ~7K-token CC system prompt cache-writes at $10/MTok; warm-session turns
  cache-read at ~0.1×). Long-lived sessions amortize this; per-utterance
  fresh sessions on Fable would be needlessly expensive.
- `rate_limit_event` exposes the subscription window live — the daemon can
  watch its own budget.
- `--max-budget-usd` exists as a per-session hard ceiling; adopt in the
  backend as a safety rail.

## Toolchain notes

- golangci-lint must be built with a Go ≥ the toolchain (2.5.0/go1.25 panics
  against Go 1.26; nixpkgs 2.12.2/go1.26.2 works: `nix shell
  nixpkgs#golangci-lint -c golangci-lint run`).
- MCP server stderr is captured by the CLI's internal logs, not the parent's
  stderr — don't rely on child stderr for daemon-side audit; log inside the
  permission tool's own process if needed.
