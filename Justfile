# Mentat development tasks

# Lint and test everything
default: lint test

# Lint Go code. Pinned through nix: golangci-lint must be built with a Go
# toolchain >= the module's (2.5.0/go1.25 panics on Go 1.26 packages).
lint:
    nix shell nixpkgs#golangci-lint -c golangci-lint run ./...

# Run all tests with the race detector
test:
    go test -race ./...

# Record a cassette from a real claude turn into testdata/cassettes/<name>.ndjson.
# Requires a claude binary (MENTAT_CLAUDE_BIN, or claude on PATH) and spends
# real tokens. Example: just record-cassette greeting "say hello in one word"
record-cassette name prompt:
    MENTAT_CLAUDE_BIN="${MENTAT_CLAUDE_BIN:-$(command -v claude)}" \
        go run ./cmd/record -out testdata/cassettes/{{name}}.ndjson {{quote(prompt)}}
