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

# Re-record spike cassettes against the pinned claude binary (costs real tokens)
record-cassettes:
    go run ./cmd/spike record
