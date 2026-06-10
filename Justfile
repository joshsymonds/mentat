# Mentat development tasks

# Lint and test everything
default: lint test

# Lint Go code
lint:
    golangci-lint run ./...

# Run all tests with the race detector
test:
    go test -race ./...

# Re-record spike cassettes against the pinned claude binary (costs real tokens)
record-cassettes:
    go run ./cmd/spike record
