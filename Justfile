# lint + test (default)
default: lint test

# eslint (strict-type-checked) + tsc --strict + knip
lint:
    npm run lint

# vitest, offline (no claude binary, no network)
test:
    npm test
