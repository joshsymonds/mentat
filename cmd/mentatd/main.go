// Command mentatd is the Mentat daemon: a Backend behind the streaming
// conversation API, run as a systemd-style foreground process. It binds
// localhost; tailnet exposure happens at deploy via tailscale serve.
package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/Veraticus/mentat/internal/api"
	"github.com/Veraticus/mentat/internal/backend"
)

const (
	readHeaderTimeout = 10 * time.Second
	shutdownTimeout   = 15 * time.Second
	minJanitorTick    = 30 * time.Second
	defaultSessionTTL = 15 * time.Minute
	defaultMaxSession = 16
)

type config struct {
	listen               string
	backendKind          string
	claudeBin            string
	model                string
	effort               string
	systemPrompt         string
	memoryDir            string
	recordDir            string
	mcpConfig            string
	cassette             string
	statePath            string
	allowedTools         string
	disallowedTools      string
	permissionPromptTool string
	allowNonLoopback     bool
	maxSessions          int
	sessionTTL           time.Duration
	maxBudgetUSD         float64
}

func main() {
	cfg := parseFlags()
	logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))

	if err := run(cfg, logger); err != nil {
		logger.Error("mentatd failed", slog.Any("error", err))
		os.Exit(1)
	}
}

func parseFlags() config {
	var cfg config
	flag.StringVar(&cfg.listen, "listen", envOr("MENTAT_LISTEN", "127.0.0.1:8484"),
		"address to bind the conversation API on")
	flag.StringVar(&cfg.backendKind, "backend", envOr("MENTAT_BACKEND", "claudecode"),
		"conversation backend: claudecode or cassette")
	flag.StringVar(&cfg.claudeBin, "claude-bin", os.Getenv("MENTAT_CLAUDE_BIN"),
		"absolute path to the claude binary (claudecode backend)")
	flag.StringVar(&cfg.model, "model", os.Getenv("MENTAT_MODEL"),
		"model for conversation sessions")
	flag.StringVar(&cfg.effort, "effort", os.Getenv("MENTAT_EFFORT"),
		"effort level for conversation sessions")
	flag.StringVar(&cfg.systemPrompt, "system-prompt", os.Getenv("MENTAT_SYSTEM_PROMPT"),
		"system prompt replacing the CLI default")
	flag.StringVar(&cfg.memoryDir, "memory-dir", os.Getenv("MENTAT_MEMORY_DIR"),
		"directory of memory files granted to sessions")
	flag.StringVar(&cfg.recordDir, "record-dir", os.Getenv("MENTAT_RECORD_DIR"),
		"directory for raw NDJSON session recordings (future cassettes)")
	flag.StringVar(&cfg.mcpConfig, "mcp-config", os.Getenv("MENTAT_MCP_CONFIG"),
		"MCP server configuration (inline JSON or path)")
	flag.StringVar(&cfg.cassette, "cassette", os.Getenv("MENTAT_CASSETTE"),
		"recorded transcript to replay (cassette backend)")
	flag.StringVar(&cfg.statePath, "state-path", os.Getenv("MENTAT_STATE_PATH"),
		"file persisting the session resume map across restarts (claudecode backend)")
	flag.StringVar(&cfg.allowedTools, "allowed-tools", os.Getenv("MENTAT_ALLOWED_TOOLS"),
		"comma-separated tool allowlist (claudecode backend)")
	flag.StringVar(&cfg.disallowedTools, "disallowed-tools", os.Getenv("MENTAT_DISALLOWED_TOOLS"),
		"comma-separated tool denylist; defaults to dangerous built-ins when no policy is set")
	flag.StringVar(&cfg.permissionPromptTool, "permission-prompt-tool", os.Getenv("MENTAT_PERMISSION_PROMPT_TOOL"),
		"MCP tool (mcp__server__tool) consulted before gated tool calls")
	flag.BoolVar(&cfg.allowNonLoopback, "allow-non-loopback", os.Getenv("MENTAT_ALLOW_NON_LOOPBACK") != "",
		"permit binding a non-loopback address (the API has no auth of its own)")
	flag.IntVar(&cfg.maxSessions, "max-sessions", envOrInt("MENTAT_MAX_SESSIONS", defaultMaxSession),
		"maximum concurrent live sessions (0 disables the cap)")
	flag.DurationVar(&cfg.sessionTTL, "session-ttl", envOrDuration("MENTAT_SESSION_TTL", defaultSessionTTL),
		"idle duration after which a session's child process is released")
	flag.Float64Var(&cfg.maxBudgetUSD, "max-budget-usd", envOrFloat("MENTAT_MAX_BUDGET_USD", 0),
		"per-session spend ceiling in USD (0 disables)")
	flag.Parse()
	return cfg
}

// validateListen refuses a non-loopback bind address unless explicitly
// allowed: the conversation API has no authentication of its own and relies
// on the deploy's tailnet ingress, so binding the open internet is a footgun.
func validateListen(addr string, allowNonLoopback bool) error {
	if allowNonLoopback {
		return nil
	}
	host, _, err := net.SplitHostPort(addr)
	if err != nil {
		return fmt.Errorf("invalid listen address %q: %w", addr, err)
	}
	if host == "localhost" {
		return nil
	}
	ip := net.ParseIP(host)
	if ip == nil {
		return fmt.Errorf("listen host %q is neither an IP nor localhost", host)
	}
	if !ip.IsLoopback() {
		return fmt.Errorf("refusing to bind non-loopback %q; pass --allow-non-loopback to override", host)
	}
	return nil
}

// toolPolicy resolves the effective allow/disallow tool lists. With no
// operator policy at all, it defaults to disallowing the dangerous built-ins.
func toolPolicy(allowed, disallowed string) (allow, disallow []string) {
	allow = splitList(allowed)
	disallow = splitList(disallowed)
	if len(allow) == 0 && len(disallow) == 0 {
		// With no operator policy, disallow the dangerous built-ins: the
		// isolated child still carries the full toolset, and a voice surface
		// must not drive Bash/Write/etc. by default. Opt into danger explicitly.
		disallow = []string{"Bash", "Write", "Edit", "NotebookEdit", "WebFetch", "WebSearch", "Task"}
	}
	return allow, disallow
}

func splitList(csv string) []string {
	if strings.TrimSpace(csv) == "" {
		return nil
	}
	parts := strings.Split(csv, ",")
	out := make([]string, 0, len(parts))
	for _, part := range parts {
		if trimmed := strings.TrimSpace(part); trimmed != "" {
			out = append(out, trimmed)
		}
	}
	return out
}

func run(cfg config, logger *slog.Logger) error {
	if err := validateListen(cfg.listen, cfg.allowNonLoopback); err != nil {
		return err
	}
	bk, err := buildBackend(cfg, logger)
	if err != nil {
		return err
	}
	defer closeBackend(bk, logger)

	server := api.NewServer(bk, logger)
	httpServer := &http.Server{
		Addr:              cfg.listen,
		Handler:           server.Handler(),
		ReadHeaderTimeout: readHeaderTimeout,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	go janitor(ctx, server, cfg.sessionTTL, logger)

	serveErr := make(chan error, 1)
	go func() { serveErr <- httpServer.ListenAndServe() }()
	logger.InfoContext(ctx, "mentatd listening",
		slog.String("addr", cfg.listen), slog.String("backend", cfg.backendKind))

	select {
	case err := <-serveErr:
		return fmt.Errorf("serving: %w", err)
	case <-ctx.Done():
	}

	logger.Info("shutting down")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), shutdownTimeout)
	defer cancel()
	if err := httpServer.Shutdown(shutdownCtx); err != nil {
		return fmt.Errorf("draining http server: %w", err)
	}
	return nil
}

func buildBackend(cfg config, logger *slog.Logger) (backend.Backend, error) {
	switch strings.ToLower(cfg.backendKind) {
	case "claudecode":
		allow, disallow := toolPolicy(cfg.allowedTools, cfg.disallowedTools)
		bk, err := backend.NewClaudeCode(backend.ClaudeCodeConfig{
			Bin:                  cfg.claudeBin,
			Model:                cfg.model,
			Effort:               cfg.effort,
			SystemPrompt:         cfg.systemPrompt,
			AddDirs:              addDirs(cfg.memoryDir),
			MCPConfig:            cfg.mcpConfig,
			PermissionPromptTool: cfg.permissionPromptTool,
			AllowedTools:         allow,
			DisallowedTools:      disallow,
			RecordDir:            cfg.recordDir,
			StatePath:            cfg.statePath,
			MaxBudgetUSD:         cfg.maxBudgetUSD,
			MaxSessions:          cfg.maxSessions,
			Logger:               logger,
		})
		if err != nil {
			return nil, fmt.Errorf("building claudecode backend: %w", err)
		}
		return bk, nil
	case "cassette":
		bk, err := backend.NewCassette(cfg.cassette)
		if err != nil {
			return nil, fmt.Errorf("building cassette backend: %w", err)
		}
		return bk, nil
	default:
		return nil, fmt.Errorf("unknown backend %q (want claudecode or cassette)", cfg.backendKind)
	}
}

func addDirs(memoryDir string) []string {
	if memoryDir == "" {
		return nil
	}
	return []string{memoryDir}
}

func closeBackend(bk backend.Backend, logger *slog.Logger) {
	if closer, ok := bk.(io.Closer); ok {
		if err := closer.Close(); err != nil {
			logger.Error("closing backend", slog.Any("error", err))
		}
	}
}

// janitor periodically releases idle sessions' child processes.
func janitor(ctx context.Context, server *api.Server, ttl time.Duration, logger *slog.Logger) {
	interval := ttl / 4
	if interval < minJanitorTick {
		interval = minJanitorTick
	}
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if expired := server.ExpireIdle(ttl); len(expired) > 0 {
				logger.InfoContext(ctx, "expired idle sessions",
					slog.Int("count", len(expired)))
			}
		}
	}
}

func envOr(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}

func envOrDuration(key string, fallback time.Duration) time.Duration {
	if value := os.Getenv(key); value != "" {
		if parsed, err := time.ParseDuration(value); err == nil {
			return parsed
		}
	}
	return fallback
}

func envOrFloat(key string, fallback float64) float64 {
	if value := os.Getenv(key); value != "" {
		if parsed, err := strconv.ParseFloat(value, 64); err == nil {
			return parsed
		}
	}
	return fallback
}

func envOrInt(key string, fallback int) int {
	if value := os.Getenv(key); value != "" {
		if parsed, err := strconv.Atoi(value); err == nil {
			return parsed
		}
	}
	return fallback
}
