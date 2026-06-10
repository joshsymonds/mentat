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
	"net/http"
	"os"
	"os/signal"
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
)

type config struct {
	listen       string
	backendKind  string
	claudeBin    string
	model        string
	effort       string
	systemPrompt string
	memoryDir    string
	recordDir    string
	mcpConfig    string
	cassette     string
	statePath    string
	sessionTTL   time.Duration
	maxBudgetUSD float64
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
	flag.DurationVar(&cfg.sessionTTL, "session-ttl", 15*time.Minute,
		"idle duration after which a session's child process is released")
	flag.Float64Var(&cfg.maxBudgetUSD, "max-budget-usd", 0,
		"per-session spend ceiling in USD (0 disables)")
	flag.Parse()
	return cfg
}

func run(cfg config, logger *slog.Logger) error {
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
		bk, err := backend.NewClaudeCode(backend.ClaudeCodeConfig{
			Bin:          cfg.claudeBin,
			Model:        cfg.model,
			Effort:       cfg.effort,
			SystemPrompt: cfg.systemPrompt,
			AddDirs:      addDirs(cfg.memoryDir),
			MCPConfig:    cfg.mcpConfig,
			RecordDir:    cfg.recordDir,
			StatePath:    cfg.statePath,
			MaxBudgetUSD: cfg.maxBudgetUSD,
			Logger:       logger,
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
