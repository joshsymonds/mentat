# NixOS module for mentatd. Lives with the daemon so deploy configuration
# evolves in the same commit as the code it deploys; the consuming flake
# (nix-config) supplies host concerns: the pinned claude binary, the agenix
# secrets file, and network placement.
{ mentatdPackage }:
{ config, lib, pkgs, ... }: let
  cfg = config.services.mentat;
in {
  options.services.mentat = {
    enable = lib.mkEnableOption "mentat personal assistant daemon";

    package = lib.mkOption {
      type = lib.types.package;
      default = mentatdPackage;
      description = "The mentatd package to run.";
    };

    # No default on purpose: the daemon must never discover claude via
    # PATH. The deploy pins the exact binary or refuses to evaluate.
    claudePackage = lib.mkOption {
      type = lib.types.package;
      description = "Package providing bin/claude (the pinned Claude Code CLI).";
    };

    listenPort = lib.mkOption {
      type = lib.types.port;
      default = 8484;
      description = "Loopback port for the conversation API. Tailnet exposure is a separate ingress concern (tailscale serve).";
    };

    model = lib.mkOption {
      type = lib.types.str;
      default = "fable";
      description = "MENTAT_MODEL for all sessions.";
    };

    effort = lib.mkOption {
      type = lib.types.nullOr (lib.types.enum [ "low" "medium" "high" "xhigh" "max" ]);
      default = null;
      description = "MENTAT_EFFORT, if set.";
    };

    promptFile = lib.mkOption {
      type = lib.types.path;
      default = ../prompt.md;
      description = "File whose contents become MENTAT_SYSTEM_PROMPT.";
    };

    mcpConfig = lib.mkOption {
      type = lib.types.nullOr (lib.types.attrsOf lib.types.anything);
      default = null;
      description = "MCP servers attrset; rendered as MENTAT_MCP_CONFIG ({\"mcpServers\": ...}).";
    };

    maxBudgetUsd = lib.mkOption {
      type = lib.types.float;
      description = "Per-turn budget ceiling (MENTAT_MAX_BUDGET_USD). Required: an unattended daemon never runs uncapped.";
    };

    environmentFile = lib.mkOption {
      type = lib.types.str;
      description = "EnvironmentFile with secrets (CLAUDE_CODE_OAUTH_TOKEN at minimum). An agenix-decrypted path, never a store path.";
    };

    extraEnv = lib.mkOption {
      type = lib.types.attrsOf lib.types.str;
      default = { };
      description = "Additional non-secret MENTAT_* environment.";
    };
  };

  config = lib.mkIf cfg.enable {
    users.users.mentat = {
      isSystemUser = true;
      group = "mentat";
    };
    users.groups.mentat = { };

    systemd.services.mentatd = {
      description = "mentat personal assistant daemon";
      after = [ "network.target" ];
      wantedBy = [ "multi-user.target" ];

      restartTriggers = [ cfg.environmentFile ];

      environment = {
        MENTAT_CLAUDE_BIN = lib.getExe' cfg.claudePackage "claude";
        MENTAT_LISTEN = "127.0.0.1:${toString cfg.listenPort}";
        MENTAT_STATE_PATH = "/var/lib/mentat/sessions.json";
        MENTAT_MODEL = cfg.model;
        MENTAT_MAX_BUDGET_USD = toString cfg.maxBudgetUsd;
        # The SDK child writes $HOME/.claude state; point it at the state
        # directory instead of weakening ProtectHome.
        HOME = "/var/lib/mentat";
      }
      // lib.optionalAttrs (cfg.effort != null) { MENTAT_EFFORT = cfg.effort; }
      // lib.optionalAttrs (cfg.mcpConfig != null) {
        MENTAT_MCP_CONFIG = builtins.toJSON { mcpServers = cfg.mcpConfig; };
      }
      // cfg.extraEnv;

      # The prompt is exported in-script rather than via `environment`:
      # systemd unit Environment= lines cannot carry multiline values.
      script = ''
        MENTAT_SYSTEM_PROMPT="$(cat ${cfg.promptFile})"
        export MENTAT_SYSTEM_PROMPT
        exec ${lib.getExe cfg.package}
      '';

      serviceConfig = {
        Type = "simple";
        User = "mentat";
        Group = "mentat";
        Restart = "always";
        RestartSec = "5s";

        EnvironmentFile = cfg.environmentFile;

        StateDirectory = "mentat";
        WorkingDirectory = "/var/lib/mentat";

        PrivateTmp = true;
        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectHome = true;
      };
    };
  };
}
