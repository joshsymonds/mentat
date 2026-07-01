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

    reminder = {
      enable = lib.mkEnableOption "the daily morning reminder timer";

      time = lib.mkOption {
        # Fail at eval, not at unit activation: OnCalendar interpolates this.
        type = lib.types.strMatching "[0-9]{2}:[0-9]{2}";
        default = "09:00";
        description = "Host-local HH:MM the reminder fires.";
      };
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

      # No restartTriggers on cfg.environmentFile here: that's the constant
      # /run/agenix path, which never changes between generations. The host
      # config triggers on the .age ciphertext store path instead.

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

    systemd.services.mentat-reminder = lib.mkIf cfg.reminder.enable {
      description = "mentat morning reminder (Morgen → mentat turn → ntfy)";
      after = [ "network-online.target" "mentatd.service" ];
      wants = [ "network-online.target" ];
      # The reminder failed silently for 18 days (Jun 13–Jul 1 2026) before
      # anyone noticed — an unattended daily job that only ever logs to a
      # journal nobody reads is invisible by construction. Fire a push on
      # every failure instead.
      onFailure = [ "mentat-reminder-alert.service" ];

      environment = {
        MENTAT_URL = "http://127.0.0.1:${toString cfg.listenPort}";
      };

      serviceConfig = {
        Type = "oneshot";
        User = "mentat";
        Group = "mentat";
        EnvironmentFile = cfg.environmentFile;
        # The script only needs MORGEN_API_KEY/NTFY_URL/NTFY_TOKEN; keep the
        # daemon credential out of this process's environment.
        UnsetEnvironment = [ "CLAUDE_CODE_OAUTH_TOKEN" ];
        ExecStart = lib.getExe' cfg.package "mentat-reminder";
        # Type=oneshot disables the default start timeout; without a bound, a
        # turn that trickles deltas forever hangs the unit silently and blocks
        # the next day's Persistent= activation. Fail loudly instead.
        TimeoutStartSec = "15min";

        PrivateTmp = true;
        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectHome = true;
      };
    };

    systemd.services.mentat-reminder-alert = lib.mkIf cfg.reminder.enable {
      description = "ntfy alert for a failed mentat-reminder run";

      # Same secrets file as the reminder itself, purely for NTFY_URL/
      # NTFY_TOKEN — the daemon credential has no business here either.
      script = ''
        if [ -n "$NTFY_TOKEN" ]; then
          exec ${lib.getExe pkgs.curl} --fail --silent --show-error \
            -H "Authorization: Bearer $NTFY_TOKEN" \
            -H "Title: mentat" -H "Priority: 5" -H "Tags: warning" \
            -d "mentat-reminder failed — journalctl -u mentat-reminder" \
            "$NTFY_URL"
        else
          exec ${lib.getExe pkgs.curl} --fail --silent --show-error \
            -H "Title: mentat" -H "Priority: 5" -H "Tags: warning" \
            -d "mentat-reminder failed — journalctl -u mentat-reminder" \
            "$NTFY_URL"
        fi
      '';

      serviceConfig = {
        Type = "oneshot";
        User = "mentat";
        Group = "mentat";
        EnvironmentFile = cfg.environmentFile;
        UnsetEnvironment = [ "CLAUDE_CODE_OAUTH_TOKEN" ];

        PrivateTmp = true;
        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectHome = true;
      };
    };

    systemd.timers.mentat-reminder = lib.mkIf cfg.reminder.enable {
      description = "Daily mentat morning reminder";
      wantedBy = [ "timers.target" ];
      timerConfig = {
        OnCalendar = "*-*-* ${cfg.reminder.time}:00";
        # Fire on next boot if the host slept through the slot.
        Persistent = true;
      };
    };
  };
}
