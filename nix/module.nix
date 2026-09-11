# NixOS module for mentatd. Lives with the daemon so deploy configuration
# evolves in the same commit as the code it deploys; the consuming flake
# (nix-config) supplies host concerns: the pinned claude binary, the agenix
# secrets file, and network placement.
{ mentatdPackage, voiceEnvPackage, publicEnvPackage }:
{ config, lib, pkgs, ... }: let
  cfg = config.services.mentat;

  # The agent imports its pure halves (stream.py, request.py) as siblings and
  # reads its persona and earcon relative to its own path, so the unit runs it
  # out of a directory, not a lone file — and every one of these is a startup
  # dependency, not an extra: without persona.md the agent raises before it
  # takes a call, and without the earcon it cannot acknowledge a heard turn.
  # Listed file by file rather than copying
  # ../voice wholesale: __pycache__, the offline test suite, and the asset
  # generator have no business in a deployed closure.
  voiceSource = lib.fileset.toSource {
    root = ../voice;
    fileset = lib.fileset.unions [
      ../voice/agent.py
      ../voice/persona.md
      ../voice/request.py
      ../voice/stream.py
      ../voice/assets/earcon.wav
    ];
  };

  # Keep the public front's source in the deployment closure without copying
  # its tests or any unrelated repository files.
  publicSource = lib.fileset.toSource {
    root = ../public;
    fileset = lib.fileset.unions [
      ../public/__init__.py
      ../public/front.py
    ];
  };
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

    public = {
      enable = lib.mkEnableOption "the OAuth-authenticated public MCP front";

      listenPort = lib.mkOption {
        type = lib.types.port;
        default = 8486;
        description = "Loopback port for the public OAuth front.";
      };

      baseUrl = lib.mkOption {
        type = lib.types.str;
        description = "Externally reachable HTTPS base URL advertised by the public OAuth front.";
      };

      accessConfigUrl = lib.mkOption {
        type = lib.types.str;
        description = "Cloudflare Access OIDC discovery URL.";
      };

      accessClientIdFile = lib.mkOption {
        type = lib.types.str;
        description = "Root-readable runtime path containing the Cloudflare Access client ID.";
      };

      accessClientSecretFile = lib.mkOption {
        type = lib.types.str;
        description = "Root-readable runtime path containing the Cloudflare Access client secret.";
      };

      jwtSecretFile = lib.mkOption {
        type = lib.types.str;
        description = "Root-readable runtime path containing the persistent FastMCP JWT signing secret.";
      };
    };

    voice = {
      enable = lib.mkEnableOption "the LiveKit voice agent";

      package = lib.mkOption {
        type = lib.types.package;
        default = voiceEnvPackage;
        description = "Python environment providing bin/python with livekit-agents.";
      };

      agentScript = lib.mkOption {
        type = lib.types.path;
        default = "${voiceSource}/agent.py";
        description = "Agent entry point. Its directory must also hold the modules it imports.";
      };

      livekitUrl = lib.mkOption {
        type = lib.types.str;
        default = "ws://127.0.0.1:7880";
        description = "LIVEKIT_URL of the SFU. Deployed separately from this module, hence a URL rather than a unit dependency.";
      };

      publicLivekitUrl = lib.mkOption {
        type = lib.types.str;
        description = "The wss URL clients are told to connect to (the tailnet-published LiveKit signal).";
      };

      mentatUrl = lib.mkOption {
        type = lib.types.str;
        default = "http://127.0.0.1:${toString cfg.listenPort}";
        description = "MENTAT_URL the agent posts turns to.";
      };

      environmentFile = lib.mkOption {
        type = lib.types.str;
        description = "EnvironmentFile supplying LIVEKIT_API_KEY/SECRET and OPENAI_API_KEY; it may also carry the optional MENTAT_PLACES_API_KEY. An agenix-decrypted path, never a store path.";
      };

      privateContextFile = lib.mkOption {
        type = lib.types.nullOr lib.types.str;
        default = null;
        description = ''
          TOML file of what the voice knows about its person — `about`
          (a paragraph folded into the instructions) and `[pronunciations]`
          (word = spoken guidance for the voice model). The repository is public,
          so this never lives in it:
          an agenix-decrypted path, root-readable, handed to the unit as a
          systemd credential. Null means the voice knows no one.
        '';
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
      // lib.optionalAttrs cfg.voice.enable {
        MENTAT_VOICE_PUBLIC_LIVEKIT_URL = cfg.voice.publicLivekitUrl;
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

        EnvironmentFile =
          if cfg.voice.enable
          then [ cfg.environmentFile cfg.voice.environmentFile ]
          else cfg.environmentFile;

        StateDirectory = "mentat";
        WorkingDirectory = "/var/lib/mentat";

        PrivateTmp = true;
        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectHome = true;
      }
      // lib.optionalAttrs cfg.voice.enable {
        # mentatd needs LIVEKIT_API_KEY/SECRET to mint join tokens and
        # MENTAT_PLACES_API_KEY for place search, but must never see the OpenAI
        # credential that shares the voice agent's secrets file.
        UnsetEnvironment = [ "OPENAI_API_KEY" ];
      };
    };

    systemd.services.mentat-public = lib.mkIf cfg.public.enable {
      description = "mentat public OAuth MCP front";
      after = [ "network-online.target" "mentatd.service" ];
      wants = [ "network-online.target" "mentatd.service" ];
      wantedBy = [ "multi-user.target" ];

      environment = {
        ACCESS_CONFIG_URL = cfg.public.accessConfigUrl;
        MCP_SERVER_URL = cfg.public.baseUrl;
        MENTAT_PUBLIC_LISTEN = "127.0.0.1:${toString cfg.public.listenPort}";
        MENTAT_PUBLIC_BACKEND = "http://127.0.0.1:${toString cfg.listenPort}/mcp";
        HOME = "/var/lib/mentat-public";
      };

      # PID 1 opens the host's agenix paths and exposes only these three files
      # to the transient service identity. Secrets never enter the Nix store or
      # the generated unit's Environment= block.
      serviceConfig = {
        Type = "simple";
        DynamicUser = true;
        Restart = "always";
        RestartSec = "5s";
        ExecStart = "${pkgs.runtimeShell} -c 'export ACCESS_CLIENT_ID=\"$(cat \"$CREDENTIALS_DIRECTORY/access-client-id\")\"; export ACCESS_CLIENT_SECRET=\"$(cat \"$CREDENTIALS_DIRECTORY/access-client-secret\")\"; export MCP_JWT_SECRET=\"$(cat \"$CREDENTIALS_DIRECTORY/jwt-secret\")\"; exec ${lib.getExe' publicEnvPackage "python"} ${publicSource}/front.py'";

        LoadCredential = [
          "access-client-id:${cfg.public.accessClientIdFile}"
          "access-client-secret:${cfg.public.accessClientSecretFile}"
          "jwt-secret:${cfg.public.jwtSecretFile}"
        ];

        StateDirectory = "mentat-public";
        WorkingDirectory = "/var/lib/mentat-public";

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

    systemd.services.mentat-voice = lib.mkIf cfg.voice.enable {
      description = "mentat LiveKit voice agent";
      # The SFU is deployed outside this flake, so its unit name is not ours
      # to depend on; the agent retries the websocket on its own either way.
      after = [ "network.target" "mentatd.service" ];
      wants = [ "mentatd.service" ];
      wantedBy = [ "multi-user.target" ];

      environment = {
        LIVEKIT_URL = cfg.voice.livekitUrl;
        MENTAT_URL = cfg.voice.mentatUrl;
        # The agents CLI writes caches under $HOME; give it the state
        # directory rather than weakening ProtectHome. Its own directory, not
        # the daemon's: nothing here belongs next to the SDK's ~/.claude.
        HOME = "/var/lib/mentat-voice";
        XDG_CACHE_HOME = "/var/lib/mentat-voice/cache";
      } // lib.optionalAttrs (cfg.voice.privateContextFile != null) {
        # %d is systemd's credentials directory, populated by LoadCredential
        # below and readable by the transient user — the only way a
        # root-owned secret reaches a DynamicUser service.
        MENTAT_VOICE_PRIVATE = "%d/private";
      };

      serviceConfig = {
        Type = "simple";
        # Its own identity, not the daemon's: the agent only ever reaches
        # mentatd over HTTP, so sharing mentat's UID would buy nothing and
        # hand a compromised agent /var/lib/mentat — the SDK's ~/.claude and
        # the session state. DynamicUser needs no users.users entry and pairs
        # with StateDirectory below, which it manages under /var/lib/private.
        # The secrets file stays readable: EnvironmentFile is opened by PID 1
        # before the unit drops to the transient UID.
        DynamicUser = true;
        Restart = "always";
        RestartSec = "5s";

        EnvironmentFile = cfg.voice.environmentFile;
        LoadCredential = lib.optional (cfg.voice.privateContextFile != null)
          "private:${cfg.voice.privateContextFile}";
        # `start` is the agents CLI's production mode (dev enables reload and
        # debug logging). livekit-agents ships no console script, so the
        # interpreter runs the file directly.
        ExecStart = "${lib.getExe' cfg.voice.package "python"} ${cfg.voice.agentScript} start";

        StateDirectory = "mentat-voice";
        WorkingDirectory = "/var/lib/mentat-voice";

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
