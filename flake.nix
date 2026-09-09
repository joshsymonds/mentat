{
  description = "mentat: personal assistant daemon — Claude as the brain, your infrastructure as the body";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }: let
    system = "x86_64-linux";
    pkgs = nixpkgs.legacyPackages.${system};
    pkgsAndroid = import nixpkgs {
      inherit system;
      config = {
        allowUnfree = true;
        android_sdk.accept_license = true;
      };
    };
    # "latest" is a moving selector: it re-resolves against whatever the
    # nixpkgs repo data happens to carry, so a pinned flake.lock would still
    # hand out a different toolchain over time. 20.0 is what this lock
    # resolves to today.
    cmdLineToolsVersion = "20.0";
    platformToolsVersion = "36.0.0";
    buildToolsVersion = "36.0.0";
    platformVersion = "36";

    androidPackages = extra: pkgsAndroid.androidenv.composeAndroidPackages ({
      inherit cmdLineToolsVersion platformToolsVersion;
      buildToolsVersions = [ buildToolsVersion ];
      platformVersions = [ platformVersion ];
    } // extra);

    # Compiling and running JVM unit tests needs the SDK only; the emulator
    # and the google_apis system image are gigabytes of closure that nothing
    # but the e2e and live lanes ever boots.
    androidUnitComposition = androidPackages { };

    androidComposition = androidPackages {
      includeEmulator = true;
      emulatorVersion = "36.1.9";
      includeSystemImages = true;
      systemImageTypes = [ "google_apis" ];
      abiVersions = [ "x86_64" ];
    };

    androidShell = composition: let
      sdk = "${composition.androidsdk}/libexec/android-sdk";
      aapt2 = "${sdk}/build-tools/${buildToolsVersion}/aapt2";
    in pkgsAndroid.mkShell {
      packages = [
        composition.androidsdk
        pkgsAndroid.gradle
        pkgsAndroid.jdk17
      ];
      ANDROID_HOME = sdk;
      ANDROID_SDK_ROOT = sdk;
      JAVA_HOME = "${pkgsAndroid.jdk17}";
      GRADLE_OPTS = "-Dandroid.aapt2FromMavenOverride=${aapt2} -Dorg.gradle.project.android.aapt2FromMavenOverride=${aapt2}";
    };

    mentatd = pkgs.buildNpmPackage {
      pname = "mentatd";
      version = "3.0.0";
      src = self;

      npmDepsHash = "sha256-sHWBsyiaqtQXsUd7qS8dvebXf5EMDdJPkJSHb7qAdUA=";

      # Install-time npm must agree with the runtime wrappers below.
      nodejs = pkgs.nodejs_24;

      # No build step: Node 24 runs the TypeScript sources directly via
      # type stripping (the repo has no compile output by design).
      dontNpmBuild = true;

      nativeBuildInputs = [ pkgs.makeWrapper ];

      installPhase = ''
        runHook preInstall
        npm prune --omit=dev
        mkdir -p $out/lib/mentat/scripts
        cp -r src node_modules package.json $out/lib/mentat/
        cp scripts/daily-reminder.ts $out/lib/mentat/scripts/
        makeWrapper ${pkgs.nodejs_24}/bin/node $out/bin/mentatd \
          --add-flags $out/lib/mentat/src/main.ts
        makeWrapper ${pkgs.nodejs_24}/bin/node $out/bin/mentat-reminder \
          --add-flags $out/lib/mentat/scripts/daily-reminder.ts
        runHook postInstall
      '';

      meta = {
        description = "Personal assistant daemon supervising Claude Code sessions";
        mainProgram = "mentatd";
      };
    };
    # Python interpreter with livekit-agents + the Silero VAD plugin, for the
    # voice agent. Built from upstream wheels; see nix/voice-env.nix.
    voice-env = import ./nix/voice-env.nix { inherit pkgs; };
    public-env = import ./nix/public-env.nix { inherit pkgs; };
  in {
    devShells.${system} = {
      # Emulator-bearing shell: the e2e and live lanes boot a device.
      android = androidShell androidComposition;
      # Lean shell for lint + JVM unit tests, in CI and `just test-android`.
      android-unit = androidShell androidUnitComposition;
    };

    packages.${system} = {
      mentatd = mentatd;
      default = mentatd;
      voice-env = voice-env;
      public-env = public-env;
    };

    nixosModules.default = import ./nix/module.nix {
      mentatdPackage = mentatd;
      voiceEnvPackage = voice-env;
      publicEnvPackage = public-env;
    };

    checks.${system} = {
      build = mentatd;

      # Pure-eval smoke test of the NixOS module: instantiating the config
      # forces every option default and the env-assembly logic without
      # building a system. Wrong types, missing reads, or bad merges fail
      # here at `nix flake check` time.
      module-eval = let
        lib = nixpkgs.lib;

        # What ultraviolet already deploys; each case below adds to it, so a
        # regression in the base config fails every case at once.
        base = {
          enable = true;
          claudePackage = pkgs.hello; # any package with a bin; eval-only
          environmentFile = "/run/agenix/mentat-env";
          maxBudgetUsd = 2.0;
          mcpConfig.shimmer = {
            type = "http";
            url = "http://127.0.0.1:8001/mcp";
          };
          reminder.enable = true;
        };

        evalMentat = extra: (lib.nixosSystem {
          inherit system;
          modules = [
            self.nixosModules.default
            { services.mentat = base // extra; }
          ];
        }).config;

        deployed = evalMentat { };
        withVoice = evalMentat {
          voice = {
            enable = true;
            environmentFile = "/run/agenix/mentat-voice-env";
            publicLivekitUrl = "wss://ultraviolet.tail82223.ts.net:7443";
          };
        };

        withPublic = evalMentat {
          public = {
            enable = true;
            baseUrl = "https://mcp.example.com";
            accessConfigUrl = "https://mcp.example.com/.well-known/openid-configuration";
            accessClientIdFile = "/run/agenix/mentat-public-client-id";
            accessClientSecretFile = "/run/agenix/mentat-public-client-secret";
            jwtSecretFile = "/run/agenix/mentat-public-jwt-secret";
          };
        };

        observed = {
          daemonEnv = deployed.systemd.services.mentatd.environment;
          daemonService = deployed.systemd.services.mentatd.serviceConfig;
          daemonUser = deployed.systemd.services.mentatd.serviceConfig.User;
          reminderEnv = deployed.systemd.services.mentat-reminder.environment;
          reminderTimer = deployed.systemd.timers.mentat-reminder.timerConfig;
          voiceDaemonEnv = withVoice.systemd.services.mentatd.environment;
          voiceDaemonService = withVoice.systemd.services.mentatd.serviceConfig;
          voiceEnv = withVoice.systemd.services.mentat-voice.environment;
          voiceExecStart = withVoice.systemd.services.mentat-voice.serviceConfig.ExecStart;
          voiceService = withVoice.systemd.services.mentat-voice.serviceConfig;
          publicEnv = withPublic.systemd.services.mentat-public.environment;
          publicService = withPublic.systemd.services.mentat-public.serviceConfig;
          publicExecStart = withPublic.systemd.services.mentat-public.serviceConfig.ExecStart;
        };
      in
      # The voice sub-block defaults OFF: the config ultraviolet deploys today
      # must gain no unit until it opts in explicitly.
      assert lib.assertMsg (!(deployed.systemd.services ? mentat-voice))
        "services.mentat.voice must default off; enabling the daemon rendered mentat-voice";
      assert lib.assertMsg (!(observed.daemonEnv ? MENTAT_VOICE_PUBLIC_LIVEKIT_URL))
        "voice-off mentatd unexpectedly received MENTAT_VOICE_PUBLIC_LIVEKIT_URL";
      assert lib.assertMsg (observed.daemonService.EnvironmentFile == "/run/agenix/mentat-env")
        "voice-off mentatd EnvironmentFile changed: ${builtins.toJSON observed.daemonService.EnvironmentFile}";
      assert lib.assertMsg (!(observed.daemonService ? UnsetEnvironment))
        "voice-off mentatd unexpectedly strips inference credentials";
      # The public front is opt-in and keeps all credentials in systemd's
      # credential directory rather than the generated Environment= block.
      assert lib.assertMsg (withPublic.systemd.services ? mentat-public)
        "enabling services.mentat.public did not render mentat-public";
      assert lib.assertMsg (builtins.removeAttrs observed.publicEnv [ "PATH" ] == {
        ACCESS_CONFIG_URL = "https://mcp.example.com/.well-known/openid-configuration";
        MCP_SERVER_URL = "https://mcp.example.com";
        MENTAT_PUBLIC_LISTEN = "127.0.0.1:8486";
        MENTAT_PUBLIC_BACKEND = "http://127.0.0.1:8484/mcp";
        HOME = "/var/lib/mentat-public";
      })
        "public front environment changed: ${builtins.toJSON observed.publicEnv}";
      assert lib.assertMsg (observed.publicService.DynamicUser or false)
        "public front must run as a DynamicUser";
      assert lib.assertMsg (observed.publicService.StateDirectory == "mentat-public")
        "public front StateDirectory changed: ${toString observed.publicService.StateDirectory}";
      assert lib.assertMsg (observed.publicService.LoadCredential == [
        "access-client-id:/run/agenix/mentat-public-client-id"
        "access-client-secret:/run/agenix/mentat-public-client-secret"
        "jwt-secret:/run/agenix/mentat-public-jwt-secret"
      ])
        "public front credentials changed: ${builtins.toJSON observed.publicService.LoadCredential}";
      assert lib.assertMsg (observed.publicService.PrivateTmp && observed.publicService.NoNewPrivileges && observed.publicService.ProtectSystem == "strict" && observed.publicService.ProtectHome)
        "public front hardening changed";
      assert lib.assertMsg (lib.hasSuffix "/front.py'" observed.publicExecStart)
        "public front does not execute the packaged front.py: ${observed.publicExecStart}";
      assert lib.assertMsg (observed.publicEnv.MENTAT_PUBLIC_LISTEN == "127.0.0.1:8486")
        "public front is not loopback-bound: ${observed.publicEnv.MENTAT_PUBLIC_LISTEN}";
      # Voice opts mentatd into the signing credentials and public client URL,
      # while keeping the inference credentials confined to the agent.
      assert lib.assertMsg
        (observed.voiceDaemonEnv.MENTAT_VOICE_PUBLIC_LIVEKIT_URL == "wss://ultraviolet.tail82223.ts.net:7443")
        "mentatd public LiveKit URL mismatch: ${observed.voiceDaemonEnv.MENTAT_VOICE_PUBLIC_LIVEKIT_URL}";
      assert lib.assertMsg
        (observed.voiceDaemonService.EnvironmentFile == [
          "/run/agenix/mentat-env"
          "/run/agenix/mentat-voice-env"
        ])
        "voice mentatd EnvironmentFile mismatch: ${builtins.toJSON observed.voiceDaemonService.EnvironmentFile}";
      assert lib.assertMsg
        (observed.voiceDaemonService.UnsetEnvironment == [
          "LIVEKIT_INFERENCE_API_KEY"
          "LIVEKIT_INFERENCE_API_SECRET"
        ])
        "voice mentatd UnsetEnvironment mismatch: ${builtins.toJSON observed.voiceDaemonService.UnsetEnvironment}";
      # The agent-side URLs still come from their own settings, and the unit
      # runs the agent as a livekit worker.
      assert lib.assertMsg (observed.voiceEnv.LIVEKIT_URL == "ws://127.0.0.1:7880")
        "voice LIVEKIT_URL default changed: ${observed.voiceEnv.LIVEKIT_URL}";
      assert lib.assertMsg (observed.voiceEnv.MENTAT_URL == "http://127.0.0.1:8484")
        "voice MENTAT_URL default changed: ${observed.voiceEnv.MENTAT_URL}";
      assert lib.assertMsg (lib.hasSuffix "/agent.py start" observed.voiceExecStart)
        "voice unit does not start the agent worker: ${observed.voiceExecStart}";
      # The agent reaches mentatd over HTTP and shares nothing else with it, so
      # it gets its own identity: under the daemon's UID a compromised agent
      # would inherit /var/lib/mentat — the SDK's ~/.claude and the session
      # state — for free.
      assert lib.assertMsg (observed.voiceService.DynamicUser or false)
        "voice unit must run under its own identity (DynamicUser)";
      assert lib.assertMsg
        (!(observed.voiceService ? User) && !(observed.voiceService ? Group))
        "voice unit pins a static User/Group; it must not share mentatd's identity";
      # Guards the two above from going vacuous: they only separate anything
      # while mentatd itself is still the static mentat user.
      assert lib.assertMsg (observed.daemonUser == "mentat")
        "mentatd no longer runs as mentat: ${observed.daemonUser}";

      pkgs.runCommand "mentat-module-eval" {} ''
        cat > $out <<'EOF'
        ${builtins.toJSON observed}
        EOF
      '';
    };
  };
}
