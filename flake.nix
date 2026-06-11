{
  description = "mentat: personal assistant daemon — Claude as the brain, your infrastructure as the body";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }: let
    system = "x86_64-linux";
    pkgs = nixpkgs.legacyPackages.${system};

    mentatd = pkgs.buildNpmPackage {
      pname = "mentatd";
      version = "3.0.0";
      src = self;

      npmDepsHash = "sha256-paKWa25wqIqRiMTvYe+nPEEXE1FPspBN1eXJs1oK76w=";

      # No build step: Node 24 runs the TypeScript sources directly via
      # type stripping (the repo has no compile output by design).
      dontNpmBuild = true;

      nativeBuildInputs = [ pkgs.makeWrapper ];

      installPhase = ''
        runHook preInstall
        npm prune --omit=dev
        mkdir -p $out/lib/mentat
        cp -r src scripts node_modules package.json $out/lib/mentat/
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
  in {
    packages.${system} = {
      mentatd = mentatd;
      default = mentatd;
    };

    nixosModules.default = import ./nix/module.nix { mentatdPackage = mentatd; };

    checks.${system} = {
      build = mentatd;

      # Pure-eval smoke test of the NixOS module: instantiating the config
      # forces every option default and the env-assembly logic without
      # building a system. Wrong types, missing reads, or bad merges fail
      # here at `nix flake check` time.
      module-eval = let
        eval = nixpkgs.lib.nixosSystem {
          inherit system;
          modules = [
            self.nixosModules.default
            {
              services.mentat = {
                enable = true;
                claudePackage = pkgs.hello; # any package with a bin; eval-only
                environmentFile = "/run/agenix/mentat-env";
                maxBudgetUsd = 2.0;
                mcpConfig.shimmer = {
                  type = "http";
                  url = "http://127.0.0.1:8001/mcp";
                };
              };
            }
          ];
        };
        env = eval.config.systemd.services.mentatd.environment;
      in pkgs.runCommand "mentat-module-eval" {} ''
        cat > $out <<'EOF'
        ${builtins.toJSON env}
        EOF
      '';
    };
  };
}
