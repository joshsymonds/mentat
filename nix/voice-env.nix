# Python environment for the LiveKit voice agent.
#
# livekit-agents 1.8.x and most of the LiveKit Python family are either absent
# from nixpkgs or pinned older than the agent line accepts, so they are built
# here from upstream wheels. The OpenTelemetry family and OpenAI client are here
# for the same reason. Everything else comes from nixpkgs.
#
# Wheels, not sdists, on purpose: the rtc SDK ships a prebuilt Rust FFI object,
# blingfire/local-inference ship prebuilt C++ extensions, and the pure-Python
# packages are pinned to exact upstream releases. Building the native packages
# from source means vendoring extra toolchains to arrive at the same bytes.
{ pkgs }:

let
  py = pkgs.python3Packages;

  turnDetectorModelRevision = "v0.4.1-intl";
  turnDetectorModelCommit = "87e35fcb1e60a569bea70346191c4886ea92e281";
  turnDetectorModelFile = name: hash: pkgs.fetchurl {
    url = "https://huggingface.co/livekit/turn-detector/resolve/${turnDetectorModelRevision}/${name}";
    inherit hash;
  };
  turnDetectorModelFiles = [
    { name = "added_tokens.json"; hash = "sha256-WLVLvjb8dS95okonHvZqCggwBUtN+tlL3nV9hRloBgs="; }
    { name = "config.json"; hash = "sha256-zhc5WaBK9+CjIZz5fucMxGNflycu6IXo/IVb3mASx4c="; }
    { name = "languages.json"; hash = "sha256-BaFuEeQoIXMzHb8UBiQEjlgacnukXDYb7TXjAUrWgYc="; }
    { name = "merges.txt"; hash = "sha256-iDHk8aBERxNA98CoPXvXEwaluGfpX9hw900MUwipBNU="; }
    { name = "onnx/model_q8.onnx"; hash = "sha256-vSwwd2iCE4odlaB/rdwTdW/ho1vvYyNQXxEk/KNJvJw="; }
    { name = "special_tokens_map.json"; hash = "sha256-doYudlJmuFqpRZdn4zy68Tlw8yeg6I0cZYRsLd06Hs0="; }
    { name = "tokenizer.json"; hash = "sha256-nFrgDmAriGDL14S6gqiqFOj+7OxpLnB2WQ0BTXt/2vo="; }
    { name = "tokenizer_config.json"; hash = "sha256-g99shhBsfzMplbPEVrxqQkzEbuNIv04K4bVBDGP0OT8="; }
    { name = "vocab.json"; hash = "sha256-yhDX6fs+0YV13R4neiV5wW0QjjLydDloSvoOELFECRA="; }
  ];
  turnDetectorModel = pkgs.runCommand "livekit-turn-detector-model-${turnDetectorModelRevision}" { } ''
    cache="$out/hub/models--livekit--turn-detector"
    snapshot="$cache/snapshots/${turnDetectorModelCommit}"
    mkdir -p "$snapshot" "$cache/refs"
    printf '%s' "${turnDetectorModelCommit}" > "$cache/refs/${turnDetectorModelRevision}"
    ${builtins.concatStringsSep "\n" (map (asset: ''
      install -D "${turnDetectorModelFile asset.name asset.hash}" "$snapshot/${asset.name}"
    '') turnDetectorModelFiles)}
  '';

  # Two of the wheels below are cp314 ABI-tagged. Nothing downstream would
  # explain a bare ImportError if nixpkgs' default interpreter moved, so say it
  # here instead.
  pythonVersionOk =
    pkgs.lib.assertMsg (py.python.pythonVersion == "3.14")
      "voice-env pins cp314 wheels; nixpkgs python3 is ${py.python.pythonVersion}. Re-pin the ABI-tagged wheels in nix/voice-env.nix.";

  # PyPI's wheel filenames use the underscored distribution name, which differs
  # from the package name for nearly everything here, hence `wheelName`.
  # Remaining attrs pass through to buildPythonPackage untouched.
  wheelPackage =
    {
      wheelName,
      version,
      hash,
      python ? "py3",
      abi ? "none",
      platform ? "any",
      ...
    }@args:
    py.buildPythonPackage (
      (builtins.removeAttrs args [
        "wheelName"
        "hash"
        "python"
        "abi"
        "platform"
      ])
      // {
        format = "wheel";
        src = py.fetchPypi {
          inherit
            version
            python
            abi
            platform
            hash
            ;
          pname = wheelName;
          format = "wheel";
          dist = python;
        };
      }
    );

  # Prebuilt native objects link only against libc and the gcc runtime
  # (verified with `patchelf --print-needed`), so autoPatchelfHook plus the
  # C++ standard library is the whole story.
  nativeWheel = {
    nativeBuildInputs = [ pkgs.autoPatchelfHook ];
    buildInputs = [ pkgs.stdenv.cc.cc.lib ];
  };

  # --- OpenTelemetry -------------------------------------------------------
  #
  # nixpkgs is on 1.34; livekit-agents needs >=1.39,<1.45 and imports
  # `ReadWriteLogRecord`, which does not exist before 1.39. The whole family
  # version-locks to itself (`opentelemetry-api==1.44.0` and friends), so it
  # moves as one set.

  otelVersion = "1.44.0";

  opentelemetry-api = wheelPackage {
    pname = "opentelemetry-api";
    wheelName = "opentelemetry_api";
    version = otelVersion;
    hash = "sha256-lLmMiTqRuIZX6qweO6iWGM24W+aRgZZwU1TzRyiyze8=";
    dependencies = [ py.typing-extensions ];
    pythonImportsCheck = [ "opentelemetry.trace" ];
  };

  opentelemetry-semantic-conventions = wheelPackage {
    pname = "opentelemetry-semantic-conventions";
    wheelName = "opentelemetry_semantic_conventions";
    version = "0.65b0"; # the 1.44.0 generation
    hash = "sha256-HKzeewrTBvhMXvCMPb4buvIBZbum+L/0O2cOVVoIa8s=";
    dependencies = [ opentelemetry-api py.typing-extensions ];
    pythonImportsCheck = [ "opentelemetry.semconv" ];
  };

  opentelemetry-sdk = wheelPackage {
    pname = "opentelemetry-sdk";
    wheelName = "opentelemetry_sdk";
    version = otelVersion;
    hash = "sha256-3wgcTGvP2xIR4+hhQDdnkmQxKKJfjXLR0nZ1k25+lq0=";
    dependencies = [
      opentelemetry-api
      opentelemetry-semantic-conventions
      py.typing-extensions
    ];
    pythonImportsCheck = [ "opentelemetry.sdk.trace" ];
  };

  opentelemetry-proto = wheelPackage {
    pname = "opentelemetry-proto";
    wheelName = "opentelemetry_proto";
    version = otelVersion;
    hash = "sha256-iYsVWg4VV6/YZ0ePthWOgSKkYynKC7jcU8xV6Y8Bf1Y=";
    dependencies = [ py.protobuf ];
    pythonImportsCheck = [ "opentelemetry.proto" ];
  };

  opentelemetry-exporter-otlp-proto-common = wheelPackage {
    pname = "opentelemetry-exporter-otlp-proto-common";
    wheelName = "opentelemetry_exporter_otlp_proto_common";
    version = otelVersion;
    hash = "sha256-mp/mG7pz2AKQS8mJ8da0p7HuQPBsQOmNb4WvZarrtpQ=";
    dependencies = [ opentelemetry-proto ];
    pythonImportsCheck = [ "opentelemetry.exporter.otlp.proto.common" ];
  };

  opentelemetry-exporter-otlp-proto-grpc = wheelPackage {
    pname = "opentelemetry-exporter-otlp-proto-grpc";
    wheelName = "opentelemetry_exporter_otlp_proto_grpc";
    version = otelVersion;
    hash = "sha256-ahpkXqGCovWUQMUfqDAdMJ8zJKj51l+DlVhLBktn7k4=";
    dependencies = [
      opentelemetry-api
      opentelemetry-exporter-otlp-proto-common
      opentelemetry-proto
      opentelemetry-sdk
      py.googleapis-common-protos
      py.grpcio
      py.typing-extensions
    ];
    pythonImportsCheck = [ "opentelemetry.exporter.otlp.proto.grpc" ];
  };

  opentelemetry-exporter-otlp-proto-http = wheelPackage {
    pname = "opentelemetry-exporter-otlp-proto-http";
    wheelName = "opentelemetry_exporter_otlp_proto_http";
    version = otelVersion;
    hash = "sha256-g4WS/Od0wci7e5oKf6y/qC4XvlqKTpTO8Qy4SuAmuuM=";
    dependencies = [
      opentelemetry-api
      opentelemetry-exporter-otlp-proto-common
      opentelemetry-proto
      opentelemetry-sdk
      py.googleapis-common-protos
      py.requests
      py.typing-extensions
    ];
    pythonImportsCheck = [ "opentelemetry.exporter.otlp.proto.http" ];
  };

  # Metadata-only package pulling in both transports; livekit-agents depends on
  # it by name even though it only imports the http one.
  opentelemetry-exporter-otlp = wheelPackage {
    pname = "opentelemetry-exporter-otlp";
    wheelName = "opentelemetry_exporter_otlp";
    version = otelVersion;
    hash = "sha256-SkmPqNj9i+no4tF1/lUko/5YHM/63YUJ24ZSal+5cFE=";
    dependencies = [
      opentelemetry-exporter-otlp-proto-grpc
      opentelemetry-exporter-otlp-proto-http
    ];
  };

  # --- LiveKit -------------------------------------------------------------

  livekit-protocol = wheelPackage {
    pname = "livekit-protocol";
    wheelName = "livekit_protocol";
    version = "1.1.22";
    hash = "sha256-XC7chDpI/iHQW4LGN8Pp65KoijS6Hiw5hXrKDmEFqE8=";
    dependencies = [ py.protobuf py.types-protobuf ];
    pythonImportsCheck = [ "livekit.protocol" ];
  };

  livekit-api = wheelPackage {
    pname = "livekit-api";
    wheelName = "livekit_api";
    version = "1.2.0";
    hash = "sha256-MH+OXPsDWMPKCRgUq3aK9ViWAiFRvNf5UZVMzvoDaiQ=";
    dependencies = [
      livekit-protocol
      py.aiohttp
      py.protobuf
      py.pyjwt
      py.types-protobuf
    ];
    pythonImportsCheck = [ "livekit.api" ];
  };

  # The rtc SDK. `livekit` is the PyPI name; the module is `livekit.rtc`, and
  # the 27MB liblivekit_ffi.so under its resources/ is the Rust core.
  livekit-rtc = wheelPackage (
    nativeWheel
    // {
      pname = "livekit";
      wheelName = "livekit";
      version = "1.1.18";
      platform = "manylinux_2_28_x86_64";
      hash = "sha256-3CRhtmj/pmylBNBciK9+pkA3L1NnKuNnFFQS43TWr/A=";
      dependencies = [
        py.aiofiles
        py.numpy
        py.protobuf
        py.types-protobuf
      ];
      pythonImportsCheck = [ "livekit.rtc" ];
    }
  );

  # Sentence segmentation for the agent's text pipeline. C++ extension.
  livekit-blingfire = wheelPackage (
    nativeWheel
    // {
      pname = "livekit-blingfire";
      wheelName = "livekit_blingfire";
      version = "1.1.0";
      python = "cp314";
      abi = "cp314";
      platform = "manylinux_2_24_x86_64.manylinux_2_28_x86_64";
      hash = "sha256-21k8BEo6/zivC01PPXOarUskx0DIAlW/+soHry+eZyE=";
      pythonImportsCheck = [ "livekit.blingfire" ];
    }
  );

  # Core (non-optional) dependency of livekit-agents. Self-contained: the
  # models it runs are compiled into the extension, so nothing is fetched at
  # runtime. Distinct from livekit-plugins-turn-detector, which pulls
  # transformers and downloads weights from HuggingFace on first use.
  livekit-local-inference = wheelPackage (
    nativeWheel
    // {
      pname = "livekit-local-inference";
      wheelName = "livekit_local_inference";
      version = "0.2.7";
      python = "cp314";
      abi = "cp314";
      platform = "manylinux_2_27_x86_64.manylinux_2_28_x86_64";
      hash = "sha256-epe8KTK3P3LFJGy4NQzyzX0h/GI3BT/HwaxkU1LtDz8=";
      pythonImportsCheck = [ "livekit.local_inference" ];
    }
  );

  # nixpkgs has 0.55.2; livekit-agents pins json-repair exactly.
  json-repair = wheelPackage {
    pname = "json-repair";
    wheelName = "json_repair";
    version = "0.60.1";
    hash = "sha256-um/5dPKovvL3doFEp/A/hwqBZEPwPaJ6Sc3Q7DGngEk=";
    pythonImportsCheck = [ "json_repair" ];
  };

  # OpenAI's realtime extra requires websockets <16; nixpkgs is already on 16,
  # so pin the newest compatible pure wheel instead of inheriting that version.
  websockets = wheelPackage {
    pname = "websockets";
    wheelName = "websockets";
    version = "15.0.1";
    hash = "sha256-96hm+8Hpe1xhfuQRbaqgm3IhAdSjwXDHh0ULpAn5c28=";
    pythonImportsCheck = [ "websockets" ];
  };

  # nixpkgs has 2.41.1; the OpenAI realtime plugin requires >=2.50, so this
  # pure wheel pins the requested 2.54.0 client and supplies its realtime extra.
  openai = wheelPackage {
    pname = "openai";
    wheelName = "openai";
    version = "2.54.0";
    hash = "sha256-iQiXiRl8zbh/FzoDFF7RWY0AeVIgyT6Wz3ErHL9eXys=";
    dependencies = [
      py.anyio
      py.distro
      py.httpx
      py.jiter
      py.pydantic
      py.sniffio
      py.tqdm
      py.typing-extensions
      websockets
    ];
    pythonImportsCheck = [ "openai" ];
  };

  livekit-agents = wheelPackage {
    pname = "livekit-agents";
    wheelName = "livekit_agents";
    version = "1.8.1";
    hash = "sha256-Pp5Ocy7OvPBRLV7sJWIaNlw1ciidHxNjltihxep44qw=";

    # nixpkgs builds upstream's nest-asyncio v1.6.0 tag, but that tag's own
    # metadata still reports 1.5.9 (upstream forgot the bump). The bound is
    # satisfied by the code; only the version string it self-reports is stale.
    pythonRelaxDeps = [ "nest-asyncio" ];

    dependencies = [
      json-repair
      livekit-api
      livekit-blingfire
      livekit-local-inference
      livekit-protocol
      livekit-rtc
      opentelemetry-api
      opentelemetry-exporter-otlp
      opentelemetry-sdk
      py.aiofiles
      py.aiohttp
      py.av
      py.certifi
      py.click
      py.colorama
      py.docstring-parser
      py.eval-type-backport
      py.nest-asyncio
      py.numpy
      py.pillow
      openai
      py.prometheus-client
      py.protobuf
      py.psutil
      py.pydantic
      py.pyjwt
      py.pyyaml
      py.sounddevice
      py.typer
      py.types-protobuf
      py.typing-extensions
      py.watchfiles
    ];

    pythonImportsCheck = [ "livekit.agents" ];
  };

  # Ships resources/silero_vad.onnx (2.3MB) inside the wheel, so VAD needs no
  # network at import or at first use.
  livekit-plugins-silero = wheelPackage {
    pname = "livekit-plugins-silero";
    wheelName = "livekit_plugins_silero";
    version = "1.8.1";
    hash = "sha256-CkGrNFSgECBzyzZFejsAUTPX6tHoNvVUHv1q8Wz7SmE=";
    dependencies = [
      livekit-agents
      py.numpy
      py.onnxruntime
    ];
    pythonImportsCheck = [ "livekit.plugins.silero" ];
  };

  # The OpenAI plugin is pure Python; pin its wheel with the agent line so the
  # realtime transport and its model helpers stay in lockstep.
  livekit-plugins-openai = wheelPackage {
    pname = "livekit-plugins-openai";
    wheelName = "livekit_plugins_openai";
    version = "1.8.1";
    hash = "sha256-Ftnhwf6qBq/elnhT89KHCb9RCaVh5KwzgWQfFiKoI+k=";
    dependencies = [
      livekit-agents
      openai
    ];
    pythonImportsCheck = [ "livekit.plugins.openai.realtime" ];
  };

  livekit-plugins-turn-detector = wheelPackage {
    pname = "livekit-plugins-turn-detector";
    wheelName = "livekit_plugins_turn_detector";
    version = "1.8.1";
    hash = "sha256-358qOnFoZZt6nrvkLtTzAdU9+WUfckAzT6BiwAOX47c=";
    dependencies = [
      livekit-agents
      py.jinja2
      py.onnxruntime
      py.transformers
    ];
    pythonImportsCheck = [ "livekit.plugins.turn_detector" ];
  };

  # The ElevenLabs plugin is pure Python and shares the pinned agent runtime.
  # Its upstream wheel is py3-none-any, so it is compatible with this cp314 env.
  livekit-plugins-elevenlabs = wheelPackage {
    pname = "livekit-plugins-elevenlabs";
    wheelName = "livekit_plugins_elevenlabs";
    version = "1.8.1";
    hash = "sha256-2HW1pViPU3ZZcgeDJOEtxsS49zHunhjF/PeTBV86w/o=";
    dependencies = [ livekit-agents ];
    pythonImportsCheck = [ "livekit.plugins.elevenlabs" ];
  };

  # Self-hosted noise suppression (DTLN, MIT) run in-process on the agent's
  # inbound audio. LiveKit's own Krisp models need LiveKit Cloud transport,
  # which this SFU is not. The ~4MB ONNX weights ship inside the wheel.
  livekit-plugins-dtln = wheelPackage {
    pname = "livekit-plugins-dtln";
    wheelName = "livekit_plugins_dtln";
    version = "0.1.5";
    hash = "sha256-lc8OrAug6ChDUii1p7GRvyXITodpJf9KmtvohVWsk0Q=";
    dependencies = [
      livekit-agents
      py.numpy
      py.onnxruntime
    ];
    pythonImportsCheck = [ "livekit.plugins.dtln" ];
  };

in
assert pythonVersionOk;

# livekit-agents ships no console scripts — agents are run as
# `python <agent>.py start`, so `bin/python` is the entry point that matters.
(pkgs.python3.withPackages (_: [
  livekit-agents
  livekit-plugins-silero
  livekit-plugins-openai
  livekit-plugins-dtln
  livekit-plugins-elevenlabs
  livekit-plugins-turn-detector
])).overrideAttrs
  (old: {
    # Nix builds are sandboxed without network access, so loading the VAD here
    # is what proves silero_vad.onnx is read out of the store rather than
    # downloaded on first use. A regression would otherwise only surface as a
    # hang on the deploy host, with no route to the network to recover.
    postBuild = (old.postBuild or "") + ''
      echo "checking the voice env resolves offline..."
      mv "$out/bin/python3.14" "$out/bin/python3.14.unwrapped"
      cat > "$out/bin/python3.14" <<'SH'
      #!/bin/sh
      export HF_HUB_CACHE="''${HF_HUB_CACHE:-${turnDetectorModel}/hub}"
      exec "$(dirname "$0")/python3.14.unwrapped" "$@"
      SH
      chmod +x "$out/bin/python3.14"
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $out/bin/python - <<'PY'
      import json
      from importlib.resources import files
      from pathlib import Path

      import livekit.agents
      from huggingface_hub import hf_hub_download
      from livekit.plugins import dtln, elevenlabs, silero
      from livekit.plugins.turn_detector.multilingual import _EUORunnerMultilingual

      assert livekit.agents.__version__.startswith("1.8."), livekit.agents.__version__
      detector = _EUORunnerMultilingual()
      detector.initialize()
      assert detector._session is not None
      assert detector._tokenizer is not None
      languages = hf_hub_download(
          "livekit/turn-detector", "languages.json", revision="${turnDetectorModelRevision}",
          local_files_only=True,
      )
      assert json.loads(Path(languages).read_text())
      # The key is synthetic; constructing the plugin must not make a request.
      elevenlabs.TTS(api_key="synthetic-test-key")

      onnx = Path(str(files("livekit.plugins.silero.resources") / "silero_vad.onnx")).resolve()
      assert onnx.is_file(), f"silero_vad.onnx missing: {onnx}"
      assert str(onnx).startswith("/nix/store/"), f"silero_vad.onnx outside the store: {onnx}"

      silero.VAD.load()
      # Same proof for the DTLN weights: constructing the processor runs its
      # warmup pass, which needs both ONNX files out of the store.
      dtln.noise_suppression()
      PY
    '';
  })
