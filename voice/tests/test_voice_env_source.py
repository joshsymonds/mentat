"""Source contract for the packaged LiveKit voice environment."""

import re
import unittest
from pathlib import Path


class VoiceEnvironmentSourceContractTest(unittest.TestCase):
    def test_turn_detector_plugin_and_model_are_pinned_for_offline_use(self):
        source = (Path(__file__).resolve().parents[2] / "nix" / "voice-env.nix").read_text()

        package = re.search(
            r"livekit-plugins-turn-detector\s*=\s*wheelPackage\s*\{(?P<body>.*?)\n  \};",
            source,
            re.DOTALL,
        )
        self.assertIsNotNone(package, "turn detector plugin must be packaged as an upstream wheel")
        body = package.group("body")
        self.assertRegex(body, r'pname\s*=\s*"livekit-plugins-turn-detector";')
        self.assertRegex(body, r'wheelName\s*=\s*"livekit_plugins_turn_detector";')
        self.assertRegex(body, r'version\s*=\s*"1\.8\.1";')
        self.assertRegex(
            body,
            r'hash\s*=\s*"sha256-358qOnFoZZt6nrvkLtTzAdU9\+WUfckAzT6BiwAOX47c=";',
        )
        self.assertIn('pythonImportsCheck = [ "livekit.plugins.turn_detector" ];', body)
        for dependency in ("livekit-agents", "py.jinja2", "py.onnxruntime", "py.transformers"):
            self.assertIn(dependency, body)

        runtime = source.split("(pkgs.python3.withPackages (_: [", 1)[1].split(
            "])).overrideAttrs", 1
        )[0]
        self.assertIn("livekit-plugins-turn-detector", runtime)
        self.assertIn('turnDetectorModelRevision = "v0.4.1-intl";', source)
        self.assertIn('name = "onnx/model_q8.onnx";', source)
        self.assertIn('name = "languages.json";', source)
        self.assertIn('name = "tokenizer.json";', source)
        self.assertIn('turnDetectorEnglishModelRevision = "v1.2.2-en";', source)
        self.assertIn('turnDetectorEnglishModelCommit = "ebcab0c09c2b62d926e92180d364df3aaae68a09";', source)
        self.assertIn("turnDetectorEnglishModelFiles", source)
        self.assertIn("--set HF_HUB_CACHE", source)
        self.assertIn('os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")', source)
        self.assertIn("runner.initialize()", source)

    def test_interpreter_wrapper_and_offline_detector_proof_are_isolated(self):
        source = (Path(__file__).resolve().parents[2] / "nix" / "voice-env.nix").read_text()

        self.assertIn("pkgs.makeWrapper", source)
        self.assertNotIn('$(dirname "$0")', source)
        self.assertIn(
            "from livekit.plugins.turn_detector.multilingual import MultilingualModel, _EUORunnerMultilingual",
            source,
        )
        self.assertIn("return MultilingualModel()", source)
        self.assertIn("runner = _EUORunnerMultilingual()", source)
        self.assertIn("runner.initialize()", source)
        self.assertIn("subprocess.run(", source)
        self.assertIn('cat > "$out/lib/python3.14/site-packages/livekit-model-cache.pth"', source)
        self.assertIn('os.environ["HF_HUB_CACHE"] = "${turnDetectorModel}/hub"', source)
        self.assertIn('"${pkgs.coreutils}/bin/env"', source)
        self.assertIn('Path(sys.executable).name == ".python3.14-wrapped"', source)
        self.assertIn("InferenceProcExecutor", source)
        self.assertIn('get_context("forkserver")', source)
        self.assertIn("executor.do_inference(", source)
        self.assertIn("runners = _InferenceRunner.registered_runners", source)
        self.assertIn('required_runners = {"lk_end_of_utterance_en", "lk_end_of_utterance_multilingual"}', source)
        self.assertIn("for runner_name in runners:", source)
        self.assertIn('os.environ.pop("HF_HUB_CACHE", None)', source)
        self.assertRegex(source, r'env -i .*HOME=.*XDG_CACHE_HOME=.*PATH=.*"\$out/bin/python"')
        self.assertIn('os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")', source)

    def test_elevenlabs_plugin_is_pinned_in_the_production_interpreter(self):
        source = (Path(__file__).resolve().parents[2] / "nix" / "voice-env.nix").read_text()

        package = re.search(
            r"livekit-plugins-elevenlabs\s*=\s*wheelPackage\s*\{(?P<body>.*?)\n  \};",
            source,
            re.DOTALL,
        )
        self.assertIsNotNone(package, "ElevenLabs plugin must be packaged as an upstream wheel")
        body = package.group("body")
        self.assertRegex(body, r'pname\s*=\s*"livekit-plugins-elevenlabs";')
        self.assertRegex(body, r'wheelName\s*=\s*"livekit_plugins_elevenlabs";')
        self.assertRegex(body, r'version\s*=\s*"1\.8\.1";')
        self.assertRegex(
            body,
            r'hash\s*=\s*"sha256-2HW1pViPU3ZZcgeDJOEtxsS49zHunhjF/PeTBV86w/o=";',
        )
        self.assertRegex(body, r"dependencies\s*=\s*\[\s*livekit-agents\s*\];")
        self.assertIn("pythonImportsCheck = [ \"livekit.plugins.elevenlabs\" ];", body)

        runtime = source.split("(pkgs.python3.withPackages (_: [", 1)[1].split(
            "])).overrideAttrs", 1
        )[0]
        self.assertIn("livekit-plugins-elevenlabs", runtime)
        self.assertIn("from livekit.plugins import dtln, elevenlabs, silero", source)
        self.assertIn('elevenlabs.TTS(api_key="synthetic-test-key")', source)


if __name__ == "__main__":
    unittest.main()
