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
        self.assertIn("HF_HUB_CACHE=", source)
        self.assertIn("HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1", source)
        self.assertIn("detector.initialize()", source)

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
