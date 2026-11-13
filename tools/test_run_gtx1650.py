"""Launcher config checks; no GPU, downloads or model weights required."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "run_gtx1650.py"


class LauncherTest(unittest.TestCase):
    def test_missing_inputs_do_not_write_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = subprocess.run([sys.executable, str(SCRIPT), "--data-dir", str(root),
                                     "--engine", str(root / "strata"), "--write-config-only"],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing", result.stderr)
            self.assertFalse((root / "gtx1650.json").exists())

    def test_config_uses_existing_files_and_opt_in(self):
        with tempfile.TemporaryDirectory(prefix="strata paths ") as tmp:
            root = Path(tmp)
            names = ["strata", "packs/iq3_xxs/dense.bin", "packs/iq3_xxs/index.txt",
                     "packs/iq3_xxs/tokenizer/tokenizer.json"]
            names += [f"models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-0000{i}-of-00002.gguf"
                      for i in (1, 2)]
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            original = root / "strata-iq3_xxs.json"
            original.write_text("user config")
            subprocess.run([sys.executable, str(SCRIPT), "--data-dir", str(root), "--engine", str(root / "strata"),
                            "--port", "18080", "--write-config-only"], check=True, capture_output=True)
            config = json.loads((root / "gtx1650.json").read_text())
            self.assertEqual(config["env"]["STRATA_WEIGHT_ARENA_HOST"], "1")
            self.assertEqual(config["host"], "127.0.0.1")
            self.assertEqual(config["port"], 18080)
            self.assertEqual(config["args"][config["args"].index("--max-context") + 1], "8192")
            self.assertNotIn("--vision", config["args"])
            self.assertNotIn("--draft-model", config["args"])
            self.assertNotIn("--mtp", config["args"])
            self.assertEqual(config["args"][config["args"].index("--prefill") + 1], "32")
            self.assertEqual(original.read_text(), "user config")
            stable_bytes = (root / "gtx1650.json").read_bytes()
            subprocess.run([sys.executable, str(SCRIPT), "--data-dir", str(root), "--engine", str(root / "strata"),
                            "--profile", "fast", "--no-prompt-cache", "--write-config-only"], check=True, capture_output=True)
            fast = json.loads((root / "gtx1650-fast.json").read_text())
            self.assertEqual((root / "gtx1650.json").read_bytes(), stable_bytes)
            self.assertEqual(fast["env"]["STRATA_SM75_PERF"], "1")
            self.assertNotIn("STRATA_HEAD_HOST", fast["env"])
            self.assertNotIn("STRATA_HC_Q8", fast["env"])
            self.assertEqual(fast["args"][fast["args"].index("--prefill") + 1], "256")
            self.assertEqual(fast["args"][fast["args"].index("--short-read") + 1], "0")
            self.assertEqual(fast["args"][fast["args"].index("--max-context") + 1], "8192")
            self.assertNotIn("--prompt-cache", config["args"])
            self.assertEqual(fast["args"][fast["args"].index("--prompt-cache") + 1], "0")


if __name__ == "__main__":
    unittest.main()
