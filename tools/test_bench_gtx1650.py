"""Benchmark output checks; no live model or GPU required."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_gtx1650 as bench


class BenchmarkTest(unittest.TestCase):
    def test_summary_keeps_errors_caching_and_variability(self):
        rows = []
        for rate, cached in [(4.0, 0), (6.0, 12)]:
            rows.append({"case": "english", "wall_s": 3.0, "response": {
                "timings": {"predicted_per_second": rate, "prompt_per_second": 20.0, "cache_n": cached},
                "usage": {"completion_tokens": 64}}})
        rows.append({"case": "english", "wall_s": 1.0, "error": "request failed"})
        summary = bench.summarize(rows)["english"]
        self.assertEqual(summary["median_decode_tok_s"], 5.0)
        self.assertEqual(summary["decode_tok_s"], [4.0, 6.0])
        self.assertEqual(summary["cached_tokens"], [0, 12])
        self.assertEqual(summary["runs"], 2)
        self.assertEqual(summary["errors"], 1)

    def test_failed_requests_are_saved_and_existing_results_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.jsonl"
            argv = ["bench", "--label", "test", "--rounds", "1", "--out", str(path)]
            def fake_request(url, body=None, timeout=240):
                if body is None:
                    return {"loaded": True, "model": "test-model"}
                raise RuntimeError("simulated error")
            with patch.object(sys, "argv", argv), patch.object(bench, "request", fake_request), patch("builtins.print"):
                self.assertEqual(bench.main(), 1)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(sum("error" in r for r in rows), len(bench.CASES))
            original = path.read_bytes()
            with patch.object(sys, "argv", argv), patch.object(bench, "request", fake_request):
                with self.assertRaises(FileExistsError):
                    bench.main()
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
