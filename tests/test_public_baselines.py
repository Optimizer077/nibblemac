"""Offline tests for pinned public weight setup and argument validation."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from bench_public_macos import parse_metrics
spec = importlib.util.spec_from_file_location("public_setup", ROOT / "tools/setup_public_baselines.py")
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


class PublicSetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.payload = b"test public model fixture"
        self.digest = hashlib.sha256(self.payload).hexdigest()
        self.directory = self.root / "models/public-baselines"
        self.directory.mkdir(parents=True)
        self.target = self.directory / setup.FILE
        self.root_patch = patch.object(setup, "ROOT", self.root)
        self.hash_patch = patch.object(setup, "EXPECTED", self.digest)
        self.root_patch.start()
        self.hash_patch.start()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(self.root_patch.stop)
        self.addCleanup(self.hash_patch.stop)

    def test_existing_verified_file_does_not_download(self):
        self.target.write_bytes(self.payload)
        with patch.object(setup.subprocess, "run") as run:
            setup.main()
            run.assert_not_called()
        source = json.loads((self.directory / "gguf-source.json").read_text())
        self.assertEqual(source["sha256"], self.digest)
        self.assertEqual(source["bytes"], len(self.payload))
        self.assertIn(setup.REVISION, source["url"])

    def test_bad_digest_preserves_file_without_provenance(self):
        self.target.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            setup.main()
        self.assertEqual(self.target.read_bytes(), b"corrupt")
        self.assertFalse((self.directory / "gguf-source.json").exists())

    def test_download_is_pinned_and_resumable(self):
        def download(argv, check):
            self.assertTrue(check)
            self.assertIn("--fail", argv)
            self.assertIn("--continue-at", argv)
            self.assertIn(setup.REVISION, argv[-1])
            Path(argv[argv.index("--output") + 1]).write_bytes(self.payload)
        with patch.object(setup.subprocess, "run", side_effect=download):
            setup.main()
        self.assertEqual(self.target.read_bytes(), self.payload)
        self.assertFalse(self.target.with_name(self.target.name + ".partial").exists())

    def test_failed_download_keeps_partial(self):
        partial = self.target.with_name(self.target.name + ".partial")
        partial.write_bytes(b"partial")
        with patch.object(setup.subprocess, "run", side_effect=subprocess.CalledProcessError(22, "curl")):
            with self.assertRaises(subprocess.CalledProcessError):
                setup.main()
        self.assertEqual(partial.read_bytes(), b"partial")
        self.assertFalse(self.target.exists())

    def test_recorder_rejects_invalid_protocol(self):
        for flags in (("--rounds", "0"), ("--generate", "1"),
                      ("--quality-chunks", "-1"), ("--label", "../unsafe")):
            with self.subTest(flags=flags):
                result = subprocess.run([sys.executable, str(ROOT / "tools/bench_public_macos.py"), *flags],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertIn("invalid arguments", result.stderr)


class PublicMetricsTests(unittest.TestCase):
    def metrics(self, **changes):
        values = {"prefill_tps": 100, "decode_tps": 50, "peak_rss_mb": 400, "generated_ids": [1, 2]}
        values.update(changes)
        return "RESULT " + json.dumps(values)

    def test_valid_public_metrics(self):
        self.assertEqual(parse_metrics(self.metrics(), "", "mlx-q4-g32", 2, 0)["decode_tps"], 50)

    def test_missing_or_duplicated_result(self):
        for stdout in ("", self.metrics() + "\n" + self.metrics()):
            with self.assertRaises(ValueError):
                parse_metrics(stdout, "", "mlx-q4-g32", 2, 0)

    def test_gpu_execution_must_be_confirmed(self):
        for cfg, stderr in (("llama-metal", ""), ("llama-metal", "offloaded 0/25 layers to GPU"),
                            ("nibblemac-metal-head6", "")):
            with self.subTest(cfg=cfg, stderr=stderr), self.assertRaises(ValueError):
                parse_metrics(self.metrics(), stderr, cfg, 2, 0)
        parse_metrics(self.metrics(), "offloaded 25/25 layers to GPU", "llama-metal", 2, 0)

    def test_bad_count_or_ids(self):
        for ids in ([1], [1, -1], [1, 2.5]):
            with self.assertRaises(ValueError):
                parse_metrics(self.metrics(generated_ids=ids), "", "mlx-q4-g32", 2, 0)

    def test_nonfinite_or_nonpositive_metric(self):
        for value in (float("nan"), float("inf"), 0, -1):
            with self.assertRaises(ValueError):
                parse_metrics(self.metrics(decode_tps=value), "", "mlx-q4-g32", 2, 0)

    def test_nibblemac_requires_generated_line(self):
        with self.assertRaisesRegex(ValueError, "missing generated"):
            parse_metrics(self.metrics(), "", "nibblemac-cpu6", 2, 0)
        parse_metrics("generated: 1,2\n" + self.metrics(), "", "nibblemac-cpu6", 2, 0)

    def test_quality_count_and_finiteness(self):
        parse_metrics('RESULT {"ppl":18.5,"tokens_scored":2040}', "", "nibblemac-cpu6", 128, 8)
        for values in ({"ppl": 18.5, "tokens_scored": 4096}, {"ppl": float("nan"), "tokens_scored": 2040}):
            with self.assertRaises(ValueError):
                parse_metrics("RESULT " + json.dumps(values), "", "nibblemac-cpu6", 128, 8)


if __name__ == "__main__":
    unittest.main()
