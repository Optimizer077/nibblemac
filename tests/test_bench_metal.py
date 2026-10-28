"""Standard-library tests for benchmark gating and failure preservation; no GPU required."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("bench_metal", ROOT / "tools/bench_metal.py")
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


class RecorderTests(unittest.TestCase):
    def run_recorder(self, responses):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(sys, "argv", ["bench_metal.py", "--rounds", "1", "--output-dir", directory]), \
                 patch.object(bench.platform, "system", return_value="Darwin"), \
                 patch.object(bench.platform, "machine", return_value="arm64"), \
                 patch.object(bench, "text", return_value="test-metadata"), \
                 patch.object(bench, "command", side_effect=responses) as command, \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                rc = bench.main()
            files = list(Path(directory).glob("*.json"))
            self.assertEqual(len(files), 1)
            return rc, json.loads(files[0].read_text()), command.call_count

    @staticmethod
    def proc(rc=0, stdout="", stderr=""):
        return subprocess.CompletedProcess([], rc, stdout, stderr)

    def test_success_requires_correctness_and_all_measurements(self):
        measurements = "\n".join(json.dumps({"correct": True}) for _ in range(5))
        rc, record, calls = self.run_recorder([self.proc(), self.proc(), self.proc(stdout=measurements)])
        self.assertEqual(rc, 0)
        self.assertTrue(record["valid"])
        self.assertEqual(calls, 3)
        self.assertEqual(len(record["rounds"][0]), 5)
        self.assertIn("engine/metal/kernels.metal", record["source_sha256"])

    def test_build_failure_is_saved_and_stops(self):
        rc, record, calls = self.run_recorder([self.proc(2, stderr=f"{ROOT}/engine failed")])
        self.assertEqual(rc, 1)
        self.assertFalse(record["valid"])
        self.assertEqual(calls, 1)
        self.assertEqual(record["steps"][0]["stderr"], "<repo>/engine failed")

    def test_correctness_failure_prevents_timing(self):
        rc, record, calls = self.run_recorder([self.proc(), self.proc(1, stderr="bad output")])
        self.assertEqual(rc, 1)
        self.assertFalse(record["valid"])
        self.assertEqual(calls, 2)
        self.assertEqual(record["rounds"], [])

    def test_bad_json_is_saved_as_failure(self):
        rc, record, _ = self.run_recorder([self.proc(), self.proc(), self.proc(stdout="broken JSON")])
        self.assertEqual(rc, 1)
        self.assertFalse(record["valid"])
        self.assertIn("error", record)

    def test_incomplete_measurements_are_not_valid(self):
        rc, record, _ = self.run_recorder([
            self.proc(), self.proc(), self.proc(stdout=json.dumps({"correct": True}))
        ])
        self.assertEqual(rc, 1)
        self.assertFalse(record["valid"])

    def test_os_failure_is_saved(self):
        rc, record, _ = self.run_recorder([OSError("tool unavailable")])
        self.assertEqual(rc, 1)
        self.assertFalse(record["valid"])
        self.assertEqual(record["error"], "tool unavailable")


if __name__ == "__main__":
    unittest.main()
