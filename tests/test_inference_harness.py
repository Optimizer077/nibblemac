"""Configuration-parser checks for real-model benchmarking (no GPU/model needed)."""
import argparse
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from bench_inference_macos import config


class ConfigurationTests(unittest.TestCase):
    def test_cpu(self):
        value = config("cpu:6")
        self.assertEqual(value["backend"], "cpu")
        self.assertEqual(value["threads"], 6)
        self.assertIsNone(value["spec_k"])

    def test_baseline(self):
        self.assertEqual(config("cpu-baseline:4")["backend"], "cpu-baseline")

    def test_i8mm(self):
        self.assertEqual(config("cpu-i8mm:6")["backend"], "cpu-i8mm")
        self.assertEqual(config("cpu-i8mm-spec3:6")["spec_k"], 3)

    def test_speculative(self):
        value = config("metal-spec4:6:67108864:packed")
        self.assertEqual(value["backend"], "metal")
        self.assertEqual(value["spec_k"], 4)
        self.assertEqual(value["metal_kernel"], "packed")

    def test_default_kernel_is_measured_packed_variant(self):
        self.assertEqual(config("metal:4:67108864")["metal_kernel"], "packed")

    def test_graph_modes(self):
        for mode, mlp, rec in (("none", False, False), ("mlp", True, False),
                               ("rec", False, True), ("graph", True, True)):
            with self.subTest(mode=mode):
                value = config(f"metal:4:67108864:packed:{mode}")
                self.assertEqual(value["metal_mlp"], mlp)
                self.assertEqual(value["metal_recurrence"], rec)

    def test_tiled_variant_preserves_explicit_selection(self):
        value = config("metal:4:0:packed4:mlp")
        self.assertEqual(value["metal_kernel"], "packed4")
        self.assertTrue(value["metal_mlp"])
        self.assertFalse(value["metal_recurrence"])

    def test_malformed(self):
        for value in ("cuda:4", "cpu:0", "cpu:9", "cpu:x", "cpu:4:123", "metal:4",
                      "metal:4:-1", "metal:4:10:unknown", "cpu-spec5:4", "cpu", "cpu:4:1:row:extra",
                      "metal:4:1:packed:unknown"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                config(value)

    def test_split_percent(self):
        value = config("metal:6:67108864:packed:none:15")
        self.assertEqual(value["metal_split_percent"], 15)
        self.assertEqual(config("metal:6:67108864")["metal_split_percent"], 0)
        for invalid in ("-1", "100", "nan", "", "1.5"):
            with self.subTest(value=invalid), self.assertRaises(argparse.ArgumentTypeError):
                config(f"metal:6:67108864:packed:none:{invalid}")


if __name__ == "__main__":
    unittest.main()
