import importlib
import unittest
import numpy as np


class MetricTests(unittest.TestCase):
    def setUp(self):
        try:
            self.api = importlib.import_module("hiantitoxinpred.metrics")
        except ModuleNotFoundError:
            self.api = None
        self.assertIsNotNone(self.api, "Metrics API is not implemented")

    def test_uses_raw_scores_even_when_sigmoid_saturates(self):
        r = self.api.binary_metrics([0, 1, 0, 1], [100.0, 101.0, 102.0, 103.0])
        self.assertAlmostEqual(r["AUROC"], 0.75)
        self.assertAlmostEqual(r["AP"], 5 / 6)
        self.assertAlmostEqual(r["precision"], 0.5)
        self.assertAlmostEqual(r["recall"], 1.0)
        self.assertAlmostEqual(r["F1"], 2 / 3)
        self.assertEqual(r["MCC"], 0.0)
        self.assertEqual(r["accuracy"], 0.5)
        self.assertEqual([r[k] for k in ("TP", "FP", "TN", "FN")], [2, 2, 0, 0])

    def test_zero_logit_is_positive(self):
        r = self.api.binary_metrics([0, 1], [-1.0, 0.0])
        self.assertEqual(r["F1"], 1.0)
        self.assertEqual(r["AP"], 1.0)

    def test_reject_invalid_vectors(self):
        pairs = [
            ([0, 1], [0]),
            ([0, 2], [0, 1]),
            ([0, 1], [np.nan, 1]),
            ([1, 1], [0, 1]),
            ([], []),
            ([[0, 1]], [[0, 1]]),
        ]
        for y, z in pairs:
            with self.subTest(y=y, z=z), self.assertRaises(ValueError):
                self.api.binary_metrics(y, z)

    def test_independent_run_summary_uses_sample_sd(self):
        r = self.api.summarize_runs([{"F1": 0.8, "AP": 0.9}, {"F1": 1.0, "AP": 1.0}])
        self.assertEqual(r["n"], 2)
        self.assertAlmostEqual(r["mean"]["F1"], 0.9)
        self.assertAlmostEqual(r["sd"]["F1"], np.sqrt(0.02))
        for rows in (
            [{"F1": 1.0}],
            [{"F1": 1.0}, {"AP": 1.0}],
            [{"F1": 1.0}, {"F1": np.nan}],
        ):
            with self.assertRaises(ValueError):
                self.api.summarize_runs(rows)
