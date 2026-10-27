import unittest
import importlib
import numpy as np


class MAIAMetricTests(unittest.TestCase):
    def setUp(self):
        try:
            self.api = importlib.import_module("hiantitoxinpred.maia_metrics")
        except ModuleNotFoundError:
            self.api = None
        self.assertIsNotNone(self.api)

    def test_fractional_ties_preserve_budget(self):
        np.testing.assert_allclose(
            self.api.top_weights([4, 3, 3, 1], 1.5), [1, 0.25, 0.25, 0]
        )

    def test_localization_and_pair_identity(self):
        metrics = self.api.region_metrics([4, 3, 2, 1], [1, 1, 0, 0])
        self.assertEqual(metrics, {"region_AP": 1.0, "top_D_overlap": 1.0})
        self.assertEqual(self.api.stability([[4, 3, 2, 1]] * 5)["mean"], 1.0)

    def test_bad_scores_and_masks_rejected(self):
        with self.assertRaises(ValueError):
            self.api.top_weights([np.nan, 1], 1)
        with self.assertRaises(ValueError):
            self.api.region_metrics([1, 2], [1, 1])
