from __future__ import annotations

import unittest

import numpy as np

from hiantitoxinpred.metrics import compute_binary_classification_metrics, select_threshold_by_f1


class MetricsTest(unittest.TestCase):
    def test_confusion_matrix_fields(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.8, 0.7, 0.2])
        metrics = compute_binary_classification_metrics(y_true, scores, threshold=0.5)

        self.assertEqual(metrics["tn"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [1, 1]])

    def test_select_threshold_by_f1_returns_valid_threshold(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        scores = np.array([0.05, 0.10, 0.80, 0.90])
        threshold = select_threshold_by_f1(y_true, scores)

        self.assertGreaterEqual(threshold, 0.0)
        self.assertLessEqual(threshold, 1.0)


if __name__ == "__main__":
    unittest.main()
