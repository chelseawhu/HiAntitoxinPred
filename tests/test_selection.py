import importlib
import unittest


class SelectionTests(unittest.TestCase):
    def setUp(self):
        try:
            self.api = importlib.import_module("hiantitoxinpred.selection")
        except ModuleNotFoundError:
            self.api = None
        self.assertIsNotNone(self.api, "Loss selection API is not implemented")

    def test_small_drop_updates_checkpoint_without_resetting_wait(self):
        s = self.api.LossSelection(patience=2, min_delta=0.01)
        self.assertTrue(s.update(1, 1.0))
        self.assertTrue(s.update(2, 0.995))
        self.assertEqual(s.best_epoch, 2)
        self.assertEqual(s.wait, 1)
        self.assertTrue(s.update(3, 0.994))
        self.assertTrue(s.should_stop)
        self.assertEqual(s.best_epoch, 3)

    def test_tie_keeps_earlier_and_cumulative_improvement_resets_wait(self):
        s = self.api.LossSelection(patience=3, min_delta=0.01)
        s.update(1, 1.0)
        self.assertFalse(s.update(2, 1.0))
        self.assertEqual(s.best_epoch, 1)
        s.update(3, 0.995)
        s.update(4, 0.98)
        self.assertEqual(s.wait, 0)
        self.assertFalse(s.should_stop)

    def test_reject_invalid_or_post_stop_trajectory(self):
        for loss in (float("nan"), float("inf"), -0.1):
            with self.assertRaises(ValueError):
                self.api.LossSelection().update(1, loss)
        s = self.api.LossSelection(patience=1)
        with self.assertRaises(ValueError):
            s.update(2, 1.0)
        s.update(1, 1.0)
        s.update(2, 2.0)
        with self.assertRaises(ValueError):
            s.update(3, 0.5)

    def test_reject_invalid_control_parameters(self):
        for kwargs in (
            {"patience": 0},
            {"patience": 1.5},
            {"patience": True},
            {"min_delta": float("nan")},
            {"min_delta": -1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.api.LossSelection(**kwargs)
