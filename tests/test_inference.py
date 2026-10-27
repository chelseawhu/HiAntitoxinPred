import importlib
import unittest

import numpy as np
import torch
from hiantitoxinpred.model import HiAntitoxinPred, ModelConfig


class InferenceTests(unittest.TestCase):
    def setUp(self):
        try:
            self.api = importlib.import_module("hiantitoxinpred.inference")
        except ModuleNotFoundError:
            self.api = None
        self.assertIsNotNone(self.api, "Fixed-width inference API is not implemented")
        torch.manual_seed(42)
        self.model = HiAntitoxinPred(ModelConfig(input_dim=16, hidden_dim=32, heads=4))
        self.values = [torch.randn(n, 16) for n in (1, 2, 19, 20, 199, 200)]

    def test_fixed_width_order_and_batch_size(self):
        shapes = []
        hook = self.model.register_forward_pre_hook(
            lambda module, args: shapes.append(args[0].shape[1])
        )
        z = self.api.predict_features(self.model, self.values, batch_size=3)
        one = self.api.predict_features(self.model, self.values, batch_size=1)
        reverse = self.api.predict_features(
            self.model, self.values[::-1], batch_size=2
        )[::-1]
        hook.remove()
        self.assertEqual(z.shape, (6,))
        self.assertTrue(np.isfinite(z).all())
        self.assertEqual(set(shapes), {200})
        np.testing.assert_allclose(z, one, atol=1e-5, rtol=0)
        np.testing.assert_allclose(z, reverse, atol=1e-5, rtol=0)

    def test_inference_preserves_training_state_and_batchnorm_buffers(self):
        before = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.assertTrue(self.model.training)
        self.api.predict_features(self.model, self.values)
        self.assertTrue(self.model.training)
        for k, v in self.model.state_dict().items():
            torch.testing.assert_close(v, before[k], atol=0, rtol=0)

    def test_explicit_reference_padding(self):
        from hiantitoxinpred.data import collate_features

        values = self.values[:4]
        self.model.eval()
        x, mask = collate_features(values, width=136)
        with torch.inference_mode():
            expected = self.model(x, mask).numpy()
        actual = self.api.predict_features(
            self.model, values, batch_size=4, padding_width=136
        )
        np.testing.assert_array_equal(actual, expected)
        with self.assertRaises(ValueError):
            self.api.predict_features(self.model, values, padding_width=19)

    def test_invalid_inputs_fail_and_restore_mode(self):
        for batch_size in (0, True, 1.5):
            with self.assertRaises(ValueError):
                self.api.predict_features(
                    self.model, self.values, batch_size=batch_size
                )
        with self.assertRaises(ValueError):
            self.api.predict_features(self.model, [])
        with self.assertRaises(ValueError):
            self.api.predict_features(self.model, [torch.ones(201, 16)])
        self.assertTrue(self.model.training)
