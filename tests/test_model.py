import importlib
import unittest

import torch

torch.set_num_threads(1)


def api():
    try:
        return importlib.import_module("hiantitoxinpred.model")
    except ModuleNotFoundError:
        return None


class ModelContractTests(unittest.TestCase):
    def setUp(self):
        self.api = api()
        self.assertIsNotNone(self.api, "Standalone model API is not implemented")
        self.config = self.api.ModelConfig(
            input_dim=16, hidden_dim=32, heads=4, layers=2
        )
        self.model = self.api.HiAntitoxinPred(self.config).eval()
        torch.manual_seed(1)
        self.x = torch.randn(2, 20, 16)
        self.mask = torch.arange(20)[None, :] < torch.tensor([20, 13])[:, None]
        self.x[~self.mask] = 0

    def test_logits_and_explanation_contract(self):
        with torch.no_grad():
            y = self.model(self.x, self.mask)
            info = self.model.explain_features(self.x, self.mask)
        self.assertEqual(tuple(y.shape), (2,))
        self.assertTrue(torch.isfinite(y).all())
        torch.testing.assert_close(y, info["logits"], atol=0, rtol=0)
        self.assertEqual(tuple(info["branch_weights"].shape), (2, 6))
        torch.testing.assert_close(info["branch_weights"].sum(1), torch.ones(2))
        self.assertTrue((info["global_attention"][~self.mask] == 0).all())
        self.assertFalse(
            any(
                "aux" in n or "gate_network" in n
                for n, _ in self.model.named_parameters()
            )
        )

    def test_reject_empty_protein(self):
        self.mask[0] = False
        with self.assertRaisesRegex(ValueError, "empty"):
            self.model(self.x, self.mask)

    def test_reject_nonprefix_mask(self):
        self.mask[0, 3] = False
        with self.assertRaisesRegex(ValueError, "prefix"):
            self.model(self.x, self.mask)

    def test_reject_nonfinite_input(self):
        self.x[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            self.model(self.x, self.mask)

    def test_reject_nonzero_padding(self):
        self.x[1, -1, 0] = 1
        with self.assertRaisesRegex(ValueError, "zero"):
            self.model(self.x, self.mask)

    def test_reject_overlength(self):
        with self.assertRaisesRegex(ValueError, "200"):
            self.model(torch.zeros(2, 201, 16), torch.ones(2, 201, dtype=torch.bool))

    def test_reject_unsupported_one_position_geometry(self):
        # This extraction preserves the registered even-kernel boundary.
        # The later collator must provide width>=2 for one-residue inputs.
        with self.assertRaisesRegex(ValueError, "width"):
            self.model(torch.zeros(2, 1, 16), torch.ones(2, 1, dtype=torch.bool))

    def test_reject_incompatible_config(self):
        with self.assertRaises(ValueError):
            self.api.ModelConfig(hidden_dim=30, heads=8)

    def test_effective_parameters_receive_gradients(self):
        self.model.train()
        self.model(self.x, self.mask).sum().backward()
        missing = [n for n, p in self.model.named_parameters() if p.grad is None]
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
