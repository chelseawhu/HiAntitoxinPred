from __future__ import annotations

import unittest


class ImportSmokeTest(unittest.TestCase):
    def test_package_imports(self) -> None:
        import hiantitoxinpred
        from hiantitoxinpred.model import TrackBSecondGenRescueNet
        from hiantitoxinpred.training import CNNRescueTrainingConfig, build_cnn_rescue_model

        self.assertTrue(hiantitoxinpred.__version__)
        config = CNNRescueTrainingConfig(
            esm_dim=1280,
            hidden_dim=64,
            num_heads=4,
            num_layers=1,
            model_family="track_b_second_gen_rescue",
        )
        model = build_cnn_rescue_model(config, device="cpu")
        self.assertIsInstance(model, TrackBSecondGenRescueNet)


if __name__ == "__main__":
    unittest.main()
