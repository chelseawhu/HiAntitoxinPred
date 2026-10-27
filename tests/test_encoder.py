import importlib, unittest
import numpy as np


class EncoderTests(unittest.TestCase):
    def setUp(self):
        try:
            self.api = importlib.import_module("hiantitoxinpred.encoder")
        except ModuleNotFoundError:
            self.api = None
        self.assertIsNotNone(self.api, "Encoder missing")

    def test_special_and_padding_removed(self):
        h = np.arange(2 * 4 * 1024, dtype=np.float32).reshape(2, 4, 1024)
        a = np.array([[1, 1, 1, 1], [1, 1, 0, 0]])
        s = np.array([[0, 0, 0, 1], [0, 1, 1, 1]])
        r = self.api.residue_arrays(h, a, s, [3, 1])
        np.testing.assert_array_equal(r[0], h[0, :3])
        np.testing.assert_array_equal(r[1], h[1, :1])
        with self.assertRaises(ValueError):
            self.api.residue_arrays(h, a, s, [2, 1])

    def test_cache_identity_contains_raw_sequence_and_encoder_mode(self):
        f = self.api.cache_key
        self.assertNotEqual(f("AB", {"dtype": "cpu"}), f("AX", {"dtype": "cpu"}))
        self.assertNotEqual(f("AB", {"dtype": "cpu"}), f("AB", {"dtype": "gpu"}))
        self.assertEqual(f("AB", {"a": 1, "b": 2}), f("AB", {"b": 2, "a": 1}))

    def test_loader_uses_the_hashed_bin_even_with_alternative_files(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        import torch
        from pathlib import Path

        model = torch.nn.Linear(1, 1)
        model.config = SimpleNamespace(d_model=1024)
        with (
            patch.object(
                self.api,
                "file_sha256",
                side_effect=lambda p: self.api.REGISTERED_CHECKSUMS[Path(p).name],
            ),
            patch("transformers.T5Tokenizer.from_pretrained"),
            patch(
                "transformers.T5EncoderModel.from_pretrained", return_value=(model, {})
            ) as loader,
        ):
            self.api.ProtT5Encoder("/unused-fixture", device="cpu")
            self.assertIs(loader.call_args.kwargs.get("use_safetensors"), False)

    def test_extra_tokenizer_files_are_rejected(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from types import SimpleNamespace
        import torch

        model = torch.nn.Linear(1, 1)
        model.config = SimpleNamespace(d_model=1024)
        for name in ("added_tokens.json", "tokenizer.json", "tokenizer.4.0.0.json"):
            with tempfile.TemporaryDirectory() as tmp:
                (Path(tmp) / name).write_text("{}")
                with (
                    patch.object(
                        self.api,
                        "file_sha256",
                        side_effect=lambda p: self.api.REGISTERED_CHECKSUMS[
                            Path(p).name
                        ],
                    ),
                    patch("transformers.T5Tokenizer.from_pretrained"),
                    patch(
                        "transformers.T5EncoderModel.from_pretrained",
                        return_value=(model, {}),
                    ),
                ):
                    with self.assertRaisesRegex(ValueError, "Unregistered tokenizer"):
                        self.api.ProtT5Encoder(tmp, device="cpu")

    def test_partial_cache_keeps_original_encoder_batch_geometry(self):
        import tempfile, json, torch
        from pathlib import Path
        from types import SimpleNamespace

        class Tokenizer:
            def __call__(self, strings, **kwargs):
                lengths = [len(s.split()) for s in strings]
                width = max(lengths) + 1
                a = torch.zeros(len(strings), width, dtype=torch.long)
                s = torch.ones_like(a)
                for i, n in enumerate(lengths):
                    a[i, : n + 1] = 1
                    s[i, :n] = 0
                return dict(
                    input_ids=torch.zeros_like(a),
                    attention_mask=a,
                    special_tokens_mask=s,
                )

        class Model:
            def __call__(self, **tokens):
                a = tokens["attention_mask"]
                return SimpleNamespace(
                    last_hidden_state=torch.full((*a.shape, 1024), float(a.shape[1]))
                )

        with tempfile.TemporaryDirectory() as tmp:
            encoder = self.api.ProtT5Encoder.__new__(self.api.ProtT5Encoder)
            encoder.device = torch.device("cpu")
            encoder.identity = {"fixture": True}
            encoder.cache = Path(tmp)
            encoder.tokenizer = Tokenizer()
            encoder.model = Model()
            expected = encoder.encode(["ACDE", "ACD"])
            for p in Path(tmp).glob("*.json"):
                if json.loads(p.read_text())["length"] == 3:
                    p.unlink()
            actual = encoder.encode(["ACDE", "ACD"])
            np.testing.assert_array_equal(actual[1], expected[1])
