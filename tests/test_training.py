import importlib, json, tempfile, unittest
from pathlib import Path
import numpy as np
import torch
from hiantitoxinpred.model import ModelConfig


class Tiny(torch.utils.data.Dataset):
    def __init__(self, split):
        self.split = split
        self.ids = [split + str(i) for i in range(4 if split == "train" else 3)]
        self.labels = [1, 0, 0, 0] if split == "train" else [1, 0, 1]
        self.identity = {"fixture": split}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.ids[i], torch.full((5 + i, 16), float(i) / 10), self.labels[i]


class TrainingTests(unittest.TestCase):
    def setUp(self):
        try:
            self.api = importlib.import_module("hiantitoxinpred.training")
        except ModuleNotFoundError:
            self.api = None
        self.assertIsNotNone(self.api, "Trainer not implemented")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name) / "run"

    def test_small_fit_l_lock_and_weighted_validation(self):
        c = self.api.TrainingConfig(max_epochs=2, batch_size=2)
        r = self.api.fit(
            Tiny("train"),
            Tiny("val"),
            self.out,
            c,
            ModelConfig(input_dim=16, hidden_dim=32, heads=4),
            seed=42,
        )
        self.assertEqual(r["stop_reason"], "max_epochs")
        self.assertAlmostEqual(r["pos_weight"], np.sqrt(3))
        lock = json.loads((self.out / "SELECTION_LOCK.json").read_text())
        self.assertFalse(lock["test_evaluated"])
        self.assertEqual(
            lock["L_epoch"],
            min(r["history"], key=lambda x: (x["val_loss"], x["epoch"]))["epoch"],
        )
        self.assertTrue((self.out / "L.pt").exists())
        self.assertEqual(len(list(self.out.glob("*.pt"))), 1)
        z = np.load(self.out / "validation" / f"epoch{lock['L_epoch']:03d}.npz")
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.from_numpy(z["raw_logits"]),
            torch.tensor(Tiny("val").labels, dtype=torch.float32),
            pos_weight=torch.tensor(np.sqrt(3), dtype=torch.float32),
        )
        self.assertAlmostEqual(lock["val_loss"], loss.item(), places=6)

    def test_refuses_test_dataset_and_overlap(self):
        with self.assertRaises(ValueError):
            self.api.fit(Tiny("train"), Tiny("test"), self.out)
        v = Tiny("val")
        v.ids[0] = "train0"
        with self.assertRaises(ValueError):
            self.api.fit(Tiny("train"), v, self.out)
        self.assertFalse(self.out.exists())

    def test_invalid_config(self):
        for kw in (
            {"batch_size": 1},
            {"learning_rate": float("nan")},
            {"max_epochs": 0},
            {"drop_last": False},
        ):
            with self.assertRaises((ValueError, TypeError)):
                self.api.TrainingConfig(**kw)
