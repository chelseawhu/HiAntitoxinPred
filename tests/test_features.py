import importlib, json, tempfile, unittest, hashlib
from pathlib import Path
import numpy as np


class FeatureTests(unittest.TestCase):
    def setUp(self):
        try:
            self.api = importlib.import_module("hiantitoxinpred.features")
        except ModuleNotFoundError:
            self.api = None
        self.assertIsNotNone(self.api, "Feature store missing")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.p = Path(self.tmp.name)
        np.save(self.p / "per_residue.npy", np.ones((2, 1024), np.float32))
        self.row = {
            "id": "a",
            "sequence": "AC",
            "sequence_sha256": hashlib.sha256(b"AC").hexdigest(),
            "label": 1,
            "split": "train",
        }
        (self.p / "index.json").write_text(
            json.dumps({"records": [dict(self.row, start=0, end=2, length=2)]})
        )
        self.identity = {
            n: hashlib.sha256((self.p / n).read_bytes()).hexdigest()
            for n in ("index.json", "per_residue.npy")
        }
        (self.p / "FEATURES.json").write_text(json.dumps(self.identity))

    def test_bound_identity_and_dataset(self):
        s = self.api.FeatureStore(self.p)
        d = self.api.ResidueDataset([self.row], s, "train")
        self.assertEqual(d[0][1].shape, (2, 1024))
        self.assertEqual(d.ids, ["a"])
        bad = dict(self.row, sequence="AD")
        with self.assertRaises(ValueError):
            self.api.ResidueDataset([bad], s, "train")

    def test_modified_array_rejected(self):
        np.save(self.p / "per_residue.npy", np.zeros((2, 1024), np.float32))
        with self.assertRaises(ValueError):
            self.api.FeatureStore(self.p)
