"""Hash-verified residue storage and split-specific dataset adapters."""

import json
from pathlib import Path
import numpy as np
import torch
from .checkpoints import file_sha256
from .data import sequence_sha256


class FeatureStore:
    def __init__(self, directory, expected=None):
        self.root = Path(directory)
        self.identity = expected or json.loads(
            (self.root / "FEATURES.json").read_text()
        )
        for name in ("index.json", "per_residue.npy"):
            if (
                name not in self.identity
                or file_sha256(self.root / name) != self.identity[name]
            ):
                raise ValueError("Feature file SHA256 mismatch: " + name)
        rows = json.loads((self.root / "index.json").read_text())["records"]
        self.records = {r["id"]: r for r in rows}
        if len(rows) != len(self.records):
            raise ValueError("Duplicate feature ID")
        self.array = np.load(
            self.root / "per_residue.npy", mmap_mode="r", allow_pickle=False
        )
        if (
            self.array.ndim != 2
            or self.array.shape[1] != 1024
            or self.array.dtype != np.float32
        ):
            raise ValueError("Expected FP32 residue features with 1024 dimensions")
        for r in rows:
            if (
                not 0 <= r["start"] < r["end"] <= len(self.array)
                or r["end"] - r["start"] != r["length"]
            ):
                raise ValueError("Invalid feature offset")

    def get(self, id_):
        r = self.records[id_]
        x = np.array(self.array[r["start"] : r["end"]], copy=True)
        if not np.isfinite(x).all():
            raise ValueError("Non-finite residue features")
        return torch.from_numpy(x)


class ResidueDataset(torch.utils.data.Dataset):
    def __init__(self, rows, store, split):
        self.rows = rows
        self.store = store
        self.split = split
        self.ids = [r["id"] for r in rows]
        self.labels = [r["label"] for r in rows]
        if not rows or len(set(self.ids)) != len(rows):
            raise ValueError("Empty/duplicate dataset")
        for r in rows:
            v = store.records.get(r["id"])
            if (
                v is None
                or r["split"] != split
                or r["label"] not in (0, 1)
                or sequence_sha256(r["sequence"]) != r["sequence_sha256"]
                or v["sequence_sha256"] != r["sequence_sha256"]
                or v["length"] != len(r["sequence"])
            ):
                raise ValueError("Feature/member identity mismatch")
        self.identity = dict(
            feature_files=store.identity,
            split=split,
            records=[
                {k: r[k] for k in ("id", "label", "sequence_sha256")} for r in rows
            ],
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.ids[i], self.store.get(self.ids[i]), self.labels[i]
