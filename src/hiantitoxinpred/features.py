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
                any(type(r[k]) is not int for k in ("start", "end", "length"))
                or not 0 <= r["start"] < r["end"] <= len(self.array)
                or r["end"] - r["start"] != r["length"]
            ):
                raise ValueError("Invalid feature offset")

    def get(self, id_):
        r = self.records[id_]
        x = np.array(self.array[r["start"] : r["end"]], copy=True)
        if not np.isfinite(x).all():
            raise ValueError("Non-finite residue features")
        return torch.from_numpy(x)


def load_main_features(directory, data_dir):
    """Accept the original registered cache or verified, locally regenerated Main.

    Regenerated arrays have their own file identities. They are never described
    as byte-identical to the original reference cache.
    """
    from .data import read_registered_main
    from .feature_extraction import DATA_FILES, validate_encoder_identity

    data_dir, directory = Path(data_dir), Path(directory)
    rows, original = read_registered_main(data_dir)
    # Early registered caches predate FEATURES.json; their immutable identities
    # remain bound by DATA_IDENTITY.json. Generated caches require the receipt.
    store = FeatureStore(
        directory,
        expected=original if not (directory / "FEATURES.json").exists() else None,
    )
    flat = [r for split in ("train", "val", "test") for r in rows[split]]
    if set(store.records) != {r["id"] for r in flat}:
        raise ValueError("Feature members differ from the registered Main inputs")
    offset = 0
    for record in sorted(store.records.values(), key=lambda r: r["start"]):
        if record["start"] != offset:
            raise ValueError("Overlapping or incomplete feature offsets")
        offset = record["end"]
    if offset != len(store.array):
        raise ValueError("Extra residue rows outside the feature index")
    for split in ("train", "val", "test"):
        ResidueDataset(rows[split], store, split)
    if store.identity != original:
        receipt = json.loads((directory / "EXTRACTION.json").read_text())
        if (
            receipt.get("schema_version") != 1
            or receipt.get("status") != "COMPLETE"
            or receipt.get("kind") != "generated_main_features"
            or receipt.get("feature_files") != store.identity
            or receipt.get("data_files")
            != {n: file_sha256(data_dir / n) for n in DATA_FILES}
        ):
            raise ValueError(
                "Generated feature data provenance is incomplete or differs"
            )
        encoder_file = directory / "ENCODER_IDENTITY.json"
        if file_sha256(encoder_file) != receipt.get("encoder_identity_sha256"):
            raise ValueError("Encoder identity SHA256 mismatch")
        encoder = json.loads(encoder_file.read_text())
        validate_encoder_identity(encoder)
        batch_size = receipt.get("encoder_batch_size")
        if (
            type(batch_size) is not int
            or batch_size < 1
            or receipt.get("order") != "train,val,test; split_manifest order"
            or list(store.records) != [r["id"] for r in flat]
            or receipt.get("count") != len(flat)
            or receipt.get("residue_count") != len(store.array)
        ):
            raise ValueError("Generated feature order, counts or batch size differ")
        store.provenance = receipt
    # Scan without holding another full copy of the residue array in memory.
    for start in range(0, len(store.array), 4096):
        if not np.isfinite(store.array[start : start + 4096]).all():
            raise ValueError("Non-finite residue features")
    return rows, store


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
                or v["label"] != r["label"]
                or v["split"] != split
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
        if getattr(store, "provenance", None) is not None:
            self.identity["feature_provenance"] = store.provenance

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.ids[i], self.store.get(self.ids[i]), self.labels[i]
