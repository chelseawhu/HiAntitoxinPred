"""Sequence identity and explicit batching for Main and user FASTA inputs."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch

ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWYXBZOU")


def sequence_sha256(sequence):
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def _sequence(sequence):
    if not sequence or any(c not in ALPHABET for c in sequence):
        raise ValueError("Empty sequence or unsupported amino-acid character")
    return sequence


@dataclass(frozen=True)
class Protein:
    id: str
    original_sequence: str
    sequence: str

    @property
    def original_length(self):
        return len(self.original_sequence)

    @property
    def input_length(self):
        return len(self.sequence)

    @property
    def truncated(self):
        return self.input_length < self.original_length

    @property
    def sequence_sha256(self):
        # Identity precedes the encoder's ambiguous-residue mapping.
        return sequence_sha256(self.sequence)

    @property
    def encoder_sequence(self):
        return self.sequence.translate(str.maketrans(dict.fromkeys("UZOB", "X")))


def read_fasta(path):
    """Normalize FASTA case/whitespace, preserve order, then truncate to cap200.

    The first whitespace-delimited header token is the prediction ID. Gaps,
    stops, digits and unsupported letters are rejected rather than removed.
    """
    proteins, seen, identifier, fragments = [], set(), None, []

    def finish():
        if identifier is None:
            return
        raw = "".join(fragments)
        if not raw.isascii():
            raise ValueError("Residue symbols must be ASCII letters")
        seq = _sequence(raw.upper())
        proteins.append(Protein(identifier, seq, seq[:200]))

    with Path(path).open() as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                finish()
                header = line[1:].split()
                if not header:
                    raise ValueError("Empty FASTA identifier")
                identifier, fragments = header[0], []
                if identifier in seen:
                    raise ValueError(f"Duplicate FASTA ID: {identifier}")
                seen.add(identifier)
            else:
                if identifier is None:
                    raise ValueError("Sequence before FASTA header")
                fragments.append("".join(line.split()))
    finish()
    if not proteins:
        raise ValueError("Empty FASTA file")
    return proteins


def collate_features(values, *, width=None):
    """Pad FP32 CPU residue tensors; no truncation or feature normalization.

    None uses the registered longest-in-batch geometry (minimum width two for
    the even convolution). An explicit width is an input policy chosen by the
    caller, not a claim of equivalence to the registered evaluation geometry.
    """
    if not values:
        raise ValueError("Empty feature batch")
    dims = set()
    for value in values:
        if not isinstance(value, torch.Tensor) or value.ndim != 2:
            raise ValueError("Features must be two-dimensional tensors")
        if value.device.type != "cpu" or value.dtype != torch.float32:
            raise ValueError("Collation requires CPU float32 features")
        if (
            not 1 <= len(value) <= 200
            or value.shape[1] < 1
            or not torch.isfinite(value).all()
        ):
            raise ValueError("Features must have 1..200 finite residue rows")
        dims.add(value.shape[1])
    if len(dims) != 1:
        raise ValueError("Feature dimensions differ")
    longest = max(map(len, values))
    width = max(2, longest) if width is None else width
    if (
        isinstance(width, bool)
        or not isinstance(width, int)
        or not max(2, longest) <= width <= 200
    ):
        raise ValueError("Invalid padded width")
    x = torch.zeros(len(values), width, next(iter(dims)), dtype=torch.float32)
    mask = torch.zeros(len(values), width, dtype=torch.bool)
    for i, value in enumerate(values):
        x[i, : len(value)] = value
        mask[i, : len(value)] = True
    return x, mask


def read_main(csv_path, manifest_path):
    """Validate registered Main records without renormalizing or re-splitting.

    Cross-split sequence equality is rejected. This membership check is not a
    replacement for the separately archived sequence-similarity audit.
    """
    with Path(csv_path).open(newline="") as f:
        rows = list(csv.DictReader(f))
    manifest = json.loads(Path(manifest_path).read_text())
    keys = {"ids_train", "ids_val", "ids_test"}
    if set(manifest) != keys or any(not isinstance(manifest[k], list) for k in keys):
        raise ValueError("Expected train/val/test member lists")
    by_id, sequence_splits = {}, {}
    for row in rows:
        if not {"id", "sequence", "label", "split", "sequence_sha256"} <= set(row):
            raise ValueError("Missing Main record fields")
        id_, seq = row["id"], _sequence(row["sequence"])
        if not id_ or id_ in by_id or len(seq) > 200:
            raise ValueError("Invalid or duplicate Main ID / input length")
        if row["label"] not in ("0", "1") or row["split"] not in (
            "train",
            "val",
            "test",
        ):
            raise ValueError("Invalid Main label or split")
        if sequence_sha256(seq) != row["sequence_sha256"]:
            raise ValueError(f"Sequence identity mismatch: {id_}")
        if seq in sequence_splits and sequence_splits[seq] != row["split"]:
            raise ValueError("Identical sequence appears across Main splits")
        sequence_splits[seq] = row["split"]
        by_id[id_] = dict(row, label=int(row["label"]))
    all_ids = [id_ for key in keys for id_ in manifest[key]]
    if not rows or len(set(all_ids)) != len(all_ids) or set(all_ids) != set(by_id):
        raise ValueError("Missing, duplicated, overlapping or extra split members")
    result = {}
    for split in ("train", "val", "test"):
        ids = manifest[f"ids_{split}"]
        if not ids or any(by_id[id_]["split"] != split for id_ in ids):
            raise ValueError("CSV and manifest split assignment disagree")
        result[split] = [by_id[id_] for id_ in ids]
    return result


def read_registered_main(directory):
    """Bind Main labels, member order and feature identities before evaluation."""
    from .checkpoints import file_sha256

    root = Path(directory)
    identity = json.loads((root / "DATA_IDENTITY.json").read_text())
    for name in ("processed_sequences.csv", "split_manifest.json"):
        if file_sha256(root / name) != identity["files"].get(name):
            raise ValueError("Registered Main SHA256 mismatch: " + name)
    expected = identity["feature_files"]
    if set(expected) != {"index.json", "per_residue.npy"}:
        raise ValueError("Incomplete registered Main feature identities")
    return (
        read_main(root / "processed_sequences.csv", root / "split_manifest.json"),
        expected,
    )
