"""Generate a reusable Main feature store with the pinned frozen encoder."""

import json
from pathlib import Path

import numpy as np

from .checkpoints import file_sha256
from .data import read_registered_main
from .encoder import REGISTERED_CHECKSUMS

DATA_FILES = ("processed_sequences.csv", "split_manifest.json")
PREPROCESSING = (
    "cap200-before-UZOB-to-X; spaced tokens; exclude special/padding; float32 storage"
)


def validate_encoder_identity(identity):
    if (
        identity.get("name") != "ProtT5-XL-U50-half-encoder"
        or identity.get("files") != REGISTERED_CHECKSUMS
        or identity.get("precision") not in ("cpu_fp32", "cuda_fp16_fp32wo")
        or identity.get("preprocessing") != PREPROCESSING
        or identity.get("version") != 1
    ):
        raise ValueError(
            "Encoder identity or preprocessing differs from the pinned protocol"
        )


def _json(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def extract_main_features(data_dir, output, encoder, *, batch_size=2):
    """Stream all fixed members to disk; only a successful run writes its receipt.

    This is frozen representation generation, without fitting on any split.
    Partial output is preserved after an error and cannot be used for training.
    """
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError("Encoder batch size must be a positive integer")
    output, data_dir = Path(output), Path(data_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    splits, _ = read_registered_main(data_dir)
    rows = [r for split in ("train", "val", "test") for r in splits[split]]
    data_files = {n: file_sha256(data_dir / n) for n in DATA_FILES}
    validate_encoder_identity(encoder.identity)
    records, offset = [], 0
    for row in rows:
        length = len(row["sequence"])
        records.append(
            dict(
                id=row["id"],
                label=row["label"],
                split=row["split"],
                sequence_sha256=row["sequence_sha256"],
                length=length,
                start=offset,
                end=offset + length,
            )
        )
        offset += length
    output.mkdir(parents=True)
    array = np.lib.format.open_memmap(
        output / "per_residue.npy", mode="w+", dtype=np.float32, shape=(offset, 1024)
    )
    try:
        for start in range(0, len(rows), batch_size):
            group = rows[start : start + batch_size]
            values = encoder.encode(
                [r["sequence"] for r in group], batch_size=batch_size
            )
            if len(values) != len(group):
                raise ValueError("Encoder returned the wrong number of proteins")
            for row, rec, value in zip(
                group, records[start : start + batch_size], values
            ):
                if (
                    not isinstance(value, np.ndarray)
                    or value.dtype != np.float32
                    or value.shape != (len(row["sequence"]), 1024)
                    or not np.isfinite(value).all()
                ):
                    raise ValueError("Invalid or nonfinite residue features")
                array[rec["start"] : rec["end"]] = value
            done = start + len(group)
            if start == 0 or done == len(rows) or start // 100 != done // 100:
                print(f"Encoded {done}/{len(rows)} proteins", flush=True)
        array.flush()
    finally:
        del array
    if {n: file_sha256(data_dir / n) for n in DATA_FILES} != data_files:
        raise ValueError("Input data changed during extraction")
    _json(
        output / "index.json",
        dict(
            schema_version=1,
            count=len(rows),
            residue_count=offset,
            dim=1024,
            dtype="float32",
            records=records,
        ),
    )
    _json(output / "ENCODER_IDENTITY.json", encoder.identity)
    features = {n: file_sha256(output / n) for n in ("index.json", "per_residue.npy")}
    _json(output / "FEATURES.json", features)
    receipt = dict(
        schema_version=1,
        status="COMPLETE",
        kind="generated_main_features",
        data_files=data_files,
        feature_files=features,
        encoder_identity_sha256=file_sha256(output / "ENCODER_IDENTITY.json"),
        encoder_batch_size=batch_size,
        order="train,val,test; split_manifest order",
        count=len(rows),
        residue_count=offset,
    )
    # Written last: a partial directory must never look like completed features.
    _json(output / "EXTRACTION.json", receipt)
    return receipt
