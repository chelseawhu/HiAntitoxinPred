"""Independently recompute a completed run's validation loss and selection."""

import argparse
import json
from pathlib import Path
import numpy as np
from hiantitoxinpred.checkpoints import file_sha256
from hiantitoxinpred.metrics import binary_metrics


def verify(root):
    root = Path(root)
    config = json.loads((root / "CONFIG.json").read_text())
    history = json.loads((root / "history.json").read_text())
    lock = json.loads((root / "SELECTION_LOCK.json").read_text())
    identity = json.loads((root / "INPUT_IDENTITY.json").read_text())
    control = config["training"]
    records = identity["val"]["records"]
    expected_ids = [r["id"] for r in records]
    expected_labels = [r["label"] for r in records]
    labels = [r["label"] for r in identity["train"]["records"]]
    weight = ((len(labels) - sum(labels)) / sum(labels)) ** 0.5
    if abs(weight - config["pos_weight"]) > 1e-12:
        raise ValueError("Incorrect training-class weight")
    if file_sha256(root / "L.pt") != lock["weights_sha256"]:
        raise ValueError("L identity mismatch")
    reference = float("inf")
    best = float("inf")
    best_epoch = 0
    wait = 0
    max_difference = 0.0
    for epoch, row in enumerate(history, 1):
        if row["epoch"] != epoch or wait >= control["patience"]:
            raise ValueError("Invalid epoch/early-stop trajectory")
        with np.load(
            root / "validation" / f"epoch{epoch:03d}.npz", allow_pickle=False
        ) as v:
            if (
                v["ids"].tolist() != expected_ids
                or v["labels"].tolist() != expected_labels
            ):
                raise ValueError("Validation identity mismatch")
            z = v["raw_logits"].astype(np.float64)
            y = v["labels"]
            if not np.isfinite(z).all():
                raise ValueError("Non-finite scores")
            loss = float(
                np.mean((1 - y) * np.logaddexp(0, z) + weight * y * np.logaddexp(0, -z))
            )
            difference = abs(loss - row["val_loss"])
            max_difference = max(max_difference, difference)
            if difference > 1e-6:
                raise ValueError("Archived objective loss mismatch")
            metrics = binary_metrics(y, z)
            for a, b in [("val_F1", "F1"), ("val_MCC", "MCC")]:
                if abs(row[a] - metrics[b]) > 1e-12:
                    raise ValueError("Validation metric mismatch")
        measured = row["val_loss"]
        if measured < reference - control["min_delta"]:
            reference = measured
            wait = 0
        else:
            wait += 1
        if measured < best:
            best = measured
            best_epoch = epoch
        if (
            row["wait"] != wait
            or row["L_epoch"] != best_epoch
            or row["reference_loss"] != reference
        ):
            raise ValueError("Selection history mismatch")
    reason = "patience" if wait >= control["patience"] else "max_epochs"
    if reason == "max_epochs" and len(history) != control["max_epochs"]:
        raise ValueError("Premature termination")
    if (
        lock["L_epoch"] != best_epoch
        or lock["val_loss"] != best
        or lock["stop_epoch"] != len(history)
        or lock["stop_reason"] != reason
    ):
        raise ValueError("Selection lock mismatch")
    return dict(
        status="PASS",
        epochs=len(history),
        L_epoch=best_epoch,
        stop_reason=reason,
        max_cpu_loss_difference=max_difference,
        weight=weight,
        weights_sha256=lock["weights_sha256"],
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    result = verify(args.run)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
