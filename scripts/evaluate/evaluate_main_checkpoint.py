#!/usr/bin/env python3
"""Reproduce the HiAntitoxinPred main-benchmark test metrics from a released checkpoint.

This is the lightweight reproduction entry point: it loads a trained Full Model
checkpoint (by default the released ``pretrained/best_model.pt``, which is the
locked main-benchmark seed_42 model), runs it on the main val/test split, and
reports the test metrics next to the locked paper numbers.

It does NOT retrain anything. It reuses the project's own ``evaluate_model`` and
the validation-MaxF1 threshold protocol, so the numbers match the locked
evaluation exactly:

  1. score validation with the model, pick the MaxF1 threshold on the
     0.05..0.95 (step 0.01) grid,
  2. freeze that threshold and apply it to the test set,
  3. print the test metrics and the per-metric difference vs. the locked
     seed_42 row.

ESM2 features
-------------
The model consumes ESM2-1280 per-residue embeddings. This script reads them from
a feature pickle (``--features``). If you do not have it locally, generate it
once with ``scripts/feature_extraction/extract_esm2_embeddings.py`` (or download
the released feature archive). See README "Reproduce the main experiment".

Example
-------
    python scripts/evaluate/evaluate_main_checkpoint.py --device cpu
    python scripts/evaluate/evaluate_main_checkpoint.py --device cuda:0 \
        --checkpoint pretrained/best_model.pt
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hiantitoxinpred.data import ESMOnlyDataset, window_collate_fn  # noqa: E402
from hiantitoxinpred.training import (  # noqa: E402
    CNNRescueTrainingConfig,
    build_cnn_rescue_model,
    evaluate_model,
)

# Locked main-benchmark seed_42 test row (source: results/locked_metrics/
# final_experiment_results_locked.md). The released best_model.pt is this model
# converted to a float16 state_dict, so evaluation should match these to <1e-3.
LOCKED_SEED42 = {
    "accuracy": 0.9892,
    "precision": 0.9833,
    "recall": 0.8939,
    "specificity": 0.9985,
    "f1": 0.9365,
    "mcc": 0.9319,
    "auc_roc": 0.9648,
    "threshold_prob": 0.09,
}
REPORT_KEYS = ["f1", "mcc", "precision", "recall", "specificity", "auc_roc", "accuracy"]
TOLERANCE = 2e-3  # float16 round-trip; threshold-based metrics are typically exact


def default(path: str) -> Path:
    return REPO_ROOT / path


def load_split(manifest: Path) -> Tuple[List[str], List[str]]:
    man = json.loads(manifest.read_text())
    return list(man["ids_val"]), list(man["ids_test"])


def load_sequences(sequences_csv: Path) -> Dict[str, Tuple[str, int]]:
    out: Dict[str, Tuple[str, int]] = {}
    with sequences_csv.open() as handle:
        for row in csv.DictReader(handle):
            out[row["id"]] = (str(row["sequence"]), int(float(row["label"])))
    return out


def load_features(features_pkl: Path, needed: set) -> Dict[str, np.ndarray]:
    if not features_pkl.exists():
        raise FileNotFoundError(
            f"ESM2 feature file not found:\n  {features_pkl}\n"
            "Generate it once with:\n"
            "  python scripts/feature_extraction/extract_esm2_embeddings.py \\\n"
            "    --config configs/feature_extraction/esm2_1280_final_benchmarks.json \\\n"
            "    --stages all --device cuda:0\n"
            "or download the released feature archive (see README)."
        )
    records = pickle.loads(features_pkl.read_bytes())["records"]
    feats = {r["id"]: np.asarray(r["features"], dtype=np.float32) for r in records if r["id"] in needed}
    missing = needed - set(feats)
    if missing:
        raise KeyError(f"{len(missing)} ids missing from feature file, e.g. {sorted(missing)[:5]}")
    return feats


def make_loader(ids: List[str], feats: Dict[str, np.ndarray], seqs: Dict[str, Tuple[str, int]], batch_size: int) -> DataLoader:
    dataset = ESMOnlyDataset([feats[i] for i in ids], [seqs[i][1] for i in ids], ids)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=window_collate_fn)


def build_model(config_path: Path, checkpoint: Path, device: str) -> torch.nn.Module:
    overrides = dict(json.loads(config_path.read_text())["main_model_overrides"])
    overrides.update(
        {
            "esm_dim": 1280,
            "random_seed": 42,
            "deterministic": False,
            "use_onehot": False,
            "use_aaindex": False,
            "use_cnn": True,
            "use_transformer": True,
        }
    )
    model = build_cnn_rescue_model(CNNRescueTrainingConfig(**overrides), device)
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)  # float16 -> float32 handled by copy_
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=default("pretrained/best_model.pt"))
    parser.add_argument("--config", type=Path, default=default("configs/model_runs/main_full_model_5seed.json"))
    parser.add_argument("--features", type=Path,
                        default=default("features/main_esm2_1280.pkl"))
    parser.add_argument("--manifest", type=Path, default=default("data/manifests/main/split_manifest.json"))
    parser.add_argument("--sequences", type=Path, default=default("data/benchmark_datasets/main/processed_sequences.csv"))
    parser.add_argument("--device", default="cpu", help="cpu (default) or cuda:N")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=8, help="CPU threads (cpu device only)")
    parser.add_argument("--output", type=Path, default=default("reproduction/results/main_checkpoint_eval/seed42_metrics.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cpu":
        torch.set_num_threads(int(args.threads))
    device = args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"

    ids_val, ids_test = load_split(args.manifest)
    seqs = load_sequences(args.sequences)
    feats = load_features(args.features, set(ids_val) | set(ids_test))

    model = build_model(args.config, args.checkpoint, device)

    val_eval = evaluate_model(model, make_loader(ids_val, feats, seqs, args.batch_size), device, fixed_threshold=None)
    thr_logit = float(val_eval["threshold_logit"])
    test_eval = evaluate_model(model, make_loader(ids_test, feats, seqs, args.batch_size), device, fixed_threshold=thr_logit)

    n_pos = sum(seqs[i][1] for i in ids_test)
    print(f"\n=== HiAntitoxinPred main-benchmark reproduction ===")
    print(f"checkpoint : {args.checkpoint}")
    print(f"device     : {device}   features: {'cached pkl' if args.features.exists() else 'n/a'}")
    print(f"test set   : n={len(ids_test)} ({n_pos} pos / {len(ids_test) - n_pos} neg)")
    print(f"threshold  : prob={test_eval['threshold_prob']:.4f} (selected on validation, MaxF1)\n")

    header = f"{'metric':<12}{'reproduced':>12}{'locked 42':>12}{'|diff|':>10}   status"
    print(header)
    print("-" * len(header))
    all_ok = True
    for key in REPORT_KEYS:
        got = float(test_eval[key])
        ref = LOCKED_SEED42.get(key)
        if ref is None:
            print(f"{key:<12}{got:>12.4f}{'-':>12}{'-':>10}")
            continue
        diff = abs(got - ref)
        ok = diff <= TOLERANCE
        all_ok = all_ok and ok
        print(f"{key:<12}{got:>12.4f}{ref:>12.4f}{diff:>10.4f}   {'OK' if ok else 'CHECK'}")
    print(f"\nconfusion (TP/FP/TN/FN): {test_eval['tp']}/{test_eval['fp']}/{test_eval['tn']}/{test_eval['fn']}"
          f"   locked: 59/1/671/7")
    print("=> reproduction MATCHES locked seed_42" if all_ok
          else "=> some metrics differ beyond tolerance (see CHECK rows)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(args.checkpoint),
        "device": device,
        "threshold_prob": float(test_eval["threshold_prob"]),
        "reproduced": {k: float(test_eval[k]) for k in REPORT_KEYS},
        "locked_seed42": LOCKED_SEED42,
        "confusion": {"tp": test_eval["tp"], "fp": test_eval["fp"], "tn": test_eval["tn"], "fn": test_eval["fn"]},
        "matches_locked": bool(all_ok),
    }
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"\nsaved: {args.output}")


if __name__ == "__main__":
    main()
