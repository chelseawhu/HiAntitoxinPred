#!/usr/bin/env python3
"""Reproduce the HiAntitoxinPred main-benchmark 5-seed headline from released checkpoints.

For each seed checkpoint this script follows the locked evaluation protocol:
(1) select the decision threshold on the main **validation** split by MaxF1, then
(2) fix that threshold on the main **test** split. It aggregates the per-seed test
metrics as mean +/- sample std (ddof=1, matching scripts/train/run_full_model_benchmarks.py)
and compares the aggregate to the locked headline reported in the manuscript
(F1 0.9206 +/- 0.0161, MCC 0.9145 +/- 0.0175; main 1:10 benchmark, seeds 42-46).

The full 5-seed checkpoint collection is distributed in the Zenodo artifact
package (see README). After extracting it, the per-seed weights live at
``checkpoints/final_hybrid_suite_full_5seed/main/seed_<seed>/best_model.pt``;
run the headline reproduction with:

    python scripts/evaluate/evaluate_5seed_checkpoints.py --device cpu

This reuses the exact evaluation code path of the single-seed cross-check in
``scripts/evaluate/evaluate_main_checkpoint.py`` (which verifies that the released
``pretrained/best_model.pt`` reproduces the locked seed_42 row), so the per-seed
numbers here are produced by the same protocol.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

# Reuse the single-seed reproduction helpers so both scripts share one protocol.
from evaluate_main_checkpoint import (  # noqa: E402
    REPORT_KEYS,
    build_model,
    load_features,
    load_sequences,
    load_split,
    make_loader,
)
from hiantitoxinpred.training import evaluate_model  # noqa: E402

DEFAULT_SEEDS = [42, 43, 44, 45, 46]

# Locked headline: results/locked_metrics/final_experiment_results_locked.md
# (main 1:10 benchmark, n=5 seeds). (mean, sample-std with ddof=1).
LOCKED_HEADLINE: Dict[str, Tuple[float, float]] = {
    "f1": (0.9206, 0.0161),
    "mcc": (0.9145, 0.0175),
    "precision": (0.9667, 0.0166),
    "recall": (0.8788, 0.0186),
    "specificity": (0.9970, 0.0015),
    "auc_roc": (0.9501, 0.0365),
    "accuracy": (0.9864, 0.0027),
}


def default(path: str) -> Path:
    return REPO_ROOT / path


def find_checkpoints(checkpoints_dir: Path, seeds: List[int]) -> List[Tuple[int, Path]]:
    found: List[Tuple[int, Path]] = []
    missing: List[int] = []
    for seed in seeds:
        ckpt = checkpoints_dir / f"seed_{seed}" / "best_model.pt"
        if ckpt.exists():
            found.append((seed, ckpt))
        else:
            missing.append(seed)
    if missing:
        raise FileNotFoundError(
            f"missing checkpoints for seeds {missing} under:\n  {checkpoints_dir}\n"
            "Expected layout: <dir>/seed_<seed>/best_model.pt\n"
            "Download and extract the Zenodo artifact package (see README), which ships\n"
            "checkpoints/final_hybrid_suite_full_5seed/main/seed_*/best_model.pt, or point\n"
            "--checkpoints-dir at a directory that follows the seed_<seed>/best_model.pt layout."
        )
    return found


def evaluate_one(checkpoint: Path, config: Path, device: str, ids_val: List[str],
                 ids_test: List[str], feats: Dict[str, np.ndarray],
                 seqs: Dict[str, Tuple[str, int]], batch_size: int) -> Dict[str, float]:
    model = build_model(config, checkpoint, device)
    val_eval = evaluate_model(model, make_loader(ids_val, feats, seqs, batch_size), device, fixed_threshold=None)
    thr_logit = float(val_eval["threshold_logit"])
    test_eval = evaluate_model(model, make_loader(ids_test, feats, seqs, batch_size), device, fixed_threshold=thr_logit)
    row: Dict[str, float] = {key: float(test_eval[key]) for key in REPORT_KEYS}
    row["threshold_prob"] = float(test_eval["threshold_prob"])
    for key in ("tp", "fp", "tn", "fn"):
        row[key] = int(test_eval[key])
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoints-dir", type=Path,
                        default=default("checkpoints/final_hybrid_suite_full_5seed/main"),
                        help="Directory holding seed_<seed>/best_model.pt (Zenodo artifact layout).")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--config", type=Path, default=default("configs/model_runs/main_full_model_5seed.json"))
    parser.add_argument("--features", type=Path,
                        default=default("features/esm2_1280/final_hybrid_suite_1to10/global_cap200_unique/esm2_per_residue_features.pkl"))
    parser.add_argument("--manifest", type=Path, default=default("data/manifests/main/split_manifest.json"))
    parser.add_argument("--sequences", type=Path, default=default("data/benchmark_datasets/main/processed_sequences.csv"))
    parser.add_argument("--device", default="cpu", help="cpu (default) or cuda:N")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=8, help="CPU threads (cpu device only)")
    parser.add_argument("--output", type=Path, default=default("reproduction/results/main_checkpoint_eval/5seed_metrics.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cpu":
        torch.set_num_threads(int(args.threads))
    device = args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"

    checkpoints = find_checkpoints(args.checkpoints_dir, list(args.seeds))
    ids_val, ids_test = load_split(args.manifest)
    seqs = load_sequences(args.sequences)
    feats = load_features(args.features, set(ids_val) | set(ids_test))  # loaded once, shared across seeds
    n_pos = sum(seqs[i][1] for i in ids_test)

    print("\n=== HiAntitoxinPred main-benchmark 5-seed reproduction ===")
    print(f"checkpoints: {args.checkpoints_dir}")
    print(f"device     : {device}   test set: n={len(ids_test)} ({n_pos} pos / {len(ids_test) - n_pos} neg)")
    print(f"seeds      : {', '.join(str(s) for s in args.seeds)}\n")

    per_seed: List[Dict[str, float]] = []
    head = f"{'seed':>5}{'thr_prob':>10}" + "".join(f"{k:>10}" for k in REPORT_KEYS)
    print("per-seed (validation-MaxF1 threshold, fixed on test):")
    print(head)
    print("-" * len(head))
    for seed, ckpt in checkpoints:
        row = evaluate_one(ckpt, args.config, device, ids_val, ids_test, feats, seqs, args.batch_size)
        row["seed"] = seed
        per_seed.append(row)
        cells = "".join(f"{row[k]:>10.4f}" for k in REPORT_KEYS)
        print(f"{seed:>5}{row['threshold_prob']:>10.4f}{cells}")

    summary: Dict[str, Dict[str, float]] = {}
    for key in REPORT_KEYS:
        vals = np.array([row[key] for row in per_seed], dtype=np.float64)
        summary[key] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1))}

    print("\naggregate vs locked headline (mean +/- sample std, ddof=1):")
    head2 = f"{'metric':<12}{'mean':>9}{'std':>9}{'locked mean':>13}{'locked std':>12}{'|Dmean|':>9}   status"
    print(head2)
    print("-" * len(head2))
    within_1std = True
    for key in REPORT_KEYS:
        m, s = summary[key]["mean"], summary[key]["std"]
        lm, ls = LOCKED_HEADLINE[key]
        dmean = abs(m - lm)
        ok = dmean <= max(ls, 2e-3)  # consistent if aggregate mean sits within ~1 locked std
        within_1std = within_1std and ok
        print(f"{key:<12}{m:>9.4f}{s:>9.4f}{lm:>13.4f}{ls:>12.4f}{dmean:>9.4f}   {'OK' if ok else 'CHECK'}")

    print("\n=> 5-seed mean is consistent with the locked headline (all metrics within ~1 std)"
          if within_1std else
          "\n=> some aggregate means fall outside ~1 locked std (see CHECK rows)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoints_dir": str(args.checkpoints_dir),
        "seeds": list(args.seeds),
        "device": device,
        "per_seed": per_seed,
        "summary": summary,
        "locked_headline": {k: {"mean": v[0], "std": v[1]} for k, v in LOCKED_HEADLINE.items()},
        "consistent_within_1std": bool(within_1std),
    }
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"\nsaved: {args.output}")


if __name__ == "__main__":
    main()
