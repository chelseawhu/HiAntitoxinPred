#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hiantitoxinpred.data import (  # noqa: E402
    build_datasets_from_manifest,
    dataset_records,
    dataset_stats,
    load_feature_records,
    window_collate_fn,
)
from hiantitoxinpred.utils import dataframe_to_markdown, ensure_dir, read_json, write_json  # noqa: E402
from hiantitoxinpred.training import CNNRescueTrainingConfig, run_cnn_rescue_train_val_test  # noqa: E402


ABLATION_VARIANTS: Dict[str, Dict[str, Any]] = {
    "Full_Model": {"use_cnn": True, "use_transformer": True},
}
COMPARISON_VARIANTS: List[str] = []
STAGE_ORDER = ["main", "cross_phylum", "cdhit", "imbalance"]
SUMMARY_METRICS = ["accuracy", "precision", "recall", "f1", "auc_roc", "auc_pr", "mcc", "specificity"]
SEED_METRICS = ["f1", "mcc", "precision", "recall"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HiAntitoxinPred Full Model benchmark suite.")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["main", "cross_phylum", "cdhit", "imbalance"],
        choices=["preflight", "main", "cross_phylum", "cdhit", "imbalance", "all"],
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--trials", nargs="+", default=None)
    parser.add_argument("--train-seeds", nargs="+", type=int, default=None)
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def load_config(path: str | Path) -> Dict[str, Any]:
    with resolve_path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_stages(stages: Sequence[str]) -> List[str]:
    if "all" in stages:
        return ["preflight", *STAGE_ORDER]
    return list(stages)


def selected_train_seeds(config: Mapping[str, Any], override: Optional[Sequence[int]]) -> List[int]:
    if override:
        return [int(seed) for seed in override]
    return [int(seed) for seed in config.get("train_seeds", [42, 43, 44, 45, 46])]


def selected_specs(config: Mapping[str, Any], stages: Iterable[str], trials: Optional[Iterable[str]]) -> List[Dict[str, Any]]:
    trial_filter = {str(item) for item in (trials or [])}
    specs: List[Dict[str, Any]] = []
    datasets = config.get("datasets") or {}
    for stage in stages:
        if stage == "preflight":
            continue
        for raw_spec in datasets.get(stage, []):
            trial_name = str(raw_spec["trial_name"])
            if trial_filter and trial_name not in trial_filter:
                continue
            spec = dict(raw_spec)
            spec["stage"] = stage
            spec["processed_sequences"] = str(resolve_path(spec["processed_sequences"]))
            spec["manifest"] = str(resolve_path(spec["manifest"]))
            spec["variants"] = [str(item) for item in spec.get("variants", ["Full_Model"])]
            invalid = [name for name in spec["variants"] if name not in ABLATION_VARIANTS]
            if invalid:
                raise ValueError(f"{trial_name} has unsupported variants: {invalid}")
            specs.append(spec)
    return specs


def read_manifest(path: str | Path) -> Dict[str, Any]:
    return read_json(path)


def manifest_ids(manifest: Mapping[str, Any]) -> List[str]:
    return [str(item) for key in ["ids_train", "ids_val", "ids_test"] for item in manifest[key]]


def load_sequence_table(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"id", "sequence", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    out = df[["id", "sequence", "label"]].copy()
    out["id"] = out["id"].astype(str)
    out["sequence"] = out["sequence"].astype(str)
    out["label"] = out["label"].astype(int)
    return out


def load_global_features(feature_path: str | Path) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    records = load_feature_records(feature_path)
    feature_map = {str(item["id"]): item for item in records}
    if len(feature_map) != len(records):
        raise ValueError("Global feature file contains duplicate ids")
    return records, feature_map


def subset_features_for_manifest(feature_map: Mapping[str, Dict[str, Any]], manifest: Mapping[str, Any]) -> List[Dict[str, Any]]:
    ids = manifest_ids(manifest)
    missing = [sample_id for sample_id in ids if sample_id not in feature_map]
    if missing:
        raise ValueError(f"Missing {len(missing)} feature records, e.g. {missing[:5]}")
    seen = set()
    subset = []
    for sample_id in ids:
        if sample_id in seen:
            continue
        seen.add(sample_id)
        subset.append(feature_map[sample_id])
    return subset


def run_preflight(config: Mapping[str, Any], specs: Sequence[Mapping[str, Any]], results_root: Path) -> None:
    report_dir = ensure_dir(results_root / "preflight")
    feature_path = resolve_path(config["global_feature_path"])
    inventory_path = resolve_path(config["dataset_feature_inventory"])
    expected_feature_dim = int(config.get("expected_feature_dim", 1280))
    if not feature_path.exists():
        raise FileNotFoundError(f"global_feature_path missing: {feature_path}")
    if not inventory_path.exists():
        raise FileNotFoundError(f"dataset_feature_inventory missing: {inventory_path}")

    records, feature_map = load_global_features(feature_path)
    feature_dim = int(records[0]["features"].shape[1]) if records else 0
    rows: List[Dict[str, Any]] = []
    missing_rows: List[Dict[str, Any]] = []
    mismatch_rows: List[Dict[str, Any]] = []

    for spec in specs:
        seq_df = load_sequence_table(spec["processed_sequences"])
        manifest = read_manifest(spec["manifest"])
        ids = manifest_ids(manifest)
        seq_lookup = {row.id: row for row in seq_df.itertuples(index=False)}
        missing_from_csv = [sample_id for sample_id in ids if sample_id not in seq_lookup]
        if missing_from_csv:
            raise ValueError(f"{spec['trial_name']} manifest ids missing from processed_sequences: {missing_from_csv[:5]}")
        for sample_id in ids:
            item = feature_map.get(sample_id)
            if item is None:
                missing_rows.append({"stage": spec["stage"], "trial_name": spec["trial_name"], "id": sample_id})
                continue
            expected_len = len(str(seq_lookup[sample_id].sequence))
            actual_len = int(item["features"].shape[0])
            actual_dim = int(item["features"].shape[1])
            if actual_len != expected_len or actual_dim != expected_feature_dim:
                mismatch_rows.append(
                    {
                        "stage": spec["stage"],
                        "trial_name": spec["trial_name"],
                        "id": sample_id,
                        "expected_length": expected_len,
                        "actual_length": actual_len,
                        "feature_dim": actual_dim,
                    }
                )
        rows.append(
            {
                "stage": spec["stage"],
                "trial_name": spec["trial_name"],
                "processed_sequences": spec["processed_sequences"],
                "manifest": spec["manifest"],
                "rows": int(len(seq_df)),
                "manifest_ids": int(len(ids)),
                "positive_count": int(seq_df["label"].sum()),
                "negative_count": int(len(seq_df) - seq_df["label"].sum()),
                "variants": ",".join(spec["variants"]),
            }
        )

    pd.DataFrame(rows).to_csv(report_dir / "dataset_preflight_summary.csv", index=False)
    pd.DataFrame(missing_rows).to_csv(report_dir / "missing_feature_records.csv", index=False)
    pd.DataFrame(mismatch_rows).to_csv(report_dir / "feature_length_mismatches.csv", index=False)
    write_json(
        {
            "global_feature_path": str(feature_path),
            "dataset_feature_inventory": str(inventory_path),
            "feature_record_count": int(len(records)),
            "feature_dim": int(feature_dim),
            "expected_feature_dim": int(expected_feature_dim),
            "trial_spec_count": int(len(rows)),
            "unique_dataset_count": int(pd.DataFrame(rows)["processed_sequences"].nunique()) if rows else 0,
            "missing_feature_count": int(len(missing_rows)),
            "feature_length_mismatch_count": int(len(mismatch_rows)),
        },
        report_dir / "preflight_report.json",
    )
    if feature_dim != expected_feature_dim or missing_rows or mismatch_rows:
        raise ValueError(
            f"Preflight failed: feature_dim={feature_dim}, "
            f"expected_feature_dim={expected_feature_dim}, "
            f"missing={len(missing_rows)}, mismatches={len(mismatch_rows)}"
        )
    unique_dataset_count = int(pd.DataFrame(rows)["processed_sequences"].nunique()) if rows else 0
    print(f"Preflight passed: {len(records)} feature records, {unique_dataset_count} unique datasets, {len(rows)} trial specs.")


def build_training_overrides(config: Mapping[str, Any], spec: Mapping[str, Any], variant: str) -> Dict[str, Any]:
    overrides = dict(config.get("main_model_overrides") or {})
    overrides.update(ABLATION_VARIANTS[variant])
    overrides["threshold_policy"] = "validation_max_f1"
    overrides["battlefield"] = f"HiAntitoxinPred::{spec['stage']}::{spec['trial_name']}"
    overrides["evaluation_protocols"] = ["validation_max_f1"]
    return overrides


def load_completed_row(run_dir: Path) -> Optional[Dict[str, Any]]:
    required = [
        "metrics.json",
        "validation_metrics.json",
        "training_meta.json",
        "split_manifest.json",
        "evaluation_artifacts.npz",
        "best_model.pt",
    ]
    if not all((run_dir / name).exists() for name in required):
        return None
    metrics = read_json(run_dir / "metrics.json")
    val_metrics = read_json(run_dir / "validation_metrics.json")
    meta = read_json(run_dir / "training_meta.json")
    manifest = read_json(run_dir / "split_manifest.json")

    def split_stat(split_name: str, field: str) -> int:
        return int((manifest.get(split_name) or {}).get(field, 0))

    row: Dict[str, Any] = {
        "status": "success",
        "split_seed": int(meta.get("split_seed", manifest.get("split_seed", 42))),
        "train_seed": int(meta.get("train_seed", manifest.get("train_seed", 42))),
        "feature_path": str(manifest.get("feature_path", "")),
        "run_dir": str(run_dir.resolve()),
        "checkpoint_path": str((run_dir / "best_model.pt").resolve()),
        "training_curve_path": str((run_dir / "training_curves.png").resolve()),
        "metrics_path": str((run_dir / "metrics.json").resolve()),
        "validation_metrics_path": str((run_dir / "validation_metrics.json").resolve()),
        "split_manifest_path": str((run_dir / "split_manifest.json").resolve()),
        "training_meta_path": str((run_dir / "training_meta.json").resolve()),
        "evaluation_artifacts_path": str((run_dir / "evaluation_artifacts.npz").resolve()),
        "best_validation_f1": float(meta.get("best_validation_f1", 0.0)),
        "best_epoch": int(meta.get("best_epoch", 0)),
        "best_stage": str(meta.get("best_stage", "")),
        "best_validation_loss": float(meta.get("best_validation_loss", float("nan"))),
        "best_validation_mcc": float(meta.get("best_validation_mcc", float("nan"))),
        "best_validation_loss_at_best_f1": float(meta.get("best_validation_loss_at_best_f1", float("nan"))),
        "best_validation_mcc_at_best_f1": float(meta.get("best_validation_mcc_at_best_f1", float("nan"))),
        "best_validation_f1_at_best_loss": float(meta.get("best_validation_f1_at_best_loss", float("nan"))),
        "best_validation_mcc_at_best_loss": float(meta.get("best_validation_mcc_at_best_loss", float("nan"))),
        "checkpoint_selection_metric": str(meta.get("checkpoint_selection_metric", "")),
        "effective_training_schedule": str(meta.get("effective_training_schedule", "")),
        "validation_threshold_logit": float(meta.get("validation_threshold_logit", metrics.get("threshold_logit", 0.0))),
        "validation_threshold_prob": float(meta.get("validation_threshold_prob", metrics.get("threshold_prob", 0.0))),
        "threshold_policy": str(meta.get("threshold_policy", "validation_max_f1")),
        "duration_seconds": float(meta.get("duration_seconds", 0.0)),
        "train_size": split_stat("train", "size"),
        "val_size": split_stat("val", "size"),
        "test_size": split_stat("test", "size"),
        "train_pos": split_stat("train", "positive_count"),
        "val_pos": split_stat("val", "positive_count"),
        "test_pos": split_stat("test", "positive_count"),
        "train_neg": split_stat("train", "negative_count"),
        "val_neg": split_stat("val", "negative_count"),
        "test_neg": split_stat("test", "negative_count"),
        "val_f1": float(val_metrics.get("f1", 0.0)),
        "val_mcc": float(val_metrics.get("mcc", 0.0)),
        "model_family": str(meta.get("model_family", "")),
        "main_pooling_mode": str(meta.get("main_pooling_mode", "dynamic")),
        "cnn_only_mode": bool(meta.get("cnn_only_mode", False)),
        "pos_weight": float(meta.get("pos_weight", float("nan"))),
        "pos_weight_cap": float(meta.get("pos_weight_cap", float("nan"))),
        "cnn_scalar_gate_init_prob": float(meta.get("cnn_scalar_gate_init_prob", 0.0)),
        "cnn_gate_mean": metrics.get("cnn_gate_mean"),
        "cnn_gate_std": metrics.get("cnn_gate_std"),
        "branch_gate_vector": metrics.get("branch_gate_vector"),
        "cnn_gate_bias_init_prob": meta.get("cnn_gate_bias_init_prob"),
        "cnn_gate_bias_final_prob": meta.get("cnn_gate_bias_final_prob"),
    }
    for key in SUMMARY_METRICS + ["threshold_prob", "threshold_logit", "tn", "fp", "fn", "tp"]:
        if key in metrics:
            row[key] = float(metrics[key]) if key not in {"tn", "fp", "fn", "tp"} else int(metrics[key])
    row["threshold_source"] = str(metrics.get("threshold_source", "unknown"))
    row["confusion_matrix"] = str(metrics.get("confusion_matrix"))
    return row


def write_split_payload(
    *,
    output_dir: Path,
    split_seed: int,
    train_seed: int,
    feature_path: Path,
    manifest: Mapping[str, Any],
    train_ds: Any,
    val_ds: Any,
    test_ds: Any,
) -> None:
    payload = {
        **{key: value for key, value in manifest.items() if key not in {"train", "val", "test"}},
        "split_seed": int(split_seed),
        "train_seed": int(train_seed),
        "feature_path": str(feature_path.resolve()),
        "ids_train": list(manifest["ids_train"]),
        "ids_val": list(manifest["ids_val"]),
        "ids_test": list(manifest["ids_test"]),
        "train": {**dataset_stats(train_ds), "records": dataset_records(train_ds)},
        "val": {**dataset_stats(val_ds), "records": dataset_records(val_ds)},
        "test": {**dataset_stats(test_ds), "records": dataset_records(test_ds)},
    }
    write_json(payload, output_dir / "split_manifest.json")


def run_one_seed(
    *,
    spec: Mapping[str, Any],
    variant: str,
    train_seed: int,
    manifest: Mapping[str, Any],
    feature_records: Sequence[Dict[str, Any]],
    feature_path: Path,
    output_dir: Path,
    config: Mapping[str, Any],
    device: str,
    skip_completed: bool,
) -> Dict[str, Any]:
    run_dir = ensure_dir(output_dir / spec["stage"] / spec["trial_name"] / variant / f"seed_{int(train_seed)}")
    row = load_completed_row(run_dir) if skip_completed else None
    if row is None:
        train_ds, val_ds, test_ds, meta = build_datasets_from_manifest(feature_records, dict(manifest))
        training_config = CNNRescueTrainingConfig(
            esm_dim=int(meta["esm_dim"]),
            random_seed=int(train_seed),
            deterministic=False,
            **build_training_overrides(config, spec, variant),
        )
        result = run_cnn_rescue_train_val_test(
            train_ds,
            val_ds,
            test_ds,
            training_config,
            device,
            collate_fn=window_collate_fn,
            checkpoint_path=run_dir / "best_model.pt",
            training_curve_path=run_dir / "training_curves.png",
            evaluation_artifacts_path=run_dir / "evaluation_artifacts.npz",
            verbose=True,
        )
        write_split_payload(
            output_dir=run_dir,
            split_seed=int(config.get("split_seed", 42)),
            train_seed=int(train_seed),
            feature_path=feature_path,
            manifest=manifest,
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
        )
        test_metrics = dict(result["test_metrics"])
        val_metrics = dict(result["val_metrics"])
        test_metrics.pop("logits_array", None)
        test_metrics.pop("labels_array", None)
        val_metrics.pop("logits_array", None)
        val_metrics.pop("labels_array", None)
        write_json(test_metrics, run_dir / "metrics.json")
        write_json(val_metrics, run_dir / "validation_metrics.json")
        write_json(
            {
                "split_seed": int(config.get("split_seed", 42)),
                "train_seed": int(train_seed),
                "duration_seconds": float(result["duration_seconds"]),
                "best_validation_f1": float(result["best_validation_f1"]),
                "best_epoch": int(result["training_logs"].get("best_epoch", 0)),
                "best_stage": str(result["training_logs"].get("best_stage", "")),
                "best_validation_loss": float(result["training_logs"].get("best_validation_loss", float("nan"))),
                "best_validation_mcc": float(result["training_logs"].get("best_validation_mcc", float("nan"))),
                "best_validation_loss_at_best_f1": float(
                    result["training_logs"].get("best_validation_loss_at_best_f1", float("nan"))
                ),
                "best_validation_mcc_at_best_f1": float(
                    result["training_logs"].get("best_validation_mcc_at_best_f1", float("nan"))
                ),
                "best_validation_f1_at_best_loss": float(
                    result["training_logs"].get("best_validation_f1_at_best_loss", float("nan"))
                ),
                "best_validation_mcc_at_best_loss": float(
                    result["training_logs"].get("best_validation_mcc_at_best_loss", float("nan"))
                ),
                "checkpoint_selection_metric": str(
                    result["training_logs"].get("checkpoint_selection_metric", training_config.checkpoint_selection_metric)
                ),
                "effective_training_schedule": str(
                    result["training_logs"].get("effective_training_schedule", "single_stage")
                ),
                "validation_threshold_logit": float(result["validation_threshold_logit"]),
                "validation_threshold_prob": float(result["validation_threshold_prob"]),
                "threshold_policy": str(result.get("threshold_policy", "validation_max_f1")),
                "threshold_source": str(test_metrics.get("threshold_source", "unknown")),
                "weight_decay": float(training_config.weight_decay),
                "label_smoothing": float(training_config.label_smoothing),
                "pos_weight": float(result["training_logs"].get("pos_weight", float("nan"))),
                "pos_weight_cap": float(training_config.pos_weight_cap),
                "model_family": str(training_config.model_family),
                "battlefield": str(training_config.battlefield),
                "evaluation_protocols": list(training_config.evaluation_protocols),
                "cnn_branch_mode": str(training_config.cnn_branch_mode),
                "cnn_pooling_mode": str(training_config.cnn_pooling_mode),
                "main_pooling_mode": str(training_config.main_pooling_mode),
                "cnn_only_mode": bool(training_config.cnn_only_mode),
                "cnn_branch_gate_mode": str(training_config.cnn_branch_gate_mode),
                "training_schedule": dict(training_config.training_schedule),
                "top_k": int(training_config.top_k),
                "cnn_scalar_gate_init_prob": float(training_config.cnn_scalar_gate_init_prob),
                "cnn_gate_bias_init_prob": result.get("cnn_gate_bias_init_prob"),
                "cnn_gate_bias_final_prob": result.get("cnn_gate_bias_final_prob"),
            },
            run_dir / "training_meta.json",
        )
        row = load_completed_row(run_dir)
        if row is None:
            raise RuntimeError(f"Run finished but output reload failed: {run_dir}")

    row["stage"] = str(spec["stage"])
    row["trial_name"] = str(spec["trial_name"])
    row["group_label"] = str(spec.get("group_label", spec["trial_name"]))
    row["model"] = str(variant)
    print(
        "[RESULT] "
        f"stage={row['stage']} "
        f"trial={row['trial_name']} "
        f"model={row['model']} "
        f"seed={int(row['train_seed'])} "
        f"F1={float(row.get('f1', float('nan'))):.4f} "
        f"MCC={float(row.get('mcc', float('nan'))):.4f}",
        flush=True,
    )
    return row


def summarize_runs(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    rows = []
    for group_values, group_df in df.groupby(list(group_cols), dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        row = {col: value for col, value in zip(group_cols, group_values)}
        row["n_runs"] = int(len(group_df))
        for metric in SUMMARY_METRICS:
            values = pd.to_numeric(group_df.get(metric), errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if not values.empty else float("nan")
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_min"] = float(values.min()) if not values.empty else float("nan")
            row[f"{metric}_max"] = float(values.max()) if not values.empty else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values(list(group_cols)).reset_index(drop=True)


def build_per_seed_wide(per_run_df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for metric in SEED_METRICS:
        pivot_df = per_run_df.pivot_table(index=["trial_name", "train_seed"], columns="model", values=metric, aggfunc="first").reset_index()
        pivot_df.columns = [col if col in {"trial_name", "train_seed"} else f"{col}_{metric}" for col in pivot_df.columns]
        frames.append(pivot_df)
    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on=["trial_name", "train_seed"], how="inner")
    return out.sort_values(["trial_name", "train_seed"]).reset_index(drop=True)


def build_full_vs_variant_tables(per_run_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    delta_rows = []
    for trial_name, trial_df in per_run_df.groupby("trial_name"):
        models = set(trial_df["model"].astype(str))
        if "Full_Model" not in models:
            continue
        pivot_f1 = trial_df.pivot_table(index="train_seed", columns="model", values="f1", aggfunc="first")
        pivot_mcc = trial_df.pivot_table(index="train_seed", columns="model", values="mcc", aggfunc="first")
        for train_seed in sorted(trial_df["train_seed"].unique()):
            row: Dict[str, Any] = {"trial_name": trial_name, "train_seed": int(train_seed)}
            for variant in COMPARISON_VARIANTS:
                if variant not in models:
                    continue
                delta_f1 = float(pivot_f1.loc[train_seed, "Full_Model"] - pivot_f1.loc[train_seed, variant])
                delta_mcc = float(pivot_mcc.loc[train_seed, "Full_Model"] - pivot_mcc.loc[train_seed, variant])
                row[f"full_minus_{variant}_f1"] = delta_f1
                row[f"full_minus_{variant}_mcc"] = delta_mcc
                if delta_f1 > 0 and delta_mcc > 0:
                    winner = "Full_Model"
                elif delta_f1 < 0 and delta_mcc < 0:
                    winner = variant
                else:
                    winner = "Mixed"
                row[f"better_model_vs_{variant}"] = winner
            delta_rows.append(row)
    delta_df = pd.DataFrame(delta_rows)
    win_rows = []
    if not delta_df.empty:
        for trial_name, trial_df in delta_df.groupby("trial_name"):
            for variant in COMPARISON_VARIANTS:
                f1_col = f"full_minus_{variant}_f1"
                mcc_col = f"full_minus_{variant}_mcc"
                if f1_col not in trial_df.columns:
                    continue
                full_better = int(((trial_df[f1_col] > 0) & (trial_df[mcc_col] > 0)).sum())
                variant_better = int(((trial_df[f1_col] < 0) & (trial_df[mcc_col] < 0)).sum())
                win_rows.append(
                    {
                        "trial_name": trial_name,
                        "comparison": f"Full_Model vs {variant}",
                        "full_better_both_f1_mcc": full_better,
                        "variant_better_both_f1_mcc": variant_better,
                        "mixed_or_tied": int(len(trial_df) - full_better - variant_better),
                        "mean_delta_f1": float(trial_df[f1_col].mean()),
                        "mean_delta_mcc": float(trial_df[mcc_col].mean()),
                    }
                )
    return delta_df, pd.DataFrame(win_rows)


def write_stage_outputs(stage_dir: Path, rows: Sequence[Mapping[str, Any]], title: str) -> Dict[str, pd.DataFrame]:
    stage_dir = ensure_dir(stage_dir)
    per_run_df = pd.DataFrame(rows).sort_values(["trial_name", "model", "train_seed"]).reset_index(drop=True)
    summary_df = summarize_runs(per_run_df, ["stage", "trial_name", "group_label", "model"])
    per_run_df.to_csv(stage_dir / "per_run_metrics.csv", index=False)
    summary_df.to_csv(stage_dir / "summary_metrics.csv", index=False)
    selected_cols = [
        "stage",
        "trial_name",
        "model",
        "train_seed",
        "threshold_policy",
        "threshold_source",
        "threshold_prob",
        "threshold_logit",
        "validation_threshold_prob",
        "validation_threshold_logit",
        "val_f1",
        "val_mcc",
    ]
    per_run_df[[col for col in selected_cols if col in per_run_df.columns]].to_csv(stage_dir / "selected_thresholds.csv", index=False)
    report_cols = [
        "trial_name",
        "group_label",
        "model",
        "train_seed",
        "f1",
        "mcc",
        "precision",
        "recall",
        "threshold_prob",
        "threshold_source",
        "confusion_matrix",
    ]
    lines = [
        f"# {title}",
        "",
        "## Summary",
        "",
        dataframe_to_markdown(summary_df),
        "",
        "## Per-Run Metrics",
        "",
        dataframe_to_markdown(per_run_df[[col for col in report_cols if col in per_run_df.columns]]),
        "",
    ]
    (stage_dir / "seed_detail_report.md").write_text("\n".join(lines), encoding="utf-8")
    if set(per_run_df["model"].astype(str)) & set(COMPARISON_VARIANTS):
        build_per_seed_wide(per_run_df).to_csv(stage_dir / "per_seed_wide_metrics.csv", index=False)
        delta_df, win_df = build_full_vs_variant_tables(per_run_df)
        delta_df.to_csv(stage_dir / "full_vs_variant_seed_deltas.csv", index=False)
        win_df.to_csv(stage_dir / "full_vs_variant_win_counts.csv", index=False)
    return {"per_run": per_run_df, "summary": summary_df}


def fmt_mean_std(row: Mapping[str, Any], metric: str) -> str:
    return f"{float(row[f'{metric}_mean']):.4f} +/- {float(row[f'{metric}_std']):.4f}"


def write_paper_table(path: Path, title: str, summary_df: pd.DataFrame) -> None:
    rows = []
    for _, row in summary_df.iterrows():
        rows.append(
            {
                "experiment": row["group_label"],
                "model": row["model"],
                "n": int(row["n_runs"]),
                "F1": fmt_mean_std(row, "f1"),
                "MCC": fmt_mean_std(row, "mcc"),
                "Precision": fmt_mean_std(row, "precision"),
                "Recall": fmt_mean_std(row, "recall"),
                "AUC-ROC": fmt_mean_std(row, "auc_roc"),
                "AUC-PR": fmt_mean_std(row, "auc_pr"),
            }
        )
    path.write_text("\n".join([f"# {title}", "", dataframe_to_markdown(pd.DataFrame(rows)), ""]), encoding="utf-8")


def write_global_outputs(results_root: Path, config: Mapping[str, Any]) -> None:
    summaries = []
    for stage in STAGE_ORDER:
        summary_path = results_root / stage / "summary_metrics.csv"
        if summary_path.exists():
            df = pd.read_csv(summary_path)
            df["suite_stage"] = stage
            summaries.append(df)
            table_title = {
                "main": "Comprehensive Benchmark",
                "cross_phylum": "Cross-Phylum Benchmark",
                "cdhit": "CD-HIT Threshold Benchmark",
                "imbalance": "Imbalance Benchmark",
                "ablation": "Ablation Study",
            }[stage]
            write_paper_table(results_root / f"paper_table_{stage}.md", table_title, df)
    if summaries:
        pd.concat(summaries, ignore_index=True).to_csv(results_root / "global_summary_metrics.csv", index=False)
    write_json(
        {
            "experiment_set_name": config.get("experiment_set_name"),
            "results_root": str(results_root.resolve()),
            "threshold_policy": config.get("threshold_policy", "validation_max_f1"),
            "train_seeds": [int(seed) for seed in config.get("train_seeds", [])],
            "global_feature_path": str(resolve_path(config["global_feature_path"])),
            "runner": str((THIS_DIR / "run_final_model_suite.py").resolve()),
        },
        results_root / "suite_manifest.json",
    )


def write_trial_manifest(
    *,
    trial_dir: Path,
    spec: Mapping[str, Any],
    config: Mapping[str, Any],
    train_seeds: Sequence[int],
    feature_path: Path,
) -> None:
    write_json(
        {
            "experiment_set_name": config.get("experiment_set_name"),
            "stage": spec["stage"],
            "trial_name": spec["trial_name"],
            "group_label": spec.get("group_label", spec["trial_name"]),
            "processed_sequences": spec["processed_sequences"],
            "manifest": spec["manifest"],
            "global_feature_path": str(feature_path.resolve()),
            "variants": list(spec["variants"]),
            "train_seeds": [int(seed) for seed in train_seeds],
            "threshold_policy": config.get("threshold_policy", "validation_max_f1"),
            "model_overrides": config.get("main_model_overrides"),
        },
        trial_dir / "trial_manifest.json",
    )


def run_stage(
    *,
    stage: str,
    specs: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    results_root: Path,
    feature_map: Mapping[str, Dict[str, Any]],
    feature_path: Path,
    train_seeds: Sequence[int],
    device: str,
    skip_completed: bool,
) -> Dict[str, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    stage_dir = ensure_dir(results_root / stage)
    for spec in specs:
        manifest = read_manifest(spec["manifest"])
        feature_records = subset_features_for_manifest(feature_map, manifest)
        write_trial_manifest(
            trial_dir=ensure_dir(stage_dir / str(spec["trial_name"])),
            spec=spec,
            config=config,
            train_seeds=train_seeds,
            feature_path=feature_path,
        )
        for variant in spec["variants"]:
            for train_seed in train_seeds:
                print(f"[{stage}] {spec['trial_name']} | {variant} | seed={train_seed}", flush=True)
                rows.append(
                    run_one_seed(
                        spec=spec,
                        variant=variant,
                        train_seed=int(train_seed),
                        manifest=manifest,
                        feature_records=feature_records,
                        feature_path=feature_path,
                        output_dir=results_root,
                        config=config,
                        device=device,
                        skip_completed=skip_completed,
                    )
                )
    return write_stage_outputs(stage_dir, rows, title=f"HiAntitoxinPred Full Model suite: {stage}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.train_seeds:
        config["train_seeds"] = [int(seed) for seed in args.train_seeds]
    stages = normalize_stages(args.stages)
    train_seeds = selected_train_seeds(config, args.train_seeds)
    results_root = ensure_dir(resolve_path(config["results_dir"]))
    specs = selected_specs(config, STAGE_ORDER, args.trials)

    if "preflight" in stages:
        run_preflight(config, specs, results_root)
        if stages == ["preflight"]:
            return

    training_stages = [stage for stage in stages if stage in STAGE_ORDER]
    selected = selected_specs(config, training_stages, args.trials)
    if not selected:
        raise ValueError("No training trials selected")
    specs_by_stage: Dict[str, List[Dict[str, Any]]] = {}
    for spec in selected:
        specs_by_stage.setdefault(str(spec["stage"]), []).append(spec)

    feature_path = resolve_path(config["global_feature_path"])
    if not feature_path.exists():
        raise FileNotFoundError(f"global_feature_path missing: {feature_path}")
    _, feature_map = load_global_features(feature_path)

    for stage in STAGE_ORDER:
        if stage not in specs_by_stage:
            continue
        run_stage(
            stage=stage,
            specs=specs_by_stage[stage],
            config=config,
            results_root=results_root,
            feature_map=feature_map,
            feature_path=feature_path,
            train_seeds=train_seeds,
            device=str(args.device),
            skip_completed=bool(args.skip_completed),
        )
    write_global_outputs(results_root, config)
    print(f"Saved HiAntitoxinPred Full Model suite outputs to {results_root}")


if __name__ == "__main__":
    main()
