#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hiantitoxinpred.data import build_datasets_from_manifest, load_feature_records, window_collate_fn  # noqa: E402
from hiantitoxinpred.training import CNNRescueTrainingConfig, build_cnn_rescue_model  # noqa: E402


TSNE_CLASS_COLORS = {
    0: "#E64B35",  # old NPJ-style non-antitoxin color
    1: "#4DBBD5",  # old NPJ-style antitoxin color
}
TSNE_CLASS_NAMES = {
    0: "Non-Antitoxin",
    1: "Antitoxin",
}
LINE_COLORS = {
    "F1-score": "#6FAED6",
    "MCC": "#E8A95A",
}


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_json(path: str | Path) -> Dict:
    with repo_path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_source(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def build_model_from_config(config_path: Path, feature_dim: int, seed: int = 42) -> torch.nn.Module:
    config = load_json(config_path)
    overrides = dict(config["main_model_overrides"])
    overrides.update(
        {
            "esm_dim": int(feature_dim),
            "random_seed": int(seed),
            "deterministic": False,
            "use_onehot": False,
            "use_aaindex": False,
        }
    )
    train_config = CNNRescueTrainingConfig(**overrides)
    return build_cnn_rescue_model(train_config, "cpu")


def masked_mean(features: np.ndarray, length: int) -> np.ndarray:
    length = max(1, min(int(length), features.shape[0]))
    return features[:length].mean(axis=0)


def extract_before_after_representations(
    *,
    feature_path: Path,
    split_manifest_path: Path,
    model_config_path: Path,
    checkpoint_path: Path,
    batch_size: int = 32,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    feature_records = load_feature_records(feature_path)
    manifest = load_json(split_manifest_path)
    _, _, test_ds, meta = build_datasets_from_manifest(feature_records, manifest)

    model = build_model_from_config(model_config_path, meta["esm_dim"], seed=42)
    try:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    captured: List[np.ndarray] = []

    def capture_classifier_input(_module, inputs):
        captured.append(inputs[0].detach().cpu().numpy())

    hook = model.classifier.register_forward_pre_hook(capture_classifier_input)
    loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=window_collate_fn,
    )

    before_rows: List[np.ndarray] = []
    labels: List[int] = []
    ids: List[str] = []
    lengths: List[int] = []
    with torch.no_grad():
        for batch in loader:
            esm = batch["esm"]
            esm_len = batch["esm_len"]
            one_hot = batch["one_hot"]
            aaindex = batch["aaindex"]
            model.compute_outputs(esm, one_hot, aaindex, esm_len=esm_len)
            for idx, sample_id in enumerate(batch["id"]):
                cur_len = int(esm_len[idx].item())
                before_rows.append(masked_mean(esm[idx].numpy(), cur_len))
                labels.append(int(batch["label"][idx].item()))
                ids.append(str(sample_id))
                lengths.append(cur_len)

    hook.remove()
    before = np.vstack(before_rows).astype(np.float32)
    after = np.vstack(captured).astype(np.float32)
    meta_df = pd.DataFrame(
        {
            "id": ids,
            "label": labels,
            "class_name": [TSNE_CLASS_NAMES[int(x)] for x in labels],
            "sequence_length": lengths,
            "split": "test",
        }
    )
    if before.shape[0] != after.shape[0]:
        raise RuntimeError(f"Before/after representation count mismatch: {before.shape} vs {after.shape}")
    return meta_df, before, after


def run_tsne(features: np.ndarray, *, perplexity: int = 50, random_state: int = 42) -> np.ndarray:
    features = StandardScaler().fit_transform(features)
    actual_perplexity = min(int(perplexity), max(5, features.shape[0] - 1))
    kwargs = dict(
        n_components=2,
        perplexity=actual_perplexity,
        init="pca",
        learning_rate=400.0,
        early_exaggeration=24.0,
        random_state=int(random_state),
        verbose=0,
    )
    try:
        return TSNE(max_iter=3000, **kwargs).fit_transform(features)
    except TypeError:
        return TSNE(n_iter=3000, **kwargs).fit_transform(features)


def get_or_create_tsne_cache(cache_path: Path, before: np.ndarray, after: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if cache_path.exists():
        payload = np.load(cache_path)
        return payload["before_2d"], payload["after_2d"]
    before_2d = run_tsne(before, perplexity=50, random_state=42)
    after_2d = run_tsne(after, perplexity=50, random_state=42)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, before_2d=before_2d, after_2d=after_2d)
    return before_2d, after_2d


def prepare_cdhit_source(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    df["threshold"] = df["trial_name"].str.extract(r"(\d+)").astype(int)
    df = df.sort_values("threshold", ascending=False)
    return df[
        [
            "trial_name",
            "group_label",
            "threshold",
            "f1_mean",
            "f1_std",
            "mcc_mean",
            "mcc_std",
            "n_runs",
        ]
    ]


def prepare_imbalance_source(path: Path, main_summary_path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    main = pd.read_csv(main_summary_path).iloc[0]
    main_row = {
        "stage": "imbalance",
        "trial_name": "ratio_1to10",
        "group_label": "1:10",
        "model": "Full_Model",
        "n_runs": int(main["n_runs"]),
        "f1_mean": float(main["f1_mean"]),
        "f1_std": float(main["f1_std"]),
        "mcc_mean": float(main["mcc_mean"]),
        "mcc_std": float(main["mcc_std"]),
    }
    if "ratio_1to10" not in set(df["trial_name"]):
        df = pd.concat([df, pd.DataFrame([main_row])], ignore_index=True)
    else:
        for key, value in main_row.items():
            df.loc[df["trial_name"] == "ratio_1to10", key] = value
    order = ["ratio_1to1", "ratio_1to10", "ratio_1to20", "ratio_1to30", "ratio_1to40", "ratio_1to50"]
    df["ratio_order"] = df["trial_name"].map({name: idx for idx, name in enumerate(order)})
    df = df.sort_values("ratio_order")
    return df[
        [
            "trial_name",
            "group_label",
            "ratio_order",
            "f1_mean",
            "f1_std",
            "mcc_mean",
            "mcc_std",
            "n_runs",
        ]
    ]


def load_or_prepare_cdhit_source(args: argparse.Namespace) -> pd.DataFrame:
    locked_source = repo_path(args.cdhit_source)
    if locked_source.exists() and not args.refresh_summary_source:
        df = pd.read_csv(locked_source).copy()
    else:
        df = prepare_cdhit_source(repo_path(args.cdhit_summary))
    expected = {80, 70, 60, 50, 40}
    observed = set(df["threshold"].astype(int))
    if observed != expected:
        raise ValueError(f"Unexpected CD-HIT thresholds in Figure 3 source: {sorted(observed)}")
    return df.sort_values("threshold", ascending=False)


def load_or_prepare_imbalance_source(args: argparse.Namespace) -> pd.DataFrame:
    locked_source = repo_path(args.imbalance_source)
    if locked_source.exists() and not args.refresh_summary_source:
        df = pd.read_csv(locked_source).copy()
    else:
        df = prepare_imbalance_source(repo_path(args.imbalance_summary), repo_path(args.main_summary))
    expected = {"ratio_1to1", "ratio_1to10", "ratio_1to20", "ratio_1to30", "ratio_1to40", "ratio_1to50"}
    observed = set(df["trial_name"].astype(str))
    if observed != expected:
        raise ValueError(f"Unexpected imbalance ratios in Figure 3 source: {sorted(observed)}")
    order = ["ratio_1to1", "ratio_1to10", "ratio_1to20", "ratio_1to30", "ratio_1to40", "ratio_1to50"]
    df["ratio_order"] = df["trial_name"].map({name: idx for idx, name in enumerate(order)})
    return df.sort_values("ratio_order")


def load_locked_tsne_source(path: Path) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    required = {"id", "label", "panel", "tsne_1", "tsne_2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in locked t-SNE source {path}: {sorted(missing)}")
    raw = df[df["panel"].eq("Raw ESM-2")].copy()
    after = df[df["panel"].eq("After model")].copy()
    if len(raw) == 0 or len(raw) != len(after):
        raise ValueError(f"Unexpected locked t-SNE panel sizes in {path}: raw={len(raw)}, after={len(after)}")
    raw = raw.sort_values("id").reset_index(drop=True)
    after = after.sort_values("id").reset_index(drop=True)
    if raw["id"].tolist() != after["id"].tolist():
        raise ValueError(f"Locked t-SNE source has mismatched Raw/After sample IDs: {path}")
    meta_cols = [col for col in ["id", "label", "class_name", "sequence_length", "split"] if col in raw.columns]
    meta_df = raw[meta_cols].copy()
    if "class_name" not in meta_df.columns:
        meta_df["class_name"] = meta_df["label"].map(TSNE_CLASS_NAMES)
    before_2d = raw[["tsne_1", "tsne_2"]].to_numpy(float)
    after_2d = after[["tsne_1", "tsne_2"]].to_numpy(float)
    return meta_df, before_2d, after_2d


def style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
    ax.tick_params(labelsize=7, width=0.6, length=2.5)


def plot_metric_lines(ax, x: Iterable, labels: List[str], f1: np.ndarray, f1_sd: np.ndarray, mcc: np.ndarray, mcc_sd: np.ndarray) -> None:
    x = np.asarray(list(x), dtype=float)
    ax.plot(x, f1, color=LINE_COLORS["F1-score"], marker="o", markersize=3.5, linewidth=1.35, label="F1-score")
    ax.fill_between(x, f1 - f1_sd, f1 + f1_sd, color=LINE_COLORS["F1-score"], alpha=0.16, linewidth=0)
    ax.plot(x, mcc, color=LINE_COLORS["MCC"], marker="s", markersize=3.3, linewidth=1.35, label="MCC")
    ax.fill_between(x, mcc - mcc_sd, mcc + mcc_sd, color=LINE_COLORS["MCC"], alpha=0.18, linewidth=0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0.78, 0.96)
    ax.set_ylabel("Score", fontsize=7.5)
    style_axis(ax)
    ax.legend(loc="lower left", fontsize=6.8, frameon=False, handlelength=1.5)


def plot_tsne_with_marginals(fig, outer_spec, coords: np.ndarray, labels: np.ndarray, bottom_label: str) -> None:
    inner = gridspec.GridSpecFromSubplotSpec(
        2,
        2,
        subplot_spec=outer_spec,
        width_ratios=[5.5, 0.50],
        height_ratios=[0.46, 5.5],
        wspace=0.012,
        hspace=0.012,
    )
    ax_top = fig.add_subplot(inner[0, 0])
    ax_main = fig.add_subplot(inner[1, 0])
    ax_right = fig.add_subplot(inner[1, 1])

    for class_id in [0, 1]:
        mask = labels == class_id
        ax_main.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=TSNE_CLASS_COLORS[class_id],
            alpha=0.6,
            s=24,
            label=TSNE_CLASS_NAMES[class_id],
            edgecolors="white",
            linewidth=0.3,
        )
        if mask.sum() > 2:
            sns.kdeplot(
                x=coords[mask, 0],
                ax=ax_top,
                color=TSNE_CLASS_COLORS[class_id],
                fill=True,
                alpha=0.28,
                linewidth=1.2,
            )
            sns.kdeplot(
                y=coords[mask, 1],
                ax=ax_right,
                color=TSNE_CLASS_COLORS[class_id],
                fill=True,
                alpha=0.28,
                linewidth=1.2,
            )

    ax_main.set_xlabel("t-SNE 1", fontsize=7.5, fontweight="bold", labelpad=5.0)
    ax_main.set_ylabel("t-SNE 2", fontsize=7.5, fontweight="bold")
    ax_main.tick_params(labelsize=6.8)
    ax_main.grid(True, alpha=0.2, linestyle="--", linewidth=0.6)
    ax_main.spines["top"].set_visible(False)
    ax_main.spines["right"].set_visible(False)
    ax_main.legend(
        loc="lower left",
        bbox_to_anchor=(0.02, 0.03),
        fontsize=6.8,
        frameon=False,
        markerscale=0.72,
        handletextpad=0.25,
        borderaxespad=0.0,
    )
    ax_main.text(
        0.5,
        -0.300,
        bottom_label,
        transform=ax_main.transAxes,
        fontsize=8.0,
        fontweight="bold",
        ha="center",
        va="top",
    )

    for edge_ax in [ax_top, ax_right]:
        edge_ax.set_xlabel("")
        edge_ax.set_ylabel("")
        edge_ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
        for spine in edge_ax.spines.values():
            spine.set_visible(False)
        edge_ax.grid(False)


def build_figure(args: argparse.Namespace) -> None:
    output_dir = repo_path(args.output_dir)
    source_dir = output_dir / "source_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    cdhit_df = load_or_prepare_cdhit_source(args)
    imbalance_df = load_or_prepare_imbalance_source(args)
    save_source(cdhit_df, source_dir / "figure3_panelA_cdhit_source.csv")
    save_source(imbalance_df, source_dir / "figure3_panelB_imbalance_source.csv")

    locked_tsne_source = repo_path(args.tsne_source)
    if locked_tsne_source.exists() and not args.refresh_tsne_source:
        meta_df, before_2d, after_2d = load_locked_tsne_source(locked_tsne_source)
        tsne_source = pd.read_csv(locked_tsne_source)
    else:
        meta_df, before, after = extract_before_after_representations(
            feature_path=repo_path(args.feature_path),
            split_manifest_path=repo_path(args.split_manifest),
            model_config_path=repo_path(args.model_config),
            checkpoint_path=repo_path(args.checkpoint),
            batch_size=int(args.batch_size),
        )
        before_2d, after_2d = get_or_create_tsne_cache(output_dir / "figure3_tsne_cache_seed42.npz", before, after)
        tsne_source = pd.concat(
            [
                meta_df.assign(panel="Raw ESM-2", tsne_1=before_2d[:, 0], tsne_2=before_2d[:, 1]),
                meta_df.assign(panel="After model", tsne_1=after_2d[:, 0], tsne_2=after_2d[:, 1]),
            ],
            ignore_index=True,
        )
    save_source(tsne_source, source_dir / "figure3_panelC_tsne_source.csv")

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.7,
            "axes.edgecolor": "#222222",
        }
    )
    sns.set_style("white")

    # Keep a wide, non-square double-column-style aspect while preserving
    # the old two-panel t-SNE visual language.
    fig = plt.figure(figsize=(8.05, 4.34), constrained_layout=False)
    outer = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[0.96, 1.30],
        width_ratios=[1, 1],
        left=0.060,
        right=0.992,
        top=0.925,
        bottom=0.105,
        hspace=0.30,
        wspace=0.24,
    )

    ax_a = fig.add_subplot(outer[0, 0])
    plot_metric_lines(
        ax_a,
        cdhit_df["threshold"],
        [str(x) for x in cdhit_df["threshold"]],
        cdhit_df["f1_mean"].to_numpy(float),
        cdhit_df["f1_std"].to_numpy(float),
        cdhit_df["mcc_mean"].to_numpy(float),
        cdhit_df["mcc_std"].to_numpy(float),
    )
    ax_a.set_xlabel("CD-HIT identity threshold (%)", fontsize=7.5)
    ax_a.set_title("Similarity-threshold robustness", fontsize=8.4, fontweight="bold", pad=4)
    ax_a.invert_xaxis()

    ax_b = fig.add_subplot(outer[0, 1])
    plot_metric_lines(
        ax_b,
        imbalance_df["ratio_order"],
        list(imbalance_df["group_label"]),
        imbalance_df["f1_mean"].to_numpy(float),
        imbalance_df["f1_std"].to_numpy(float),
        imbalance_df["mcc_mean"].to_numpy(float),
        imbalance_df["mcc_std"].to_numpy(float),
    )
    ax_b.set_xlabel("Positive:negative ratio", fontsize=7.5)
    ax_b.set_title("Class-imbalance robustness", fontsize=8.4, fontweight="bold", pad=4)

    c_area = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1, :], wspace=0.17)
    labels_arr = meta_df["label"].to_numpy(int)
    plot_tsne_with_marginals(fig, c_area[0, 0], before_2d, labels_arr, "(a) Raw ESM-2")
    plot_tsne_with_marginals(fig, c_area[0, 1], after_2d, labels_arr, "(b) After model")

    fig.text(0.006, 0.955, "A", fontsize=10.2, fontweight="bold", ha="left", va="top")
    fig.text(0.505, 0.955, "B", fontsize=10.2, fontweight="bold", ha="left", va="top")
    fig.text(0.006, 0.555, "C", fontsize=10.2, fontweight="bold", ha="left", va="top")

    base = output_dir / "Figure3_main_robustness_tsne"
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved Figure 3 outputs to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot final Figure 3: robustness and before/after t-SNE.")
    parser.add_argument("--output-dir", default="figures/main_text/Figure3_main_combination")
    parser.add_argument("--main-summary", default="results/model_runs/final_hybrid_suite_full_5seed/main/summary_metrics.csv")
    parser.add_argument("--cdhit-summary", default="results/model_runs/cdhit_anchor80_threshold_compliant_full_5seed/cdhit/summary_metrics.csv")
    parser.add_argument("--imbalance-summary", default="results/model_runs/final_hybrid_imbalance_anchor_main_1to10_5seed/imbalance/summary_metrics.csv")
    parser.add_argument("--cdhit-source", default="results/figure_source_data/Figure3_main_combination/source_data/figure3_panelA_cdhit_source.csv")
    parser.add_argument("--imbalance-source", default="results/figure_source_data/Figure3_main_combination/source_data/figure3_panelB_imbalance_source.csv")
    parser.add_argument("--tsne-source", default="results/figure_source_data/Figure3_main_combination/source_data/figure3_panelC_tsne_source.csv")
    parser.add_argument("--feature-path", default="features/esm2_1280/final_hybrid_suite_1to10/global_cap200_unique/esm2_per_residue_features.pkl")
    parser.add_argument("--split-manifest", default="data/manifests/main/split_manifest.json")
    parser.add_argument("--checkpoint", default="pretrained/best_model.pt")
    parser.add_argument("--model-config", default="configs/model_runs/main_full_model_5seed.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--refresh-summary-source",
        action="store_true",
        help="Regenerate Panel A/B source data from run summaries instead of using locked paper-workspace source data.",
    )
    parser.add_argument(
        "--refresh-tsne-source",
        action="store_true",
        help="Regenerate Panel C t-SNE from embeddings/checkpoint instead of using locked source data.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    build_figure(parse_args())
