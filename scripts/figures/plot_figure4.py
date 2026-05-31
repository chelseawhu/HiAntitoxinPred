#!/usr/bin/env python3
"""Plot Figure 4: strict cross-source generalization radar panels.

By default, the figure is regenerated from the locked source-data CSV shipped
with this release. Use --refresh-source only when the full per-run result
tables are available locally.
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures/main_text/Figure4_cross_radar"

CROSS_ROOT = PROJECT_ROOT / "results/model_runs/strict_cross_source_hybrid_full_5seed/cross_phylum"
BASELINE_ROOT = PROJECT_ROOT / "results/baselines/main_cross_full_rerun_v1"

LABELS = ["label1", "label2", "label3", "label4"]
LABEL_TITLES = {
    "label1": "Test on\nProteobacteria",
    "label2": "Test on\nFirmicutes",
    "label3": "Test on\nActinobacteria",
    "label4": "Test on\nOther sources",
}

METRICS = [
    ("accuracy", "Acc"),
    ("precision", "Prec"),
    ("recall", "Recall"),
    ("specificity", "Spec"),
    ("mcc", "MCC"),
    ("f1", "F1"),
]

METHOD_ORDER = [
    "Proposed model",
    "ESM2 + MLP",
    "ESM2 + BiLSTM",
    "BLASTP",
    "Foldseek",
]

MAIN_TABLE_METHOD_ORDER = [
    "HiAntitoxinPred",
    "ESM2 + MLP",
    "ESM2 + Random Forest",
    "ProtBERT + MLP",
    "ESM2 per-residue + BiLSTM",
    "BLASTP",
    "PSI-BLAST",
    "HMMER/phmmer",
    "SProtFP",
    "AMP Scanner v2",
    "Foldseek",
    "TM-align",
]

ALL_METHOD_ORDER = [
    "Proposed model",
    "ESM2 + LogReg",
    "ESM2 + MLP",
    "ESM2 + RF",
    "ESM2 + BiLSTM",
    "ProtBERT + LogReg",
    "ProtBERT + MLP",
    "ProtBERT + RF",
    "ProtT5 + LogReg",
    "ProtT5 + MLP",
    "ProtT5 + RF",
    "BLASTP",
    "PSI-BLAST",
    "HMMER",
    "SProtFP",
    "AMP Scanner",
    "Foldseek",
    "TM-align",
]

METHOD_COLORS = {
    "Proposed model": "#E64B35",
    "HiAntitoxinPred": "#E64B35",
    "ESM2 + LogReg": "#8ECBE8",
    "ESM2 + MLP": "#56B4E9",
    "ESM2 + RF": "#2C7FB8",
    "ESM2 + Random Forest": "#2C7FB8",
    "ESM2 + BiLSTM": "#3C5488",
    "ESM2 per-residue + BiLSTM": "#3C5488",
    "ProtBERT + LogReg": "#B8E186",
    "ProtBERT + MLP": "#7FBF7B",
    "ProtBERT + RF": "#4D9221",
    "ProtT5 + LogReg": "#D7BDE2",
    "ProtT5 + MLP": "#B57EDC",
    "ProtT5 + RF": "#7B3294",
    "BLASTP": "#F39B7F",
    "PSI-BLAST": "#E07A5F",
    "HMMER": "#C8553D",
    "HMMER/phmmer": "#C8553D",
    "SProtFP": "#A6A6A6",
    "AMP Scanner": "#F2C14E",
    "AMP Scanner v2": "#F2C14E",
    "Foldseek": "#8491B4",
    "TM-align": "#5B6C9E",
}

METHOD_MARKERS = {
    "Proposed model": "o",
    "HiAntitoxinPred": "o",
    "ESM2 + LogReg": "v",
    "ESM2 + MLP": "s",
    "ESM2 + RF": "^",
    "ESM2 + Random Forest": "^",
    "ESM2 + BiLSTM": "D",
    "ESM2 per-residue + BiLSTM": "D",
    "ProtBERT + LogReg": "v",
    "ProtBERT + MLP": "s",
    "ProtBERT + RF": "^",
    "ProtT5 + LogReg": "v",
    "ProtT5 + MLP": "s",
    "ProtT5 + RF": "^",
    "BLASTP": "P",
    "PSI-BLAST": "X",
    "HMMER": "*",
    "HMMER/phmmer": "*",
    "SProtFP": "h",
    "AMP Scanner": "8",
    "AMP Scanner v2": "8",
    "Foldseek": "X",
    "TM-align": "p",
}

PRIMARY_METHODS = {"Proposed model", "HiAntitoxinPred"}

PLM_BASELINE_NAME_MAP = {
    "esm2+LogReg": "ESM2 + LogReg",
    "esm2+MLP": "ESM2 + MLP",
    "esm2+RandomForest": "ESM2 + RF",
    "protbert+LogReg": "ProtBERT + LogReg",
    "protbert+MLP": "ProtBERT + MLP",
    "protbert+RandomForest": "ProtBERT + RF",
    "prott5+LogReg": "ProtT5 + LogReg",
    "prott5+MLP": "ProtT5 + MLP",
    "prott5+RandomForest": "ProtT5 + RF",
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.linewidth": 0.6,
            "axes.unicode_minus": False,
            "legend.frameon": False,
        }
    )


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def mean_std_records(df: pd.DataFrame, method_name: str, label: str, *, experiment: str) -> list[dict[str, object]]:
    sub = df[df["experiment"].eq(experiment)]
    records: list[dict[str, object]] = []
    for metric_key, metric_label in METRICS:
        values = sub[metric_key].astype(float)
        records.append(
            {
                "test_label": LABEL_TITLES[label],
                "method": method_name,
                "metric": metric_label,
                "metric_key": metric_key,
                "mean": values.mean(),
                "std": values.std(ddof=1) if len(values) > 1 else 0.0,
                "n": len(values),
            }
        )
    return records


def collect_proposed_model(method_name: str = "Proposed model") -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label in LABELS:
        per_seed = []
        for seed in range(42, 47):
            metrics_path = CROSS_ROOT / f"strict_cross_{label}_test/Full_Model/seed_{seed}/metrics.json"
            with metrics_path.open() as handle:
                metrics = json.load(handle)
            metrics["experiment"] = f"strict_cross_{label}_test"
            per_seed.append(metrics)
        df = pd.DataFrame(per_seed)
        rows.extend(mean_std_records(df, method_name, label, experiment=f"strict_cross_{label}_test"))
    return rows


def collect_seeded_baseline(csv_path: Path, source_baseline: str, method_name: str) -> list[dict[str, object]]:
    df = pd.read_csv(csv_path)
    df = df[df["baseline"].eq(source_baseline)].copy()
    rows: list[dict[str, object]] = []
    for label in LABELS:
        experiment = f"strict_cross_{label}_test"
        rows.extend(mean_std_records(df, method_name, label, experiment=experiment))
    return rows


def collect_single_run_baseline(csv_path: Path, source_baseline: str, method_name: str) -> list[dict[str, object]]:
    df = pd.read_csv(csv_path)
    df = df[df["baseline"].eq(source_baseline)].copy()
    rows: list[dict[str, object]] = []
    for label in LABELS:
        experiment = f"strict_cross_{label}_test"
        rows.extend(mean_std_records(df, method_name, label, experiment=experiment))
    return rows


def build_source_table(method_set: str = "selected") -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if method_set == "main_table":
        rows.extend(collect_proposed_model("HiAntitoxinPred"))
        rows.extend(collect_seeded_baseline(BASELINE_ROOT / "plm/per_run_metrics.csv", "esm2+MLP", "ESM2 + MLP"))
        rows.extend(collect_seeded_baseline(BASELINE_ROOT / "plm/per_run_metrics.csv", "esm2+RandomForest", "ESM2 + Random Forest"))
        rows.extend(collect_seeded_baseline(BASELINE_ROOT / "plm/per_run_metrics.csv", "protbert+MLP", "ProtBERT + MLP"))
        rows.extend(collect_seeded_baseline(BASELINE_ROOT / "bilstm/per_run_metrics.csv", "ESM2 per-residue + BiLSTM", "ESM2 per-residue + BiLSTM"))
        for source_name, display_name in [
            ("BLASTP", "BLASTP"),
            ("PSI-BLAST", "PSI-BLAST"),
            ("HMMER (phmmer)", "HMMER/phmmer"),
        ]:
            rows.extend(collect_single_run_baseline(BASELINE_ROOT / "retrieval/per_run_metrics.csv", source_name, display_name))
        rows.extend(collect_single_run_baseline(BASELINE_ROOT / "sprotfp/per_run_metrics.csv", "SProtFP direct prediction", "SProtFP"))
        rows.extend(collect_single_run_baseline(BASELINE_ROOT / "amp_scanner/per_run_metrics.csv", "AMP Scanner v2 retrained", "AMP Scanner v2"))
        rows.extend(collect_single_run_baseline(BASELINE_ROOT / "structure/global_structure_baseline_summary.csv", "Foldseek", "Foldseek"))
        rows.extend(collect_single_run_baseline(BASELINE_ROOT / "structure/global_structure_baseline_summary.csv", "TM-align", "TM-align"))
        method_order = MAIN_TABLE_METHOD_ORDER
    else:
        rows.extend(collect_proposed_model())
    if method_set == "all":
        for source_name, display_name in PLM_BASELINE_NAME_MAP.items():
            rows.extend(collect_seeded_baseline(BASELINE_ROOT / "plm/per_run_metrics.csv", source_name, display_name))
        rows.extend(collect_seeded_baseline(BASELINE_ROOT / "bilstm/per_run_metrics.csv", "ESM2 per-residue + BiLSTM", "ESM2 + BiLSTM"))
        for source_name, display_name in [
            ("BLASTP", "BLASTP"),
            ("PSI-BLAST", "PSI-BLAST"),
            ("HMMER (phmmer)", "HMMER"),
        ]:
            rows.extend(collect_single_run_baseline(BASELINE_ROOT / "retrieval/per_run_metrics.csv", source_name, display_name))
        rows.extend(collect_single_run_baseline(BASELINE_ROOT / "sprotfp/per_run_metrics.csv", "SProtFP direct prediction", "SProtFP"))
        rows.extend(collect_single_run_baseline(BASELINE_ROOT / "amp_scanner/per_run_metrics.csv", "AMP Scanner v2 retrained", "AMP Scanner"))
        rows.extend(collect_single_run_baseline(BASELINE_ROOT / "structure/global_structure_baseline_summary.csv", "Foldseek", "Foldseek"))
        rows.extend(collect_single_run_baseline(BASELINE_ROOT / "structure/global_structure_baseline_summary.csv", "TM-align", "TM-align"))
        method_order = ALL_METHOD_ORDER
    else:
        if method_set == "selected":
            rows.extend(collect_seeded_baseline(BASELINE_ROOT / "plm/per_run_metrics.csv", "esm2+MLP", "ESM2 + MLP"))
            rows.extend(collect_seeded_baseline(BASELINE_ROOT / "bilstm/per_run_metrics.csv", "ESM2 per-residue + BiLSTM", "ESM2 + BiLSTM"))
            rows.extend(collect_single_run_baseline(BASELINE_ROOT / "retrieval/per_run_metrics.csv", "BLASTP", "BLASTP"))
            rows.extend(collect_single_run_baseline(BASELINE_ROOT / "structure/global_structure_baseline_summary.csv", "Foldseek", "Foldseek"))
            method_order = METHOD_ORDER
    source = pd.DataFrame(rows)
    source["method"] = pd.Categorical(source["method"], method_order, ordered=True)
    source["metric"] = pd.Categorical(source["metric"], [label for _, label in METRICS], ordered=True)
    source["test_label"] = pd.Categorical(source["test_label"], [LABEL_TITLES[x] for x in LABELS], ordered=True)
    return source.sort_values(["test_label", "method", "metric"]).reset_index(drop=True)


def values_for(source: pd.DataFrame, test_label: str, method: str) -> list[float]:
    sub = source[source["test_label"].eq(test_label) & source["method"].eq(method)]
    metric_order = {label: idx for idx, (_, label) in enumerate(METRICS)}
    sub = sub.assign(metric_order=sub["metric"].map(metric_order)).sort_values("metric_order")
    return sub["mean"].astype(float).tolist()


def validate_source(source: pd.DataFrame, method_order: list[str]) -> None:
    expected_methods = set(method_order)
    expected_labels = {LABEL_TITLES[x] for x in LABELS}
    expected_metrics = {label for _, label in METRICS}
    observed_methods = set(source["method"].astype(str))
    observed_labels = set(source["test_label"].astype(str))
    observed_metrics = set(source["metric"].astype(str))
    if observed_methods != expected_methods:
        raise ValueError(f"Unexpected Figure 4 methods: {sorted(observed_methods)}")
    if observed_labels != expected_labels:
        raise ValueError(f"Unexpected Figure 4 labels: {sorted(observed_labels)}")
    if observed_metrics != expected_metrics:
        raise ValueError(f"Unexpected Figure 4 metrics: {sorted(observed_metrics)}")


def plot_radar(source: pd.DataFrame, output_dir: Path, *, method_order: list[str], output_name: str) -> None:
    num_vars = len(METRICS)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    is_dense_legend = len(method_order) > 6

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(8.05, 2.44 if is_dense_legend else 2.30),
        subplot_kw={"polar": True},
        constrained_layout=False,
    )
    fig.patch.set_facecolor("white")

    for ax, label in zip(axes, [LABEL_TITLES[x] for x in LABELS]):
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_facecolor("white")
        ax.spines["polar"].set_visible(False)
        ax.grid(color="#cfd6dc", linestyle="--", linewidth=0.40, alpha=0.72)
        ax.set_ylim(0.35, 1.0)
        ax.set_yticks([0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.4", "0.6", "0.8", "1.0"], fontsize=4.3, color="#6a6a6a")
        ax.set_rlabel_position(88)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([label for _, label in METRICS], fontsize=4.8)
        ax.tick_params(axis="x", pad=1.0)

        for method in method_order:
            vals = values_for(source, label, method)
            vals_closed = vals + vals[:1]
            is_primary = method in PRIMARY_METHODS
            line_width = 2.15 if is_primary else 0.62
            alpha = 1.0 if is_primary else 0.48
            zorder = 10 if is_primary else 2
            ax.plot(
                angles,
                vals_closed,
                color=METHOD_COLORS[method],
                linewidth=line_width,
                alpha=alpha,
                zorder=zorder,
                solid_capstyle="round",
            )
            ax.scatter(
                angles[:-1],
                vals,
                color=METHOD_COLORS[method],
                s=18 if is_primary else 4.2,
                marker=METHOD_MARKERS[method],
                edgecolor="white",
                linewidth=0.55 if is_primary else 0.35,
                alpha=alpha,
                zorder=zorder + 1,
            )
            if is_primary:
                ax.fill(angles, vals_closed, color=METHOD_COLORS[method], alpha=0.055, zorder=1)

    handles = [
        plt.Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linewidth=2.35 if method in PRIMARY_METHODS else 1.15,
            markersize=4.6 if method in PRIMARY_METHODS else 3.1,
            label=method,
        )
        for method in method_order
    ]
    ncol = 5 if not is_dense_legend else 6
    bottom = 0.19 if not is_dense_legend else 0.350
    legend_anchor = (0.035, 0.024, 0.930, 0.135) if is_dense_legend else (0.5, 0.0)
    legend_loc = "lower left" if is_dense_legend else "lower center"
    fig.legend(
        handles=handles,
        loc=legend_loc,
        bbox_to_anchor=legend_anchor,
        mode="expand" if is_dense_legend else None,
        ncol=ncol,
        fontsize=5.35 if not is_dense_legend else 5.85,
        columnspacing=1.10 if not is_dense_legend else 1.00,
        handlelength=1.52 if not is_dense_legend else 1.62,
        handletextpad=0.40,
        labelspacing=0.52 if is_dense_legend else 0.35,
        borderaxespad=0.0,
    )
    for text in fig.legends[0].get_texts():
        if text.get_text() in PRIMARY_METHODS:
            text.set_fontweight("bold")
    fig.subplots_adjust(left=0.030, right=0.992, top=0.765 if is_dense_legend else 0.790, bottom=bottom, wspace=0.30)

    # Place titles in figure coordinates instead of polar-axis coordinates.
    # This decouples the panel titles from the "Acc" radial tick label and
    # lets the radar circles sit lower without title overlap.
    title_y = 0.885 if is_dense_legend else 0.900
    for ax, label in zip(axes, [LABEL_TITLES[x] for x in LABELS]):
        bbox = ax.get_position()
        fig.text(
            (bbox.x0 + bbox.x1) / 2,
            title_y,
            label,
            ha="center",
            va="bottom",
            fontsize=7.8 if is_dense_legend else 7.2,
            fontweight="bold",
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / output_name
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Figure 4 strict cross-source radar panels.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--source-data",
        default=None,
        help="Locked Figure 4 source data. If omitted, the script uses the source CSV matching --method-set.",
    )
    parser.add_argument(
        "--refresh-source",
        action="store_true",
        help="Rebuild source data from run outputs instead of using the locked paper-workspace CSV.",
    )
    parser.add_argument("--method-set", choices=["selected", "main_table", "all"], default="selected")
    parser.add_argument(
        "--output-name",
        default=None,
        help="Output basename without extension. Defaults to selected/all-baseline Figure 4 names.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    source_dir = output_dir / "source_data"
    if args.method_set == "all":
        method_order = ALL_METHOD_ORDER
    elif args.method_set == "main_table":
        method_order = MAIN_TABLE_METHOD_ORDER
    else:
        method_order = METHOD_ORDER
    output_name = args.output_name
    if output_name is None:
        if args.method_set == "all":
            output_name = "Figure4_strict_cross_source_radar_all_baselines"
        elif args.method_set == "main_table":
            output_name = "Figure4_strict_cross_source_radar_main_table_baselines"
        else:
            output_name = "Figure4_strict_cross_source_radar"
    source_file = f"{output_name}_source.csv" if args.method_set in {"all", "main_table"} else "figure4_strict_cross_radar_source.csv"
    if args.source_data is None:
        source_path = PROJECT_ROOT / "results/figure_source_data/Figure4_cross_radar/source_data" / source_file
    else:
        source_path = resolve_path(args.source_data)
    setup_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    if source_path.exists() and not args.refresh_source:
        source = pd.read_csv(source_path)
    else:
        source = build_source_table(args.method_set)
    validate_source(source, method_order)
    source.to_csv(source_dir / source_file, index=False)
    plot_radar(source, output_dir, method_order=method_order, output_name=output_name)
    print(f"Saved Figure 4 outputs to {output_dir}")


if __name__ == "__main__":
    main()
