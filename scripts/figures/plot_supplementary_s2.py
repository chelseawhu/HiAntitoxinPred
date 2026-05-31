#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
TABLE_PATH = (
    REPO_ROOT
    / "results"
    / "table_source_data"
    / "supplementary"
    / "Supplementary_Table_S3_ablation_study.csv"
)
OUT_DIR = (
    REPO_ROOT
    / "figures"
    / "supplementary"
    / "Supplementary_Figure_S2_ablation_bar"
)
SOURCE_DIR = OUT_DIR / "source_data"

MODEL_ORDER = ["Full_Model", "No_CNN", "No_Transformer_CNNOnly", "No_Both"]
MODEL_LABELS = {
    "Full_Model": "Full",
    "No_CNN": "w/o\nDMCME",
    "No_Transformer_CNNOnly": "w/o\nTGCE",
    "No_Both": "No_Both",
}
ABLATION_DISPLAY_NAMES = {
    "Full_Model": "HiAntitoxinPred (PERE + DMCME + TGCE + GIAC)",
    "No_CNN": "w/o DMCME",
    "No_Transformer_CNNOnly": "w/o TGCE",
    "No_Both": "No_Both",
}
MODULE_GLOSSARY = {
    "PERE": "PLM-derived Evolutionary Residue Encoder",
    "DMCME": "Dynamic Multi-scale CNN Motif Encoder",
    "TGCE": "Transformer-based Global Context Encoder",
    "GIAC": "Gated Imbalance-Adaptive Classifier",
    "MAIA": "Mutation-Augmented Interpretability Analysis",
}
MODEL_COLORS = {
    "Full_Model": "#557FA8",
    "No_CNN": "#8FB8D8",
    "No_Transformer_CNNOnly": "#D8A272",
    "No_Both": "#D77F71",
}
METRIC_COLORS = {
    "F1": "#557FA8",
    "MCC": "#D8A272",
}


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "font.size": 7.0,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "legend.frameon": False,
            "figure.dpi": 180,
        }
    )


def load_ablation_table() -> pd.DataFrame:
    df = pd.read_csv(TABLE_PATH)
    required = {"Model", "Structure", "Threshold", "F1", "MCC", "Precision", "Recall", "TP/FP/TN/FN"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {TABLE_PATH}: {sorted(missing)}")
    unknown = sorted(set(df["Model"]) - set(MODEL_ORDER))
    if unknown:
        raise ValueError(f"Unexpected ablation models: {unknown}")
    df = df.copy()
    df["model_order"] = df["Model"].map({name: idx for idx, name in enumerate(MODEL_ORDER)})
    df = df.sort_values("model_order").reset_index(drop=True)
    df["ablation_display_name"] = df["Model"].map(ABLATION_DISPLAY_NAMES)
    df["plot_label"] = df["Model"].map(MODEL_LABELS).str.replace("\n", " ", regex=False)
    return df


def save_source_data(df: pd.DataFrame) -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SOURCE_DIR / "suppfig_s2_ablation_metrics_source.csv", index=False)
    pd.DataFrame(
        [{"abbreviation": key, "module_name": value} for key, value in MODULE_GLOSSARY.items()]
    ).to_csv(SOURCE_DIR / "suppfig_s2_module_glossary.csv", index=False)


def annotate_bars(ax: plt.Axes, bars, values: np.ndarray, *, y_offset: float = 0.003) -> None:
    for bar, value in zip(bars, values):
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        ax.text(
            x,
            y + y_offset,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=5.9,
            color="#333333",
            linespacing=0.9,
        )


def build_figure(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.35, 2.8))
    x = np.arange(len(df))
    width = 0.34
    f1_values = df["F1"].astype(float).to_numpy()
    mcc_values = df["MCC"].astype(float).to_numpy()

    bars_f1 = ax.bar(
        x - width / 2,
        f1_values,
        color=METRIC_COLORS["F1"],
        width=width,
        edgecolor="white",
        linewidth=0.45,
        label="F1-score",
        zorder=3,
    )
    bars_mcc = ax.bar(
        x + width / 2,
        mcc_values,
        color=METRIC_COLORS["MCC"],
        width=width,
        edgecolor="white",
        linewidth=0.45,
        label="MCC",
        zorder=3,
    )
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[name] for name in df["Model"]])
    ax.set_ylim(0.82, 0.96)
    ax.grid(axis="y", color="#E8E8E8", linewidth=0.45, zorder=1)
    ax.legend(loc="upper right", ncol=2, handlelength=1.3, columnspacing=1.0, borderaxespad=0.2)
    annotate_bars(ax, bars_f1, f1_values)
    annotate_bars(ax, bars_mcc, mcc_values)
    fig.subplots_adjust(bottom=0.24)
    return fig


def main() -> None:
    configure_matplotlib()
    df = load_ablation_table()
    save_source_data(df)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = build_figure(df)
    stem = OUT_DIR / "Supplementary_Figure_S2_ablation_bar"
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {stem}.pdf")
    print(f"Saved: {stem}.png")
    print(df[["Model", "F1", "MCC"]].to_string(index=False))


if __name__ == "__main__":
    main()
