#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = (
    REPO_ROOT
    / "data"
    / "benchmark_datasets"
    / "main"
    / "processed_sequences.csv"
)
OUT_DIR = (
    REPO_ROOT
    / "figures"
    / "supplementary"
    / "Supplementary_Figure_S1_data_distribution"
)
SOURCE_DIR = OUT_DIR / "source_data"
AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
CLASS_NAMES = {1: "Antitoxin", 0: "Non-antitoxin"}
COLORS = {"Antitoxin": "#D98C7A", "Non-antitoxin": "#7BAED6"}


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


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    required = {"id", "label", "sequence", "cap200_length"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {DATA_PATH}: {sorted(missing)}")
    df = df.copy()
    df["label"] = df["label"].astype(int)
    df["class_name"] = df["label"].map(CLASS_NAMES)
    df["cap200_length"] = df["cap200_length"].astype(int)
    if df["class_name"].isna().any():
        raise ValueError("Unexpected label values found; expected labels 0 and 1 only.")
    return df


def ecdf(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    values = np.sort(values.astype(float))
    y = np.arange(1, len(values) + 1, dtype=float) / float(len(values))
    return values, y


def ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(a.astype(float))
    b = np.sort(b.astype(float))
    grid = np.sort(np.unique(np.concatenate([a, b])))
    cdf_a = np.searchsorted(a, grid, side="right") / len(a)
    cdf_b = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def wasserstein_1d(a: np.ndarray, b: np.ndarray, n_quantiles: int = 1000) -> float:
    quantiles = np.linspace(0.0, 1.0, n_quantiles)
    qa = np.quantile(a.astype(float), quantiles)
    qb = np.quantile(b.astype(float), quantiles)
    return float(np.mean(np.abs(qa - qb)))


def build_length_sources(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    length_rows = df[["id", "label", "class_name", "cap200_length", "length_bin"]].copy()
    bin_order = ["bin1_42_77", "bin2_78_85", "bin3_86_95", "bin4_96_100", "bin5_101_132", "bin6_133_200"]
    bin_labels = {
        "bin1_42_77": "42-77",
        "bin2_78_85": "78-85",
        "bin3_86_95": "86-95",
        "bin4_96_100": "96-100",
        "bin5_101_132": "101-132",
        "bin6_133_200": "133-200",
    }
    bin_rows = []
    for class_name in ["Antitoxin", "Non-antitoxin"]:
        sub = df[df["class_name"] == class_name]
        total = len(sub)
        for idx, bin_name in enumerate(bin_order):
            count = int((sub["length_bin"] == bin_name).sum())
            bin_rows.append(
                {
                    "class_name": class_name,
                    "length_bin": bin_name,
                    "length_range": bin_labels[bin_name],
                    "bin_order": idx,
                    "count": count,
                    "percentage": float(count / total * 100.0),
                }
            )
    bin_df = pd.DataFrame(bin_rows)
    pos = df.loc[df["label"] == 1, "cap200_length"].to_numpy()
    neg = df.loc[df["label"] == 0, "cap200_length"].to_numpy()
    summary = {
        "positive_n": int(len(pos)),
        "negative_n": int(len(neg)),
        "positive_mean_length": float(np.mean(pos)),
        "negative_mean_length": float(np.mean(neg)),
        "positive_median_length": float(np.median(pos)),
        "negative_median_length": float(np.median(neg)),
        "length_ks_distance": ks_distance(pos, neg),
        "length_wasserstein_distance_aa": wasserstein_1d(pos, neg),
    }
    return length_rows, bin_df, summary


def iter_standard_aas(sequence: str) -> Iterable[str]:
    for aa in str(sequence).upper():
        if aa in AA_ORDER:
            yield aa


def build_aa_sources(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows = []
    for class_name in ["Antitoxin", "Non-antitoxin"]:
        sub = df[df["class_name"] == class_name]
        counts = {aa: 0 for aa in AA_ORDER}
        total = 0
        for sequence in sub["sequence"]:
            for aa in iter_standard_aas(sequence):
                counts[aa] += 1
                total += 1
        for aa in AA_ORDER:
            rows.append(
                {
                    "class_name": class_name,
                    "amino_acid": aa,
                    "count": int(counts[aa]),
                    "frequency": float(counts[aa] / total),
                    "frequency_percent": float(counts[aa] / total * 100.0),
                    "total_standard_residues": int(total),
                }
            )
    aa_df = pd.DataFrame(rows)
    pivot = aa_df.pivot(index="amino_acid", columns="class_name", values="frequency_percent").reset_index()
    pivot["delta_antitoxin_minus_non_antitoxin"] = pivot["Antitoxin"] - pivot["Non-antitoxin"]
    summary = {
        "aa_mean_abs_delta_percentage_points": float(
            pivot["delta_antitoxin_minus_non_antitoxin"].abs().mean()
        ),
        "aa_max_abs_delta_percentage_points": float(
            pivot["delta_antitoxin_minus_non_antitoxin"].abs().max()
        ),
        "aa_pearson_r": float(np.corrcoef(pivot["Antitoxin"], pivot["Non-antitoxin"])[0, 1]),
    }
    return pivot, summary


def save_sources(
    length_rows: pd.DataFrame,
    length_bins: pd.DataFrame,
    aa_composition: pd.DataFrame,
    summary: Dict[str, float],
) -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    stale_histogram = SOURCE_DIR / "suppfig_s1_length_histogram_source.csv"
    if stale_histogram.exists():
        stale_histogram.unlink()
    length_rows.to_csv(SOURCE_DIR / "suppfig_s1_length_values_source.csv", index=False)
    length_bins.to_csv(SOURCE_DIR / "suppfig_s1_length_bin_composition_source.csv", index=False)
    aa_composition.to_csv(SOURCE_DIR / "suppfig_s1_amino_acid_composition_source.csv", index=False)
    pd.DataFrame([summary]).to_csv(SOURCE_DIR / "suppfig_s1_distribution_summary.csv", index=False)


def plot_panel_length(ax: plt.Axes, bin_df: pd.DataFrame, summary: Dict[str, float]) -> None:
    ordered = bin_df.sort_values("bin_order")
    labels = ordered.drop_duplicates("length_bin")["length_range"].tolist()
    x = np.arange(len(labels))
    width = 0.34
    for offset, class_name in [(-width / 2, "Non-antitoxin"), (width / 2, "Antitoxin")]:
        sub = ordered[ordered["class_name"] == class_name].sort_values("bin_order")
        ax.bar(
            x + offset,
            sub["percentage"],
            width=width,
            color=COLORS[class_name],
            alpha=0.82 if class_name == "Non-antitoxin" else 0.88,
            label=class_name,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
    ax.set_title("A  Length-matched distribution", loc="left", fontweight="bold", pad=7)
    ax.set_xlabel("Length bin after 200-aa cap")
    ax.set_ylabel("Sequences (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_ylim(0, max(1.0, float(bin_df["percentage"].max()) * 1.30))
    ax.grid(axis="y", color="#E8E8E8", linewidth=0.45, zorder=1)
    ax.legend(loc="upper right", handlelength=1.6, borderaxespad=0.2)
    text = (
        f"n={summary['positive_n']} / {summary['negative_n']}\n"
        "1:10 in each bin\n"
        f"median={summary['positive_median_length']:.0f}/{summary['negative_median_length']:.0f} aa"
    )
    ax.text(
        0.03,
        0.96,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.1,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#D9D9D9", linewidth=0.45),
    )


def plot_panel_aa(ax: plt.Axes, aa_df: pd.DataFrame, summary: Dict[str, float]) -> None:
    aa_df = aa_df.copy()
    aa_df["sort_value"] = aa_df[["Antitoxin", "Non-antitoxin"]].mean(axis=1)
    aa_df = aa_df.sort_values("sort_value", ascending=True).reset_index(drop=True)
    y = np.arange(len(aa_df))
    for _, row in aa_df.iterrows():
        yi = int(row.name)
        ax.plot(
            [row["Non-antitoxin"], row["Antitoxin"]],
            [yi, yi],
            color="#CFCFCF",
            linewidth=0.75,
            zorder=1,
        )
    ax.scatter(
        aa_df["Non-antitoxin"],
        y,
        s=19,
        color=COLORS["Non-antitoxin"],
        edgecolor="white",
        linewidth=0.35,
        label="Non-antitoxin",
        zorder=3,
    )
    ax.scatter(
        aa_df["Antitoxin"],
        y,
        s=19,
        color=COLORS["Antitoxin"],
        edgecolor="white",
        linewidth=0.35,
        label="Antitoxin",
        zorder=4,
    )
    ax.set_title("B  Amino-acid composition", loc="left", fontweight="bold", pad=7)
    ax.set_xlabel("Residue frequency (%)")
    ax.set_ylabel("Amino acid")
    ax.set_yticks(y)
    ax.set_yticklabels(aa_df["amino_acid"])
    ax.set_xlim(0, max(float(aa_df[["Antitoxin", "Non-antitoxin"]].max().max()) * 1.14, 10.0))
    ax.grid(axis="x", color="#E8E8E8", linewidth=0.45, zorder=0)
    ax.legend(loc="lower right", handletextpad=0.35, borderaxespad=0.2)
    text = (
        f"r={summary['aa_pearson_r']:.03f}\n"
        f"mean |Δ|={summary['aa_mean_abs_delta_percentage_points']:.02f} pp\n"
        f"max |Δ|={summary['aa_max_abs_delta_percentage_points']:.02f} pp"
    )
    ax.text(
        0.03,
        0.96,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.1,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#D9D9D9", linewidth=0.45),
    )


def build_figure(bin_df: pd.DataFrame, aa_df: pd.DataFrame, summary: Dict[str, float]) -> plt.Figure:
    fig = plt.figure(figsize=(7.05, 3.05))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.06, 1.0], wspace=0.33)
    ax_len = fig.add_subplot(gs[0, 0])
    ax_aa = fig.add_subplot(gs[0, 1])
    plot_panel_length(ax_len, bin_df, summary)
    plot_panel_aa(ax_aa, aa_df, summary)
    fig.subplots_adjust(bottom=0.22)
    return fig


def main() -> None:
    configure_matplotlib()
    df = load_dataset()
    length_rows, length_bins, length_summary = build_length_sources(df)
    aa_df, aa_summary = build_aa_sources(df)
    summary = {**length_summary, **aa_summary}
    save_sources(length_rows, length_bins, aa_df, summary)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = build_figure(length_bins, aa_df, summary)
    stem = OUT_DIR / "Supplementary_Figure_S1_data_distribution"
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {stem}.pdf")
    print(f"Saved: {stem}.png")
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()
