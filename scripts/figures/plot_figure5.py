#!/usr/bin/env python3
"""Plot Figure 5: AT6247 motif-level interpretability case study."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DATA_DIR = PROJECT_ROOT / "results/figure_source_data/Figure5_interpretability/source_data"
OUTPUT_DIR = PROJECT_ROOT / "figures/main_text/Figure5_interpretability"
SOURCE_DIR = OUTPUT_DIR / "source_data"

SAMPLE_ID = "AT6247"
DOMAIN_NAME = "RHH/CopG domain"
DOMAIN_START = 2
DOMAIN_END = 39
DOMAIN_COLOR = "#F2A7DA"
PROFILE_COLOR = "#C55A9F"


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
            "legend.frameon": False,
        }
    )


def read_profiles() -> tuple[pd.DataFrame, pd.DataFrame]:
    mean_df = pd.read_csv(SOURCE_DATA_DIR / "figure5_at6247_mean_saliency_source.csv")
    per_seed_df = pd.read_csv(SOURCE_DATA_DIR / "figure5_at6247_per_seed_saliency_source.csv")
    return mean_df, per_seed_df


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.070, 1.06, label, transform=ax.transAxes, fontsize=8.0, fontweight="bold", ha="left", va="top")


def plot_domain_track(ax: plt.Axes, positions: np.ndarray) -> None:
    ax.set_xlim(0.5, positions.max() + 0.5)
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="#c9c9c9", linewidth=1.5, zorder=1)
    rect = FancyBboxPatch(
        (DOMAIN_START, 0.23),
        DOMAIN_END - DOMAIN_START,
        0.54,
        boxstyle="round,pad=0.02,rounding_size=0.22",
        facecolor=DOMAIN_COLOR,
        edgecolor=PROFILE_COLOR,
        linewidth=0.7,
        alpha=0.95,
        zorder=2,
    )
    ax.add_patch(rect)
    ax.text(
        (DOMAIN_START + DOMAIN_END) / 2,
        0.50,
        DOMAIN_NAME,
        ha="center",
        va="center",
        fontsize=6.4,
        fontweight="bold",
        zorder=3,
    )
    for x in (DOMAIN_START, DOMAIN_END):
        ax.axvline(x=x, color="#1f1f1f", linestyle="--", linewidth=0.9, alpha=0.85, zorder=4)
    ax.text(-0.016, 0.5, "Domain", transform=ax.transAxes, ha="right", va="center", fontsize=6.3, fontweight="bold")
    ax.set_yticks([])
    ax.tick_params(axis="x", length=0, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_saliency_profile(ax: plt.Axes, mean_df: pd.DataFrame, per_seed_df: pd.DataFrame) -> None:
    positions = mean_df["position"].to_numpy()
    mean_saliency = mean_df["mean_saliency"].to_numpy()
    std_saliency = mean_df["std_saliency"].to_numpy()

    for _, seed_df in per_seed_df.groupby("model_seed"):
        ax.plot(seed_df["position"], seed_df["saliency"], color="#8b8b8b", linewidth=0.55, alpha=0.34, zorder=1)

    ax.fill_between(
        positions,
        mean_saliency - std_saliency,
        mean_saliency + std_saliency,
        color=PROFILE_COLOR,
        alpha=0.16,
        linewidth=0,
        zorder=2,
    )
    ax.plot(positions, mean_saliency, color=PROFILE_COLOR, linewidth=1.65, zorder=3)
    ax.scatter(
        positions,
        mean_saliency,
        c=mean_saliency,
        cmap="coolwarm",
        norm=TwoSlopeNorm(
            vcenter=0,
            vmin=min(-0.01, float(mean_saliency.min())),
            vmax=max(0.01, float(mean_saliency.max())),
        ),
        s=12,
        edgecolors="white",
        linewidths=0.15,
        zorder=4,
    )
    ax.axvspan(DOMAIN_START - 0.5, DOMAIN_END + 0.5, color=DOMAIN_COLOR, alpha=0.12, zorder=0)
    for x in (DOMAIN_START, DOMAIN_END):
        ax.axvline(x=x, color="#1f1f1f", linestyle="--", linewidth=0.9, alpha=0.85, zorder=5)
    ax.axhline(0, color="#202020", linestyle="--", linewidth=0.7, alpha=0.60)

    y_min = min(per_seed_df["saliency"].min(), (mean_saliency - std_saliency).min())
    y_max = max(per_seed_df["saliency"].max(), (mean_saliency + std_saliency).max())
    pad = max((y_max - y_min) * 0.12, 0.012)
    ax.set_xlim(0.5, positions.max() + 0.5)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_ylabel("Perturbation importance\n(delta logit)", fontsize=6.5, fontweight="bold")
    ax.tick_params(axis="x", length=0, labelbottom=False)
    ax.tick_params(axis="y", labelsize=5.7)
    ax.grid(axis="y", color="#dddddd", linewidth=0.45, alpha=0.75)
    legend_handles = [
        Line2D([0], [0], color="#8b8b8b", linewidth=0.7, alpha=0.55, label="individual seeds"),
        Patch(facecolor=PROFILE_COLOR, alpha=0.16, edgecolor="none", label="mean +/- SD"),
        Line2D([0], [0], color=PROFILE_COLOR, linewidth=1.7, label="seed mean"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=5.4, handlelength=1.55, borderaxespad=0.25)


def plot_heatmap(ax: plt.Axes, mean_df: pd.DataFrame) -> mpl.image.AxesImage:
    positions = mean_df["position"].to_numpy()
    values = mean_df["mean_saliency"].to_numpy()
    full = np.full(int(positions.max()), np.nan)
    full[positions.astype(int) - 1] = values
    abs_max = max(abs(np.nanmin(full)), abs(np.nanmax(full)), 1e-6)
    im = ax.imshow(
        full.reshape(1, -1),
        aspect="auto",
        cmap="coolwarm",
        norm=TwoSlopeNorm(vcenter=0, vmin=-abs_max, vmax=abs_max),
        extent=[0.5, positions.max() + 0.5, 0, 1],
        interpolation="nearest",
    )
    for x in (DOMAIN_START, DOMAIN_END):
        ax.axvline(x=x, color="#1f1f1f", linestyle="--", linewidth=0.9, alpha=0.85)
    ticks = np.array([1, 20, 40, 60, int(positions.max())])
    ticks = np.unique(ticks[ticks <= positions.max()])
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(int(x)) for x in ticks], fontsize=5.8)
    ax.set_yticks([0.5])
    ax.set_yticklabels(["Mean"], fontsize=5.8, fontweight="bold")
    ax.set_xlabel("Amino-acid position", fontsize=6.5, fontweight="bold", labelpad=1.3)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#8a8a8a")
        spine.set_linewidth(0.45)
    return im


def plot_summary_panel(ax: plt.Axes) -> None:
    display_metrics = pd.read_csv(SOURCE_DATA_DIR / "figure5_at6247_display_metrics_source.csv").set_index("metric")
    rows = [
        ("Domain AUPRC", display_metrics.loc["Domain AUPRC", "mean"], display_metrics.loc["Domain AUPRC", "std"], "#F2A7DA"),
        ("Top-L precision", display_metrics.loc["Top-L precision", "mean"], display_metrics.loc["Top-L precision", "std"], "#C55A9F"),
        ("Top-L overlap", display_metrics.loc["Top-L overlap", "mean"], display_metrics.loc["Top-L overlap", "std"], "#74C476"),
    ]
    y = np.arange(len(rows))[::-1]
    vals = np.array([r[1] for r in rows], dtype=float)
    errs = np.array([r[2] for r in rows], dtype=float)
    colors = [r[3] for r in rows]

    ax.barh(y, vals, xerr=errs, color=colors, edgecolor="#333333", linewidth=0.45, height=0.62, error_kw={"elinewidth": 0.65, "capsize": 2})
    ax.set_xlim(0, 1.08)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=5.8)
    ax.tick_params(axis="x", labelsize=5.7)
    ax.grid(axis="x", color="#dddddd", linewidth=0.45, alpha=0.75)
    ax.set_xlabel("Score", fontsize=6.4, fontweight="bold")
    ax.set_title("Domain enrichment and stability", fontsize=6.9, fontweight="bold", pad=2.5)
    for yi, val in zip(y, vals):
        ax.text(min(val + 0.030, 1.04), yi, f"{val:.2f}", ha="left", va="center", fontsize=5.2)



def main() -> None:
    setup_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    mean_df, per_seed_df = read_profiles()
    mean_df.to_csv(SOURCE_DIR / "figure5_at6247_mean_saliency_source.csv", index=False)
    per_seed_df.to_csv(SOURCE_DIR / "figure5_at6247_per_seed_saliency_source.csv", index=False)
    for filename in [
        "figure5_at6247_alignment_metrics_source.csv",
        "figure5_at6247_seed_stability_source.csv",
        "figure5_at6247_display_metrics_source.csv",
    ]:
        pd.read_csv(SOURCE_DATA_DIR / filename).to_csv(SOURCE_DIR / filename, index=False)

    positions = mean_df["position"].to_numpy()
    fig = plt.figure(figsize=(7.45, 3.55), constrained_layout=False)
    gs = gridspec.GridSpec(
        3,
        2,
        figure=fig,
        width_ratios=[4.75, 1.55],
        height_ratios=[0.36, 2.05, 0.36],
        hspace=0.075,
        wspace=0.35,
        left=0.075,
        right=0.970,
        top=0.88,
        bottom=0.145,
    )

    ax_domain = fig.add_subplot(gs[0, 0])
    ax_profile = fig.add_subplot(gs[1, 0], sharex=ax_domain)
    ax_heat = fig.add_subplot(gs[2, 0], sharex=ax_domain)
    ax_summary = fig.add_subplot(gs[:, 1])

    add_panel_label(ax_domain, "A")
    ax_summary.text(-0.28, 1.035, "B", transform=ax_summary.transAxes, fontsize=8.0, fontweight="bold", ha="left", va="top")
    plot_domain_track(ax_domain, positions)
    plot_saliency_profile(ax_profile, mean_df, per_seed_df)
    plot_heatmap(ax_heat, mean_df)
    plot_summary_panel(ax_summary)

    fig.suptitle("AT6247 mutation-augmented interpretability profile", fontsize=8.2, fontweight="bold", y=0.975)
    base = OUTPUT_DIR / "Figure5_AT6247_interpretability"
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved Figure 5 outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
