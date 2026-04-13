"""Figures V1–V6 for steering validation (margin analysis + random control)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.visualization.style import (
    BLUE,
    CATEGORY_COLORS,
    CATEGORY_LABELS,
    DPI,
    GRAY,
    GREEN,
    ORANGE,
    RED_ORANGE,
    WONG_PALETTE,
    apply_style,
)
from src.utils.io import ensure_dir
from src.utils.logging import log

try:
    import pandas as pd
except ImportError:
    pd = None

VERMILLION = RED_ORANGE
apply_style()

MARGIN_BIN_ORDER = [
    "near_indifferent", "low_confidence", "moderate", "confident", "very_confident",
]
MARGIN_BIN_LABELS = [
    "<0.5", "0.5–1", "1–2", "2–5", ">5",
]


def _save_both(fig: plt.Figure, path: str | Path, tight: bool = True) -> None:
    path = Path(path)
    if tight:
        fig.tight_layout()
    fig.savefig(str(path), dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(str(path.with_suffix(".pdf")), bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# V1: Correction by margin
# ---------------------------------------------------------------------------


def fig_correction_by_margin(
    margin_summary: dict[str, Any],
    random_summary: dict[str, Any] | None,
    output_dir: Path,
) -> None:
    """Bar chart: correction rate per margin bin, per category."""
    per_cat = margin_summary.get("per_category", {})

    for cat, data in per_cat.items():
        corr = data.get("correction_by_bin", {})
        if not corr:
            continue

        bins = [b for b in MARGIN_BIN_ORDER if b in corr]
        if not bins:
            continue

        rates = [corr[b].get("correction_rate", 0) for b in bins]
        fracs = [corr[b].get("fraction_of_corrections", 0) for b in bins]
        labels = [MARGIN_BIN_LABELS[MARGIN_BIN_ORDER.index(b)] for b in bins]

        fig, ax1 = plt.subplots(figsize=(8, 5))
        x = np.arange(len(bins))
        width = 0.4

        bars = ax1.bar(x - width / 2, rates, width, color=BLUE, label="Correction rate")
        ax1.set_ylabel("Correction rate", color=BLUE)

        # Fraction as second bar set
        ax1.bar(x + width / 2, fracs, width, color=GREEN, alpha=0.7,
                label="Fraction of all corrections")

        # Random control baseline (if available)
        # The random control is at the category level, not per-bin from the summary
        # We'll overlay as a dashed line if we have per-alpha random data
        if random_summary:
            rc = random_summary.get("per_category", {}).get(cat, {})
            rand_mean = rc.get("random_correction_mean", None)
            if rand_mean is not None:
                ax1.axhline(rand_mean, ls="--", color=GRAY, lw=1.5,
                            label=f"Random baseline ({rand_mean:.2f})")

        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=9)
        ax1.set_xlabel("Logit margin (nats)")
        cat_label = CATEGORY_LABELS.get(cat, cat)
        ax1.set_title(f"Correction rate by logit margin \u2014 {cat_label}")
        ax1.legend(fontsize=8, loc="upper right")

        _save_both(fig, output_dir / f"fig_correction_by_margin_{cat}.png")
        log(f"    Saved fig_correction_by_margin_{cat}")


# ---------------------------------------------------------------------------
# V2: Margin distribution
# ---------------------------------------------------------------------------


def fig_margin_distribution(
    margins_df: "pd.DataFrame",
    sweep_df: "pd.DataFrame | None",
    alpha: float,
    output_dir: Path,
) -> None:
    """Histogram of margins for corrected vs uncorrected items."""
    if margins_df is None or margins_df.empty:
        return

    for cat in sorted(margins_df["category"].unique()):
        cat_m = margins_df[margins_df["category"] == cat]
        stereo = cat_m[cat_m["is_stereotyped"] == True]  # noqa: E712
        if stereo.empty:
            continue

        # Determine which items were corrected
        corrected_idx = set()
        if sweep_df is not None and not sweep_df.empty:
            at_alpha = sweep_df[
                (sweep_df["alpha"] == alpha) & (sweep_df["category"] == cat)
            ]
            corrected_idx = set(
                at_alpha.loc[
                    at_alpha["flipped"] & at_alpha["steered_role"].isin(["non_stereotyped", "unknown"]),
                    "item_idx",
                ].values
            )

        corr_mask = stereo["item_idx"].isin(corrected_idx)
        margins_corr = stereo.loc[corr_mask, "margin"].values
        margins_uncorr = stereo.loc[~corr_mask, "margin"].values

        if len(margins_corr) == 0 and len(margins_uncorr) == 0:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        all_m = np.concatenate([margins_corr, margins_uncorr])
        bins = np.linspace(0, min(all_m.max() + 0.5, 10), 30)

        if len(margins_corr) > 0:
            ax.hist(margins_corr, bins=bins, color=BLUE, alpha=0.6,
                    label=f"Corrected (n={len(margins_corr)}, med={np.median(margins_corr):.2f})",
                    density=True)
        if len(margins_uncorr) > 0:
            ax.hist(margins_uncorr, bins=bins, color=VERMILLION, alpha=0.6,
                    label=f"Uncorrected (n={len(margins_uncorr)}, med={np.median(margins_uncorr):.2f})",
                    density=True)

        # Bin boundaries
        for _, lo, hi in [("", 0.5, 0.5), ("", 1.0, 1.0), ("", 2.0, 2.0), ("", 5.0, 5.0)]:
            if hi <= bins[-1]:
                ax.axvline(hi, ls=":", color=GRAY, lw=0.7, alpha=0.5)

        ax.set_xlabel("Logit margin (nats)")
        ax.set_ylabel("Density")
        cat_label = CATEGORY_LABELS.get(cat, cat)
        ax.set_title(f"Margin distribution: corrected vs. uncorrected \u2014 {cat_label}")
        ax.legend(fontsize=8)

        _save_both(fig, output_dir / f"fig_margin_distribution_{cat}.png")
        log(f"    Saved fig_margin_distribution_{cat}")


# ---------------------------------------------------------------------------
# V3: Random control overlay
# ---------------------------------------------------------------------------


def fig_random_control_overlay(
    random_summary: dict[str, Any],
    feature_results: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    """Feature steering vs random control curves."""
    per_cat = random_summary.get("per_category", {})

    for cat, rdata in per_cat.items():
        per_alpha = rdata.get("per_alpha_random", {})
        feat = feature_results.get(cat, {})
        feat_per_alpha = feat.get("per_alpha", {})

        if not per_alpha or not feat_per_alpha:
            continue

        # Build curves
        alphas_r = sorted(float(a) for a in per_alpha.keys())
        rand_mean = [per_alpha[str(a)]["random_flip_mean"] for a in alphas_r]
        rand_std = [per_alpha[str(a)]["random_flip_std"] for a in alphas_r]

        # Feature curve (use absolute alpha for x-axis)
        alphas_f = sorted(float(a) for a in feat_per_alpha.keys())
        feat_corr = [
            feat_per_alpha[str(a)].get("correction_rate", feat_per_alpha[str(a)].get("flip_rate", 0))
            for a in alphas_f
        ]

        # Use |alpha| for x-axis
        x_r = [abs(a) for a in alphas_r]
        x_f = [abs(a) for a in alphas_f]
        rand_mean_a = np.array(rand_mean)
        rand_std_a = np.array(rand_std)

        fig, ax = plt.subplots(figsize=(8, 5))

        # Random band
        ax.fill_between(x_r, rand_mean_a - rand_std_a, rand_mean_a + rand_std_a,
                        color=GRAY, alpha=0.25, label="Random \u00b11\u03c3")
        ax.plot(x_r, rand_mean, "o--", color=GRAY, markersize=4, label="Random mean")

        # Feature curve
        ax.plot(x_f, feat_corr, "s-", color=BLUE, markersize=6, label="Bias features")

        # Shade attributable gap
        # Interpolate random to feature alpha grid for shading
        rand_interp = np.interp(x_f, x_r, rand_mean)
        feat_arr = np.array(feat_corr)
        ax.fill_between(x_f, rand_interp, feat_arr,
                        where=(feat_arr > rand_interp),
                        color=BLUE, alpha=0.1, label="Attributable effect")

        ax.set_xlabel("|\u03b1| (steering coefficient magnitude)")
        ax.set_ylabel("Flip rate")
        cat_label = CATEGORY_LABELS.get(cat, cat)
        ax.set_title(f"Feature steering vs. random control \u2014 {cat_label}")
        ax.legend(fontsize=8)

        _save_both(fig, output_dir / f"fig_random_control_overlay_{cat}.png")
        log(f"    Saved fig_random_control_overlay_{cat}")


# ---------------------------------------------------------------------------
# V4: Attributable correction
# ---------------------------------------------------------------------------


def fig_attributable_correction(
    random_summary: dict[str, Any],
    output_dir: Path,
) -> None:
    """Grouped bars: total, random baseline, attributable correction per category."""
    per_cat = random_summary.get("per_category", {})
    cats = sorted(per_cat.keys())
    if not cats:
        return

    feat_vals = [per_cat[c].get("feature_correction", 0) for c in cats]
    rand_vals = [per_cat[c].get("random_correction_mean", 0) for c in cats]
    attr_vals = [per_cat[c].get("attributable_correction", 0) for c in cats]
    p_vals = [per_cat[c].get("correction_ttest_p", 1.0) for c in cats]
    labels = [CATEGORY_LABELS.get(c, c) for c in cats]

    x = np.arange(len(cats))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, feat_vals, width, color=BLUE, label="Total correction")
    ax.bar(x, rand_vals, width, color=GRAY, label="Random baseline")
    ax.bar(x + width, attr_vals, width, color=GREEN, label="Attributable")

    # Annotate p-values
    for i, p in enumerate(p_vals):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax.text(x[i] + width, max(attr_vals[i], 0) + 0.02, sig,
                ha="center", fontsize=8, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Rate")
    ax.set_title("Attributable correction rate after random control subtraction")
    ax.legend(fontsize=8)

    _save_both(fig, output_dir / "fig_attributable_correction.png")
    log("    Saved fig_attributable_correction")


# ---------------------------------------------------------------------------
# V5: Margin-stratified attribution heatmap
# ---------------------------------------------------------------------------


def fig_margin_stratified_attribution(
    margin_summary: dict[str, Any],
    random_summary: dict[str, Any] | None,
    output_dir: Path,
) -> None:
    """Heatmap: category × margin bin → attributable correction rate."""
    per_cat_m = margin_summary.get("per_category", {})
    cats = sorted(per_cat_m.keys())
    if not cats:
        return

    # Build matrix: for each (cat, bin), feature_correction - random baseline
    # Use category-level random as proxy for per-bin random
    bins_present = MARGIN_BIN_ORDER
    mat = np.full((len(cats), len(bins_present)), np.nan)

    for ci, cat in enumerate(cats):
        corr = per_cat_m[cat].get("correction_by_bin", {})
        rand_baseline = 0.0
        if random_summary:
            rand_baseline = (
                random_summary.get("per_category", {}).get(cat, {})
                .get("random_correction_mean", 0)
            )
        for bi, bname in enumerate(bins_present):
            if bname in corr:
                feat_rate = corr[bname].get("correction_rate", 0)
                mat[ci, bi] = feat_rate - rand_baseline

    fig, ax = plt.subplots(figsize=(9, max(4, len(cats) * 0.7)))

    vabs = max(abs(np.nanmin(mat)), abs(np.nanmax(mat)), 0.01)
    im = ax.imshow(mat, cmap="RdYlGn", vmin=-vabs, vmax=vabs, aspect="auto")

    ax.set_xticks(range(len(bins_present)))
    ax.set_xticklabels(MARGIN_BIN_LABELS, fontsize=9)
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels([CATEGORY_LABELS.get(c, c) for c in cats], fontsize=9)

    for ci in range(len(cats)):
        for bi in range(len(bins_present)):
            v = mat[ci, bi]
            if not np.isnan(v):
                color = "white" if abs(v) > vabs * 0.6 else "black"
                ax.text(bi, ci, f"{v:+.2f}", ha="center", va="center",
                        fontsize=8, color=color)

    fig.colorbar(im, ax=ax, label="Attributable correction rate", shrink=0.8)
    ax.set_xlabel("Logit margin bin")
    ax.set_title("Feature-attributable correction by logit margin")

    _save_both(fig, output_dir / "fig_margin_stratified_attribution.png")
    log("    Saved fig_margin_stratified_attribution")


# ---------------------------------------------------------------------------
# V6: Correction/corruption with random overlay
# ---------------------------------------------------------------------------


def fig_correction_corruption_with_random(
    random_summary: dict[str, Any],
    output_dir: Path,
) -> None:
    """Updated correction vs corruption chart with random baselines."""
    per_cat = random_summary.get("per_category", {})
    cats = sorted(per_cat.keys())
    if not cats:
        return

    feat_corr = [per_cat[c].get("feature_correction", 0) for c in cats]
    rand_corr = [per_cat[c].get("random_correction_mean", 0) for c in cats]
    feat_corrupt = [per_cat[c].get("feature_corruption", 0) for c in cats]
    rand_corrupt = [per_cat[c].get("random_corruption_mean", 0) for c in cats]
    labels = [CATEGORY_LABELS.get(c, c) for c in cats]

    x = np.arange(len(cats))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    # Hatched random baseline behind
    ax.bar(x - width / 2, rand_corr, width, color=GRAY, alpha=0.4,
           hatch="//", edgecolor=GRAY, label="Random (correction)")
    ax.bar(x + width / 2, rand_corrupt, width, color=GRAY, alpha=0.4,
           hatch="\\\\", edgecolor=GRAY, label="Random (corruption)")

    # Feature bars on top
    ax.bar(x - width / 2, feat_corr, width, color=BLUE, alpha=0.8,
           label="Feature correction")
    ax.bar(x + width / 2, feat_corrupt, width, color=VERMILLION, alpha=0.8,
           label="Feature corruption")

    # Annotate attributable ratios
    for i in range(len(cats)):
        attr_c = feat_corr[i] - rand_corr[i]
        attr_p = feat_corrupt[i] - rand_corrupt[i]
        if attr_c > 0.01:
            ratio = attr_p / attr_c if attr_c > 0.01 else float("inf")
            ax.text(i, max(feat_corr[i], feat_corrupt[i]) + 0.02,
                    f"attr: {ratio:.1f}x", ha="center", fontsize=7, color=GRAY)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Rate")
    ax.set_title("Correction vs. corruption with random baseline")
    ax.legend(fontsize=7, ncol=2)

    _save_both(fig, output_dir / "fig_correction_corruption_with_random.png")
    log("    Saved fig_correction_corruption_with_random")


# ---------------------------------------------------------------------------
# Generate all
# ---------------------------------------------------------------------------


def generate_validation_figures(
    margin_summary: dict[str, Any],
    random_summary: dict[str, Any] | None,
    margins_df: Any | None,
    sweep_dfs: dict[str, Any] | None,
    feature_results: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    """Generate all validation figures (V1–V6)."""
    fig_dir = ensure_dir(output_dir / "figures")
    log("  Generating validation figures ...")

    # V1
    fig_correction_by_margin(margin_summary, random_summary, fig_dir)

    # V2
    if margins_df is not None and sweep_dfs:
        for cat, sweep_df in sweep_dfs.items():
            # Get optimal alpha for this category
            feat = feature_results.get(cat, {})
            alpha = feat.get("optimal_alpha", -10)
            cat_margins = margins_df[margins_df["category"] == cat] if hasattr(margins_df, "empty") else None
            if cat_margins is not None and not cat_margins.empty:
                fig_margin_distribution(cat_margins, sweep_df, alpha, fig_dir)

    # V3
    if random_summary:
        fig_random_control_overlay(random_summary, feature_results, fig_dir)

    # V4
    if random_summary:
        fig_attributable_correction(random_summary, fig_dir)

    # V5
    fig_margin_stratified_attribution(margin_summary, random_summary, fig_dir)

    # V6
    if random_summary:
        fig_correction_corruption_with_random(random_summary, fig_dir)

    log("  Validation figures complete.")
