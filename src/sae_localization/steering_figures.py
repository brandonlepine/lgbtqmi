"""Figure generation for SAE steering experiments (Figures 20–28).

All figures use Wong colorblind-safe palette, save as PNG + PDF.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

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
from src.utils.logging import log

try:
    import pandas as pd
except ImportError:
    pd = None


def _require_pandas() -> None:
    global pd
    if pd is not None:
        return
    try:
        import pandas as _pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required for SAE steering figure generation. Install with: pip install pandas"
        ) from exc
    pd = _pd

VERMILLION = RED_ORANGE
ANTI_BIAS_BLUE = BLUE

apply_style()


def _save_both(fig: plt.Figure, path: str | Path, tight: bool = True) -> None:
    path = Path(path)
    if tight:
        fig.tight_layout()
    fig.savefig(str(path), dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(str(path.with_suffix(".pdf")), bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 20: Alpha sweep
# ---------------------------------------------------------------------------


def fig_alpha_sweep(
    exp_a_results: dict[str, Any],
    category: str,
    output_dir: Path,
) -> None:
    """Line plot: correction_rate and degeneration_rate vs alpha."""
    per_alpha = exp_a_results.get("per_alpha", {})
    if not per_alpha:
        return

    # per_alpha keys may be "-50" or "-50.0" depending on how JSON was written.
    alpha_keys = list(per_alpha.keys())
    try:
        alpha_keys_sorted = sorted(alpha_keys, key=lambda k: float(k))
    except Exception:
        # Fallback: stable order if keys are unexpected
        alpha_keys_sorted = alpha_keys

    alphas = []
    correction = []
    degen = []
    for k in alpha_keys_sorted:
        try:
            a = float(k)
        except Exception:
            continue
        d = per_alpha.get(k, {}) or {}
        alphas.append(a)
        correction.append(d.get("correction_rate", 0))
        degen.append(d.get("degeneration_rate", 0))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alphas, correction, "o-", color=BLUE, label="Correction rate", markersize=6)
    ax.plot(alphas, degen, "s-", color=VERMILLION, label="Degeneration rate", markersize=6)

    # Mark optimal
    opt = exp_a_results.get("optimal_alpha", 0)
    if opt != 0:
        ax.axvline(opt, ls="--", color=GRAY, lw=0.8, alpha=0.6)
        ax.annotate(f"optimal α={opt}", xy=(opt, 0.02), fontsize=8, color=GRAY)

    ax.set_xlabel("Steering coefficient (α)")
    ax.set_ylabel("Rate")
    cat_label = CATEGORY_LABELS.get(category, category)
    ax.set_title(f"Steering coefficient sweep \u2014 {cat_label}")
    ax.legend()
    ax.set_ylim(-0.02, 1.02)

    _save_both(fig, output_dir / f"fig_alpha_sweep_{category}.png")
    log(f"    Saved fig_alpha_sweep_{category}")


# ---------------------------------------------------------------------------
# Figure 21: Correction vs corruption
# ---------------------------------------------------------------------------


def fig_correction_vs_corruption(
    exp_a_all: dict[str, dict],
    exp_b_all: dict[str, dict],
    output_dir: Path,
) -> None:
    """Grouped bar chart: correction (A) vs corruption (B) per category."""
    cats = sorted(set(exp_a_all.keys()) | set(exp_b_all.keys()))
    if not cats:
        return

    labels = [CATEGORY_LABELS.get(c, c) for c in cats]
    correction = [
        exp_a_all.get(c, {}).get("optimal_rates", {}).get("correction_rate", 0) for c in cats
    ]
    corruption = [
        exp_b_all.get(c, {}).get("optimal_rates", {}).get("corruption_rate", 0) for c in cats
    ]

    x = np.arange(len(cats))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, correction, width, color=BLUE, label="Correction (Exp A)")
    ax.bar(x + width / 2, corruption, width, color=VERMILLION, label="Corruption (Exp B)")

    for i in range(len(cats)):
        if correction[i] > 0:
            ratio = corruption[i] / correction[i]
            ax.text(i, max(correction[i], corruption[i]) + 0.02, f"{ratio:.1f}x",
                    ha="center", fontsize=7, color=GRAY)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Rate")
    ax.set_title("Correction vs. corruption rates at optimal α")
    ax.legend()

    _save_both(fig, output_dir / "fig_correction_vs_corruption.png")
    log("    Saved fig_correction_vs_corruption")


# ---------------------------------------------------------------------------
# Figure 22: Individual feature causal
# ---------------------------------------------------------------------------


def fig_individual_feature_causal(
    indiv_df: "pd.DataFrame",
    category: str,
    output_dir: Path,
) -> None:
    """Bar chart of correction_rate per individual feature."""
    if indiv_df is None or indiv_df.empty:
        return

    df = indiv_df.sort_values("correction_rate", ascending=False)
    n = len(df)
    if n == 0:
        return

    fig, ax = plt.subplots(figsize=(max(8, n * 0.4), 5))
    x = np.arange(n)
    colors = [BLUE if r > 0.05 else GRAY for r in df["correction_rate"]]

    ax.bar(x, df["correction_rate"].values, color=colors, width=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"F{int(f)}" for f in df["feature_idx"]], rotation=90, fontsize=6)
    ax.set_ylabel("Correction rate")
    cat_label = CATEGORY_LABELS.get(category, category)
    ax.set_title(f"Individual feature causal validation \u2014 {cat_label}")

    _save_both(fig, output_dir / f"fig_individual_feature_causal_{category}.png")
    log(f"    Saved fig_individual_feature_causal_{category}")


# ---------------------------------------------------------------------------
# Figure 23: CrowS-Pairs transfer
# ---------------------------------------------------------------------------


def fig_crows_pairs_transfer(
    exp_d: dict[str, Any],
    output_dir: Path,
) -> None:
    """Paired bars + flip rate for CrowS-Pairs categories."""
    per_type = exp_d.get("per_bias_type", {})
    if not per_type:
        return

    types = sorted(per_type.keys())
    n = len(types)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(n)
    width = 0.35
    orig = [per_type[t]["stereo_rate_orig"] for t in types]
    steered = [per_type[t]["stereo_rate_steered"] for t in types]
    flips = [per_type[t]["flip_rate"] for t in types]

    ax1.bar(x - width / 2, orig, width, color=VERMILLION, alpha=0.7, label="Original")
    ax1.bar(x + width / 2, steered, width, color=BLUE, alpha=0.7, label="Steered")
    ax1.set_xticks(x)
    ax1.set_xticklabels(types, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Stereotype preference rate")
    ax1.set_title("Stereotype preference")
    ax1.legend(fontsize=8)

    ax2.bar(x, flips, color=GREEN, width=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(types, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Flip rate")
    ax2.set_title("Preference flip rate")

    fig.suptitle("Cross-dataset transfer to CrowS-Pairs", fontsize=12)
    _save_both(fig, output_dir / "fig_crows_pairs_transfer.png")
    log("    Saved fig_crows_pairs_transfer")


# ---------------------------------------------------------------------------
# Figure 24: MMLU side effects
# ---------------------------------------------------------------------------


def fig_side_effects_mmlu(
    exp_e: dict[str, Any],
    output_dir: Path,
) -> None:
    """Scatter: accuracy delta per MMLU subject (colored by flip rate)."""
    mmlu = exp_e.get("mmlu", {})
    per_subject = mmlu.get("per_subject", {})
    if not per_subject:
        return

    subjects = sorted(per_subject.keys())
    deltas = [per_subject[s]["delta"] for s in subjects]
    flip_rates = [per_subject[s].get("flip_rate", 0.0) for s in subjects]
    overall_delta = mmlu.get("delta", 0)
    overall_flip = mmlu.get("flip_rate", None)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(subjects))
    # Color by flip rate (if present), else fallback to delta magnitude.
    if any(fr > 0 for fr in flip_rates):
        colors = flip_rates
        sc = ax.scatter(x, deltas, c=colors, s=35, zorder=3, cmap="viridis", vmin=0.0, vmax=max(max(flip_rates), 1e-6))
        cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
        cbar.set_label("Flip rate")
    else:
        colors = [VERMILLION if abs(d) > 0.02 else GRAY for d in deltas]
        ax.scatter(x, deltas, c=colors, s=30, zorder=3)

    ax.axhline(0, ls="--", color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=90, fontsize=6)
    ax.set_ylabel("Accuracy delta (steered − original)")
    extra = f", flip={overall_flip:.3f}" if isinstance(overall_flip, (float, int)) else ""
    ax.set_title(f"MMLU accuracy impact (overall Δ = {overall_delta:+.3f}{extra})")

    _save_both(fig, output_dir / "fig_side_effects_mmlu.png")
    log("    Saved fig_side_effects_mmlu")


# ---------------------------------------------------------------------------
# Figure 25: MedQA demographic
# ---------------------------------------------------------------------------


def fig_medqa_demographic(
    exp_e: dict[str, Any],
    output_dir: Path,
) -> None:
    """Legacy bar chart: MedQA accuracy split by demographic flag.

    This is a *global pooled steering* view and is not subgroup-conditioned.
    Prefer `fig_medqa_subgroup_conditional` when available.
    """
    medqa = exp_e.get("medqa", {})
    if not medqa:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ["All items", "Demographic items"]
    orig = [medqa.get("accuracy_original", 0), medqa.get("demographic_accuracy_original", 0)]
    steered = [medqa.get("accuracy_steered", 0), medqa.get("demographic_accuracy_steered", 0)]
    flips = [medqa.get("flip_rate", 0), medqa.get("demographic_flip_rate", 0)]
    ns = [medqa.get("n_items", 0), medqa.get("n_demographic", 0)]

    x = np.arange(len(labels))
    width = 0.3
    ax.bar(x - width / 2, orig, width, color=VERMILLION, alpha=0.7, label="Original")
    ax.bar(x + width / 2, steered, width, color=BLUE, alpha=0.7, label="Steered")

    # Annotate flip rates + n
    for i in range(len(labels)):
        ax.text(
            x[i],
            max(orig[i], steered[i]) + 0.02,
            f"flip={flips[i]:.2f}\\nn={ns[i]}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=GRAY,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Accuracy")
    ax.set_title("MedQA accuracy (LEGACY: global pooled steering)")
    ax.legend()

    _save_both(fig, output_dir / "fig_medqa_demographic.png")
    log("    Saved fig_medqa_demographic")


# ---------------------------------------------------------------------------
# Figure 25b: MedQA subgroup-conditioned controls (preferred)
# ---------------------------------------------------------------------------


def fig_medqa_subgroup_conditional(
    exp_e: dict[str, Any],
    output_dir: Path,
) -> None:
    """Summary of subgroup-conditioned MedQA steering + controls (E2b).

    Plots Δ accuracy and flip rate for:
    - matched_suppress / matched_amplify
    - mismatched_suppress / mismatched_amplify
    - random_on_untagged_suppress / random_on_untagged_amplify
    """
    rows = exp_e.get("medqa_subgroup_conditional", None)
    if not rows:
        return

    # Stable condition order
    order = [
        "matched_suppress",
        "matched_amplify",
        "mismatched_suppress",
        "mismatched_amplify",
        "random_on_untagged_suppress",
        "random_on_untagged_amplify",
    ]
    rows2 = [r for r in rows if r.get("condition") in order]
    if not rows2:
        return
    rows2.sort(key=lambda r: order.index(r.get("condition")))

    labels = [r["condition"].replace("_", "\n") for r in rows2]
    deltas = [float(r.get("delta", 0.0)) for r in rows2]
    flips = [float(r.get("flip_rate", 0.0)) for r in rows2]
    ns = [int(r.get("n_items", 0)) for r in rows2]
    demo_mode = rows2[0].get("demo_mode", "unknown")

    x = np.arange(len(labels))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    # Delta bars
    colors = [GREEN if d > 0 else VERMILLION for d in deltas]
    ax1.bar(x, deltas, color=colors, alpha=0.8)
    ax1.axhline(0, color=GRAY, lw=0.8)
    ax1.set_ylabel("Δ accuracy (steered − baseline)")
    ax1.set_title(f"MedQA subgroup-conditioned steering + controls (demo_mode={demo_mode})")
    for i in range(len(x)):
        ax1.text(x[i], deltas[i] + (0.002 if deltas[i] >= 0 else -0.002), f"n={ns[i]}", ha="center", va="bottom" if deltas[i] >= 0 else "top", fontsize=8, color=GRAY)

    # Flip rate bars
    ax2.bar(x, flips, color=BLUE, alpha=0.7)
    ax2.set_ylabel("Flip rate")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)

    _save_both(fig, output_dir / "fig_medqa_subgroup_conditional.png")
    log("    Saved fig_medqa_subgroup_conditional")


# ---------------------------------------------------------------------------
# Figure 26: Experiment summary table
# ---------------------------------------------------------------------------


def fig_experiment_summary(
    summary: dict[str, Any],
    output_dir: Path,
) -> None:
    """Formatted table figure summarising all experiments."""
    exp_a = summary.get("experiment_A", {}).get("per_category", {})
    exp_b = summary.get("experiment_B", {}).get("per_category", {})

    cats = sorted(set(list(exp_a.keys()) + list(exp_b.keys())))
    if not cats:
        return

    col_labels = ["Category", "Correction (A)", "Corruption (B)", "MMLU Δ"]
    rows = []
    for c in cats:
        corr = exp_a.get(c, {}).get("optimal_rates", {}).get("correction_rate", 0)
        corrupt = exp_b.get(c, {}).get("optimal_rates", {}).get("corruption_rate", 0)
        exp_e = summary.get("experiment_E", {}) or {}
        mmlu_by_cat = exp_e.get("mmlu_by_category", {}) if isinstance(exp_e, dict) else {}
        if isinstance(mmlu_by_cat, dict) and c in mmlu_by_cat:
            mmlu_d = mmlu_by_cat[c].get("delta", 0)
        else:
            # Fallback: global Experiment E delta (same for all categories)
            mmlu_d = exp_e.get("mmlu", {}).get("delta", 0) if isinstance(exp_e, dict) else 0
        rows.append([CATEGORY_LABELS.get(c, c), f"{corr:.3f}", f"{corrupt:.3f}", f"{mmlu_d:+.3f}"])

    fig, ax = plt.subplots(figsize=(8, max(3, len(rows) * 0.5 + 1)))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    ax.set_title("Experiment summary", fontsize=12, pad=20)

    _save_both(fig, output_dir / "fig_experiment_summary.png", tight=False)
    log("    Saved fig_experiment_summary")


# ---------------------------------------------------------------------------
# Figure 27: Steering asymmetry
# ---------------------------------------------------------------------------


def fig_steering_asymmetry(
    exp_a_all: dict[str, dict],
    exp_b_all: dict[str, dict],
    output_dir: Path,
) -> None:
    """Per-category subplots: correction vs corruption rate at each |alpha|."""
    cats = sorted(set(exp_a_all.keys()) & set(exp_b_all.keys()))
    if not cats:
        return

    n = len(cats)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows))
    if n == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    for idx, cat in enumerate(cats):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        a_alphas = exp_a_all[cat].get("per_alpha", {})
        b_alphas = exp_b_all[cat].get("per_alpha", {})

        # Correction rates at |alpha|
        a_vals = sorted(
            [(abs(float(a)), d.get("correction_rate", 0)) for a, d in a_alphas.items()],
        )
        b_vals = sorted(
            [(abs(float(a)), d.get("corruption_rate", 0)) for a, d in b_alphas.items()],
        )

        if a_vals:
            ax.plot([v[0] for v in a_vals], [v[1] for v in a_vals],
                    "o-", color=BLUE, label="Correction", markersize=4)
        if b_vals:
            ax.plot([v[0] for v in b_vals], [v[1] for v in b_vals],
                    "s-", color=VERMILLION, label="Corruption", markersize=4)

        cat_label = CATEGORY_LABELS.get(cat, cat)
        ax.set_title(cat_label, fontsize=9)
        ax.set_xlabel("|α|", fontsize=8)
        ax.set_ylabel("Rate", fontsize=8)
        if idx == 0:
            ax.legend(fontsize=7)

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle("Steering asymmetry: correction vs corruption", fontsize=12)
    _save_both(fig, output_dir / "fig_steering_asymmetry.png")
    log("    Saved fig_steering_asymmetry")


# ---------------------------------------------------------------------------
# Figure 28: Cross-subgroup transfer heatmap
# ---------------------------------------------------------------------------


def fig_subgroup_transfer_heatmap(
    transfer_result: dict[str, Any],
    category: str,
    output_dir: Path,
) -> None:
    """Heatmap: source subgroup features × target subgroup items → flip rate."""
    matrix = transfer_result.get("matrix", {})
    sources = transfer_result.get("sources", [])
    targets = transfer_result.get("targets", [])

    if not sources or not targets:
        return

    mat = np.zeros((len(sources), len(targets)))
    for i, s in enumerate(sources):
        for j, t in enumerate(targets):
            mat[i, j] = matrix.get(s, {}).get(t, {}).get("flip_rate", 0)

    fig, ax = plt.subplots(figsize=(max(6, len(targets) * 1.2), max(5, len(sources) * 0.8)))
    im = ax.imshow(mat, cmap="YlOrRd", vmin=0, aspect="auto")

    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(targets, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(sources)))
    ax.set_yticklabels(sources, fontsize=9)
    ax.set_xlabel("Target subgroup (items)")
    ax.set_ylabel("Source subgroup (features)")

    for i in range(len(sources)):
        for j in range(len(targets)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if mat[i, j] > 0.3 else "black")

    fig.colorbar(im, ax=ax, label="Flip rate", shrink=0.8)
    cat_label = CATEGORY_LABELS.get(category, category)
    ax.set_title(f"Cross-subgroup steering transfer \u2014 {cat_label}")

    _save_both(fig, output_dir / f"fig_subgroup_steering_transfer_{category}.png")
    log(f"    Saved fig_subgroup_steering_transfer_{category}")


# ---------------------------------------------------------------------------
# Generate all
# ---------------------------------------------------------------------------


def generate_all_steering_figures(
    summary: dict[str, Any],
    exp_a_all: dict[str, dict],
    exp_b_all: dict[str, dict],
    exp_d: dict[str, Any] | None,
    exp_e: dict[str, Any] | None,
    indiv_results: dict[str, Any] | None,
    transfer_results: dict[str, dict] | None,
    output_dir: Path,
) -> None:
    """Generate all steering experiment figures (20–28)."""
    from src.utils.io import ensure_dir
    fig_dir = ensure_dir(output_dir / "figures")

    log("  Generating steering figures ...")

    # Fig 20: Alpha sweeps
    for cat, result in exp_a_all.items():
        fig_alpha_sweep(result, cat, fig_dir)

    # Fig 21: Correction vs corruption
    fig_correction_vs_corruption(exp_a_all, exp_b_all, fig_dir)

    # Fig 22: Individual feature causal
    if indiv_results:
        for cat, df in indiv_results.items():
            if df is not None and not df.empty:
                fig_individual_feature_causal(df, cat, fig_dir)

    # Fig 23: CrowS-Pairs
    if exp_d and not exp_d.get("skipped"):
        fig_crows_pairs_transfer(exp_d, fig_dir)

    # Fig 24: MMLU
    if exp_e and "mmlu" in exp_e:
        fig_side_effects_mmlu(exp_e, fig_dir)

    # Fig 25: MedQA
    if exp_e and "medqa" in exp_e:
        fig_medqa_demographic(exp_e, fig_dir)
    # Fig 25b: MedQA subgroup-conditioned controls (preferred)
    if exp_e and "medqa_subgroup_conditional" in exp_e:
        fig_medqa_subgroup_conditional(exp_e, fig_dir)

    # Fig 26: Summary table
    fig_experiment_summary(summary, fig_dir)

    # Fig 27: Asymmetry
    fig_steering_asymmetry(exp_a_all, exp_b_all, fig_dir)

    # Fig 28: Subgroup transfer heatmaps
    if transfer_results:
        for cat, result in transfer_results.items():
            fig_subgroup_transfer_heatmap(result, cat, fig_dir)

    log("  Steering figures complete.")
