"""Cross-subgroup alpha-sweep analysis and figures.

For each category with >=2 subgroups of significant features, sweep alpha
values and produce transfer heatmaps that reveal the transition from
non-specific (high alpha) to subgroup-specific (low alpha) steering.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils.io import atomic_save_json, ensure_dir
from src.utils.logging import log
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

try:
    import pandas as pd
except ImportError:
    pd = None

apply_style()

VERMILLION = RED_ORANGE
DEFAULT_ALPHAS = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50]


def _save_both(fig: plt.Figure, path: str | Path, tight: bool = True) -> None:
    path = Path(path)
    if tight:
        fig.tight_layout()
    fig.savefig(str(path), dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(str(path.with_suffix(".pdf")), bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def get_eligible_categories(
    per_sub_df: "pd.DataFrame",
    direction: str = "pro_bias",
    min_subgroups: int = 2,
) -> dict[str, list[str]]:
    """Return categories with >=min_subgroups that have significant pro_bias features.

    Returns dict: category → list of subgroup names.
    """
    sig = per_sub_df[
        (per_sub_df["is_significant"]) & (per_sub_df["direction"] == direction)
    ]
    out: dict[str, list[str]] = {}
    for cat in sig["category"].unique():
        subs = sorted(sig.loc[sig["category"] == cat, "subcategory"].unique())
        if len(subs) >= min_subgroups:
            out[cat] = subs
    return out


def get_subgroup_features(
    per_sub_df: "pd.DataFrame",
    category: str,
    subcategory: str,
    direction: str = "pro_bias",
) -> list[int]:
    """Get feature indices for a specific subgroup."""
    mask = (
        (per_sub_df["category"] == category)
        & (per_sub_df["subcategory"] == subcategory)
        & (per_sub_df["direction"] == direction)
        & (per_sub_df["is_significant"])
    )
    return per_sub_df.loc[mask, "feature_idx"].tolist()


def load_jaccard_pairs(
    overlap_path: Path, category: str,
) -> list[tuple[str, str, float]]:
    """Load pairwise Jaccard values for a category's subgroups."""
    with open(overlap_path) as f:
        data = json.load(f)
    spec = data.get("subgroup_specificity", {}).get(category, {})
    jaccard = spec.get("jaccard", {})
    pairs = []
    for src, targets in jaccard.items():
        for tgt, val in targets.items():
            if src != tgt and isinstance(val, (int, float)):
                pairs.append((src, tgt, float(val)))
    return pairs


def partition_items_by_subgroup(
    items: list[dict[str, Any]],
    stereotyped_only: bool = True,
    max_per_subgroup: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Partition items by stereotyped_groups, keeping only stereotyped responses."""
    by_sub: dict[str, list[dict[str, Any]]] = {}
    for it in items:
        if stereotyped_only:
            role = it.get("model_answer_role", "")
            is_stereo = it.get("is_stereotyped_response", False)
            if role != "stereotyped_target" and not is_stereo:
                continue
        for sg in it.get("stereotyped_groups", []):
            by_sub.setdefault(sg, []).append(it)

    if max_per_subgroup is not None:
        by_sub = {k: v[:max_per_subgroup] for k, v in by_sub.items()}
    return by_sub


# ---------------------------------------------------------------------------
# Core sweep logic
# ---------------------------------------------------------------------------


def run_subgroup_alpha_sweep(
    steerer: Any,
    items_by_subgroup: dict[str, list[dict[str, Any]]],
    features_by_subgroup: dict[str, list[int]],
    alpha_values: list[float],
    prompt_formatter: Callable,
    category: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Run cross-subgroup transfer at each alpha value.

    Returns dict with:
      - flip_rates: np.ndarray (n_alphas, n_sources, n_targets)
      - alpha_values, source_subgroups, target_subgroups
    """
    sources = sorted(features_by_subgroup.keys())
    targets = sorted(items_by_subgroup.keys())
    n_alphas = len(alpha_values)
    n_sources = len(sources)
    n_targets = len(targets)

    flip_rates = np.full((n_alphas, n_sources, n_targets), np.nan)

    log(f"  Sweep: {category} — {n_sources} sources x {n_targets} targets "
        f"x {n_alphas} alphas")

    for ai, alpha in enumerate(alpha_values):
        # Check for resume
        checkpoint_path = output_dir / f"{category}_alpha_{alpha}.json"
        if checkpoint_path.exists():
            with open(checkpoint_path) as f:
                ckpt = json.load(f)
            for si, src in enumerate(sources):
                for ti, tgt in enumerate(targets):
                    val = ckpt.get("matrix", {}).get(src, {}).get(tgt, {}).get("flip_rate")
                    if val is not None:
                        flip_rates[ai, si, ti] = val
            log(f"    alpha={alpha}: loaded from checkpoint")
            continue

        log(f"    alpha={alpha} ...")
        alpha_matrix: dict[str, dict[str, dict[str, Any]]] = {}

        for si, src in enumerate(sources):
            features = features_by_subgroup[src]
            if not features:
                continue

            # Dampening: negative alpha
            vec = steerer.get_composite_steering(features, -abs(alpha))
            alpha_matrix[src] = {}

            for ti, tgt in enumerate(targets):
                tgt_items = items_by_subgroup.get(tgt, [])
                if not tgt_items:
                    alpha_matrix[src][tgt] = {"n_items": 0, "flip_rate": 0.0, "n_flipped": 0}
                    flip_rates[ai, si, ti] = 0.0
                    continue

                n_flipped = 0
                n_items = len(tgt_items)

                for item in tgt_items:
                    prompt = prompt_formatter(item)
                    baseline = steerer.evaluate_baseline(prompt)
                    result = steerer.steer_and_evaluate(prompt, vec)

                    if result["model_answer"] != baseline["model_answer"]:
                        n_flipped += 1

                rate = n_flipped / max(n_items, 1)
                flip_rates[ai, si, ti] = rate
                alpha_matrix[src][tgt] = {
                    "n_items": n_items,
                    "flip_rate": rate,
                    "n_flipped": n_flipped,
                }

            log(f"      {src}: " + ", ".join(
                f"{tgt}={alpha_matrix[src].get(tgt, {}).get('flip_rate', 0):.2f}"
                for tgt in targets
            ))

        # Save checkpoint
        atomic_save_json(
            {"alpha": alpha, "category": category, "matrix": alpha_matrix,
             "sources": sources, "targets": targets},
            checkpoint_path,
        )

        # Memory management
        import torch
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save full sweep
    np.savez(
        output_dir / f"{category}.npz",
        flip_rates=flip_rates,
        alpha_values=np.array(alpha_values),
        source_subgroups=np.array(sources),
        target_subgroups=np.array(targets),
    )

    return {
        "flip_rates": flip_rates,
        "alpha_values": alpha_values,
        "source_subgroups": sources,
        "target_subgroups": targets,
        "category": category,
    }


def load_sweep_results(sweep_dir: Path, category: str) -> dict[str, Any] | None:
    """Load previously saved sweep results."""
    path = sweep_dir / f"{category}.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return {
        "flip_rates": data["flip_rates"],
        "alpha_values": data["alpha_values"].tolist(),
        "source_subgroups": data["source_subgroups"].tolist(),
        "target_subgroups": data["target_subgroups"].tolist(),
        "category": category,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_specificity_ratios(
    flip_rates: np.ndarray,
    alpha_values: list[float],
    sources: list[str],
    targets: list[str],
) -> dict[str, np.ndarray]:
    """Compute per-source specificity ratio at each alpha.

    specificity_ratio = on-diagonal flip rate / mean off-diagonal flip rate
    Returns dict: source_name → array of shape (n_alphas,).
    """
    # Map source→target index (matched subgroup on diagonal)
    tgt_idx = {t: i for i, t in enumerate(targets)}
    ratios: dict[str, np.ndarray] = {}

    for si, src in enumerate(sources):
        if src not in tgt_idx:
            continue  # source not in targets
        diag_ti = tgt_idx[src]
        n_alphas = flip_rates.shape[0]
        r = np.ones(n_alphas)

        for ai in range(n_alphas):
            on_diag = flip_rates[ai, si, diag_ti]
            off_diag = [
                flip_rates[ai, si, ti]
                for ti in range(len(targets))
                if ti != diag_ti and not np.isnan(flip_rates[ai, si, ti])
            ]
            if off_diag and np.mean(off_diag) > 0.01:
                r[ai] = on_diag / np.mean(off_diag)
            elif on_diag > 0:
                r[ai] = float("inf")
            else:
                r[ai] = 1.0
        ratios[src] = r
    return ratios


def compute_diagonal_gap(
    flip_rates: np.ndarray,
    sources: list[str],
    targets: list[str],
) -> np.ndarray:
    """Compute mean(diagonal) - mean(off-diagonal) at each alpha.

    Returns shape (n_alphas,).
    """
    tgt_idx = {t: i for i, t in enumerate(targets)}
    n_alphas = flip_rates.shape[0]
    gaps = np.zeros(n_alphas)

    for ai in range(n_alphas):
        diag_vals = []
        off_diag_vals = []
        for si, src in enumerate(sources):
            if src not in tgt_idx:
                continue
            diag_ti = tgt_idx[src]
            for ti in range(len(targets)):
                val = flip_rates[ai, si, ti]
                if np.isnan(val):
                    continue
                if ti == diag_ti:
                    diag_vals.append(val)
                else:
                    off_diag_vals.append(val)
        if diag_vals and off_diag_vals:
            gaps[ai] = np.mean(diag_vals) - np.mean(off_diag_vals)
    return gaps


# ---------------------------------------------------------------------------
# Figure A: Alpha grid heatmaps
# ---------------------------------------------------------------------------


def fig_transfer_alpha_grid(
    result: dict[str, Any], output_dir: Path,
) -> None:
    """Grid of source×target heatmaps, one per alpha."""
    fr = result["flip_rates"]
    alphas = result["alpha_values"]
    sources = result["source_subgroups"]
    targets = result["target_subgroups"]
    cat = result["category"]

    n = len(alphas)
    ncols = min(5, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3.2 * ncols, 2.8 * nrows + 0.8),
        squeeze=False,
    )

    vmin, vmax = 0.0, np.nanmax(fr) if not np.all(np.isnan(fr)) else 1.0
    vmax = max(vmax, 0.01)

    for ai, alpha in enumerate(alphas):
        row, col = divmod(ai, ncols)
        ax = axes[row, col]
        mat = fr[ai]
        im = ax.imshow(mat, cmap="YlOrRd", vmin=vmin, vmax=vmax, aspect="auto")

        ax.set_xticks(range(len(targets)))
        ax.set_xticklabels(targets, rotation=60, ha="right", fontsize=5)
        ax.set_yticks(range(len(sources)))
        ax.set_yticklabels(sources, fontsize=5)
        ax.set_title(f"\u03b1 = {alpha}", fontsize=8)

        for si in range(len(sources)):
            for ti in range(len(targets)):
                v = mat[si, ti]
                if not np.isnan(v):
                    color = "white" if v > vmax * 0.6 else "black"
                    ax.text(ti, si, f"{v:.2f}", ha="center", va="center",
                            fontsize=4, color=color)

    for ai in range(n, nrows * ncols):
        row, col = divmod(ai, ncols)
        axes[row, col].set_visible(False)

    fig.colorbar(im, ax=axes.ravel().tolist(), label="Flip rate", shrink=0.6, pad=0.02)
    cat_label = CATEGORY_LABELS.get(cat, cat)
    fig.suptitle(
        f"Cross-subgroup steering transfer across \u03b1 \u2014 {cat_label}",
        fontsize=11, y=1.01,
    )
    _save_both(fig, output_dir / f"fig_subgroup_transfer_alpha_grid_{cat}.png", tight=False)
    log(f"    Saved fig_subgroup_transfer_alpha_grid_{cat}")


# ---------------------------------------------------------------------------
# Figure B: Specificity ratio
# ---------------------------------------------------------------------------


def fig_specificity_ratio(
    result: dict[str, Any], output_dir: Path,
) -> None:
    """Line plot: specificity ratio vs alpha per source subgroup."""
    fr = result["flip_rates"]
    alphas = result["alpha_values"]
    sources = result["source_subgroups"]
    targets = result["target_subgroups"]
    cat = result["category"]

    ratios = compute_specificity_ratios(fr, alphas, sources, targets)
    if not ratios:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (src, r) in enumerate(sorted(ratios.items())):
        color = WONG_PALETTE[i % len(WONG_PALETTE)]
        marker = ["o", "s", "^", "D", "v", "<", ">", "p"][i % 8]
        # Clip inf for plotting
        r_plot = np.clip(r, 0, 20)
        ax.plot(alphas, r_plot, f"{marker}-", color=color, label=src, markersize=5)

    ax.axhline(1.0, ls="--", color=GRAY, lw=1, label="Non-specific (ratio=1)")
    ax.set_xlabel("Steering coefficient |\u03b1|")
    ax.set_ylabel("Specificity ratio (on-diag / off-diag)")
    cat_label = CATEGORY_LABELS.get(cat, cat)
    ax.set_title(f"Steering specificity across \u03b1 \u2014 {cat_label}")
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(bottom=0)

    _save_both(fig, output_dir / f"fig_specificity_ratio_{cat}.png")
    log(f"    Saved fig_specificity_ratio_{cat}")


# ---------------------------------------------------------------------------
# Figure C: Specificity vs correction
# ---------------------------------------------------------------------------


def fig_specificity_vs_correction(
    all_results: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    """Scatter: overall correction rate vs mean specificity ratio, colored by category."""
    points: list[dict[str, Any]] = []

    for cat, result in all_results.items():
        fr = result["flip_rates"]
        alphas = result["alpha_values"]
        sources = result["source_subgroups"]
        targets = result["target_subgroups"]

        ratios = compute_specificity_ratios(fr, alphas, sources, targets)
        if not ratios:
            continue

        for ai, alpha in enumerate(alphas):
            overall_corr = np.nanmean(fr[ai])
            spec_vals = [r[ai] for r in ratios.values() if np.isfinite(r[ai])]
            if not spec_vals:
                continue
            mean_spec = np.mean(spec_vals)
            points.append({
                "category": cat, "alpha": alpha,
                "correction": overall_corr, "specificity": min(mean_spec, 20),
            })

    if not points:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    for cat in sorted(set(p["category"] for p in points)):
        cat_pts = [p for p in points if p["category"] == cat]
        color = CATEGORY_COLORS.get(cat, GRAY)
        ax.scatter(
            [p["correction"] for p in cat_pts],
            [p["specificity"] for p in cat_pts],
            c=color, s=25, alpha=0.7,
            label=CATEGORY_LABELS.get(cat, cat),
        )

    ax.axhline(1.0, ls="--", color=GRAY, lw=0.8, alpha=0.6)
    # Mark sweet spot region
    ax.axhspan(1.5, 20, xmin=0, xmax=1, alpha=0.05, color=GREEN)
    ax.set_xlabel("Overall correction rate")
    ax.set_ylabel("Mean specificity ratio")
    ax.set_title("Correction rate vs. steering specificity")
    ax.legend(fontsize=8)

    _save_both(fig, output_dir / "fig_specificity_vs_correction.png")
    log("    Saved fig_specificity_vs_correction")


# ---------------------------------------------------------------------------
# Figure D: Transfer vs Jaccard
# ---------------------------------------------------------------------------


def fig_transfer_vs_jaccard(
    all_results: dict[str, dict[str, Any]],
    all_jaccard: dict[str, list[tuple[str, str, float]]],
    output_dir: Path,
    low_alpha_idx: int | None = None,
    high_alpha_idx: int = -1,
) -> None:
    """Scatter: flip rate vs Jaccard overlap at low and high alpha."""
    from scipy.stats import linregress

    for label, alpha_idx in [("low", low_alpha_idx), ("high", high_alpha_idx)]:
        points_x: list[float] = []
        points_y: list[float] = []
        points_cat: list[str] = []
        alpha_used = None

        for cat, result in all_results.items():
            fr = result["flip_rates"]
            alphas = result["alpha_values"]
            sources = result["source_subgroups"]
            targets = result["target_subgroups"]
            jac_pairs = all_jaccard.get(cat, [])

            if alpha_idx is None:
                # Pick alpha where mean specificity is maximized
                ratios = compute_specificity_ratios(fr, alphas, sources, targets)
                if ratios:
                    mean_specs = []
                    for ai in range(len(alphas)):
                        vals = [r[ai] for r in ratios.values() if np.isfinite(r[ai])]
                        mean_specs.append(np.mean(vals) if vals else 1.0)
                    ai = int(np.argmax(mean_specs))
                else:
                    ai = 0
            else:
                ai = alpha_idx if alpha_idx >= 0 else len(alphas) + alpha_idx

            alpha_used = alphas[ai] if ai < len(alphas) else alphas[-1]

            src_idx = {s: i for i, s in enumerate(sources)}
            tgt_idx = {t: i for i, t in enumerate(targets)}

            # Normalize names for matching
            src_norm = {s.lower().strip(): i for i, s in enumerate(sources)}
            tgt_norm = {t.lower().strip(): i for i, t in enumerate(targets)}

            for s_name, t_name, jac_val in jac_pairs:
                si = src_norm.get(s_name.lower().strip())
                ti = tgt_norm.get(t_name.lower().strip())
                if si is None or ti is None:
                    continue
                val = fr[ai, si, ti]
                if np.isnan(val):
                    continue
                points_x.append(jac_val)
                points_y.append(val)
                points_cat.append(cat)

        if len(points_x) < 3:
            log(f"    Skipping transfer_vs_jaccard ({label}): <3 data points")
            continue

        fig, ax = plt.subplots(figsize=(8, 6))

        for cat in sorted(set(points_cat)):
            mask = [c == cat for c in points_cat]
            px = [points_x[i] for i in range(len(mask)) if mask[i]]
            py = [points_y[i] for i in range(len(mask)) if mask[i]]
            color = CATEGORY_COLORS.get(cat, GRAY)
            ax.scatter(px, py, c=color, s=30, alpha=0.7,
                       label=CATEGORY_LABELS.get(cat, cat))

        # Regression
        x_arr = np.array(points_x)
        y_arr = np.array(points_y)
        if len(x_arr) >= 5:
            slope, intercept, r_val, p_val, _ = linregress(x_arr, y_arr)
            x_line = np.linspace(x_arr.min(), x_arr.max(), 50)
            ax.plot(x_line, slope * x_line + intercept, "--", color="black", lw=1.5)
            ax.text(
                0.95, 0.05,
                f"r\u00b2 = {r_val**2:.3f}\np = {p_val:.2e}\nn = {len(x_arr)}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
            )

        ax.set_xlabel("Jaccard similarity (Stage 2 feature overlap)")
        ax.set_ylabel("Flip rate (steering transfer)")
        ax.set_title(
            f"Feature overlap predicts steering transfer "
            f"(\u03b1={alpha_used}, {label})"
        )
        ax.legend(fontsize=8, loc="upper left")

        _save_both(fig, output_dir / f"fig_transfer_vs_jaccard_{label}.png")
        log(f"    Saved fig_transfer_vs_jaccard_{label}")


# ---------------------------------------------------------------------------
# Figure E: Diagonal emergence
# ---------------------------------------------------------------------------


def fig_diagonal_emergence(
    result: dict[str, Any], output_dir: Path,
) -> None:
    """Diagonal gap as a function of alpha."""
    fr = result["flip_rates"]
    alphas = result["alpha_values"]
    sources = result["source_subgroups"]
    targets = result["target_subgroups"]
    cat = result["category"]

    gaps = compute_diagonal_gap(fr, sources, targets)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alphas, gaps, "o-", color=BLUE, markersize=6)
    ax.fill_between(alphas, 0, gaps, where=(gaps > 0), alpha=0.15, color=BLUE)
    ax.axhline(0, ls="--", color=GRAY, lw=0.8)

    # Annotate max
    max_idx = int(np.argmax(gaps))
    ax.annotate(
        f"max gap = {gaps[max_idx]:.3f}\nat \u03b1 = {alphas[max_idx]}",
        xy=(alphas[max_idx], gaps[max_idx]),
        xytext=(alphas[max_idx] + 3, gaps[max_idx] + 0.02),
        arrowprops=dict(arrowstyle="->", color=GRAY),
        fontsize=8,
    )

    ax.set_xlabel("Steering coefficient |\u03b1|")
    ax.set_ylabel("Diagonal gap (on-diag \u2212 off-diag)")
    cat_label = CATEGORY_LABELS.get(cat, cat)
    ax.set_title(f"Emergence of subgroup specificity \u2014 {cat_label}")

    _save_both(fig, output_dir / f"fig_diagonal_emergence_{cat}.png")
    log(f"    Saved fig_diagonal_emergence_{cat}")


# ---------------------------------------------------------------------------
# Generate all figures from saved results
# ---------------------------------------------------------------------------


def generate_sweep_figures(
    all_results: dict[str, dict[str, Any]],
    all_jaccard: dict[str, list[tuple[str, str, float]]],
    output_dir: Path,
) -> None:
    """Generate all alpha-sweep figures (A-E)."""
    fig_dir = ensure_dir(output_dir / "figures")
    log("  Generating alpha-sweep figures ...")

    for cat, result in all_results.items():
        fig_transfer_alpha_grid(result, fig_dir)
        fig_specificity_ratio(result, fig_dir)
        fig_diagonal_emergence(result, fig_dir)

    fig_specificity_vs_correction(all_results, fig_dir)
    fig_transfer_vs_jaccard(all_results, all_jaccard, fig_dir)

    log("  Alpha-sweep figures complete.")
