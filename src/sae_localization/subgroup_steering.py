"""Subgroup-specific stepwise steering: incrementally add features and evaluate.

For each subgroup, builds steering vectors from top-1, top-2, ..., top-K
features and sweeps alpha to find the optimal (k, alpha) pair.  Also computes
margin-conditioned correction rates.
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
import torch

from src.utils.io import atomic_save_json, ensure_dir
from src.utils.logging import log
from src.visualization.style import (
    BLUE,
    CATEGORY_COLORS,
    CATEGORY_LABELS,
    DPI,
    GRAY,
    GREEN,
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
K_STEPS = [1, 2, 3, 5, 8, 13, 21]
DEFAULT_ALPHA_SWEEP = [-80, -60, -40, -20, -10, -5, 5, 10, 20, 40, 60, 80]
MARGIN_BINS = [
    ("near_indifferent", 0.0, 1.0),
    ("moderate", 1.0, 2.5),
    ("confident", 2.5, float("inf")),
]


def _save_both(fig: plt.Figure, path: str | Path, tight: bool = True) -> None:
    path = Path(path)
    if tight:
        fig.tight_layout()
    fig.savefig(str(path), dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(str(path.with_suffix(".pdf")), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _bin_margin(margin: float) -> str:
    for name, lo, hi in MARGIN_BINS:
        if lo <= margin < hi:
            return name
    return MARGIN_BINS[-1][0]


# ---------------------------------------------------------------------------
# Steering vector construction
# ---------------------------------------------------------------------------


def build_subgroup_steering_vector(
    feature_list: list[dict[str, Any]],
    sae_cache: dict[int, Any],
    k: int,
    alpha: float,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, int]:
    """Build a steering vector from the top-k ranked features.

    Features may span multiple layers / SAEs.  We take the MEAN of the
    unit-normalised decoder columns scaled by alpha.

    Parameters
    ----------
    feature_list : list
        Ranked feature dicts with ``feature_idx`` and ``layer``.
    sae_cache : dict
        Maps layer → loaded SAEWrapper instance.
    k : int
        Number of top features to use.
    alpha : float
        Steering coefficient (negative to dampen, positive to amplify).
    device, dtype : torch device/dtype for the output.

    Returns
    -------
    (steering_vector, injection_layer)
        steering_vector shape ``(hidden_dim,)``; injection_layer is the mode
        layer among the selected features.
    """
    top_k = feature_list[:k]
    if not top_k:
        # Return zero vector at layer 0
        first_sae = next(iter(sae_cache.values()))
        return torch.zeros(first_sae.hidden_dim, dtype=dtype, device=device), 0

    # Determine injection layer (mode of layers in top-k)
    layer_counts: dict[int, int] = {}
    for f in top_k:
        layer_counts[f["layer"]] = layer_counts.get(f["layer"], 0) + 1
    injection_layer = max(layer_counts, key=layer_counts.get)

    # Collect decoder directions (all already in residual-stream space)
    directions: list[torch.Tensor] = []
    for f in top_k:
        sae = sae_cache.get(f["layer"])
        if sae is None:
            continue
        d = torch.from_numpy(sae.get_feature_direction(f["feature_idx"]))
        directions.append(d)

    if not directions:
        first_sae = next(iter(sae_cache.values()))
        return torch.zeros(first_sae.hidden_dim, dtype=dtype, device=device), injection_layer

    # Mean of unit-normalised directions, scaled by alpha
    stacked = torch.stack(directions)  # (k, hidden_dim)
    mean_dir = stacked.mean(dim=0)
    vec = alpha * mean_dir

    return vec.to(dtype=dtype, device=device), injection_layer


# ---------------------------------------------------------------------------
# Stepwise sweep
# ---------------------------------------------------------------------------


def run_stepwise_sweep(
    wrapper: Any,
    sae_cache: dict[int, Any],
    feature_list: list[dict[str, Any]],
    items: list[dict[str, Any]],
    alpha_values: list[float],
    k_steps: list[int],
    prompt_formatter: Callable,
    subgroup: str,
    category: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Run the k × alpha grid for one subgroup.

    Returns dict with ``grid`` (list of per-(k,alpha) result dicts),
    ``optimal`` config, and ``margin_bins`` breakdown.
    """
    from src.sae_localization.steering import SAESteerer

    grid: list[dict[str, Any]] = []
    best_correction = -1.0
    best_config: dict[str, Any] = {}

    n_items = len(items)
    log(f"    Sweep {category}/{subgroup}: {n_items} items, "
        f"{len(feature_list)} features, k_steps={k_steps}")

    for k in k_steps:
        if k > len(feature_list):
            break

        for alpha in alpha_values:
            # Resume checkpoint
            ckpt_path = output_dir / f"{category}_{subgroup}_k{k}_a{alpha}.json"
            if ckpt_path.exists():
                with open(ckpt_path) as f:
                    rec = json.load(f)
                grid.append(rec)
                # Check optimality
                cr = rec.get("correction_rate", 0)
                corr_rate = rec.get("corruption_rate", 0)
                if cr > best_correction and corr_rate < 0.05:
                    best_correction = cr
                    best_config = rec
                continue

            vec, inj_layer = build_subgroup_steering_vector(
                feature_list, sae_cache, k, alpha,
                device=wrapper.device, dtype=wrapper.model.dtype,
            )

            # Build a temporary steerer at the injection layer
            first_sae = sae_cache.get(inj_layer, next(iter(sae_cache.values())))
            steerer = SAESteerer(wrapper, first_sae, inj_layer)

            n_corrected = 0
            n_corrupted = 0
            n_unknown = 0
            n_degen = 0
            margin_bins: dict[str, dict[str, int]] = {
                b[0]: {"n": 0, "corrected": 0} for b in MARGIN_BINS
            }

            for item in items:
                prompt = prompt_formatter(item)
                baseline = steerer.evaluate_baseline(prompt)
                result = steerer.steer_and_evaluate(prompt, vec)

                orig = baseline["model_answer"]
                steered_ans = result["model_answer"]
                orig_role = item.get("answer_roles", {}).get(orig, "unknown")
                steered_role = item.get("answer_roles", {}).get(steered_ans, "unknown")

                flipped = orig != steered_ans
                if result["degenerated"]:
                    n_degen += 1

                # Correction: was stereotyped, now non-stereo/unknown
                if orig_role == "stereotyped_target" and steered_role in ("non_stereotyped", "unknown"):
                    n_corrected += 1
                # Corruption: was non-stereo, now stereotyped
                if orig_role == "non_stereotyped" and steered_role == "stereotyped_target":
                    n_corrupted += 1
                if steered_role == "unknown":
                    n_unknown += 1

                # Margin bin
                logits = baseline.get("answer_logits", {})
                try:
                    vals = {lk: float(lv) for lk, lv in logits.items()}
                    top = vals.get(orig, 0)
                    others = [v for lk, v in vals.items() if lk != orig]
                    margin = top - max(others) if others else 0
                except (ValueError, TypeError):
                    margin = 0
                mbin = _bin_margin(margin)
                margin_bins[mbin]["n"] += 1
                if orig_role == "stereotyped_target" and steered_role in ("non_stereotyped", "unknown"):
                    margin_bins[mbin]["corrected"] += 1

            rec = {
                "category": category,
                "subgroup": subgroup,
                "k": k,
                "alpha": alpha,
                "injection_layer": inj_layer,
                "n_items": n_items,
                "n_corrected": n_corrected,
                "n_corrupted": n_corrupted,
                "n_unknown": n_unknown,
                "n_degenerated": n_degen,
                "correction_rate": n_corrected / max(n_items, 1),
                "corruption_rate": n_corrupted / max(n_items, 1),
                "unknown_rate": n_unknown / max(n_items, 1),
                "degeneration_rate": n_degen / max(n_items, 1),
                "margin_bins": {
                    mb: {
                        "n": d["n"],
                        "corrected": d["corrected"],
                        "correction_rate": d["corrected"] / max(d["n"], 1),
                    }
                    for mb, d in margin_bins.items()
                },
                "features_used": [
                    {"feature_idx": f["feature_idx"], "layer": f["layer"]}
                    for f in feature_list[:k]
                ],
            }
            grid.append(rec)
            atomic_save_json(rec, ckpt_path)

            cr = rec["correction_rate"]
            corr_r = rec["corruption_rate"]
            if cr > best_correction and corr_r < 0.05:
                best_correction = cr
                best_config = rec

            log(f"      k={k} alpha={alpha}: corr={cr:.3f} corrupt={corr_r:.3f} "
                f"degen={rec['degeneration_rate']:.3f}")

        # Memory cleanup
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Tie-break: among configs within 0.005 of best, prefer smaller |alpha| then smaller k
    if grid:
        eligible = [
            r for r in grid
            if r["corruption_rate"] < 0.05
            and r["correction_rate"] >= (best_correction - 0.005)
        ]
        if eligible:
            eligible.sort(key=lambda r: (abs(r["alpha"]), r["k"], -r["correction_rate"]))
            best_config = eligible[0]

    return {
        "category": category,
        "subgroup": subgroup,
        "grid": grid,
        "optimal": best_config,
    }


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def build_steering_manifest(
    optimal: dict[str, Any],
    vec_norm: float,
    medqa_delta: float | None = None,
    mmlu_delta: float | None = None,
) -> dict[str, Any]:
    """Build the per-subgroup steering manifest for reproducibility."""
    return {
        "subgroup": optimal.get("subgroup", ""),
        "category": optimal.get("category", ""),
        "optimal_k": optimal.get("k", 0),
        "optimal_alpha": optimal.get("alpha", 0),
        "injection_layer": optimal.get("injection_layer", 0),
        "features": optimal.get("features_used", []),
        "steering_vector_norm": round(vec_norm, 4),
        "bbq_correction_rate": optimal.get("correction_rate", 0),
        "bbq_corruption_rate": optimal.get("corruption_rate", 0),
        "margin_bins": optimal.get("margin_bins", {}),
        "medqa_matched_accuracy_delta": medqa_delta,
        "mmlu_accuracy_delta": mmlu_delta,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_stepwise_correction(
    results: dict[str, dict[str, Any]],
    category: str,
    output_dir: Path,
) -> None:
    """Per subgroup: correction rate vs k at optimal alpha."""
    subs = sorted(results.keys())
    if not subs:
        return

    n = len(subs)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False)

    for idx, sub in enumerate(subs):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        data = results[sub]
        grid = data.get("grid", [])
        if not grid:
            ax.set_visible(False)
            continue

        # Group by k, pick best alpha (highest correction with corruption<0.05)
        by_k: dict[int, dict] = {}
        for r in grid:
            k = r["k"]
            if k not in by_k or (
                r["corruption_rate"] < 0.05
                and r["correction_rate"] > by_k[k].get("correction_rate", 0)
            ):
                by_k[k] = r

        ks = sorted(by_k.keys())
        corrs = [by_k[k]["correction_rate"] for k in ks]
        corrupts = [by_k[k]["corruption_rate"] for k in ks]

        ax.plot(ks, corrs, "o-", color=BLUE, label="Correction", markersize=5)
        ax.plot(ks, corrupts, "s--", color=VERMILLION, label="Corruption", markersize=4, alpha=0.7)
        ax.set_title(sub, fontsize=9)
        ax.set_xlabel("k (features)", fontsize=8)
        ax.set_ylabel("Rate", fontsize=8)
        if idx == 0:
            ax.legend(fontsize=7)

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    cat_label = CATEGORY_LABELS.get(category, category)
    fig.suptitle(f"Stepwise steering — {cat_label}", fontsize=11)
    _save_both(fig, output_dir / f"fig_stepwise_correction_{category}.png")
    log(f"    Saved fig_stepwise_correction_{category}")


def fig_optimal_k_distribution(
    all_optimals: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    """Histogram of optimal k across all subgroups."""
    ks = [o.get("k", 0) for o in all_optimals if o.get("k")]
    if not ks:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(ks, bins=range(0, max(ks) + 2), color=BLUE, edgecolor="white", alpha=0.8)
    ax.set_xlabel("Optimal k (number of features)")
    ax.set_ylabel("Count (subgroups)")
    ax.set_title("Distribution of optimal feature count across subgroups")

    _save_both(fig, output_dir / "fig_optimal_k_distribution.png")
    log("    Saved fig_optimal_k_distribution")


def fig_alpha_vs_k_heatmap(
    results: dict[str, dict[str, Any]],
    category: str,
    output_dir: Path,
) -> None:
    """Heatmap of correction rate across (k, alpha) grid per subgroup."""
    subs = sorted(results.keys())
    if not subs:
        return

    n = len(subs)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False)

    for idx, sub in enumerate(subs):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        grid = results[sub].get("grid", [])
        if not grid:
            ax.set_visible(False)
            continue

        ks = sorted(set(r["k"] for r in grid))
        alphas = sorted(set(r["alpha"] for r in grid))
        mat = np.full((len(ks), len(alphas)), np.nan)
        k_idx = {k: i for i, k in enumerate(ks)}
        a_idx = {a: i for i, a in enumerate(alphas)}

        for r in grid:
            mat[k_idx[r["k"]], a_idx[r["alpha"]]] = r["correction_rate"]

        im = ax.imshow(mat, cmap="YlGnBu", aspect="auto", vmin=0)
        ax.set_xticks(range(len(alphas)))
        ax.set_xticklabels([str(a) for a in alphas], fontsize=5, rotation=90)
        ax.set_yticks(range(len(ks)))
        ax.set_yticklabels([str(k) for k in ks], fontsize=6)
        ax.set_xlabel("alpha", fontsize=7)
        ax.set_ylabel("k", fontsize=7)
        ax.set_title(sub, fontsize=8)

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    cat_label = CATEGORY_LABELS.get(category, category)
    fig.suptitle(f"Correction rate landscape — {cat_label}", fontsize=11)
    _save_both(fig, output_dir / f"fig_alpha_vs_k_heatmaps_{category}.png")
    log(f"    Saved fig_alpha_vs_k_heatmaps_{category}")


def fig_margin_conditioned_correction(
    all_results: dict[str, dict[str, dict[str, Any]]],
    output_dir: Path,
) -> None:
    """Grouped bars: correction rate per margin bin per subgroup."""
    for cat, subs in all_results.items():
        sub_names = sorted(subs.keys())
        if not sub_names:
            continue

        fig, ax = plt.subplots(figsize=(max(8, len(sub_names) * 2), 5))

        bin_names = [b[0] for b in MARGIN_BINS]
        bin_labels = ["<1.0", "1.0–2.5", ">2.5"]
        n_bins = len(bin_names)
        width = 0.8 / n_bins
        x = np.arange(len(sub_names))

        for bi, (bname, blabel) in enumerate(zip(bin_names, bin_labels)):
            vals = []
            for sub in sub_names:
                opt = subs[sub].get("optimal", {})
                mb = opt.get("margin_bins", {}).get(bname, {})
                vals.append(mb.get("correction_rate", 0))

            color = WONG_PALETTE[bi % len(WONG_PALETTE)]
            ax.bar(x + bi * width - 0.4 + width / 2, vals, width,
                   color=color, label=blabel, alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(sub_names, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Correction rate")
        cat_label = CATEGORY_LABELS.get(cat, cat)
        ax.set_title(f"Margin-conditioned correction — {cat_label}")
        ax.legend(title="Logit margin", fontsize=7)

        _save_both(fig, output_dir / f"fig_margin_conditioned_correction_{cat}.png")
        log(f"    Saved fig_margin_conditioned_correction_{cat}")
