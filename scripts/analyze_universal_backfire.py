#!/usr/bin/env python3
"""Universal backfire prediction: cosine structure predicts cross-subgroup transfer.

Computes pairwise subgroup direction cosines (SAE-based and DIM-based),
measures cross-subgroup steering transfer effects, and tests whether cosine
similarity predicts backfire magnitude via OLS regression.

Usage
-----
python scripts/analyze_universal_backfire.py \\
    --model_path models/llama-3.1-8b \\
    --model_id llama-3.1-8b \\
    --device mps \\
    --sae_source fnlp/Llama3_1-8B-Base-LXR-8x \\
    --sae_expansion 8 \\
    --steering_dir results/subgroup_steering/llama-3.1-8b/2026-04-13/ \\
    --ranked_features results/steering_features/llama-3.1-8b/ranked_features_by_subgroup.json \\
    --localization_dir results/sae_localization/llama-3.1-8b/2026-04-12/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import atomic_save_json, ensure_dir
from src.utils.logging import log

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.visualization.style import (
    BLUE, CATEGORY_COLORS, CATEGORY_LABELS, CATEGORY_MARKERS,
    DPI, GRAY, WONG_PALETTE, apply_style,
)
apply_style()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Universal backfire prediction")
    p.add_argument("--model_path", required=True)
    p.add_argument("--model_id", default="llama-3.1-8b")
    p.add_argument("--device", default="mps")
    p.add_argument("--sae_source", required=True)
    p.add_argument("--sae_expansion", type=int, default=8)
    p.add_argument("--steering_dir", required=True)
    p.add_argument("--ranked_features", required=True)
    p.add_argument("--localization_dir", required=True)
    p.add_argument("--categories", default=None)
    p.add_argument("--max_items", type=int, default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--skip_figures", action="store_true")
    return p.parse_args()


def _save_both(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(str(path), dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(str(path.with_suffix(".pdf")), bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Step 1: Pairwise cosines
# ---------------------------------------------------------------------------


def compute_sae_cosines(
    manifests: list[dict],
    sae_cache: dict[int, Any],
    categories: list[str],
) -> dict[str, float]:
    """Compute pairwise cosines between subgroup SAE directions.

    Returns dict keyed by "cat|sub1|sub2" -> cosine.
    """
    # Build per-subgroup mean direction
    sub_dirs: dict[str, np.ndarray] = {}  # "cat|sub" -> direction
    for m in manifests:
        cat = m.get("category", "")
        sub = m.get("subgroup", "")
        if cat not in categories:
            continue
        features = m.get("features", [])
        if not features:
            continue

        dirs = []
        for f in features:
            sae = sae_cache.get(f["layer"])
            if sae is None:
                continue
            d = sae.get_feature_direction(f["feature_idx"])
            dirs.append(d)

        if dirs:
            mean_d = np.mean(dirs, axis=0)
            norm = np.linalg.norm(mean_d)
            if norm > 1e-8:
                mean_d /= norm
            sub_dirs[f"{cat}|{sub}"] = mean_d

    # Pairwise cosines within each category
    cosines: dict[str, float] = {}
    for cat in categories:
        cat_keys = sorted(k for k in sub_dirs if k.startswith(f"{cat}|"))
        for i, k1 in enumerate(cat_keys):
            for j, k2 in enumerate(cat_keys):
                if j > i:
                    s1 = k1.split("|")[1]
                    s2 = k2.split("|")[1]
                    cos = float(np.dot(sub_dirs[k1], sub_dirs[k2]))
                    cosines[f"{cat}|{s1}|{s2}"] = round(cos, 4)

    return cosines


def load_dim_cosines(model_id: str) -> dict[str, float]:
    """Try to load DIM-based pairwise cosines from subgroup_directions.npz."""
    results_dir = PROJECT_ROOT / "results" / "runs" / model_id
    if not results_dir.is_dir():
        return {}

    # Scan for the most recent run with subgroup_directions
    for run_dir in sorted(results_dir.iterdir(), reverse=True):
        npz_path = run_dir / "analysis" / "subgroup_directions.npz"
        if npz_path.exists():
            log(f"  Found DIM directions at {npz_path}")
            try:
                data = np.load(npz_path, allow_pickle=True)
                # Expected format: directions keyed by "cat/sub", one per layer
                # Structure may vary; try to extract what we can
                return _extract_dim_cosines(data)
            except Exception as e:
                log(f"  WARNING: could not load DIM directions: {e}")
                return {}
    return {}


def _extract_dim_cosines(data: Any) -> dict[str, float]:
    """Extract pairwise cosines from DIM subgroup_directions.npz data."""
    # The .npz may contain arrays keyed by subgroup identifiers
    # Try common formats
    cosines: dict[str, float] = {}
    keys = list(data.keys())

    # Try to find direction arrays
    directions: dict[str, np.ndarray] = {}
    for k in keys:
        arr = data[k]
        if arr.ndim >= 1:
            directions[k] = arr.flatten() if arr.ndim == 1 else arr

    if len(directions) < 2:
        return cosines

    # Group by category and compute pairwise
    by_cat: dict[str, dict[str, np.ndarray]] = {}
    for k, d in directions.items():
        parts = k.split("/")
        if len(parts) == 2:
            cat, sub = parts
            by_cat.setdefault(cat, {})[sub] = d

    for cat, subs in by_cat.items():
        sub_names = sorted(subs.keys())
        for i, s1 in enumerate(sub_names):
            for j, s2 in enumerate(sub_names):
                if j > i:
                    d1 = subs[s1]
                    d2 = subs[s2]
                    if d1.shape == d2.shape:
                        n1 = np.linalg.norm(d1)
                        n2 = np.linalg.norm(d2)
                        if n1 > 1e-8 and n2 > 1e-8:
                            cos = float(np.dot(d1 / n1, d2 / n2))
                            cosines[f"{cat}|{s1}|{s2}"] = round(cos, 4)
    return cosines


# ---------------------------------------------------------------------------
# Step 2: Cross-subgroup transfer effects
# ---------------------------------------------------------------------------


def _load_merged_items(
    loc_dir: Path, cat: str,
) -> list[dict]:
    """Load processed items merged with Stage-1 metadata."""
    proc_dir = PROJECT_ROOT / "data" / "processed"
    files = sorted(proc_dir.glob(f"stimuli_{cat}_*.json"))
    if not files:
        return []
    with open(files[-1]) as f:
        proc = json.load(f)

    s1_by_idx: dict[int, dict] = {}
    act_dir = loc_dir / "activations" / cat
    if act_dir.is_dir():
        for npz_path in sorted(act_dir.glob("item_*.npz")):
            try:
                data = np.load(npz_path, allow_pickle=True)
                raw = data["metadata_json"]
                meta = json.loads(raw.item() if raw.shape == () else str(raw))
                s1_by_idx[meta.get("item_idx", -1)] = meta
            except Exception:
                continue

    merged = []
    for item in proc:
        m = dict(item)
        s1 = s1_by_idx.get(item.get("item_idx", -1), {})
        m["model_answer"] = s1.get("model_answer", "")
        m["model_answer_role"] = s1.get("model_answer_role", "")
        m["is_stereotyped_response"] = s1.get("is_stereotyped_response", False)
        merged.append(m)
    return merged


def compute_transfer_effects(
    wrapper: Any,
    sae_cache: dict[int, Any],
    vectors: dict[str, dict[str, Any]],
    items_by_cat: dict[str, list[dict]],
    categories: list[str],
    output_dir: Path,
    max_items: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Compute cross-subgroup steering transfer for all (source, target) pairs."""
    import torch
    from src.extraction.activations import format_prompt
    from src.metrics.bias_metrics import compute_all_metrics, compute_margin
    from src.sae_localization.steering import SAESteerer

    transfer_dir = ensure_dir(output_dir / "transfer_effects")
    results: dict[str, dict[str, Any]] = {}  # "cat|src|tgt" -> result

    for cat in categories:
        items = items_by_cat.get(cat, [])
        if not items:
            continue

        # Find all subgroups with vectors in this category
        cat_vec_keys = [k for k in vectors if k.startswith(f"{cat}_")]
        cat_subs = [k.split("_", 1)[1] for k in cat_vec_keys]

        for source in cat_subs:
            src_key = f"{cat}_{source}"
            vec_data = vectors.get(src_key)
            if vec_data is None:
                continue

            layer = vec_data["injection_layer"]
            sae = sae_cache.get(layer, next(iter(sae_cache.values())))
            steerer = SAESteerer(wrapper, sae, layer)
            vec = vec_data["vector"].to(dtype=wrapper.model.dtype, device=wrapper.device)

            for target in cat_subs:
                result_key = f"{cat}|{source}|{target}"
                ckpt_path = transfer_dir / f"transfer_{cat}_{source}_to_{target}.json"

                if ckpt_path.exists():
                    with open(ckpt_path) as f:
                        results[result_key] = json.load(f)
                    continue

                # Get items where model gave stereotyped response targeting this subgroup
                target_items = [
                    it for it in items
                    if it.get("model_answer_role") == "stereotyped_target"
                    and target in it.get("stereotyped_groups", [])
                ]
                if max_items:
                    target_items = target_items[:max_items]

                if len(target_items) < 3:
                    log(f"    {source} -> {target}: skipped ({len(target_items)} items)")
                    continue

                log(f"    {source} -> {target}: {len(target_items)} items ...")

                item_results: list[dict[str, Any]] = []
                for item in target_items:
                    prompt = format_prompt(item)
                    baseline = steerer.evaluate_baseline(prompt)
                    steered = steerer.steer_and_evaluate(prompt, vec)

                    orig = baseline["model_answer"]
                    steered_ans = steered["model_answer"]
                    orig_role = item.get("answer_roles", {}).get(orig, "unknown")
                    steered_role = item.get("answer_roles", {}).get(steered_ans, "unknown")

                    corrected = (
                        orig_role == "stereotyped_target"
                        and steered_role in ("non_stereotyped", "unknown")
                    )
                    corrupted = (
                        orig_role in ("non_stereotyped", "unknown")
                        and steered_role == "stereotyped_target"
                    )

                    logits_b = {lk: float(lv) for lk, lv in baseline.get("answer_logits", {}).items()}
                    logits_s = {lk: float(lv) for lk, lv in steered.get("answer_logits", {}).items()}
                    margin = compute_margin(logits_b, orig) if orig in logits_b else 0.0

                    answer_roles = item.get("answer_roles", {})
                    stereo_opt = next((l for l, r in answer_roles.items() if r == "stereotyped_target"), "")

                    item_results.append({
                        "corrected": corrected,
                        "corrupted": corrupted,
                        "margin": margin,
                        "logit_baseline": logits_b,
                        "logit_steered": logits_s,
                        "stereotyped_option": stereo_opt,
                    })

                metrics = compute_all_metrics(item_results)

                # Compute bias_change: for non-unknown steered responses, fraction stereotyped
                non_unknown = [r for r in item_results if "unknown" not in str(r.get("steered_role", ""))]
                n_stereo_steered = sum(1 for r in item_results if r.get("corrupted", False))
                n_corrected = sum(1 for r in item_results if r.get("corrected", False))
                n = len(item_results)

                # Bias change: negative = debiasing, positive = backfire
                # Use correction/corruption rates directly
                bias_change = (n_stereo_steered - n_corrected) / max(n, 1)

                entry = {
                    "source": source,
                    "target": target,
                    "category": cat,
                    "n_items": n,
                    "bias_change": round(bias_change, 4),
                    "mean_logit_shift": metrics.get("logit_shift", {}).get("mean_shift", 0),
                    "rcr_1.0": metrics.get("rcr_1.0", {}).get("rcr", 0),
                    "raw_correction_rate": metrics.get("raw_correction_rate", 0),
                    "raw_corruption_rate": metrics.get("raw_corruption_rate", 0),
                    "is_self": source == target,
                }
                results[result_key] = entry
                atomic_save_json(entry, ckpt_path)

            # Memory cleanup
            import torch as _torch
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
            elif hasattr(_torch, "mps") and _torch.backends.mps.is_available():
                _torch.mps.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Step 3: Regression
# ---------------------------------------------------------------------------


def run_regression(
    cosines: dict[str, float],
    transfer: dict[str, dict[str, Any]],
    exclude_categories: list[str] | None = None,
) -> dict[str, Any]:
    """OLS regression: cosine -> bias_change."""
    from scipy.stats import linregress

    x_vals: list[float] = []
    y_vals: list[float] = []
    labels: list[str] = []

    for key, cos_val in cosines.items():
        parts = key.split("|")
        if len(parts) != 3:
            continue
        cat, s1, s2 = parts

        if exclude_categories and cat in exclude_categories:
            continue

        # Get transfer in both directions
        for src, tgt in [(s1, s2), (s2, s1)]:
            tkey = f"{cat}|{src}|{tgt}"
            if tkey in transfer:
                x_vals.append(cos_val)
                y_vals.append(transfer[tkey]["bias_change"])
                labels.append(tkey)

    if len(x_vals) < 3:
        return {"r_squared": None, "p_value": None, "n_pairs": len(x_vals),
                "slope": None, "intercept": None}

    x = np.array(x_vals)
    y = np.array(y_vals)
    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    # Bootstrap 95% CI
    rng = np.random.default_rng(42)
    n_boot = 1000
    boot_slopes = []
    for _ in range(n_boot):
        idx = rng.choice(len(x), size=len(x), replace=True)
        try:
            s, _, _, _, _ = linregress(x[idx], y[idx])
            boot_slopes.append(s)
        except Exception:
            pass

    if len(boot_slopes) < n_boot * 0.8:
        log(f"  WARNING: only {len(boot_slopes)}/{n_boot} bootstrap resamples succeeded — "
            f"CI estimates may be unreliable")
    ci_low = float(np.percentile(boot_slopes, 2.5)) if boot_slopes else None
    ci_high = float(np.percentile(boot_slopes, 97.5)) if boot_slopes else None

    return {
        "slope": round(float(slope), 4),
        "intercept": round(float(intercept), 4),
        "r_squared": round(float(r_value ** 2), 4),
        "r_value": round(float(r_value), 4),
        "p_value": float(p_value),
        "std_err": round(float(std_err), 4),
        "n_pairs": len(x_vals),
        "slope_ci_95": [ci_low, ci_high],
        "x_values": [round(v, 4) for v in x_vals],
        "y_values": [round(v, 4) for v in y_vals],
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_universal_scatter(
    reg_all: dict[str, Any],
    reg_no_disability: dict[str, Any],
    transfer: dict[str, dict[str, Any]],
    cosines: dict[str, float],
    output_dir: Path,
) -> None:
    """THE key figure: cosine vs bias_change with regression."""
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 5))

    for panel_idx, (ax, reg, title_suffix) in enumerate([
        (ax_a, reg_all, "all categories"),
        (ax_b, reg_no_disability, "excl. Disability"),
    ]):
        x = np.array(reg.get("x_values", []))
        y = np.array(reg.get("y_values", []))
        labels = reg.get("labels", [])

        if len(x) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        # Color/marker by category
        for i, lbl in enumerate(labels):
            cat = lbl.split("|")[0]
            color = CATEGORY_COLORS.get(cat, GRAY)
            marker = CATEGORY_MARKERS.get(cat, "o")
            ax.scatter(x[i], y[i], c=color, marker=marker, s=40, alpha=0.8, zorder=2)

        # Regression line
        if reg.get("slope") is not None:
            x_line = np.linspace(x.min() - 0.1, x.max() + 0.1, 100)
            y_line = reg["slope"] * x_line + reg["intercept"]
            ax.plot(x_line, y_line, "k--", linewidth=1.5, zorder=1)

            # Bootstrap CI band
            rng = np.random.default_rng(42)
            boot_lines = []
            for _ in range(1000):
                idx = rng.choice(len(x), size=len(x), replace=True)
                try:
                    from scipy.stats import linregress as _lr
                    s, i, _, _, _ = _lr(x[idx], y[idx])
                    boot_lines.append(s * x_line + i)
                except Exception:
                    pass
            if boot_lines:
                boot_arr = np.array(boot_lines)
                lo = np.percentile(boot_arr, 2.5, axis=0)
                hi = np.percentile(boot_arr, 97.5, axis=0)
                ax.fill_between(x_line, lo, hi, color="gray", alpha=0.15, zorder=0)

        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
        ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Pairwise cosine (SAE directions)", fontsize=9)
        ax.set_ylabel("Bias change (neg = debiasing)", fontsize=9)

        r2 = reg.get("r_squared")
        p = reg.get("p_value")
        n = reg.get("n_pairs", 0)
        r2_str = f"{r2:.3f}" if r2 is not None else "N/A"
        p_str = f"{p:.4f}" if p is not None else "N/A"
        ax.annotate(f"r2 = {r2_str}\np = {p_str}\nn = {n}",
                    xy=(0.95, 0.05), xycoords="axes fraction",
                    ha="right", va="bottom", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        panel_label = chr(65 + panel_idx)
        ax.text(0.02, 0.95, panel_label, transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top")
        ax.set_title(f"Universal backfire -- {title_suffix}", fontsize=10)

    # Legend for categories
    handles = []
    for cat in sorted(CATEGORY_COLORS.keys()):
        cat_label = CATEGORY_LABELS.get(cat, cat)
        marker = CATEGORY_MARKERS.get(cat, "o")
        h = plt.Line2D([0], [0], marker=marker, color="w",
                        markerfacecolor=CATEGORY_COLORS[cat], markersize=6, label=cat_label)
        handles.append(h)
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7,
               bbox_to_anchor=(0.5, -0.02))

    _save_both(fig, output_dir / "fig_universal_backfire_scatter.png")
    log("    Saved fig_universal_backfire_scatter")


def fig_transfer_heatmaps(
    transfer: dict[str, dict[str, Any]],
    categories: list[str],
    output_dir: Path,
) -> None:
    """Per-category heatmap of cross-subgroup transfer effects."""
    cats_with_data = []
    for cat in categories:
        cat_entries = {k: v for k, v in transfer.items() if k.startswith(f"{cat}|")}
        if len(cat_entries) >= 2:
            cats_with_data.append(cat)

    if not cats_with_data:
        return

    n = len(cats_with_data)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for idx, cat in enumerate(cats_with_data):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        cat_entries = {k: v for k, v in transfer.items() if k.startswith(f"{cat}|")}
        subs = sorted(set(
            [v["source"] for v in cat_entries.values()] +
            [v["target"] for v in cat_entries.values()]
        ))

        mat = np.full((len(subs), len(subs)), np.nan)
        s_idx = {s: i for i, s in enumerate(subs)}
        for entry in cat_entries.values():
            i = s_idx.get(entry["source"])
            j = s_idx.get(entry["target"])
            if i is not None and j is not None:
                mat[i, j] = entry["bias_change"]

        vmax = max(abs(np.nanmin(mat)), abs(np.nanmax(mat)), 0.1)
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

        ax.set_xticks(range(len(subs)))
        ax.set_xticklabels(subs, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(subs)))
        ax.set_yticklabels(subs, fontsize=8)
        ax.set_xlabel("Target subgroup", fontsize=8)
        ax.set_ylabel("Source vector", fontsize=8)

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if not np.isnan(mat[i, j]):
                    color = "white" if abs(mat[i, j]) > vmax * 0.6 else "black"
                    ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                            fontsize=7, color=color)

        fig.colorbar(im, ax=ax, label="Bias change", shrink=0.8)
        cat_label = CATEGORY_LABELS.get(cat, cat)
        ax.set_title(f"Transfer effects -- {cat_label}", fontsize=9)

        ax.text(0.02, 0.95, chr(65 + idx), transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="top")

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    _save_both(fig, output_dir / "fig_cross_subgroup_transfer_heatmaps.png")
    log("    Saved fig_cross_subgroup_transfer_heatmaps")


def fig_by_category(
    transfer: dict[str, dict[str, Any]],
    cosines: dict[str, float],
    categories: list[str],
    output_dir: Path,
) -> None:
    """Per-category scatter of cosine vs bias_change with per-category regression."""
    from scipy.stats import linregress

    cats_with_data = []
    for cat in categories:
        cat_cos = {k: v for k, v in cosines.items() if k.startswith(f"{cat}|")}
        if len(cat_cos) >= 2:
            cats_with_data.append(cat)

    if not cats_with_data:
        return

    n = len(cats_with_data)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for idx, cat in enumerate(cats_with_data):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        x_vals, y_vals = [], []
        cat_cos = {k: v for k, v in cosines.items() if k.startswith(f"{cat}|")}

        for key, cos_val in cat_cos.items():
            parts = key.split("|")
            if len(parts) != 3:
                continue
            _, s1, s2 = parts
            for src, tgt in [(s1, s2), (s2, s1)]:
                tkey = f"{cat}|{src}|{tgt}"
                if tkey in transfer:
                    x_vals.append(cos_val)
                    y_vals.append(transfer[tkey]["bias_change"])

        if len(x_vals) < 2:
            ax.set_visible(False)
            continue

        color = CATEGORY_COLORS.get(cat, GRAY)
        ax.scatter(x_vals, y_vals, c=color, s=40, alpha=0.8)

        x = np.array(x_vals)
        y = np.array(y_vals)
        if len(x) >= 3:
            slope, intercept, r_value, p_value, _ = linregress(x, y)
            x_line = np.linspace(x.min() - 0.1, x.max() + 0.1, 50)
            ax.plot(x_line, slope * x_line + intercept, "k--", linewidth=1)
            r2_str = f"r2={r_value**2:.3f}"
        else:
            r2_str = "r2=N/A"

        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
        ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)

        cat_label = CATEGORY_LABELS.get(cat, cat)
        ax.set_title(f"{cat_label} ({r2_str})", fontsize=9)
        ax.set_xlabel("Pairwise cosine", fontsize=8)
        ax.set_ylabel("Bias change", fontsize=8)

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle("Backfire prediction by category", fontsize=11)
    _save_both(fig, output_dir / "fig_cosine_vs_backfire_by_category.png")
    log("    Saved fig_cosine_vs_backfire_by_category")


def fig_sae_vs_dim_comparison(
    sae_cosines: dict[str, float],
    dim_cosines: dict[str, float],
    output_dir: Path,
) -> None:
    """Scatter of SAE cosines vs DIM cosines."""
    from scipy.stats import linregress

    shared_keys = sorted(set(sae_cosines.keys()) & set(dim_cosines.keys()))
    if len(shared_keys) < 3:
        return

    x = np.array([sae_cosines[k] for k in shared_keys])
    y = np.array([dim_cosines[k] for k in shared_keys])
    cats = [k.split("|")[0] for k in shared_keys]

    fig, ax = plt.subplots(figsize=(6, 5))

    for i, k in enumerate(shared_keys):
        cat = cats[i]
        color = CATEGORY_COLORS.get(cat, GRAY)
        marker = CATEGORY_MARKERS.get(cat, "o")
        ax.scatter(x[i], y[i], c=color, marker=marker, s=40, alpha=0.8)

    slope, intercept, r_value, _, _ = linregress(x, y)
    x_line = np.linspace(x.min() - 0.1, x.max() + 0.1, 50)
    ax.plot(x_line, slope * x_line + intercept, "k--", linewidth=1)

    ax.set_xlabel("SAE-based pairwise cosine", fontsize=9)
    ax.set_ylabel("DIM-based pairwise cosine", fontsize=9)
    ax.set_title(f"SAE vs DIM cosines (r2={r_value**2:.3f})", fontsize=10)
    ax.plot([-1, 1], [-1, 1], ":", color="gray", alpha=0.5)

    _save_both(fig, output_dir / "fig_sae_vs_dim_cosine_comparison.png")
    log("    Saved fig_sae_vs_dim_cosine_comparison")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import torch

    t0 = time.time()
    args = parse_args()

    steering_dir = Path(args.steering_dir)
    loc_dir = Path(args.localization_dir)

    output_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "results" / "universal_backfire"
        / args.model_id / date.today().isoformat()
    )
    ensure_dir(output_dir)
    fig_dir = ensure_dir(output_dir / "figures")
    log(f"Output: {output_dir}")

    # Load manifests
    with open(steering_dir / "steering_manifests.json") as f:
        manifests = json.load(f)

    # Load ranked features for categories
    with open(args.ranked_features) as f:
        ranked = json.load(f)

    categories = list(ranked.keys())
    if args.categories:
        requested = [c.strip() for c in args.categories.split(",")]
        categories = [c for c in categories if c in requested]

    # Load steering vectors
    vec_dir = steering_dir / "steering_vectors"
    if not vec_dir.is_dir():
        log(f"ERROR: steering_vectors/ not found at {vec_dir}")
        sys.exit(1)

    vectors: dict[str, dict[str, Any]] = {}
    for npz_path in sorted(vec_dir.glob("*.npz")):
        data = np.load(npz_path, allow_pickle=True)
        key = npz_path.stem
        vectors[key] = {
            "vector": torch.from_numpy(data["vector"]),
            "injection_layer": int(data["injection_layer"]),
            "alpha": float(data["alpha"]),
            "k": int(data["k"]),
        }
    if args.categories:
        vectors = {k: v for k, v in vectors.items()
                   if any(k.startswith(c + "_") for c in categories)}

    if not vectors:
        log("ERROR: no steering vectors found (or all filtered out by --categories). "
            f"Checked: {vec_dir}")
        sys.exit(1)
    log(f"Loaded {len(vectors)} steering vectors")

    # Load SAEs
    needed_layers = set()
    for m in manifests:
        for f in m.get("features", []):
            needed_layers.add(f["layer"])
    for v in vectors.values():
        needed_layers.add(v["injection_layer"])

    log(f"Loading SAEs for layers: {sorted(needed_layers)}")
    from src.sae_localization.sae_wrapper import SAEWrapper
    sae_cache: dict[int, SAEWrapper] = {}
    for layer in sorted(needed_layers):
        sae_cache[layer] = SAEWrapper(
            args.sae_source, layer=layer,
            expansion=args.sae_expansion, device=args.device,
        )

    # Step 1: Compute SAE-based pairwise cosines
    log("\nStep 1: Computing pairwise SAE cosines ...")
    sae_cosines = compute_sae_cosines(manifests, sae_cache, categories)
    atomic_save_json(sae_cosines, output_dir / "sae_cosines.json")
    log(f"  {len(sae_cosines)} pairwise cosines computed")
    for k, v in sorted(sae_cosines.items()):
        log(f"    {k}: {v}")

    # Try DIM cosines
    dim_cosines = load_dim_cosines(args.model_id)
    if dim_cosines:
        atomic_save_json(dim_cosines, output_dir / "dim_cosines.json")
        log(f"  {len(dim_cosines)} DIM cosines loaded")

    # Step 2: Cross-subgroup transfer effects (requires model)
    log("\nStep 2: Computing cross-subgroup transfer effects ...")
    from src.models.wrapper import ModelWrapper
    log("Loading model ...")
    wrapper = ModelWrapper.from_pretrained(args.model_path, device=args.device)

    items_by_cat: dict[str, list[dict]] = {}
    for cat in categories:
        items_by_cat[cat] = _load_merged_items(loc_dir, cat)
        log(f"  {cat}: {len(items_by_cat[cat])} items")

    transfer = compute_transfer_effects(
        wrapper, sae_cache, vectors, items_by_cat, categories,
        output_dir, max_items=args.max_items,
    )
    log(f"  {len(transfer)} transfer pairs computed")

    # Save scatter data
    scatter_data = []
    for cos_key, cos_val in sae_cosines.items():
        parts = cos_key.split("|")
        if len(parts) != 3:
            continue
        cat, s1, s2 = parts
        for src, tgt in [(s1, s2), (s2, s1)]:
            tkey = f"{cat}|{src}|{tgt}"
            if tkey in transfer:
                scatter_data.append({
                    "cosine": cos_val,
                    "bias_change": transfer[tkey]["bias_change"],
                    "mean_logit_shift": transfer[tkey].get("mean_logit_shift", 0),
                    "category": cat,
                    "source": src,
                    "target": tgt,
                })
    atomic_save_json(scatter_data, output_dir / "universal_scatter_data.json")

    # Step 3: Regression
    log("\nStep 3: Running regression ...")
    reg_all = run_regression(sae_cosines, transfer)
    reg_no_disability = run_regression(sae_cosines, transfer, exclude_categories=["disability"])

    regression_results = {
        "all_categories": reg_all,
        "excl_disability": reg_no_disability,
    }
    atomic_save_json(regression_results, output_dir / "regression_results.json")

    log(f"  All categories: r2={reg_all.get('r_squared')}, p={reg_all.get('p_value')}")
    log(f"  Excl disability: r2={reg_no_disability.get('r_squared')}, p={reg_no_disability.get('p_value')}")

    # Step 4: DIM cross-validation
    if dim_cosines:
        log("\nStep 4: DIM cross-validation ...")
        reg_dim = run_regression(dim_cosines, transfer)
        regression_results["dim_based"] = reg_dim
        atomic_save_json(regression_results, output_dir / "regression_results.json")
        log(f"  DIM-based: r2={reg_dim.get('r_squared')}, p={reg_dim.get('p_value')}")

    # Figures
    if not args.skip_figures:
        log("\nGenerating figures ...")
        fig_universal_scatter(reg_all, reg_no_disability, transfer, sae_cosines, fig_dir)
        fig_transfer_heatmaps(transfer, categories, fig_dir)
        fig_by_category(transfer, sae_cosines, categories, fig_dir)
        if dim_cosines:
            fig_sae_vs_dim_comparison(sae_cosines, dim_cosines, fig_dir)

    total = time.time() - t0
    log(f"\nComplete in {total:.1f}s")
    log(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
