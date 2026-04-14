#!/usr/bin/env python3
"""Feature interpretability deep dive for optimal steering features.

Characterizes the features selected by the joint (k, alpha) optimization:
  A. Max-activating items (which BBQ items fire this feature most)
  B. Stereotype specificity (subgroup-specific vs category-general)
  C. Feature co-occurrence (pairwise correlation within a subgroup's set)
  D. Cross-subgroup activation matrix (block-diagonal structure test)

Uses existing logic from ``src/sae_localization/feature_characterization.py``
where applicable, extending with specificity and cross-subgroup analyses.

Usage
-----
python scripts/analyze_top_features.py \\
    --model_path models/llama-3.1-8b \\
    --model_id llama-3.1-8b \\
    --device mps \\
    --sae_source fnlp/Llama3_1-8B-Base-LXR-8x \\
    --sae_expansion 8 \\
    --steering_dir results/subgroup_steering/llama-3.1-8b/2026-04-13/ \\
    --localization_dir results/sae_localization/llama-3.1-8b/2026-04-12/ \\
    --ranked_features results/steering_features/llama-3.1-8b/ranked_features_by_subgroup.json \\
    --categories so,race,disability
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
    BLUE, CATEGORY_COLORS, CATEGORY_LABELS, DPI, GRAY, WONG_PALETTE,
    apply_style,
)
apply_style()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Feature interpretability deep dive")
    p.add_argument("--model_path", required=True)
    p.add_argument("--model_id", default="llama-3.1-8b")
    p.add_argument("--device", default="mps")
    p.add_argument("--sae_source", required=True)
    p.add_argument("--sae_expansion", type=int, default=8)
    p.add_argument("--steering_dir", required=True)
    p.add_argument("--localization_dir", required=True)
    p.add_argument("--ranked_features", required=True)
    p.add_argument("--categories", default=None)
    p.add_argument("--max_items", type=int, default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--token_level", action="store_true",
                   help="Run expensive token-level analysis for top-3 features per subgroup")
    p.add_argument("--skip_figures", action="store_true")
    return p.parse_args()


def _save_both(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(str(path), dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(str(path.with_suffix(".pdf")), bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_stage1_hidden_states(
    loc_dir: Path, cat: str, layer: int,
) -> dict[int, np.ndarray]:
    """Load per-item hidden states at a specific layer from Stage 1 .npz files.

    Returns {item_idx: hidden_state_vector} where hidden_state is float32.
    """
    act_dir = loc_dir / "activations" / cat
    if not act_dir.is_dir():
        return {}

    states: dict[int, np.ndarray] = {}
    for npz_path in sorted(act_dir.glob("item_*.npz")):
        try:
            data = np.load(npz_path, allow_pickle=True)
            hs = data["hidden_states"]  # (n_layers, hidden_dim)
            raw_norms = data["hidden_states_raw_norms"]  # (n_layers,)
            meta = json.loads(
                data["metadata_json"].item()
                if data["metadata_json"].shape == ()
                else str(data["metadata_json"])
            )
            idx = meta.get("item_idx", -1)
            if idx < 0 or layer >= hs.shape[0]:
                continue
            # De-normalize: probes operate on raw activations
            h = hs[layer].astype(np.float32) * float(raw_norms[layer])
            states[idx] = h
        except Exception:
            continue
    return states


def _load_items_metadata(
    loc_dir: Path, cat: str,
) -> dict[int, dict[str, Any]]:
    """Load metadata from Stage 1 .npz files."""
    act_dir = loc_dir / "activations" / cat
    if not act_dir.is_dir():
        return {}
    metas: dict[int, dict[str, Any]] = {}
    for npz_path in sorted(act_dir.glob("item_*.npz")):
        try:
            data = np.load(npz_path, allow_pickle=True)
            meta = json.loads(
                data["metadata_json"].item()
                if data["metadata_json"].shape == ()
                else str(data["metadata_json"])
            )
            idx = meta.get("item_idx", -1)
            if idx >= 0:
                metas[idx] = meta
        except Exception:
            continue
    return metas


def _load_processed_stimuli(cat: str) -> dict[int, dict[str, Any]]:
    """Load processed stimuli JSON for prompt text."""
    proc_dir = PROJECT_ROOT / "data" / "processed"
    files = sorted(proc_dir.glob(f"stimuli_{cat}_*.json"))
    if not files:
        return {}
    with open(files[-1]) as f:
        items = json.load(f)
    return {it.get("item_idx", -1): it for it in items if it.get("item_idx", -1) >= 0}


# ---------------------------------------------------------------------------
# Analysis A: Max-activating items
# ---------------------------------------------------------------------------


def analyze_max_activating(
    feature_idx: int,
    layer: int,
    sae: Any,
    hidden_states: dict[int, np.ndarray],
    metas: dict[int, dict[str, Any]],
    stimuli: dict[int, dict[str, Any]],
    top_k: int = 20,
) -> dict[str, Any]:
    """Find items that maximally activate a feature, compute activation stats."""
    import torch

    activations: list[tuple[int, float]] = []
    stereo_acts: list[float] = []
    non_stereo_acts: list[float] = []
    n_nonzero = 0

    for idx, hs in hidden_states.items():
        hs_t = torch.from_numpy(hs).to(device=sae._device)
        feat_acts = sae.encode(hs_t)
        if feature_idx >= feat_acts.shape[-1]:
            continue
        act_val = float(feat_acts[feature_idx])
        activations.append((idx, act_val))

        if act_val > 0:
            n_nonzero += 1

        meta = metas.get(idx, {})
        if meta.get("is_stereotyped_response"):
            stereo_acts.append(act_val)
        elif meta.get("model_answer_role") not in ("", "unknown", None):
            non_stereo_acts.append(act_val)

    # Sort by activation (descending)
    activations.sort(key=lambda x: -x[1])

    top_items = []
    for idx, act_val in activations[:top_k]:
        if act_val <= 0:
            break
        stim = stimuli.get(idx, {})
        meta = metas.get(idx, {})
        prompt = stim.get("prompt", stim.get("context", ""))
        top_items.append({
            "item_idx": idx,
            "activation": round(act_val, 4),
            "prompt_preview": prompt[:120],
            "model_answer_role": meta.get("model_answer_role", ""),
            "stereotyped_groups": meta.get("stereotyped_groups", stim.get("stereotyped_groups", [])),
            "context_condition": stim.get("context_condition", meta.get("context_condition", "")),
        })

    all_acts = [a for _, a in activations]
    arr = np.array(all_acts) if all_acts else np.array([0.0])
    stats = {
        "mean_all_items": round(float(arr.mean()), 4),
        "std_all_items": round(float(arr.std()), 4),
        "mean_stereotyped_items": round(float(np.mean(stereo_acts)), 4) if stereo_acts else 0.0,
        "mean_non_stereotyped_items": round(float(np.mean(non_stereo_acts)), 4) if non_stereo_acts else 0.0,
        "fraction_nonzero": round(n_nonzero / max(len(activations), 1), 4),
    }

    return {
        "feature_idx": feature_idx,
        "layer": layer,
        "top_items": top_items,
        "activation_stats": stats,
    }


# ---------------------------------------------------------------------------
# Analysis B: Stereotype specificity
# ---------------------------------------------------------------------------


def compute_specificity(
    feature_idx: int,
    layer: int,
    sae: Any,
    hidden_states: dict[int, np.ndarray],
    stimuli: dict[int, dict[str, Any]],
    subgroups_in_cat: list[str],
) -> dict[str, Any]:
    """Compute specificity score and cross-subgroup activation profile."""
    import torch

    # Compute activation per item
    acts_by_sub: dict[str, list[float]] = {s: [] for s in subgroups_in_cat}
    all_acts: list[float] = []

    for idx, hs in hidden_states.items():
        hs_t = torch.from_numpy(hs).to(device=sae._device)
        feat_acts = sae.encode(hs_t)
        if feature_idx >= feat_acts.shape[-1]:
            continue
        act_val = float(feat_acts[feature_idx])
        all_acts.append(act_val)

        stim = stimuli.get(idx, {})
        groups = stim.get("stereotyped_groups", [])
        for sub in subgroups_in_cat:
            if sub in groups:
                acts_by_sub[sub].append(act_val)

    mean_all = float(np.mean(all_acts)) if all_acts else 0.0
    per_sub_means: dict[str, float] = {}
    for sub in subgroups_in_cat:
        vals = acts_by_sub[sub]
        per_sub_means[sub] = round(float(np.mean(vals)), 4) if vals else 0.0

    return {
        "feature_idx": feature_idx,
        "layer": layer,
        "mean_all_items": round(mean_all, 4),
        "per_subgroup_mean": per_sub_means,
        "n_per_subgroup": {s: len(acts_by_sub[s]) for s in subgroups_in_cat},
    }


# ---------------------------------------------------------------------------
# Analysis C: Feature co-occurrence
# ---------------------------------------------------------------------------


def compute_cooccurrence(
    feature_indices: list[int],
    layer: int,
    sae: Any,
    hidden_states: dict[int, np.ndarray],
) -> dict[str, Any]:
    """Compute pairwise correlation of activations for a set of features."""
    import torch

    if len(feature_indices) < 2:
        return {"features": feature_indices, "matrix": [], "labels": []}

    # Collect activations matrix: (n_items, n_features)
    item_indices = sorted(hidden_states.keys())
    n = len(item_indices)
    k = len(feature_indices)

    if n < 3:
        labels = [f"F{f}" for f in feature_indices]
        return {"features": feature_indices, "labels": labels,
                "matrix": np.eye(k).tolist()}

    mat = np.zeros((n, k), dtype=np.float32)

    for i, idx in enumerate(item_indices):
        hs_t = torch.from_numpy(hidden_states[idx]).to(device=sae._device)
        feat_acts = sae.encode(hs_t)
        n_feats = feat_acts.shape[-1]
        for j, fidx in enumerate(feature_indices):
            if fidx < n_feats:
                mat[i, j] = float(feat_acts[fidx])

    # Pearson correlation — replace NaN with 0 (constant-activation features)
    corr = np.corrcoef(mat.T)
    if corr.ndim == 0:
        corr = np.array([[1.0]])
    corr = np.nan_to_num(corr, nan=0.0)

    labels = [f"F{f}" for f in feature_indices]
    return {
        "features": feature_indices,
        "labels": labels,
        "matrix": [[round(float(corr[i, j]), 4) for j in range(k)] for i in range(k)],
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_top_feature_activations(
    reports: list[dict[str, Any]],
    category: str,
    output_dir: Path,
) -> None:
    """Horizontal bar chart of top items by activation for each subgroup's #1 feature."""
    # Group reports by subgroup, pick #1 feature (highest cohens_d)
    by_sub: dict[str, dict] = {}
    for r in reports:
        sub = r.get("subgroup", "")
        if sub not in by_sub:
            by_sub[sub] = r

    subs = sorted(by_sub.keys())
    if not subs:
        return

    n = len(subs)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for idx, sub in enumerate(subs):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        report = by_sub[sub]
        items = report.get("top_items", [])[:10]

        if not items:
            ax.set_visible(False)
            continue

        labels = [it["prompt_preview"][:60] for it in items]
        vals = [it["activation"] for it in items]
        conditions = [it.get("context_condition", "") for it in items]
        colors = [BLUE if "ambig" in c else "#003f5c" for c in conditions]

        y_pos = np.arange(len(labels))
        ax.barh(y_pos, vals, color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=6)
        ax.invert_yaxis()
        ax.set_xlabel("Feature activation", fontsize=8)

        fidx = report.get("feature_idx", "?")
        layer = report.get("layer", "?")
        ax.set_title(f"Feature L{layer}_F{fidx} -- {sub}", fontsize=9)

        # Panel label
        ax.text(0.02, 0.95, chr(65 + idx), transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="top")

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    cat_label = CATEGORY_LABELS.get(category, category)
    fig.suptitle(f"Top activating items -- {cat_label}", fontsize=11)
    _save_both(fig, output_dir / f"fig_top_feature_activations_{category}.png")
    log(f"    Saved fig_top_feature_activations_{category}")


def fig_cross_subgroup_heatmap(
    matrix_data: dict[str, Any],
    category: str,
    output_dir: Path,
) -> None:
    """Heatmap of feature activation across target subgroups."""
    from scipy.cluster.hierarchy import dendrogram, linkage

    row_labels = matrix_data.get("row_labels", [])
    col_labels = matrix_data.get("col_labels", [])
    mat = np.array(matrix_data.get("matrix", []))

    if mat.size == 0 or len(row_labels) < 2:
        return

    fig, ax = plt.subplots(figsize=(max(5, len(col_labels) * 1.2),
                                     max(4, len(row_labels) * 0.5)))

    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            color = "white" if v > 0.5 * np.max(mat) else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color=color)

    fig.colorbar(im, ax=ax, label="Mean activation", shrink=0.8)
    cat_label = CATEGORY_LABELS.get(category, category)
    ax.set_title(f"Cross-subgroup activation -- {cat_label}", fontsize=10)

    _save_both(fig, output_dir / f"fig_cross_subgroup_activation_heatmap_{category}.png")
    log(f"    Saved fig_cross_subgroup_activation_heatmap_{category}")


def fig_specificity_distribution(
    all_specificities: list[float],
    output_dir: Path,
) -> None:
    """Histogram of specificity scores across all features."""
    if not all_specificities:
        return

    arr = np.array(all_specificities)
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.hist(arr, bins=30, color=BLUE, edgecolor="white", alpha=0.8)
    ax.axvline(x=1.0, color="black", linestyle="--", linewidth=1, label="Category-general (1.0)")

    median = float(np.median(arr))
    q25, q75 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
    ax.axvline(x=median, color="#D55E00", linestyle="-", linewidth=1.5,
               label=f"Median = {median:.2f}")
    ax.set_xlabel("Specificity score")
    ax.set_ylabel("Count")
    ax.set_title("Feature specificity distribution")
    ax.legend(fontsize=8)
    ax.annotate(f"IQR: [{q25:.2f}, {q75:.2f}]", xy=(0.95, 0.95),
                xycoords="axes fraction", ha="right", va="top", fontsize=8)

    _save_both(fig, output_dir / "fig_specificity_distribution.png")
    log("    Saved fig_specificity_distribution")


def fig_cooccurrence(
    cooccurrence_data: dict[str, dict[str, Any]],
    category: str,
    output_dir: Path,
) -> None:
    """Correlation matrix for features within each subgroup's optimal set."""
    subs = sorted(cooccurrence_data.keys())
    subs_with_data = [s for s in subs if len(cooccurrence_data[s].get("features", [])) >= 2]
    if not subs_with_data:
        return

    n = len(subs_with_data)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False)

    for idx, sub in enumerate(subs_with_data):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        data = cooccurrence_data[sub]
        mat = np.array(data["matrix"])
        labels = data["labels"]

        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=6)

        ax.set_title(sub, fontsize=9)

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    cat_label = CATEGORY_LABELS.get(category, category)
    fig.suptitle(f"Feature co-occurrence -- {cat_label}", fontsize=11)
    _save_both(fig, output_dir / f"fig_feature_cooccurrence_{category}.png")
    log(f"    Saved fig_feature_cooccurrence_{category}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import torch

    t0 = time.time()
    args = parse_args()
    loc_dir = Path(args.localization_dir)
    steering_dir = Path(args.steering_dir)

    output_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "results" / "feature_interpretability"
        / args.model_id / date.today().isoformat()
    )
    ensure_dir(output_dir)
    fig_dir = ensure_dir(output_dir / "figures")
    log(f"Output: {output_dir}")

    # Load manifests
    manifest_path = steering_dir / "steering_manifests.json"
    with open(manifest_path) as f:
        manifests = json.load(f)
    log(f"Loaded {len(manifests)} steering manifests")

    # Load ranked features
    with open(args.ranked_features) as f:
        ranked = json.load(f)

    # Filter categories
    categories = list(ranked.keys())
    if args.categories:
        requested = [c.strip() for c in args.categories.split(",")]
        categories = [c for c in categories if c in requested]

    # Load SAEs for needed layers
    needed_layers: set[int] = set()
    cat_manifests_count = 0
    for m in manifests:
        if m.get("category") not in categories:
            continue
        cat_manifests_count += 1
        for f in m.get("features", []):
            needed_layers.add(f["layer"])

    if cat_manifests_count == 0:
        log("WARNING: no steering manifests match the requested categories. "
            f"Available categories in manifests: "
            f"{sorted(set(m.get('category','') for m in manifests))}")
        log("Nothing to analyze. Exiting.")
        return

    if not needed_layers:
        log("WARNING: manifests have no features listed. Nothing to analyze.")
        return

    log(f"Loading SAEs for layers: {sorted(needed_layers)}")
    from src.sae_localization.sae_wrapper import SAEWrapper
    sae_cache: dict[int, SAEWrapper] = {}
    for layer in sorted(needed_layers):
        sae_cache[layer] = SAEWrapper(
            args.sae_source, layer=layer,
            expansion=args.sae_expansion, device=args.device,
        )

    # Process per category
    all_reports: list[dict[str, Any]] = []
    all_specificities: list[float] = []
    cross_sub_matrices: dict[str, dict[str, Any]] = {}
    all_cooccurrence: dict[str, dict[str, dict[str, Any]]] = {}

    for cat in categories:
        log(f"\n{'='*50}")
        log(f"Category: {cat}")
        log(f"{'='*50}")

        # Get manifests for this category
        cat_manifests = [m for m in manifests if m.get("category") == cat]
        if not cat_manifests:
            log(f"  No manifests for {cat}")
            continue

        # Load stimuli and metadata
        stimuli = _load_processed_stimuli(cat)
        metas = _load_items_metadata(loc_dir, cat)
        log(f"  {len(stimuli)} stimuli, {len(metas)} metadata records")

        # Collect subgroups in this category
        subgroups_in_cat = sorted(set(m["subgroup"] for m in cat_manifests))

        # Pre-load hidden states per layer
        hs_by_layer: dict[int, dict[int, np.ndarray]] = {}
        for layer in needed_layers:
            hs = _load_stage1_hidden_states(loc_dir, cat, layer)
            if hs:
                hs_by_layer[layer] = hs
        log(f"  Loaded hidden states for {len(hs_by_layer)} layers")

        # Cross-subgroup activation matrix
        all_cat_features: list[tuple[str, int, int]] = []  # (source_sub, fidx, layer)
        for m in cat_manifests:
            for f in m.get("features", []):
                all_cat_features.append((m["subgroup"], f["feature_idx"], f["layer"]))

        if all_cat_features and len(subgroups_in_cat) >= 2:
            cross_mat = np.zeros((len(all_cat_features), len(subgroups_in_cat)))
            row_labels = []

            for i, (src_sub, fidx, layer) in enumerate(all_cat_features):
                row_labels.append(f"{src_sub}: F{fidx}")
                sae = sae_cache.get(layer)
                hs = hs_by_layer.get(layer, {})
                if sae is None or not hs:
                    continue

                for j, target_sub in enumerate(subgroups_in_cat):
                    target_items = [
                        idx for idx, st in stimuli.items()
                        if target_sub in st.get("stereotyped_groups", [])
                    ]
                    acts = []
                    for idx in target_items:
                        if idx in hs:
                            hs_t = torch.from_numpy(hs[idx]).to(device=sae._device)
                            feat_acts = sae.encode(hs_t)
                            acts.append(float(feat_acts[fidx]))
                    cross_mat[i, j] = float(np.mean(acts)) if acts else 0.0

            cross_data = {
                "row_labels": row_labels,
                "col_labels": subgroups_in_cat,
                "matrix": cross_mat.tolist(),
            }
            cross_sub_matrices[cat] = cross_data
            atomic_save_json(cross_data, output_dir / f"cross_subgroup_matrix_{cat}.json")
            log(f"  Cross-subgroup matrix: {cross_mat.shape}")

        # Per-subgroup analyses
        cat_cooccurrence: dict[str, dict[str, Any]] = {}

        for m in cat_manifests:
            sub = m["subgroup"]
            features = m.get("features", [])
            if not features:
                continue

            log(f"\n  Subgroup: {sub} ({len(features)} features)")

            sub_reports: list[dict] = []

            for feat in features:
                fidx = feat["feature_idx"]
                layer = feat["layer"]
                sae = sae_cache.get(layer)
                hs = hs_by_layer.get(layer, {})
                if sae is None or not hs:
                    continue

                # Analysis A: Max-activating items
                report = analyze_max_activating(
                    fidx, layer, sae, hs, metas, stimuli, top_k=20,
                )
                report["subgroup"] = sub
                report["category"] = cat
                report["cohens_d"] = feat.get("cohens_d", 0)

                # Analysis B: Specificity
                spec = compute_specificity(fidx, layer, sae, hs, stimuli, subgroups_in_cat)
                report["specificity"] = spec

                # Compute specificity score: mean_activation(this_sub) / mean_all
                sub_mean = spec["per_subgroup_mean"].get(sub, 0)
                all_mean = spec["mean_all_items"]
                spec_score = sub_mean / max(abs(all_mean), 1e-8) if all_mean != 0 else 0.0
                report["specificity_score"] = round(spec_score, 4)
                all_specificities.append(spec_score)

                sub_reports.append(report)
                log(f"    F{fidx} L{layer}: spec={spec_score:.2f}, "
                    f"nonzero={report['activation_stats']['fraction_nonzero']:.2f}")

            # Analysis C: Co-occurrence
            feat_indices = [f["feature_idx"] for f in features]
            if len(feat_indices) >= 2:
                # Use the layer of the first feature for co-occurrence
                primary_layer = features[0]["layer"]
                hs_primary = hs_by_layer.get(primary_layer, {})
                sae_primary = sae_cache.get(primary_layer)
                if hs_primary and sae_primary:
                    cooc = compute_cooccurrence(feat_indices, primary_layer, sae_primary, hs_primary)
                    cat_cooccurrence[sub] = cooc
                    atomic_save_json(cooc, output_dir / f"cooccurrence_{cat}_{sub}.json")

            all_reports.extend(sub_reports)

        all_cooccurrence[cat] = cat_cooccurrence

    # Save all reports
    atomic_save_json(all_reports, output_dir / "feature_reports.json")
    log(f"\nSaved {len(all_reports)} feature reports")

    # Save specificity scores
    specificity_data = [
        {"feature_idx": r["feature_idx"], "layer": r["layer"],
         "subgroup": r["subgroup"], "category": r["category"],
         "specificity_score": r["specificity_score"]}
        for r in all_reports
    ]
    atomic_save_json(specificity_data, output_dir / "specificity_scores.json")

    # Save cross-subgroup matrices
    atomic_save_json(cross_sub_matrices, output_dir / "cross_subgroup_activation_matrices.json")

    # Figures
    if not args.skip_figures:
        log("\nGenerating figures ...")

        for cat in categories:
            cat_reports = [r for r in all_reports if r["category"] == cat]
            if cat_reports:
                fig_top_feature_activations(cat_reports, cat, fig_dir)

            if cat in cross_sub_matrices:
                fig_cross_subgroup_heatmap(cross_sub_matrices[cat], cat, fig_dir)

            if cat in all_cooccurrence:
                fig_cooccurrence(all_cooccurrence[cat], cat, fig_dir)

        if all_specificities:
            fig_specificity_distribution(all_specificities, fig_dir)

    total = time.time() - t0
    log(f"\nComplete in {total:.1f}s")
    log(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
