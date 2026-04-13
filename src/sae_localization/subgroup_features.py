"""Subgroup-specific feature ranking and overlap analysis.

Collects significant SAE features per subgroup across all analysed layers,
ranks by |cohen's_d|, and computes pairwise overlap matrices.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
    WONG_PALETTE,
    apply_style,
)

try:
    import pandas as pd
except ImportError:
    pd = None

apply_style()


def _save_both(fig: plt.Figure, path: str | Path, tight: bool = True) -> None:
    path = Path(path)
    if tight:
        fig.tight_layout()
    fig.savefig(str(path), dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(str(path.with_suffix(".pdf")), bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def rank_subgroup_features(
    analysis_dir: Path,
    layers: list[int],
) -> dict[str, dict[str, dict[str, list[dict[str, Any]]]]]:
    """Collect and rank significant features per subgroup across layers.

    Returns::
        {category: {subgroup: {"pro_bias": [...], "anti_bias": [...]}}}

    Each entry sorted by |cohen's_d| descending.
    """
    global pd
    if pd is None:
        import pandas as _pd
        pd = _pd

    all_rows: list["pd.DataFrame"] = []
    for layer in layers:
        path = analysis_dir / "features" / f"per_subcategory_layer_{layer}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df = df[df["is_significant"]].copy()
        if "layer" not in df.columns:
            df["layer"] = layer
        all_rows.append(df)

    if not all_rows:
        log("  WARNING: no per-subcategory parquets found")
        return {}

    combined = pd.concat(all_rows, ignore_index=True)
    combined["abs_d"] = combined["cohens_d"].abs()

    result: dict[str, dict[str, dict[str, list[dict]]]] = {}

    for cat in sorted(combined["category"].unique()):
        cat_df = combined[combined["category"] == cat]
        result[cat] = {}

        for sub in sorted(cat_df["subcategory"].unique()):
            sub_df = cat_df[cat_df["subcategory"] == sub].copy()
            sub_df = sub_df.sort_values("abs_d", ascending=False)

            # Deduplicate: same feature at same layer
            sub_df = sub_df.drop_duplicates(subset=["feature_idx", "layer"], keep="first")

            pro = sub_df[sub_df["cohens_d"] > 0]
            anti = sub_df[sub_df["cohens_d"] < 0]

            def _to_records(df: "pd.DataFrame") -> list[dict]:
                records = []
                for _, row in df.iterrows():
                    records.append({
                        "feature_idx": int(row["feature_idx"]),
                        "layer": int(row["layer"]),
                        "cohens_d": round(float(row["cohens_d"]), 4),
                        "p_fdr": round(float(row["p_value_fdr"]), 6),
                        "firing_rate_stereotyped": round(float(row.get("firing_rate_stereotyped", 0)), 4),
                        "firing_rate_non_stereotyped": round(float(row.get("firing_rate_non_stereotyped", 0)), 4),
                        "direction": "pro_bias" if row["cohens_d"] > 0 else "anti_bias",
                    })
                return records

            result[cat][sub] = {
                "pro_bias": _to_records(pro),
                "anti_bias": _to_records(anti),
            }

    return result


# ---------------------------------------------------------------------------
# Overlap matrix
# ---------------------------------------------------------------------------

def compute_overlap_matrix(
    ranked: dict[str, dict[str, dict[str, list[dict]]]],
    top_k: int = 20,
    direction: str = "pro_bias",
) -> dict[str, dict[str, Any]]:
    """Compute pairwise feature overlap for each category.

    Returns::
        {category: {"subgroups": [list], "matrix": [[float]]}}
    """
    result: dict[str, dict[str, Any]] = {}

    for cat, subs_data in ranked.items():
        subs = sorted(subs_data.keys())
        if len(subs) < 2:
            continue

        # Top-k feature sets per subgroup (as (feature_idx, layer) tuples)
        sets: dict[str, set[tuple[int, int]]] = {}
        for sub in subs:
            feats = subs_data[sub].get(direction, [])[:top_k]
            sets[sub] = {(f["feature_idx"], f["layer"]) for f in feats}

        n = len(subs)
        mat = np.zeros((n, n))
        for i, s1 in enumerate(subs):
            for j, s2 in enumerate(subs):
                if i == j:
                    mat[i, j] = len(sets[s1])
                else:
                    inter = len(sets[s1] & sets[s2])
                    union = len(sets[s1] | sets[s2])
                    mat[i, j] = inter / union if union > 0 else 0.0

        result[cat] = {
            "subgroups": subs,
            "matrix": mat.tolist(),
            "top_k": top_k,
            "direction": direction,
        }

    return result


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_feature_overlap_heatmaps(
    overlap: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    """One heatmap per category showing pairwise feature overlap."""
    for cat, data in overlap.items():
        subs = data["subgroups"]
        mat = np.array(data["matrix"])
        n = len(subs)
        if n < 2:
            continue

        fig, ax = plt.subplots(figsize=(max(5, n * 0.9), max(4, n * 0.8)))
        im = ax.imshow(mat, cmap="YlOrRd", vmin=0, aspect="auto")

        ax.set_xticks(range(n))
        ax.set_xticklabels(subs, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n))
        ax.set_yticklabels(subs, fontsize=8)

        for i in range(n):
            for j in range(n):
                v = mat[i, j]
                text = f"{int(v)}" if i == j else f"{v:.2f}"
                color = "white" if v > 0.5 * np.max(mat) else "black"
                ax.text(j, i, text, ha="center", va="center", fontsize=7, color=color)

        fig.colorbar(im, ax=ax, label="Jaccard / count", shrink=0.8)
        cat_label = CATEGORY_LABELS.get(cat, cat)
        ax.set_title(f"Top-{data['top_k']} feature overlap — {cat_label}")

        _save_both(fig, output_dir / f"fig_feature_overlap_{cat}.png")
        log(f"    Saved fig_feature_overlap_{cat}")


def fig_feature_layer_distribution(
    ranked: dict[str, dict[str, dict[str, list[dict]]]],
    output_dir: Path,
    direction: str = "pro_bias",
    top_k: int = 20,
) -> None:
    """Histogram of which layers top features come from, per subgroup."""
    # Collect data
    cat_sub_layers: list[tuple[str, str, list[int]]] = []
    for cat, subs in ranked.items():
        for sub, dirs in subs.items():
            feats = dirs.get(direction, [])[:top_k]
            layers = [f["layer"] for f in feats]
            if layers:
                cat_sub_layers.append((cat, sub, layers))

    if not cat_sub_layers:
        return

    # Find global layer range
    all_layers = [l for _, _, ls in cat_sub_layers for l in ls]
    min_l, max_l = min(all_layers), max(all_layers)

    n = len(cat_sub_layers)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 2.5 * nrows), squeeze=False)

    for idx, (cat, sub, layers) in enumerate(cat_sub_layers):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        color = CATEGORY_COLORS.get(cat, GRAY)
        ax.hist(layers, bins=range(min_l, max_l + 2), color=color, edgecolor="white", alpha=0.8)
        ax.set_title(f"{sub} ({cat})", fontsize=8)
        ax.set_xlabel("Layer", fontsize=7)
        ax.set_ylabel("Count", fontsize=7)

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle(f"Layer distribution of top-{top_k} {direction} features", fontsize=11)
    _save_both(fig, output_dir / "fig_feature_layer_distribution.png")
    log("    Saved fig_feature_layer_distribution")


def fig_ranked_effect_sizes(
    ranked: dict[str, dict[str, dict[str, list[dict]]]],
    output_dir: Path,
    direction: str = "pro_bias",
    max_rank: int = 30,
) -> None:
    """Per category, overlaid rank-effect-size curves."""
    for cat, subs in ranked.items():
        sub_names = sorted(subs.keys())
        if not sub_names:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))

        for i, sub in enumerate(sub_names):
            feats = subs[sub].get(direction, [])[:max_rank]
            if not feats:
                continue
            ranks = list(range(1, len(feats) + 1))
            d_vals = [abs(f["cohens_d"]) for f in feats]
            color = WONG_PALETTE[i % len(WONG_PALETTE)]
            marker = ["o", "s", "^", "D", "v", "<", ">", "p"][i % 8]
            ax.plot(ranks, d_vals, f"{marker}-", color=color, label=sub, markersize=4)

        ax.set_xlabel("Feature rank")
        ax.set_ylabel("|Cohen's d|")
        cat_label = CATEGORY_LABELS.get(cat, cat)
        ax.set_title(f"Ranked feature effect sizes — {cat_label}")
        ax.legend(fontsize=7, ncol=2)

        _save_both(fig, output_dir / f"fig_ranked_effect_sizes_{cat}.png")
        log(f"    Saved fig_ranked_effect_sizes_{cat}")
