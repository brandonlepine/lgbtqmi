#!/usr/bin/env python3
"""Analyze within-category subgroup fragmentation across ALL categories.

Reads subgroup_directions.npz, detects families via clustering, tests
demographic decomposition generalisation, and generates fig_60-62.

Usage:
    python scripts/analyze_subgroup_fragmentation.py \
        --run_dir results/runs/llama2-13b-hf/2026-04-11/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from scipy.spatial.distance import squareform

from src.analysis.geometry import cosine_similarity_matrix
from src.data.bbq_loader import CATEGORY_MAP
from src.utils.io import atomic_save_json, ensure_dir
from src.utils.logging import log
from src.visualization.style import (
    ANNOT_SIZE, CATEGORY_COLORS, CATEGORY_LABELS, LABEL_SIZE,
    TICK_SIZE, TITLE_SIZE, apply_style, label_panel, save_fig,
)

ALL_CATS = list(CATEGORY_MAP.keys())


def _load_subgroup_dirs(run_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    """Load subgroup directions grouped by category."""
    path = run_dir / "analysis" / "subgroup_directions.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run compute_subgroup_directions.py first.")
    data = np.load(path, allow_pickle=True)
    cat_sgs: dict[str, dict[str, np.ndarray]] = {}
    for key in data.files:
        if key.startswith("subgroup_"):
            rest = key[len("subgroup_"):]
            # First token is the category, rest is subgroup name
            for cat in ALL_CATS:
                prefix = f"{cat}_"
                if rest.startswith(prefix):
                    sg = rest[len(prefix):]
                    cat_sgs.setdefault(cat, {})[sg] = data[key]
                    break
    return cat_sgs


def _detect_families(
    sim_matrix: np.ndarray, names: list[str], n_families: int = 2,
) -> dict[int, list[str]]:
    """Hierarchical clustering into n_families."""
    n = len(names)
    if n < 2:
        return {0: list(names)}
    dist = 1.0 - sim_matrix
    np.fill_diagonal(dist, 0.0)
    dist = np.maximum(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=n_families, criterion="maxclust")
    families: dict[int, list[str]] = {}
    for name, lab in zip(names, labels):
        families.setdefault(int(lab), []).append(name)
    return families


def _classify_structure(
    sim_matrix: np.ndarray, families: dict[int, list[str]], names: list[str],
) -> str:
    """Classify internal structure type."""
    n = len(names)
    if n < 2:
        return "singleton"
    upper = []
    for i in range(n):
        for j in range(i + 1, n):
            upper.append(float(sim_matrix[i, j]))
    min_cos = min(upper)
    mean_cos = float(np.mean(upper))
    if n == 2 and min_cos < -0.5:
        return "bipolar"
    if min_cos < -0.3 and len(families) >= 2:
        return "fragmented_families"
    if mean_cos > 0.3:
        return "consistent"
    return "multipolar"


def _project_out_axis(
    sg_dirs: dict[str, np.ndarray], layer: int,
) -> tuple[np.ndarray | None, dict[str, np.ndarray]]:
    """Project out the top-1 PCA axis from subgroup directions at one layer.

    Returns the axis and the residual directions (re-normalised per subgroup).
    """
    names = sorted(sg_dirs.keys())
    vecs = np.stack([sg_dirs[n][layer] for n in names])
    if vecs.shape[0] < 2:
        return None, {}
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)
    pca.fit(vecs)
    axis = pca.components_[0]
    axis = axis / max(np.linalg.norm(axis), 1e-10)

    residuals: dict[str, np.ndarray] = {}
    for n, v in zip(names, vecs):
        r = v - float(np.dot(v, axis)) * axis
        rn = np.linalg.norm(r)
        residuals[n] = r / max(rn, 1e-10)
    return axis, residuals


# ===== Figures ==============================================================

def plot_fig60(
    cat_sgs: dict[str, dict[str, np.ndarray]],
    frag_info: dict[str, dict],
    mid_layer: int,
    path: str,
) -> None:
    """Fig 60: One cosine heatmap per category."""
    apply_style()
    cats = [c for c in ALL_CATS if c in cat_sgs and len(cat_sgs[c]) >= 2]
    cats.sort(key=lambda c: frag_info.get(c, {}).get("mean_pairwise_cos", 1.0))
    n = len(cats)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.8, nrows * 3.5))
    if n == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    structure_colors = {
        "bipolar": "#D55E00", "fragmented_families": "#E69F00",
        "consistent": "#009E73", "multipolar": "#56B4E9", "singleton": "#999999",
    }

    for idx, cat in enumerate(cats):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        sg_dirs = cat_sgs[cat]
        sim, names = cosine_similarity_matrix(sg_dirs, mid_layer)
        ns = len(names)

        im = ax.imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
        ax.set_xticks(range(ns))
        ax.set_yticks(range(ns))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=ANNOT_SIZE - 1)
        ax.set_yticklabels(names, fontsize=ANNOT_SIZE - 1)
        if ns <= 8:
            for i in range(ns):
                for j in range(ns):
                    val = sim[i, j]
                    clr = "white" if abs(val) > 0.6 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=ANNOT_SIZE - 2, color=clr)

        stype = frag_info.get(cat, {}).get("structure_type", "")
        border_color = structure_colors.get(stype, "#999999")
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(2.5)
        ax.set_title(f"{CATEGORY_LABELS.get(cat, cat)}\n({stype})",
                     fontsize=TICK_SIZE, color=border_color)

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    fig.suptitle(f"Within-category subgroup fragmentation (Layer {mid_layer})",
                 fontsize=TITLE_SIZE + 1, y=1.01)
    save_fig(fig, path)


def plot_fig61(frag_info: dict[str, dict], path: str) -> None:
    """Fig 61: Summary bars — within-family vs between-family cosine per category."""
    apply_style()
    cats = sorted(frag_info.keys())
    display = [CATEGORY_LABELS.get(c, c) for c in cats]
    n = len(cats)

    within = [frag_info[c].get("within_family_cos", 0.0) for c in cats]
    between = [frag_info[c].get("between_family_cos", 0.0) for c in cats]
    n_fam = [frag_info[c].get("n_families", 1) for c in cats]

    fig, ax = plt.subplots(figsize=(max(10, n * 1.4), 5))
    x = np.arange(n)
    w = 0.35
    ax.bar(x - w / 2, within, w, label="Within-family mean cos", color="#0072B2",
           edgecolor="black", linewidth=0.4)
    ax.bar(x + w / 2, between, w, label="Between-family mean cos", color="#D55E00",
           edgecolor="black", linewidth=0.4)

    for i in range(n):
        ax.text(x[i], max(within[i], between[i]) + 0.04,
                f"fam={n_fam[i]}", ha="center", fontsize=ANNOT_SIZE - 1)

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(display, rotation=45, ha="right")
    ax.set_ylabel("Mean cosine similarity", fontsize=LABEL_SIZE)
    ax.set_title("Family structure per category", fontsize=TITLE_SIZE)
    ax.legend(fontsize=TICK_SIZE)
    save_fig(fig, path)


def plot_fig62(
    cat_sgs: dict[str, dict[str, np.ndarray]],
    decomp_results: dict[str, dict],
    mid_layer: int,
    path: str,
) -> None:
    """Fig 62: Before/after projecting out dominant axis."""
    apply_style()
    cats_with_effect = [c for c in decomp_results if decomp_results[c].get("axis_found")]
    if not cats_with_effect:
        log("  No categories with axis projection effect, skipping fig_62")
        return

    n = len(cats_with_effect)
    fig, axes = plt.subplots(n, 2, figsize=(10, n * 3.5))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, cat in enumerate(cats_with_effect):
        sg_dirs = cat_sgs[cat]
        sim_before, names = cosine_similarity_matrix(sg_dirs, mid_layer)
        residuals = decomp_results[cat].get("residuals", {})
        if not residuals:
            axes[row, 0].set_visible(False)
            axes[row, 1].set_visible(False)
            continue

        # Recompute after-projection cosines from residuals
        ns = len(names)
        sim_after = np.eye(ns, dtype=np.float32)
        for i, na in enumerate(names):
            for j, nb in enumerate(names):
                if j > i and na in residuals and nb in residuals:
                    ra, rb = residuals[na], residuals[nb]
                    cos_val = float(np.dot(ra, rb) / max(np.linalg.norm(ra) * np.linalg.norm(rb), 1e-10))
                    sim_after[i, j] = cos_val
                    sim_after[j, i] = cos_val

        for ax, sim, title in [(axes[row, 0], sim_before, "Before"),
                                (axes[row, 1], sim_after, "After projection")]:
            im = ax.imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
            ax.set_xticks(range(ns))
            ax.set_yticks(range(ns))
            ax.set_xticklabels(names, rotation=45, ha="right", fontsize=ANNOT_SIZE - 1)
            ax.set_yticklabels(names, fontsize=ANNOT_SIZE - 1)
            if ns <= 8:
                for i in range(ns):
                    for j in range(ns):
                        val = sim[i, j]
                        clr = "white" if abs(val) > 0.6 else "black"
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                                fontsize=ANNOT_SIZE - 2, color=clr)
            ax.set_title(f"{CATEGORY_LABELS.get(cat, cat)} — {title}", fontsize=TICK_SIZE + 1)

    fig.suptitle("Demographic axis projection: before vs after",
                 fontsize=TITLE_SIZE + 1, y=1.01)
    save_fig(fig, path)


# ===== Main =================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze subgroup fragmentation.")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--model_id", type=str, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    analysis_dir = ensure_dir(run_dir / "analysis")
    fig_dir = str(ensure_dir(run_dir / "figures"))
    model_id = args.model_id or run_dir.parent.name

    log(f"Analyzing subgroup fragmentation for {model_id}")
    cat_sgs = _load_subgroup_dirs(run_dir)
    log(f"Loaded subgroups for {len(cat_sgs)} categories")

    n_layers = next(iter(next(iter(cat_sgs.values())).values())).shape[0]
    mid_layer = n_layers // 2

    frag_info: dict[str, dict] = {}
    decomp_results: dict[str, dict] = {}

    for cat in ALL_CATS:
        if cat not in cat_sgs or len(cat_sgs[cat]) < 2:
            continue
        sg_dirs = cat_sgs[cat]
        log(f"\n--- {CATEGORY_LABELS.get(cat, cat)} ({len(sg_dirs)} subgroups) ---")

        sim, names = cosine_similarity_matrix(sg_dirs, mid_layer)
        ns = len(names)

        # Summary stats
        upper = [float(sim[i, j]) for i in range(ns) for j in range(i + 1, ns)]
        mean_cos = float(np.mean(upper)) if upper else 0.0
        min_cos = float(np.min(upper)) if upper else 0.0
        frac_anti = sum(1 for c in upper if c < -0.3) / max(len(upper), 1)

        # Detect families
        families = _detect_families(sim, names, n_families=2)
        structure = _classify_structure(sim, families, names)

        # Family cosines
        name_to_fam = {}
        for fid, members in families.items():
            for m in members:
                name_to_fam[m] = fid
        within_cos_vals: list[float] = []
        between_cos_vals: list[float] = []
        for i, na in enumerate(names):
            for j, nb in enumerate(names):
                if j <= i:
                    continue
                if name_to_fam.get(na) == name_to_fam.get(nb):
                    within_cos_vals.append(float(sim[i, j]))
                else:
                    between_cos_vals.append(float(sim[i, j]))

        within_mean = float(np.mean(within_cos_vals)) if within_cos_vals else 0.0
        between_mean = float(np.mean(between_cos_vals)) if between_cos_vals else 0.0

        log(f"  Mean cos: {mean_cos:.3f}, min: {min_cos:.3f}, "
            f"frac_anti: {frac_anti:.2f}, structure: {structure}")
        log(f"  Families: {families}")
        log(f"  Within-family cos: {within_mean:.3f}, between-family: {between_mean:.3f}")

        frag_info[cat] = {
            "n_subgroups": ns,
            "subgroup_names": names,
            "mean_pairwise_cos": mean_cos,
            "min_pairwise_cos": min_cos,
            "frac_anticorrelated": frac_anti,
            "structure_type": structure,
            "families": {str(k): v for k, v in families.items()},
            "n_families": len(families),
            "within_family_cos": within_mean,
            "between_family_cos": between_mean,
        }

        # Decomposition: project out dominant axis
        axis, residuals = _project_out_axis(sg_dirs, mid_layer)
        if axis is not None and residuals:
            # Check if projection changed things
            res_names = sorted(residuals.keys())
            n_res = len(res_names)
            cos_before = []
            cos_after = []
            for i, na in enumerate(res_names):
                for nb in res_names[i + 1:]:
                    ni = names.index(na) if na in names else -1
                    nj = names.index(nb) if nb in names else -1
                    if ni >= 0 and nj >= 0:
                        cos_before.append(float(sim[ni, nj]))
                    ra, rb = residuals[na], residuals[nb]
                    cos_after.append(float(np.dot(ra, rb)))

            mean_change = float(np.mean(np.abs(np.array(cos_after) - np.array(cos_before)))) if cos_before else 0.0
            has_effect = mean_change > 0.05
            log(f"  Axis projection: mean |cos change|={mean_change:.3f}, effect={has_effect}")
            decomp_results[cat] = {
                "axis_found": has_effect,
                "residuals": residuals,
                "mean_cos_change": mean_change,
            }
        else:
            decomp_results[cat] = {"axis_found": False}

    # Save
    log(f"\n--- Saving ---")
    frag_path = analysis_dir / "subgroup_fragmentation.json"
    atomic_save_json({"model_id": model_id, "mid_layer": mid_layer,
                      "categories": frag_info}, frag_path)
    log(f"  -> {frag_path}")

    # Figures
    log(f"\n--- Figures ---")
    plot_fig60(cat_sgs, frag_info, mid_layer, f"{fig_dir}/fig_60_fragmentation_universality.png")
    log("  Saved fig_60")

    plot_fig61(frag_info, f"{fig_dir}/fig_61_fragmentation_summary_bars.png")
    log("  Saved fig_61")

    plot_fig62(cat_sgs, decomp_results, mid_layer,
              f"{fig_dir}/fig_62_decomposition_generalization.png")
    log("  Saved fig_62")

    log("\nDone!")


if __name__ == "__main__":
    main()
