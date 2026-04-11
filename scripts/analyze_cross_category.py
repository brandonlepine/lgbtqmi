#!/usr/bin/env python3
"""Cross-category geometric analysis and visualization.

Generates figures 01-08: cosine heatmaps, trajectories, PCA, dendrograms,
fragmentation grid, and shared component analysis.

Usage:
    python scripts/analyze_cross_category.py \
        --run_dir results/runs/llama2-13b/2026-04-10

    # Specific categories only
    python scripts/analyze_cross_category.py \
        --run_dir results/runs/llama2-13b/2026-04-10 \
        --categories so,gi,race,religion
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.analysis.geometry import (
    cluster_ordering,
    cosine_similarity_matrix,
    cosine_trajectory,
    hierarchical_clustering,
    run_pca,
    shared_component_analysis,
)
from src.data.bbq_loader import CATEGORY_MAP, parse_categories
from src.utils.io import atomic_save_json, ensure_dir
from src.utils.logging import log
from src.visualization.heatmaps import (
    plot_cosine_heatmap,
    plot_dual_heatmaps,
    plot_fragmentation_grid,
)
from src.visualization.trajectories import (
    plot_cosine_trajectories,
    plot_cosine_trajectories_dual,
    plot_dendrogram,
    plot_pca_variance,
    plot_variance_decomposition,
)
from src.visualization.scatter import plot_pca_loadings


def load_directions(run_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]], dict]:
    """Load directions from compute_directions.py output."""
    directions_path = run_dir / "analysis" / "directions.npz"
    if not directions_path.exists():
        raise FileNotFoundError(
            f"Directions not found at {directions_path}. "
            f"Run scripts/compute_directions.py first."
        )

    data = np.load(directions_path, allow_pickle=True)

    # Category-level directions
    category_dirs: dict[str, np.ndarray] = {}
    subgroup_dirs: dict[str, dict[str, np.ndarray]] = {}

    for key in data.files:
        if key.startswith("direction_"):
            name = key[len("direction_"):]
            category_dirs[name] = data[key]
        elif key.startswith("subgroup_"):
            parts = key[len("subgroup_"):].split("_", 1)
            if len(parts) == 2:
                cat, sg = parts
                if cat not in subgroup_dirs:
                    subgroup_dirs[cat] = {}
                subgroup_dirs[cat][sg] = data[key]

    metadata = {}
    if "_metadata" in data.files:
        metadata = json.loads(str(data["_metadata"]))

    log(f"Loaded {len(category_dirs)} category directions, "
        f"{sum(len(v) for v in subgroup_dirs.values())} sub-group directions")

    return category_dirs, subgroup_dirs, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-category geometric analysis.")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--categories", type=str, default=None,
                        help="Filter to these categories (default: all available)")
    parser.add_argument("--model_id", type=str, default=None,
                        help="Model ID (for figure titles; inferred from run_dir if omitted)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    fig_dir = ensure_dir(run_dir / "figures")
    analysis_dir = ensure_dir(run_dir / "analysis")

    model_id = args.model_id or run_dir.parent.name

    # Load directions
    category_dirs, subgroup_dirs, metadata = load_directions(run_dir)

    # Filter to requested categories
    if args.categories:
        requested = set(parse_categories(args.categories))
        category_dirs = {k: v for k, v in category_dirs.items() if k in requested}

    # Only use base category directions (not decomposition variants) for cross-category
    base_cats = {k: v for k, v in category_dirs.items()
                 if not k.startswith("so_")}
    n_layers = list(base_cats.values())[0].shape[0]
    mid_layer = n_layers // 2
    early_layer = n_layers // 5
    late_layer = int(n_layers * 0.8)

    log(f"\nModel: {model_id}, {n_layers} layers")
    log(f"Representative layers: early={early_layer}, mid={mid_layer}, late={late_layer}")
    log(f"Categories: {sorted(base_cats.keys())}")

    # ===== Fig 01: Cross-category cosine heatmap =====
    log("\n--- Fig 01: Cross-category cosine heatmap ---")
    sim_matrix, names = cosine_similarity_matrix(base_cats, mid_layer)
    order = cluster_ordering(base_cats, mid_layer)
    plot_cosine_heatmap(
        sim_matrix, names, mid_layer,
        path=str(fig_dir / "fig_01_cross_category_cosine_heatmap.png"),
        order=order,
    )
    log(f"  Saved fig_01")

    # ===== Fig 02: Cross-category cosine trajectories =====
    log("\n--- Fig 02: Cosine trajectories ---")
    raw_trajectories: dict[str, np.ndarray] = {}
    cat_names = sorted(base_cats.keys())
    for i, ca in enumerate(cat_names):
        for cb in cat_names[i + 1:]:
            pair_name = f"{ca}↔{cb}"
            raw_trajectories[pair_name] = cosine_trajectory(base_cats[ca], base_cats[cb])

    # Compute residual trajectories (after shared removal at each layer)
    residual_trajectories: dict[str, np.ndarray] = {}
    for pair_name in raw_trajectories:
        residual_trajectories[pair_name] = np.zeros(n_layers, dtype=np.float32)

    for layer in range(n_layers):
        sca = shared_component_analysis(base_cats, layer)
        residuals = sca["residuals"]
        for pair_name in raw_trajectories:
            ca, cb = pair_name.split("↔")
            if ca in residuals and cb in residuals:
                ra = residuals[ca]
                rb = residuals[cb]
                na = np.linalg.norm(ra)
                nb = np.linalg.norm(rb)
                if na > 1e-8 and nb > 1e-8:
                    residual_trajectories[pair_name][layer] = np.dot(ra, rb) / (na * nb)

    plot_cosine_trajectories_dual(
        raw_trajectories, residual_trajectories,
        path=str(fig_dir / "fig_02_cross_category_cosine_trajectories.png"),
        suptitle=f"Cross-category cosine trajectories ({model_id})",
    )
    log(f"  Saved fig_02")

    # ===== Fig 03: Within-category fragmentation =====
    log("\n--- Fig 03: Within-category fragmentation ---")
    subgroup_cosines: dict[str, tuple[np.ndarray, list[str]]] = {}
    for cat, sg_dirs in subgroup_dirs.items():
        if len(sg_dirs) >= 2:
            sim_sg, sg_names = cosine_similarity_matrix(sg_dirs, mid_layer)
            subgroup_cosines[cat] = (sim_sg, sg_names)

    if subgroup_cosines:
        plot_fragmentation_grid(
            subgroup_cosines,
            path=str(fig_dir / "fig_03_within_category_fragmentation.png"),
            title=f"Within-category fragmentation ({model_id}, Layer {mid_layer})",
        )
        log(f"  Saved fig_03")
    else:
        log("  Skipped fig_03: no categories with >=2 sub-groups")

    # ===== Fig 04: PCA variance explained =====
    log("\n--- Fig 04: PCA variance explained ---")
    pca_by_layer: dict[str, np.ndarray] = {}
    for label, layer in [("early", early_layer), ("mid", mid_layer), ("late", late_layer)]:
        pca_result = run_pca(base_cats, layer)
        pca_by_layer[f"{label} ({layer})"] = pca_result["explained_variance_ratio"]

    plot_pca_variance(
        pca_by_layer,
        path=str(fig_dir / "fig_04_pca_variance_explained.png"),
        title=f"PCA variance by component ({model_id})",
    )
    log(f"  Saved fig_04")

    # ===== Fig 05: PCA category loadings =====
    log("\n--- Fig 05: PCA category loadings ---")
    pca_mid = run_pca(base_cats, mid_layer)
    plot_pca_loadings(
        pca_mid["loadings"], pca_mid["names"],
        path=str(fig_dir / "fig_05_pca_category_loadings.png"),
        title=f"Category directions in PCA space ({model_id}, Layer {mid_layer})",
        variance_ratios=pca_mid["explained_variance_ratio"],
    )
    log(f"  Saved fig_05")

    # ===== Fig 06: Dendrogram =====
    log("\n--- Fig 06: Hierarchical clustering dendrogram ---")
    linkage_by_layer: dict[str, tuple[np.ndarray, list[str]]] = {}
    for label, layer in [("early", early_layer), ("mid", mid_layer), ("late", late_layer)]:
        Z, names = hierarchical_clustering(base_cats, layer)
        linkage_by_layer[f"{label} ({layer})"] = (Z, names)

    plot_dendrogram(
        linkage_by_layer,
        path=str(fig_dir / "fig_06_hierarchy_dendrogram.png"),
        title=f"Hierarchical clustering ({model_id})",
    )
    log(f"  Saved fig_06")

    # ===== Fig 07: Shared vs specific variance =====
    log("\n--- Fig 07: Shared vs specific variance ---")
    sca_mid = shared_component_analysis(base_cats, mid_layer)
    plot_variance_decomposition(
        sca_mid["variance_decomposition"],
        path=str(fig_dir / "fig_07_shared_vs_specific_variance.png"),
        title=f"Variance decomposition ({model_id}, Layer {mid_layer})",
    )
    log(f"  Saved fig_07")

    # ===== Fig 08: Before/after shared removal =====
    log("\n--- Fig 08: Before/after shared component removal ---")
    original_sim, orig_names = cosine_similarity_matrix(base_cats, mid_layer)
    residual_sim = sca_mid["residual_cosine_matrix"]
    residual_names = sca_mid["residual_names"]
    plot_dual_heatmaps(
        original_sim, residual_sim,
        names=orig_names,
        path=str(fig_dir / "fig_08_before_after_shared_removal.png"),
        title_left="Original cosines",
        title_right="After shared removal",
        suptitle=f"Shared component effect ({model_id}, Layer {mid_layer})",
    )
    log(f"  Saved fig_08")

    # ===== Save analysis results =====
    log("\n--- Saving analysis results ---")
    analysis_results = {
        "model_id": model_id,
        "n_layers": n_layers,
        "representative_layers": {
            "early": early_layer, "mid": mid_layer, "late": late_layer,
        },
        "categories": sorted(base_cats.keys()),
        "cross_category_cosine_mid": {
            f"{names[i]}_{names[j]}": float(sim_matrix[i, j])
            for i in range(len(names)) for j in range(i + 1, len(names))
        },
        "pca_variance_mid": pca_mid["explained_variance_ratio"].tolist(),
        "variance_decomposition_mid": sca_mid["variance_decomposition"],
        "shared_projections_mid": sca_mid["shared_projections"],
        "pca_variance_ratios_mid": sca_mid["pca_variance_ratios"],
    }

    atomic_save_json(analysis_results, analysis_dir / "cross_category_results.json")
    log(f"  Results -> {analysis_dir / 'cross_category_results.json'}")
    log("\nDone!")


if __name__ == "__main__":
    main()
