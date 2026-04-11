#!/usr/bin/env python3
"""Generate publication-ready summary figures tying the full story together.

Generates figures 20-24: representational hierarchy, RLHF mechanism, generalization,
fragmentation universality, cross-model stability.

Usage:
    python scripts/generate_summary_figures.py \
        --run_dir results/runs/llama2-13b/2026-04-10

    # Cross-model comparison
    python scripts/generate_summary_figures.py \
        --run_dirs results/runs/llama2-13b/2026-04-10,results/runs/llama3.1-8b/2026-04-10,results/runs/qwen2.5-7b/2026-04-10
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.analysis.geometry import (
    cosine_similarity_matrix,
    hierarchical_clustering,
    run_pca,
    shared_component_analysis,
)
from src.data.bbq_loader import CATEGORY_MAP, parse_categories
from src.utils.io import atomic_save_json, ensure_dir
from src.utils.logging import log
from src.visualization.heatmaps import plot_fragmentation_grid
from src.visualization.summary import (
    plot_ablation_grouped_bars,
    plot_cross_model_stability,
    plot_generalization_summary,
    plot_representational_hierarchy_summary,
    plot_rlhf_mechanism_summary,
)


def load_run_data(run_dir: Path) -> dict:
    """Load all analysis results from a run directory."""
    analysis_dir = run_dir / "analysis"
    data = {"run_dir": str(run_dir)}

    # Directions
    directions_path = analysis_dir / "directions.npz"
    if directions_path.exists():
        dir_data = np.load(directions_path, allow_pickle=True)
        cat_dirs = {}
        subgroup_dirs = {}
        for key in dir_data.files:
            if key.startswith("direction_") and not key.startswith("direction_so_"):
                name = key[len("direction_"):]
                cat_dirs[name] = dir_data[key]
            elif key.startswith("subgroup_"):
                parts = key[len("subgroup_"):].split("_", 1)
                if len(parts) == 2:
                    cat, sg = parts
                    if cat not in subgroup_dirs:
                        subgroup_dirs[cat] = {}
                    subgroup_dirs[cat][sg] = dir_data[key]
        data["category_directions"] = cat_dirs
        data["subgroup_directions"] = subgroup_dirs

    # Cross-category results
    cc_path = analysis_dir / "cross_category_results.json"
    if cc_path.exists():
        with open(cc_path) as f:
            data["cross_category"] = json.load(f)

    # Probe results
    probe_path = analysis_dir / "probe_matrices.npz"
    if probe_path.exists():
        probe_data = np.load(probe_path)
        data["probe_matrices"] = {k: probe_data[k] for k in probe_data.files}

    probe_json = analysis_dir / "probe_results.json"
    if probe_json.exists():
        with open(probe_json) as f:
            data["probe_results"] = json.load(f)

    # Generalization
    gen_path = analysis_dir / "generalization_results.json"
    if gen_path.exists():
        with open(gen_path) as f:
            data["generalization"] = json.load(f)

    # Ablation
    abl_path = analysis_dir / "ablation_results.json"
    if abl_path.exists():
        with open(abl_path) as f:
            data["ablation"] = json.load(f)

    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate summary figures.")
    parser.add_argument("--run_dir", type=str, default=None, help="Single run directory")
    parser.add_argument("--run_dirs", type=str, default=None,
                        help="Comma-separated run directories for cross-model comparison")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory for figures")
    args = parser.parse_args()

    if args.run_dirs:
        run_dirs = [Path(d.strip()) for d in args.run_dirs.split(",")]
    elif args.run_dir:
        run_dirs = [Path(args.run_dir)]
    else:
        parser.error("Provide --run_dir or --run_dirs")
        return

    primary_dir = run_dirs[0]
    fig_dir = ensure_dir(Path(args.output_dir) if args.output_dir else primary_dir / "figures")

    # Also create latest symlink
    latest_dir = Path("results/figures/latest")
    latest_dir.parent.mkdir(parents=True, exist_ok=True)
    if latest_dir.is_symlink():
        latest_dir.unlink()
    try:
        latest_dir.symlink_to(fig_dir.resolve())
        log(f"Symlinked results/figures/latest -> {fig_dir}")
    except OSError:
        log(f"Could not create symlink (not critical)")

    # Load primary run data
    log("Loading primary run data...")
    primary = load_run_data(primary_dir)
    model_id = primary.get("cross_category", {}).get("model_id", primary_dir.parent.name)

    # ===== Fig 20: Representational hierarchy summary =====
    if "category_directions" in primary and len(primary["category_directions"]) >= 2:
        log("\n--- Fig 20: Representational hierarchy summary ---")
        cat_dirs = primary["category_directions"]
        n_layers = list(cat_dirs.values())[0].shape[0]
        mid_layer = n_layers // 2

        sim_matrix, sim_names = cosine_similarity_matrix(cat_dirs, mid_layer)
        Z, link_names = hierarchical_clustering(cat_dirs, mid_layer)
        pca_result = run_pca(cat_dirs, mid_layer)
        sca = shared_component_analysis(cat_dirs, mid_layer)

        plot_representational_hierarchy_summary(
            cosine_matrix=sim_matrix,
            cosine_names=sim_names,
            linkage_Z=Z,
            linkage_names=link_names,
            pca_loadings=pca_result["loadings"],
            pca_names=pca_result["names"],
            pca_var_ratios=pca_result["explained_variance_ratio"],
            variance_decomposition=sca["variance_decomposition"],
            path=str(fig_dir / "fig_20_representational_hierarchy_summary.png"),
            layer=mid_layer,
        )
        log(f"  Saved fig_20")
    else:
        log("  Skipping fig_20: insufficient direction data")

    # ===== Fig 21: RLHF mechanism summary =====
    if "probe_matrices" in primary and "ablation" in primary:
        pm = primary["probe_matrices"]
        if "base_stereo" in pm and "chat_stereo" in pm:
            log("\n--- Fig 21: RLHF mechanism summary ---")
            ablation_data = primary["ablation"].get("ablation_results", {})
            # Construct RLHF-style results
            rlhf_results: dict[str, dict[str, float]] = {}
            for cat, abl in ablation_data.items():
                rlhf_results[cat] = {
                    "base_baseline": abl.get("baseline", 0),
                    "base_ablated": abl.get("ablate_shared", 0),
                    "chat_baseline": 0,  # placeholder if not available
                }

            plot_rlhf_mechanism_summary(
                base_stereo_matrix=pm["base_stereo"],
                chat_stereo_matrix=pm["chat_stereo"],
                ablation_results=rlhf_results,
                path=str(fig_dir / "fig_21_rlhf_mechanism_summary.png"),
            )
            log(f"  Saved fig_21")
    else:
        log("  Skipping fig_21: need probe_matrices and ablation results")

    # ===== Fig 22: Generalization summary =====
    if "generalization" in primary:
        log("\n--- Fig 22: Generalization summary ---")
        gen = primary["generalization"]
        transfer_matrix = np.array(gen.get("transfer_matrix", []))
        transfer_names = gen.get("categories", [])
        cross_bench = gen.get("cross_benchmark", {})

        if transfer_matrix.size > 0 and "category_directions" in primary:
            cat_dirs = primary["category_directions"]
            n_layers = list(cat_dirs.values())[0].shape[0]
            best_layer = gen.get("best_layer", n_layers // 2)

            # Compute cosine-transfer pairs
            cosine_vals, transfer_vals, pair_names = [], [], []
            sim_matrix, sim_names = cosine_similarity_matrix(cat_dirs, best_layer)
            for i, ca in enumerate(transfer_names):
                for j, cb in enumerate(transfer_names):
                    if i >= j:
                        continue
                    if ca in sim_names and cb in sim_names:
                        ci, cj = sim_names.index(ca), sim_names.index(cb)
                        cosine_vals.append(float(sim_matrix[ci, cj]))
                        avg_t = (transfer_matrix[i, j] + transfer_matrix[j, i]) / 2
                        transfer_vals.append(float(avg_t))
                        pair_names.append(f"{ca}↔{cb}")

            plot_generalization_summary(
                transfer_matrix=transfer_matrix,
                transfer_names=transfer_names,
                cross_benchmark_results=cross_bench if cross_bench else {c: {"accuracy": 0.5} for c in transfer_names},
                cosine_values=np.array(cosine_vals) if cosine_vals else np.array([0.0]),
                transfer_values=np.array(transfer_vals) if transfer_vals else np.array([0.5]),
                pair_names=pair_names if pair_names else [""],
                path=str(fig_dir / "fig_22_generalization_summary.png"),
            )
            log(f"  Saved fig_22")
    else:
        log("  Skipping fig_22: no generalization results")

    # ===== Fig 23: Fragmentation universality =====
    if "subgroup_directions" in primary:
        log("\n--- Fig 23: Fragmentation universality ---")
        cat_dirs = primary["category_directions"]
        sg_dirs = primary["subgroup_directions"]
        n_layers = list(cat_dirs.values())[0].shape[0]
        mid_layer = n_layers // 2

        subgroup_cosines = {}
        for cat, sgs in sg_dirs.items():
            if len(sgs) >= 2:
                sim, names = cosine_similarity_matrix(sgs, mid_layer)
                subgroup_cosines[cat] = (sim, names)

        if subgroup_cosines:
            plot_fragmentation_grid(
                subgroup_cosines,
                path=str(fig_dir / "fig_23_fragmentation_universality.png"),
                title=f"Within-category fragmentation across categories ({model_id})",
            )
            log(f"  Saved fig_23")
    else:
        log("  Skipping fig_23: no sub-group directions")

    # ===== Fig 24: Cross-model stability =====
    if len(run_dirs) > 1:
        log("\n--- Fig 24: Cross-model stability ---")
        cosine_matrices: dict[str, tuple[np.ndarray, list[str]]] = {}
        bias_scores: dict[str, dict[str, float]] = {}

        for rd in run_dirs:
            rd_data = load_run_data(rd)
            rd_model_id = rd_data.get("cross_category", {}).get("model_id", rd.parent.name)

            if "category_directions" in rd_data and len(rd_data["category_directions"]) >= 2:
                cd = rd_data["category_directions"]
                n_l = list(cd.values())[0].shape[0]
                ml = n_l // 2
                sim, names = cosine_similarity_matrix(cd, ml)
                cosine_matrices[rd_model_id] = (sim, names)

            if "ablation" in rd_data:
                abl = rd_data["ablation"].get("ablation_results", {})
                bias_scores[rd_model_id] = {
                    cat: info.get("baseline", 0.0) for cat, info in abl.items()
                }

        if len(cosine_matrices) >= 2:
            plot_cross_model_stability(
                cosine_matrices, bias_scores,
                path=str(fig_dir / "fig_24_cross_model_geometry_stability.png"),
            )
            log(f"  Saved fig_24")
        else:
            log("  Not enough models with directions for fig_24")
    else:
        log("  Skipping fig_24: single model only (provide --run_dirs for comparison)")

    log("\nSummary figure generation complete!")


if __name__ == "__main__":
    main()
