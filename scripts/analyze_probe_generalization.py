#!/usr/bin/env python3
"""Test cross-category and cross-benchmark generalization of probes.

Generates figures 14-16: transfer matrix, transfer vs cosine, cross-benchmark.

Usage:
    python scripts/analyze_probe_generalization.py \
        --run_dir results/runs/llama2-13b/2026-04-10 \
        --n_heads 40 --head_dim 128

    # With CrowS-Pairs
    python scripts/analyze_probe_generalization.py \
        --run_dir results/runs/llama2-13b/2026-04-10 \
        --n_heads 40 --head_dim 128 \
        --crows_pairs_path data/raw/crows_pairs.csv
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.linear_model import RidgeClassifier

from src.analysis.directions import load_activations_indexed
from src.analysis.geometry import cosine_similarity_matrix
from src.analysis.probes import (
    build_stereotyping_labels,
    collect_head_features,
)
from src.data.bbq_loader import CATEGORY_MAP, parse_categories
from src.utils.io import atomic_save_json, ensure_dir
from src.utils.logging import log
from src.visualization.heatmaps import plot_transfer_matrix
from src.visualization.scatter import plot_transfer_vs_cosine


def load_category_data(
    run_dir: Path,
    cat: str,
    max_items: int | None = None,
) -> tuple[list[np.ndarray], list[dict], list[dict]] | None:
    """Load activations + stimuli for a single category."""
    act_dir = run_dir / "activations" / cat
    if not act_dir.exists():
        return None

    stimuli_files = sorted((run_dir / "stimuli").glob(f"stimuli_{cat}_*.json"))
    if not stimuli_files:
        stimuli_files = sorted(Path("data/processed").glob(f"stimuli_{cat}_*.json"))
    if not stimuli_files:
        return None

    with open(stimuli_files[-1]) as f:
        stimuli = json.load(f)

    finals_by_idx, _ids_by_idx, metas_by_idx = load_activations_indexed(
        act_dir,
        max_items=max_items,
        final_key="attn_pre_o_proj_final",
    )

    aligned_finals: list[np.ndarray] = []
    aligned_metas: list[dict] = []
    aligned_stimuli: list[dict] = []
    for item in stimuli[: (max_items or len(stimuli))]:
        idx = int(item["item_idx"])
        if idx in finals_by_idx and idx in metas_by_idx:
            aligned_finals.append(finals_by_idx[idx])
            aligned_metas.append(metas_by_idx[idx])
            aligned_stimuli.append(item)

    if not aligned_finals:
        return None
    return aligned_finals, aligned_metas, aligned_stimuli


def train_stereotyping_probe(
    finals: list[np.ndarray],
    metas: list[dict],
    stimuli: list[dict],
    layer: int,
    head_idx: int,
    head_dim: int,
) -> RidgeClassifier | None:
    """Train a stereotyping probe on one category's data. Returns fitted clf or None."""
    mask, y = build_stereotyping_labels(stimuli, metas)
    if mask.sum() < 10:
        return None

    filtered_finals = [f for f, m in zip(finals, mask) if m]
    X = np.stack([
        f[layer, head_idx * head_dim:(head_idx + 1) * head_dim]
        for f in filtered_finals
    ])

    clf = RidgeClassifier(alpha=1.0)
    clf.fit(X, y)
    return clf


def evaluate_probe(
    clf: RidgeClassifier,
    finals: list[np.ndarray],
    metas: list[dict],
    stimuli: list[dict],
    layer: int,
    head_idx: int,
    head_dim: int,
) -> float:
    """Evaluate a trained probe on a target category. Returns accuracy."""
    mask, y = build_stereotyping_labels(stimuli, metas)
    if mask.sum() < 5:
        return 0.5  # chance

    filtered_finals = [f for f, m in zip(finals, mask) if m]
    X = np.stack([
        f[layer, head_idx * head_dim:(head_idx + 1) * head_dim]
        for f in filtered_finals
    ])

    return clf.score(X, y)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe generalization analysis.")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--categories", type=str, default="all")
    parser.add_argument("--n_heads", type=int, required=True)
    parser.add_argument("--head_dim", type=int, required=True)
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--crows_pairs_path", type=str, default=None)
    parser.add_argument("--model_id", type=str, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    categories = parse_categories(args.categories)
    fig_dir = ensure_dir(run_dir / "figures")
    analysis_dir = ensure_dir(run_dir / "analysis")
    model_id = args.model_id or run_dir.parent.name

    # Find best stereotyping head from probe results
    probe_path = analysis_dir / "probe_matrices.npz"
    if probe_path.exists():
        probe_data = np.load(probe_path)
        stereo_matrix = probe_data["base_stereo"]
        best_flat = stereo_matrix.argmax()
        best_layer = best_flat // args.n_heads
        best_head = best_flat % args.n_heads
        log(f"Using best stereotyping head: L{best_layer}H{best_head} "
            f"(acc={stereo_matrix[best_layer, best_head]:.3f})")
    else:
        # Default to middle layer, head 0
        n_layers_guess = 40
        best_layer = n_layers_guess // 2
        best_head = 0
        log(f"No probe_matrices.npz found, defaulting to L{best_layer}H{best_head}")

    # Load data per category
    log("\nLoading category data...")
    cat_data: dict[str, tuple] = {}
    for cat in categories:
        data = load_category_data(run_dir, cat, args.max_items)
        if data is not None:
            cat_data[cat] = data
            log(f"  {cat}: {len(data[0])} items")
        else:
            log(f"  {cat}: not found, skipping")

    available_cats = sorted(cat_data.keys())
    n_cats = len(available_cats)

    if n_cats < 2:
        log("ERROR: Need at least 2 categories for transfer analysis")
        return

    # ===== Fig 14: Cross-category transfer matrix =====
    log(f"\n--- Fig 14: Cross-category transfer matrix ---")
    transfer_matrix = np.full((n_cats, n_cats), 0.5, dtype=np.float32)

    for i, source_cat in enumerate(available_cats):
        src_finals, src_metas, src_stimuli = cat_data[source_cat]
        clf = train_stereotyping_probe(
            src_finals, src_metas, src_stimuli,
            best_layer, best_head, args.head_dim,
        )
        if clf is None:
            log(f"  Source {source_cat}: couldn't train probe, skipping")
            continue

        for j, target_cat in enumerate(available_cats):
            tgt_finals, tgt_metas, tgt_stimuli = cat_data[target_cat]
            acc = evaluate_probe(
                clf, tgt_finals, tgt_metas, tgt_stimuli,
                best_layer, best_head, args.head_dim,
            )
            transfer_matrix[i, j] = acc

        log(f"  Source {source_cat}: trained, transfer computed")

    plot_transfer_matrix(
        transfer_matrix, available_cats,
        path=str(fig_dir / "fig_14_cross_category_transfer_matrix.png"),
        title=f"Probe transfer at L{best_layer}H{best_head} ({model_id})",
    )
    log(f"  Saved fig_14")

    # ===== Fig 15: Transfer vs cosine =====
    log(f"\n--- Fig 15: Transfer vs cosine ---")
    # Load directions for cosine computation
    directions_path = analysis_dir / "directions.npz"
    if directions_path.exists():
        dir_data = np.load(directions_path, allow_pickle=True)
        cat_directions = {}
        for cat in available_cats:
            key = f"direction_{cat}"
            if key in dir_data.files:
                cat_directions[cat] = dir_data[key]

        if len(cat_directions) >= 2:
            sim_matrix, sim_names = cosine_similarity_matrix(cat_directions, best_layer)

            cosine_vals: list[float] = []
            transfer_vals: list[float] = []
            pair_names: list[str] = []

            for i, ca in enumerate(available_cats):
                for j, cb in enumerate(available_cats):
                    if i >= j:
                        continue
                    if ca in sim_names and cb in sim_names:
                        ci = sim_names.index(ca)
                        cj = sim_names.index(cb)
                        cosine_vals.append(float(sim_matrix[ci, cj]))
                        # Average both transfer directions
                        avg_transfer = (transfer_matrix[i, j] + transfer_matrix[j, i]) / 2
                        transfer_vals.append(float(avg_transfer))
                        pair_names.append(f"{ca}↔{cb}")

            if cosine_vals:
                plot_transfer_vs_cosine(
                    np.array(cosine_vals), np.array(transfer_vals), pair_names,
                    path=str(fig_dir / "fig_15_transfer_vs_cosine.png"),
                    title=f"Direction cosine vs probe transfer ({model_id})",
                )
                log(f"  Saved fig_15")
            else:
                log("  No cosine-transfer pairs available")
        else:
            log("  Not enough directions for cosine analysis")
    else:
        log(f"  No directions.npz found, skipping fig_15")

    # ===== Fig 16: Cross-benchmark (CrowS-Pairs) =====
    cross_benchmark_results: dict[str, dict] = {}
    if args.crows_pairs_path and Path(args.crows_pairs_path).exists():
        log(f"\n--- Fig 16: Cross-benchmark generalization ---")
        log(f"  CrowS-Pairs path: {args.crows_pairs_path}")
        # CrowS-Pairs integration would require additional activation extraction
        # Placeholder: save empty results with a note
        cross_benchmark_results = {
            cat: {"accuracy": 0.5, "n_items": 0, "note": "requires CrowS-Pairs activations"}
            for cat in available_cats
        }
        log("  NOTE: CrowS-Pairs analysis requires separate activation extraction. "
            "Run extract_activations.py on CrowS-Pairs stimuli first.")
    else:
        log("\n  Skipping fig_16: no CrowS-Pairs data provided")

    # Save results
    results = {
        "model_id": model_id,
        "best_layer": int(best_layer),
        "best_head": int(best_head),
        "categories": available_cats,
        "transfer_matrix": transfer_matrix.tolist(),
        "cross_benchmark": cross_benchmark_results,
    }
    atomic_save_json(results, analysis_dir / "generalization_results.json")
    log(f"\nResults saved to {analysis_dir / 'generalization_results.json'}")
    log("Done!")


if __name__ == "__main__":
    main()
