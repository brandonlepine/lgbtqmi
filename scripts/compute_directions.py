#!/usr/bin/env python3
"""Compute within-item identity deltas and category-level directions.

Usage:
    python scripts/compute_directions.py \
        --run_dir results/runs/llama2-13b/2026-04-10 \
        --categories all

    # Quick test
    python scripts/compute_directions.py \
        --run_dir results/runs/llama2-13b/2026-04-10 \
        --categories so \
        --max_items 20
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.analysis.directions import (
    compute_category_direction,
    compute_gender_decomposition,
    compute_item_delta,
    compute_subgroup_directions,
    load_activations_indexed,
)
from src.analysis.geometry import cosine_similarity_matrix
from src.data.bbq_loader import CATEGORY_MAP, parse_categories
from src.utils.io import atomic_save_json, ensure_dir
from src.utils.logging import ProgressLogger, log


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute identity directions from activations.")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to run directory")
    parser.add_argument("--categories", type=str, default="all", help="Comma-separated categories or 'all'")
    parser.add_argument("--max_items", type=int, default=None, help="Max items per category")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    categories = parse_categories(args.categories)
    analysis_dir = ensure_dir(run_dir / "analysis")

    log(f"Computing directions for {len(categories)} categories")
    log(f"Run dir: {run_dir}")

    # Check for existing output (resume safety)
    output_path = analysis_dir / "directions.npz"
    if output_path.exists():
        log(f"WARNING: {output_path} already exists, will overwrite")

    all_category_directions: dict[str, np.ndarray] = {}
    all_subgroup_directions: dict[str, dict[str, np.ndarray]] = {}
    all_deltas: dict[str, list[np.ndarray]] = {}
    summary: dict[str, dict] = {}

    for cat in categories:
        log(f"\n{'='*60}")
        log(f"Category: {cat} ({CATEGORY_MAP[cat]})")
        log(f"{'='*60}")

        act_dir = run_dir / "activations" / cat
        if not act_dir.exists():
            log(f"  WARNING: {act_dir} not found, skipping")
            continue

        # Find stimuli file
        stimuli_dir = run_dir / "stimuli"
        stimuli_files = list(stimuli_dir.glob(f"stimuli_{cat}_*.json"))
        if not stimuli_files:
            # Fall back to data/processed/
            stimuli_files = list(Path("data/processed").glob(f"stimuli_{cat}_*.json"))
        if not stimuli_files:
            log(f"  WARNING: No stimuli file found for {cat}, skipping")
            continue
        stimuli_path = sorted(stimuli_files)[-1]  # most recent

        log(f"  Loading stimuli from {stimuli_path}")
        with open(stimuli_path) as f:
            stimuli_items = json.load(f)

        log(f"  Loading activations from {act_dir}")
        _finals_by_idx, hidden_identities_by_idx, metadatas_by_idx = load_activations_indexed(
            act_dir, max_items=args.max_items
        )
        n_items = len(hidden_identities_by_idx)
        log(f"  Loaded {n_items} activation files (indexed by item_idx)")

        if n_items == 0:
            log(f"  WARNING: No activations found, skipping")
            continue

        # Compute per-item deltas
        log("  Computing per-item deltas...")
        deltas: list[np.ndarray] = []
        deltas_metas: list[dict] = []
        deltas_items: list[dict] = []
        n_skipped = 0
        progress = ProgressLogger(len(stimuli_items), prefix=f"  [{cat}]")

        for j, stim in enumerate(stimuli_items):
            idx = int(stim["item_idx"])
            if idx not in hidden_identities_by_idx or idx not in metadatas_by_idx:
                n_skipped += 1
                if (j + 1) % 200 == 0:
                    progress.count = j + 1
                    progress.step(f"({len(deltas)} valid deltas)")
                continue
            delta = compute_item_delta(
                hidden_identities_by_idx[idx], metadatas_by_idx[idx], stim
            )
            if delta is not None:
                deltas.append(delta)
                deltas_metas.append(metadatas_by_idx[idx])
                deltas_items.append(stim)
            else:
                n_skipped += 1
            if (j + 1) % 200 == 0:
                progress.count = j + 1
                progress.step(f"({len(deltas)} valid deltas)")

        log(
            f"  Valid deltas: {len(deltas)}/{len(stimuli_items)} "
            f"({n_skipped} skipped due to missing/invalid spans)"
        )

        if len(deltas) < 5:
            log(f"  WARNING: Too few deltas ({len(deltas)}), skipping")
            continue

        # Category-level direction
        cat_direction = compute_category_direction(deltas)
        all_category_directions[cat] = cat_direction
        all_deltas[cat] = deltas
        log(f"  Category direction computed: shape {cat_direction.shape}")

        # Sub-group directions
        log("  Computing sub-group directions...")
        subgroup_dirs = compute_subgroup_directions(deltas, deltas_metas, deltas_items)
        all_subgroup_directions[cat] = subgroup_dirs

        # SO-specific: gender decomposition
        decomp: dict[str, np.ndarray] = {}
        if cat == "so":
            decomp = compute_gender_decomposition(subgroup_dirs)
            for name, d in decomp.items():
                all_category_directions[f"so_{name}"] = d

        summary[cat] = {
            "n_items": n_items,
            "n_valid_deltas": len(deltas),
            "n_subgroups": len(subgroup_dirs),
            "subgroup_names": list(subgroup_dirs.keys()),
            "direction_shape": list(cat_direction.shape),
        }
        if decomp:
            summary[cat]["decomposition_keys"] = list(decomp.keys())

    # Save all directions to a single .npz
    if not all_category_directions:
        log("ERROR: No directions computed. Check that activations exist.")
        return

    log(f"\n{'='*60}")
    log("Saving directions...")
    log(f"{'='*60}")

    save_arrays: dict[str, np.ndarray] = {}
    for name, d in all_category_directions.items():
        save_arrays[f"direction_{name}"] = d
    for cat, sg_dirs in all_subgroup_directions.items():
        for sg_name, d in sg_dirs.items():
            save_arrays[f"subgroup_{cat}_{sg_name}"] = d

    # Save metadata as JSON string
    save_arrays["_metadata"] = np.array(json.dumps(summary))
    save_arrays["_category_names"] = np.array(json.dumps(list(all_category_directions.keys())))

    np.savez(output_path, **save_arrays)
    log(f"  Saved {len(save_arrays)} arrays -> {output_path}")

    # Also save summary JSON
    summary_path = analysis_dir / "directions_summary.json"
    atomic_save_json(summary, summary_path)
    log(f"  Summary -> {summary_path}")

    # Quick cross-category cosine preview
    if len(all_category_directions) >= 2:
        # Use middle layer
        n_layers = list(all_category_directions.values())[0].shape[0]
        mid_layer = n_layers // 2
        sim, names = cosine_similarity_matrix(all_category_directions, mid_layer)
        log(f"\n  Cross-category cosines at layer {mid_layer}:")
        for i, ni in enumerate(names):
            for j, nj in enumerate(names):
                if j > i:
                    log(f"    {ni} <-> {nj}: {sim[i,j]:.3f}")


if __name__ == "__main__":
    main()
