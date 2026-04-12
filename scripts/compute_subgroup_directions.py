#!/usr/bin/env python3
"""Compute per-subgroup identity directions for ALL categories.

Reads per-item activations and stimuli, groups deltas by stereotyped subgroup,
and computes unit-normalised subgroup directions.

Usage:
    python scripts/compute_subgroup_directions.py \
        --run_dir results/runs/llama2-13b-hf/2026-04-11/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.analysis.directions import (
    compute_category_direction,
    compute_item_delta,
    load_activations_indexed,
)
from src.analysis.geometry import cosine_similarity_matrix
from src.data.bbq_loader import CATEGORY_MAP, parse_categories
from src.utils.io import atomic_save_json, atomic_save_npz, ensure_dir
from src.utils.logging import log

ALL_CATS = list(CATEGORY_MAP.keys())
MIN_ITEMS = 10  # minimum items to compute a subgroup direction


def _load_stimuli(run_dir: Path, cat: str) -> list[dict] | None:
    files = sorted((run_dir / "stimuli").glob(f"stimuli_{cat}_*.json"))
    if not files:
        files = sorted(Path("data/processed").glob(f"stimuli_{cat}_*.json"))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)


def compute_subgroup_deltas(
    run_dir: Path,
    cat: str,
    max_items: int | None = None,
) -> dict[str, list[np.ndarray]]:
    """Load activations for a category and compute per-item deltas grouped by subgroup.

    Returns dict[subgroup_label -> list of (n_layers, hidden_dim) deltas].
    """
    stimuli = _load_stimuli(run_dir, cat)
    if stimuli is None:
        log(f"  WARNING: no stimuli for {cat}")
        return {}

    act_dir = run_dir / "activations" / cat
    if not act_dir.exists():
        log(f"  WARNING: {act_dir} not found")
        return {}

    _finals, id_by_idx, meta_by_idx = load_activations_indexed(act_dir, max_items=max_items)

    group_deltas: dict[str, list[np.ndarray]] = {}
    n_skipped = 0

    for item in stimuli[: (max_items or len(stimuli))]:
        idx = int(item["item_idx"])
        if idx not in id_by_idx or idx not in meta_by_idx:
            n_skipped += 1
            continue

        delta = compute_item_delta(id_by_idx[idx], meta_by_idx[idx], item)
        if delta is None:
            n_skipped += 1
            continue

        groups = item.get("stereotyped_groups", [])
        if not groups:
            n_skipped += 1
            continue
        primary = groups[0].lower()

        group_deltas.setdefault(primary, []).append(delta)

    log(f"  {cat}: {sum(len(v) for v in group_deltas.values())} valid deltas, "
        f"{n_skipped} skipped, {len(group_deltas)} subgroups")
    return group_deltas


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute per-subgroup directions.")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--categories", type=str, default="all")
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--model_id", type=str, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    analysis_dir = ensure_dir(run_dir / "analysis")
    model_id = args.model_id or run_dir.parent.name
    categories = parse_categories(args.categories)

    log(f"Computing subgroup directions for {model_id}")
    log(f"Categories: {categories}")

    save_arrays: dict[str, np.ndarray] = {}
    summary: dict[str, dict] = {}

    for cat in categories:
        log(f"\n{'='*60}")
        log(f"Category: {cat} ({CATEGORY_MAP.get(cat, cat)})")
        log(f"{'='*60}")

        group_deltas = compute_subgroup_deltas(run_dir, cat, args.max_items)
        if not group_deltas:
            continue

        cat_subgroups: dict[str, dict] = {}
        for sg_name, deltas in sorted(group_deltas.items()):
            n_items = len(deltas)
            if n_items < MIN_ITEMS:
                log(f"  {sg_name}: {n_items} items (< {MIN_ITEMS}, skipped)")
                cat_subgroups[sg_name] = {"n_items": n_items, "skipped": True}
                continue

            direction = compute_category_direction(deltas)
            save_arrays[f"subgroup_{cat}_{sg_name}"] = direction

            # Mean delta norm
            stacked = np.stack(deltas, axis=0)
            mean_norm = float(np.linalg.norm(stacked.mean(axis=0), axis=1).mean())

            cat_subgroups[sg_name] = {
                "n_items": n_items,
                "skipped": False,
                "mean_delta_norm": mean_norm,
            }
            log(f"  {sg_name}: {n_items} items, mean_norm={mean_norm:.4f}")

        # Within-category cosine matrix at mid layer
        sg_dirs = {
            sg: save_arrays[f"subgroup_{cat}_{sg}"]
            for sg in cat_subgroups if not cat_subgroups[sg].get("skipped", True)
        }
        if len(sg_dirs) >= 2:
            n_layers = next(iter(sg_dirs.values())).shape[0]
            mid = n_layers // 2
            sim, names = cosine_similarity_matrix(sg_dirs, mid)
            log(f"\n  Pairwise cosines at layer {mid}:")
            for i, na in enumerate(names):
                for j, nb in enumerate(names):
                    if j > i:
                        log(f"    {na} ↔ {nb}: {sim[i,j]:+.3f}")

            # Store cosine info in summary
            cosines_mid = {}
            for i, na in enumerate(names):
                for j, nb in enumerate(names):
                    if j > i:
                        cosines_mid[f"{na}_{nb}"] = float(sim[i, j])
            cat_subgroups["_cosines_mid"] = cosines_mid
            cat_subgroups["_subgroup_order"] = names

        summary[cat] = cat_subgroups

    # Save
    log(f"\n{'='*60}")
    log("Saving...")
    npz_path = analysis_dir / "subgroup_directions.npz"
    atomic_save_npz(npz_path, **save_arrays)
    log(f"  Arrays -> {npz_path} ({len(save_arrays)} entries)")

    json_path = analysis_dir / "subgroup_directions.json"
    atomic_save_json({"model_id": model_id, "categories": summary}, json_path)
    log(f"  Summary -> {json_path}")
    log("\nDone!")


if __name__ == "__main__":
    main()
