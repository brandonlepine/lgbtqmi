#!/usr/bin/env python3
"""Rank SAE features per subgroup across all analysed layers.

Produces ranked feature lists, overlap matrices, and diagnostic figures.

Usage
-----
python scripts/rank_subgroup_features.py \\
    --analysis_dir results/sae_analysis/2026-04-13-max1200-logits/ \\
    --layers 14,16,18 \\
    --model_id llama-3.1-8b
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import atomic_save_json, ensure_dir
from src.utils.logging import log


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rank SAE features per subgroup")
    p.add_argument("--analysis_dir", required=True)
    p.add_argument("--layers", required=True, help="Comma-separated layer indices")
    p.add_argument("--model_id", default="llama-3.1-8b")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--top_k", type=int, default=20, help="Top-K for overlap matrix")
    p.add_argument("--skip_figures", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    analysis_dir = Path(args.analysis_dir)
    layers = [int(x.strip()) for x in args.layers.split(",")]

    output_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "results" / "steering_features" / args.model_id
    )
    ensure_dir(output_dir)
    fig_dir = ensure_dir(output_dir / "figures")

    from src.sae_localization.subgroup_features import (
        rank_subgroup_features,
        compute_overlap_matrix,
        fig_feature_overlap_heatmaps,
        fig_feature_layer_distribution,
        fig_ranked_effect_sizes,
    )

    log(f"Ranking features from {analysis_dir}, layers {layers}")
    ranked = rank_subgroup_features(analysis_dir, layers)

    if not ranked:
        log("ERROR: no ranked features produced")
        sys.exit(1)

    # Summary stats
    for cat, subs in ranked.items():
        for sub, dirs in subs.items():
            n_pro = len(dirs.get("pro_bias", []))
            n_anti = len(dirs.get("anti_bias", []))
            log(f"  {cat}/{sub}: {n_pro} pro-bias, {n_anti} anti-bias features")

    # Save
    atomic_save_json(ranked, output_dir / "ranked_features_by_subgroup.json")
    log(f"Saved ranked features to {output_dir / 'ranked_features_by_subgroup.json'}")

    # Overlap matrix
    overlap = compute_overlap_matrix(ranked, top_k=args.top_k, direction="pro_bias")
    atomic_save_json(overlap, output_dir / "feature_overlap_matrix.json")
    log(f"Saved overlap matrix ({len(overlap)} categories)")

    # Figures
    if not args.skip_figures:
        log("Generating figures ...")
        fig_feature_overlap_heatmaps(overlap, fig_dir)
        fig_feature_layer_distribution(ranked, fig_dir, top_k=args.top_k)
        fig_ranked_effect_sizes(ranked, fig_dir)
        log("Figures complete.")


if __name__ == "__main__":
    main()
