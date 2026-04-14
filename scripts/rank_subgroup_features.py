#!/usr/bin/env python3
"""Rank SAE features per subgroup across all analysed layers.

Bridges Stage 2 (SAE feature analysis) to the subgroup steering pipeline.
Collects significant features per subgroup, ranks by |Cohen's d|, computes
overlap matrices and injection layers, and generates diagnostic figures.

Usage
-----
python scripts/rank_subgroup_features.py \\
    --analysis_dir results/sae_analysis/llama-3.1-8b/2026-04-13-logits/ \\
    --model_id llama-3.1-8b \\
    --top_k 20

If --layers is omitted, layers are auto-detected from per_subcategory_layer_*.parquet files.
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
    p.add_argument("--layers", default=None,
                   help="Comma-separated layer indices (auto-detected if omitted)")
    p.add_argument("--model_id", default="llama-3.1-8b")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--top_k", type=int, default=20, help="Top-K for overlap matrix")
    p.add_argument("--skip_figures", action="store_true")
    return p.parse_args()


def _detect_layers(analysis_dir: Path) -> list[int]:
    """Auto-detect analysed layers from per_subcategory parquet filenames."""
    feat_dir = analysis_dir / "features"
    layers: set[int] = set()
    for p in feat_dir.glob("per_subcategory_layer_*.parquet"):
        parts = p.stem.split("_")
        try:
            layers.add(int(parts[-1]))
        except ValueError:
            continue
    result = sorted(layers)
    log(f"Auto-detected layers: {result}")
    return result


def _compute_injection_layers(
    ranked: dict[str, dict[str, dict[str, list[dict]]]],
) -> dict[str, dict[str, object]]:
    """Determine optimal injection layer per subgroup from top-10 pro-bias features."""
    injection_layers: dict[str, dict[str, object]] = {}
    for cat, subs in ranked.items():
        for sub, dirs in subs.items():
            top_feats = dirs.get("pro_bias", [])[:10]
            if not top_feats:
                continue
            layer_counts: dict[int, int] = {}
            for f in top_feats:
                layer_counts[f["layer"]] = layer_counts.get(f["layer"], 0) + 1
            best_layer = max(layer_counts, key=layer_counts.get)
            injection_layers[f"{cat}/{sub}"] = {
                "injection_layer": best_layer,
                "layer_distribution": layer_counts,
                "n_features_total": len(dirs.get("pro_bias", [])),
            }
    return injection_layers


def main() -> None:
    args = parse_args()
    analysis_dir = Path(args.analysis_dir)

    # 1. Determine layers
    if args.layers:
        layers = [int(x.strip()) for x in args.layers.split(",")]
    else:
        layers = _detect_layers(analysis_dir)

    if not layers:
        log("ERROR: no layers found or specified")
        sys.exit(1)

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

    # 2. Rank features across all layers per subgroup
    log(f"Ranking features from {analysis_dir}, layers {layers}")
    ranked = rank_subgroup_features(analysis_dir, layers)

    if not ranked:
        log("ERROR: no ranked features produced")
        sys.exit(1)

    # 3. Determine optimal injection layer per subgroup
    injection_layers = _compute_injection_layers(ranked)

    # 4. Compute overlap matrices (both pro-bias and anti-bias)
    overlap_pro = compute_overlap_matrix(ranked, top_k=args.top_k, direction="pro_bias")
    overlap_anti = compute_overlap_matrix(ranked, top_k=args.top_k, direction="anti_bias")

    # 5. Save everything
    atomic_save_json(ranked, output_dir / "ranked_features_by_subgroup.json")
    log(f"Saved ranked features to {output_dir / 'ranked_features_by_subgroup.json'}")

    atomic_save_json(injection_layers, output_dir / "injection_layers.json")
    log(f"Saved injection layers ({len(injection_layers)} subgroups)")

    atomic_save_json(
        {"pro_bias": overlap_pro, "anti_bias": overlap_anti},
        output_dir / "feature_overlap.json",
    )
    log(f"Saved overlap matrices (pro: {len(overlap_pro)}, anti: {len(overlap_anti)} categories)")

    # 6. Generate figures
    if not args.skip_figures:
        log("Generating figures ...")
        fig_feature_overlap_heatmaps(overlap_pro, fig_dir)
        fig_feature_layer_distribution(ranked, fig_dir, direction="pro_bias", top_k=args.top_k)
        fig_ranked_effect_sizes(ranked, fig_dir, direction="pro_bias", max_rank=30)
        log("Figures complete.")

    # 7. Summary
    for cat, subs in ranked.items():
        n_subs = len(subs)
        n_with_features = sum(1 for s in subs.values() if s.get("pro_bias"))
        total_feats = sum(len(s.get("pro_bias", [])) for s in subs.values())
        log(f"  {cat}: {n_with_features}/{n_subs} subgroups with features, {total_feats} total")


if __name__ == "__main__":
    main()
