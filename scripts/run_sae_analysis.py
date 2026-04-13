#!/usr/bin/env python3
"""Stage 2: SAE Feature Discovery & Bias Decomposition Pipeline.

Uses layer(s) identified by Stage 1 to:
1. Load pre-trained SAE checkpoints and encode BBQ activations
2. Identify bias-associated SAE features (differential activation analysis)
3. Project onto DIM directions (hybrid validation, if available)
4. Characterize top features (interpretability reports)
5. Generate all figures (10–19)

Usage
-----
# Full analysis
python scripts/run_sae_analysis.py \\
    --model_path models/llama-3.1-8b \\
    --model_id llama31-8b \\
    --device mps \\
    --sae_source fnlp/Llama3_1-8B-Base-LXR-32x \\
    --sae_layers 14,16,18 \\
    --sae_expansion 32 \\
    --localization_dir results/sae_localization/llama-3.1-8b/2026-04-12/ \\
    --categories all

# Quick test
python scripts/run_sae_analysis.py \\
    --model_path models/llama-3.1-8b \\
    --model_id llama31-8b \\
    --device mps \\
    --sae_source fnlp/Llama3_1-8B-Base-LXR-8x \\
    --sae_layers 16 \\
    --sae_expansion 8 \\
    --localization_dir results/sae_localization/llama-3.1-8b/2026-04-12/ \\
    --categories so \\
    --max_items 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import atomic_save_json, ensure_dir
from src.utils.logging import log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2: SAE Feature Discovery & Bias Decomposition"
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to the language model (for logit attribution; "
        "set to 'none' to skip loading)",
    )
    parser.add_argument("--model_id", required=True, help="Model identifier")
    parser.add_argument("--device", default="cpu", help="Device (cpu/mps/cuda)")

    # SAE parameters
    parser.add_argument(
        "--sae_source",
        required=True,
        help="HuggingFace repo id or local path for SAE checkpoints",
    )
    parser.add_argument(
        "--sae_layers",
        default=None,
        help="Comma-separated layer indices to analyze (overrides layer_recommendation)",
    )
    parser.add_argument(
        "--sae_expansion",
        type=int,
        default=8,
        help="SAE expansion factor (8 = 32K features, 32 = 128K features)",
    )

    # Paths
    parser.add_argument(
        "--localization_dir",
        required=True,
        help="Stage 1 output directory with activations/ and layer_recommendation.json",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory (default: results/sae_analysis/<model_id>/<date>/)",
    )
    parser.add_argument(
        "--dim_dir",
        default=None,
        help="Directory with DIM direction .npy files (for hybrid projection)",
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        help="BBQ JSONL directory (for stimulus text in characterization), e.g. datasets/bbq/data",
    )

    # Scope
    parser.add_argument(
        "--categories",
        default="all",
        help="Categories to analyze: 'all' or comma-separated (so,gi,race,...)",
    )
    parser.add_argument(
        "--max_items",
        type=int,
        default=None,
        help="Limit items per category (for testing)",
    )
    parser.add_argument(
        "--skip_characterization",
        action="store_true",
        help="Skip Module 4 (feature characterization)",
    )
    parser.add_argument(
        "--skip_figures",
        action="store_true",
        help="Skip figure generation",
    )
    parser.add_argument(
        "--load_model",
        action="store_true",
        help="Load the LM for logit attribution (requires GPU memory)",
    )
    return parser.parse_args()


def resolve_layers(args: argparse.Namespace, loc_dir: Path) -> list[int]:
    """Determine which layers to analyze."""
    if args.sae_layers:
        layers = [int(x.strip()) for x in args.sae_layers.split(",")]
        log(f"Using CLI-specified layers: {layers}")
        return layers

    rec_path = loc_dir / "layer_recommendation.json"
    if rec_path.exists():
        with open(rec_path) as f:
            rec = json.load(f)
        overall = rec.get("overall_recommended_range", [0, 31])
        # Pick a few layers spanning the range
        lo, hi = overall[0], overall[1]
        mid = (lo + hi) // 2
        layers = sorted(set([lo, mid, hi]))
        log(f"Using layers from recommendation ({lo}-{hi}): {layers}")
        return layers

    log("WARNING: No --sae_layers and no layer_recommendation.json found. "
        "Defaulting to layer 16.")
    return [16]


def resolve_categories(args: argparse.Namespace, loc_dir: Path) -> list[str]:
    """Parse categories, falling back to available activation dirs."""
    from src.data.bbq_loader import parse_categories

    cats = parse_categories(args.categories)

    # Filter to categories that actually have activations
    act_dir = loc_dir / "activations"
    if act_dir.is_dir():
        available = {d.name for d in act_dir.iterdir() if d.is_dir()}
        filtered = [c for c in cats if c in available]
        if len(filtered) < len(cats):
            skipped = set(cats) - set(filtered)
            log(f"  Skipping categories without activations: {skipped}")
        return filtered
    return cats


def main() -> None:
    try:
        import numpy as np  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "ERROR: numpy is required for SAE analysis. Install with: pip install numpy"
        ) from exc
    try:
        import pandas as pd  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "ERROR: pandas is required for SAE analysis outputs. Install with: pip install pandas"
        ) from exc

    t0 = time.time()
    args = parse_args()

    loc_dir = Path(args.localization_dir)
    if not loc_dir.is_dir():
        log(f"ERROR: localization_dir not found: {loc_dir}")
        sys.exit(1)

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = (
            PROJECT_ROOT
            / "results"
            / "sae_analysis"
            / args.model_id
            / date.today().isoformat()
        )
    ensure_dir(output_dir)
    log(f"Output: {output_dir}")

    # Resolve layers and categories
    layers = resolve_layers(args, loc_dir)
    categories = resolve_categories(args, loc_dir)
    log(f"Layers: {layers}")
    log(f"Categories: {categories}")

    if not categories:
        log("ERROR: no categories with activations found")
        sys.exit(1)

    # --------------- Per-layer analysis ---------------
    from src.sae_localization.sae_wrapper import SAEWrapper
    from src.sae_localization.feature_discovery import (
        collect_sae_activations,
        load_sae_activations,
        run_differential_analysis,
    )
    from src.sae_localization.hybrid_projection import (
        find_dim_directions,
        run_hybrid_projection,
    )
    from src.sae_localization.feature_characterization import (
        run_feature_characterization,
    )
    from src.sae_localization.figures import generate_all_figures

    # Find DIM directions (optional)
    dim_dirs_to_scan: list[Path] = []
    if args.dim_dir:
        dim_dirs_to_scan.append(Path(args.dim_dir))
    # Also scan standard run directories
    runs_base = PROJECT_ROOT / "results" / "runs" / args.model_id
    if runs_base.is_dir():
        dim_dirs_to_scan.extend(sorted(runs_base.iterdir()))

    dim_directions = find_dim_directions(dim_dirs_to_scan)
    if dim_directions:
        log(f"Found {len(dim_directions)} DIM direction(s)")
    else:
        log("DIM directions not found. Skipping hybrid projection analysis. "
            "Run the main fragmentation pipeline first to enable this analysis.")

    # Optionally load model for logit attribution
    model = None
    tokenizer = None
    if args.load_model and args.model_path.lower() != "none":
        log("Loading language model for logit attribution ...")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            dtype=torch.float16,
        )
        model.to(args.device)
        model.eval()
        log("  Model loaded.")

    # Data dir for stimulus text (BBQ JSONL directory)
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        default_bbq = PROJECT_ROOT / "datasets" / "bbq" / "data"
        data_dir = default_bbq if default_bbq.is_dir() else (PROJECT_ROOT / "data" / "processed")
        if not default_bbq.is_dir():
            log(
                "WARNING: default BBQ data dir not found at datasets/bbq/data. "
                "Stimulus text in characterization may be unavailable. "
                "Fix by cloning BBQ to datasets/bbq or passing --data_dir datasets/bbq/data"
            )

    summary: dict[str, any] = {
        "model_id": args.model_id,
        "sae_source": args.sae_source,
        "sae_expansion": args.sae_expansion,
        "layers_analyzed": layers,
        "categories": categories,
        "per_layer": {},
    }

    for layer in layers:
        log(f"\n{'='*60}")
        log(f"Layer {layer}")
        log(f"{'='*60}")

        layer_t0 = time.time()
        sae: SAEWrapper | None = None

        def _get_sae() -> SAEWrapper:
            nonlocal sae
            if sae is None or sae.layer != layer:
                log(f"  Loading SAE for layer {layer} ...")
                sae = SAEWrapper(
                    args.sae_source,
                    layer=layer,
                    expansion=args.sae_expansion,
                    device=args.device,
                )
            return sae

        # Step 1: Load or create SAE activations
        feat_act_dir = output_dir / "feature_activations"
        existing = load_sae_activations(feat_act_dir, layer, categories)

        if len(existing) == len(categories):
            log(f"  All categories already encoded for layer {layer}, loading ...")
            cat_data = existing
        else:
            cat_data = collect_sae_activations(
                activations_dir=loc_dir / "activations",
                sae=_get_sae(),
                target_layer=layer,
                categories=categories,
                output_dir=feat_act_dir,
                max_items=args.max_items,
            )

        if not cat_data:
            log(f"  No data for layer {layer}, skipping")
            continue

        # Step 2: Differential analysis
        log(f"  Running differential feature analysis ...")
        feature_results = run_differential_analysis(
            cat_data, layer, output_dir
        )

        # Load overlap data
        overlap_path = output_dir / f"feature_overlap_layer_{layer}.json"
        overlap = {}
        if overlap_path.exists():
            with open(overlap_path) as f:
                overlap = json.load(f)

        # Step 3: Hybrid projection (if DIM directions exist)
        hybrid_summary = None
        if dim_directions:
            log("  Running hybrid DIM projection ...")
            pooled_df = feature_results.get("pooled")
            hybrid_summary = run_hybrid_projection(
                _get_sae(), dim_directions, pooled_df, layer, output_dir
            )

        # Step 4: Feature characterization
        reports: list[dict] = []
        if not args.skip_characterization:
            log("  Running feature characterization ...")
            reports = run_feature_characterization(
                cat_data=cat_data,
                feature_results=feature_results,
                sae=_get_sae(),
                target_layer=layer,
                output_dir=output_dir,
                data_dir=data_dir,
                localization_dir=loc_dir,
                model=model,
                tokenizer=tokenizer,
            )

        # Step 5: Figures
        if not args.skip_figures:
            log("  Generating figures ...")
            generate_all_figures(
                feature_results=feature_results,
                overlap=overlap,
                reports=reports,
                hybrid_summary=hybrid_summary,
                target_layer=layer,
                output_dir=output_dir,
            )

        # Layer summary
        pooled_df = feature_results.get("pooled", pd.DataFrame())
        per_cat_df = feature_results.get("per_category", pd.DataFrame())

        layer_summary: dict[str, any] = {
            "total_features": int(sae.n_features) if sae is not None else 0,
            "significant_pooled": int(pooled_df["is_significant"].sum()) if not pooled_df.empty else 0,
        }

        if not per_cat_df.empty:
            sig_per_cat = per_cat_df[per_cat_df["is_significant"]]
            layer_summary["significant_per_category"] = {
                cat: int(
                    sig_per_cat.loc[sig_per_cat["category"] == cat].shape[0]
                )
                for cat in sig_per_cat["category"].unique()
            }
        else:
            layer_summary["significant_per_category"] = {}

        layer_summary["cross_category_jaccard_mean"] = _mean_jaccard(overlap)
        layer_summary["broad_features_5plus_categories"] = (
            overlap.get("feature_breadth", {}).get("broad_5plus_cats", 0)
        )
        layer_summary["narrow_features_1_category_only"] = (
            overlap.get("feature_breadth", {}).get("narrow_1_cat", 0)
        )

        if hybrid_summary:
            for proj in hybrid_summary.get("directions_analysed", []):
                if "spearman_r" in proj:
                    layer_summary["hybrid_spearman_r"] = proj["spearman_r"]
                    layer_summary["hybrid_spearman_p"] = proj["spearman_p"]
                    break

        summary["per_layer"][str(layer)] = layer_summary

        elapsed = time.time() - layer_t0
        log(f"  Layer {layer} complete in {elapsed:.1f}s")

    # Save summary
    atomic_save_json(summary, output_dir / "sae_analysis_summary.json")

    total = time.time() - t0
    log(f"\nPipeline complete in {total:.1f}s")
    log(f"Results: {output_dir}")


def _mean_jaccard(overlap: dict) -> float:
    """Compute mean off-diagonal Jaccard from overlap dict."""
    jaccard = overlap.get("cross_category_jaccard", {})
    if not jaccard:
        return 0.0
    vals = []
    cats = list(jaccard.keys())
    for i, c1 in enumerate(cats):
        for j, c2 in enumerate(cats):
            if i < j:
                v = jaccard[c1].get(c2, 0)
                if isinstance(v, (int, float)):
                    vals.append(v)
    return float(sum(vals) / len(vals)) if vals else 0.0


if __name__ == "__main__":
    main()
