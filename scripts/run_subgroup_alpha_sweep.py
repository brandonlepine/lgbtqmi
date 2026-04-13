#!/usr/bin/env python3
"""Cross-subgroup alpha-sweep: produce transfer heatmaps at multiple alpha values.

Tests whether subgroup fragmentation is causally operative: at low alpha only
matched-subgroup features should flip answers; at high alpha everything flips.

Usage
-----
# Full sweep
python scripts/run_subgroup_alpha_sweep.py \\
    --model_path models/llama-3.1-8b \\
    --model_id llama-3.1-8b \\
    --device mps \\
    --sae_source fnlp/Llama3_1-8B-Base-LXR-8x \\
    --sae_layer 14 \\
    --sae_expansion 8 \\
    --analysis_dir results/sae_analysis/2026-04-13-max1200-logits/ \\
    --localization_dir results/sae_localization/llama-3.1-8b/2026-04-12/ \\
    --categories disability,so \\
    --alpha_values "1,2,3,5,7,10,15,20,30,50"

# Generate figures from saved results (no model needed)
python scripts/run_subgroup_alpha_sweep.py \\
    --analyze_only \\
    --analysis_dir results/sae_analysis/2026-04-13-max1200-logits/ \\
    --sweep_dir results/sae_steering/llama-3.1-8b/2026-04-12/subgroup_alpha_sweep/ \\
    --sae_layer 14
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import atomic_save_json, ensure_dir
from src.utils.logging import log


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cross-subgroup alpha-sweep for causal specificity analysis"
    )

    # Model (not required for --analyze_only)
    p.add_argument("--model_path", default=None)
    p.add_argument("--model_id", default="llama-3.1-8b")
    p.add_argument("--device", default="mps")

    # SAE
    p.add_argument("--sae_source", default=None)
    p.add_argument("--sae_layer", type=int, required=True)
    p.add_argument("--sae_expansion", type=int, default=8)

    # Data
    p.add_argument("--analysis_dir", required=True,
                   help="Stage 2 output dir with features/ parquets and overlap JSONs")
    p.add_argument("--localization_dir", default=None,
                   help="Stage 1 dir with activations/ (for behavioral metadata)")

    # Scope
    p.add_argument("--categories", default=None,
                   help="Comma-separated categories (default: auto-detect eligible)")
    p.add_argument("--alpha_values", default="1,2,3,5,7,10,15,20,30,50")
    p.add_argument("--max_items_per_subgroup", type=int, default=None,
                   help="Cap items per target subgroup (uniform subsampling)")

    # Output
    p.add_argument("--output_dir", default=None)
    p.add_argument("--sweep_dir", default=None,
                   help="For --analyze_only: dir with saved .npz results")

    # Modes
    p.add_argument("--analyze_only", action="store_true",
                   help="Generate figures from saved results (no model needed)")
    p.add_argument("--skip_figures", action="store_true")

    return p.parse_args()


def _load_stage1_metadata(
    loc_dir: Path, categories: list[str],
) -> dict[str, list[dict]]:
    """Load item metadata from Stage-1 activation .npz files."""
    import numpy as np
    items: dict[str, list[dict]] = {}
    act_dir = loc_dir / "activations"
    if not act_dir.is_dir():
        return items
    for cat in categories:
        cat_dir = act_dir / cat
        if not cat_dir.is_dir():
            continue
        cat_items = []
        for npz_path in sorted(cat_dir.glob("item_*.npz")):
            try:
                data = np.load(npz_path, allow_pickle=True)
                raw = data["metadata_json"]
                meta_str = raw.item() if raw.shape == () else str(raw)
                cat_items.append(json.loads(meta_str))
            except Exception:
                continue
        if cat_items:
            items[cat] = cat_items
    return items


def _merge_items(
    processed: list[dict], stage1: list[dict],
) -> list[dict]:
    """Merge processed stimuli with Stage-1 behavioral metadata."""
    s1_by_idx = {m.get("item_idx", -1): m for m in stage1}
    merged = []
    for item in processed:
        m = dict(item)
        s1 = s1_by_idx.get(item.get("item_idx", -1), {})
        m["model_answer"] = s1.get("model_answer", "")
        m["model_answer_role"] = s1.get("model_answer_role", "")
        m["is_stereotyped_response"] = s1.get("is_stereotyped_response", False)
        merged.append(m)
    return merged


def main() -> None:
    import numpy as np
    import pandas as pd

    t0 = time.time()
    args = parse_args()
    layer = args.sae_layer
    alpha_values = [float(x) for x in args.alpha_values.split(",")]

    analysis_dir = Path(args.analysis_dir)

    # ---- Load Stage-2 features ----
    from src.sae_localization.subgroup_sweep import (
        get_eligible_categories,
        get_subgroup_features,
        partition_items_by_subgroup,
        load_jaccard_pairs,
        load_sweep_results,
        generate_sweep_figures,
        run_subgroup_alpha_sweep,
    )
    from src.sae_localization.steering import load_significant_features

    per_sub_df = load_significant_features(analysis_dir, layer, "per_subcategory")
    if per_sub_df.empty:
        log("ERROR: no significant subcategory features found")
        sys.exit(1)

    eligible = get_eligible_categories(per_sub_df)
    log(f"Eligible categories (>=2 subgroups): {list(eligible.keys())}")

    if args.categories:
        requested = [c.strip() for c in args.categories.split(",")]
        eligible = {k: v for k, v in eligible.items() if k in requested}

    if not eligible:
        log("ERROR: no eligible categories after filtering")
        sys.exit(1)

    # ---- Output dir ----
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.sweep_dir:
        output_dir = Path(args.sweep_dir).parent
    else:
        output_dir = (
            PROJECT_ROOT / "results" / "sae_steering"
            / args.model_id / date.today().isoformat()
        )
    sweep_dir = ensure_dir(
        Path(args.sweep_dir) if args.sweep_dir
        else output_dir / "subgroup_alpha_sweep"
    )

    # ---- Load Jaccard data for Figure D ----
    all_jaccard: dict[str, list[tuple[str, str, float]]] = {}
    overlap_path = analysis_dir / f"feature_overlap_layer_{layer}.json"
    if not overlap_path.exists():
        # Try under features/
        overlap_path = analysis_dir / "features" / f"feature_overlap_layer_{layer}.json"
    if overlap_path.exists():
        for cat in eligible:
            pairs = load_jaccard_pairs(overlap_path, cat)
            if pairs:
                all_jaccard[cat] = pairs
                log(f"  Loaded {len(pairs)} Jaccard pairs for {cat}")
    else:
        log(f"  WARNING: overlap JSON not found at {overlap_path}")

    # ---- Analyze-only mode ----
    if args.analyze_only:
        log("Analyze-only mode: loading saved sweep results ...")
        all_results: dict[str, dict] = {}
        for cat in eligible:
            result = load_sweep_results(sweep_dir, cat)
            if result:
                all_results[cat] = result
                log(f"  Loaded {cat}: {len(result['alpha_values'])} alphas")
            else:
                log(f"  No saved results for {cat}")

        if all_results and not args.skip_figures:
            generate_sweep_figures(all_results, all_jaccard, output_dir)

        log(f"Done in {time.time() - t0:.1f}s")
        return

    # ---- Full sweep mode: load model + SAE ----
    if not args.model_path:
        log("ERROR: --model_path required for sweep mode (use --analyze_only for figures only)")
        sys.exit(1)
    if not args.sae_source:
        log("ERROR: --sae_source required for sweep mode")
        sys.exit(1)

    log("Loading model ...")
    from src.models.wrapper import ModelWrapper
    wrapper = ModelWrapper.from_pretrained(args.model_path, device=args.device)

    log("Loading SAE ...")
    from src.sae_localization.sae_wrapper import SAEWrapper
    sae = SAEWrapper(
        args.sae_source, layer=layer,
        expansion=args.sae_expansion, device=args.device,
    )

    from src.sae_localization.steering import SAESteerer
    steerer = SAESteerer(wrapper, sae, layer)

    # ---- Load items ----
    loc_dir = Path(args.localization_dir) if args.localization_dir else None
    proc_dir = PROJECT_ROOT / "data" / "processed"

    from src.extraction.activations import format_prompt

    all_results: dict[str, dict] = {}

    for cat, subgroups in eligible.items():
        log(f"\n{'='*50}")
        log(f"Category: {cat} — subgroups: {subgroups}")
        log(f"{'='*50}")

        # Load processed items
        proc_files = sorted(proc_dir.glob(f"stimuli_{cat}_*.json"))
        if not proc_files:
            log(f"  No processed stimuli for {cat}, skipping")
            continue
        with open(proc_files[-1]) as f:
            proc_items = json.load(f)

        # Merge with Stage-1 metadata
        s1_items = []
        if loc_dir:
            s1_data = _load_stage1_metadata(loc_dir, [cat])
            s1_items = s1_data.get(cat, [])
        merged = _merge_items(proc_items, s1_items)

        n_annotated = sum(1 for m in merged if m.get("model_answer_role"))
        log(f"  {len(merged)} items, {n_annotated} with behavioral annotation")

        # Partition by subgroup (stereotyped only)
        items_by_sub = partition_items_by_subgroup(
            merged,
            stereotyped_only=True,
            max_per_subgroup=args.max_items_per_subgroup,
        )
        log(f"  Subgroups with stereo items: " + ", ".join(
            f"{k}={len(v)}" for k, v in sorted(items_by_sub.items())
        ))

        # Build feature sets
        features_by_sub = {
            sub: get_subgroup_features(per_sub_df, cat, sub, "pro_bias")
            for sub in subgroups
        }
        features_by_sub = {k: v for k, v in features_by_sub.items() if v}
        log(f"  Subgroups with features: " + ", ".join(
            f"{k}={len(v)}" for k, v in sorted(features_by_sub.items())
        ))

        if len(features_by_sub) < 2:
            log(f"  <2 subgroups with features, skipping")
            continue

        # Filter targets to subgroups that have items
        valid_targets = {
            k: v for k, v in items_by_sub.items()
            if len(v) >= 3  # need at least a few items
        }
        if len(valid_targets) < 2:
            log(f"  <2 target subgroups with items, skipping")
            continue

        result = run_subgroup_alpha_sweep(
            steerer=steerer,
            items_by_subgroup=valid_targets,
            features_by_subgroup=features_by_sub,
            alpha_values=alpha_values,
            prompt_formatter=format_prompt,
            category=cat,
            output_dir=sweep_dir,
        )
        all_results[cat] = result

    # ---- Figures ----
    if all_results and not args.skip_figures:
        generate_sweep_figures(all_results, all_jaccard, output_dir)

    # ---- Summary ----
    summary: dict[str, Any] = {
        "model_id": args.model_id,
        "sae_layer": layer,
        "alpha_values": alpha_values,
        "categories": list(all_results.keys()),
    }
    for cat, result in all_results.items():
        from src.sae_localization.subgroup_sweep import (
            compute_diagonal_gap, compute_specificity_ratios,
        )
        gaps = compute_diagonal_gap(
            result["flip_rates"], result["source_subgroups"], result["target_subgroups"],
        )
        summary[cat] = {
            "n_sources": len(result["source_subgroups"]),
            "n_targets": len(result["target_subgroups"]),
            "max_diagonal_gap": float(np.nanmax(gaps)),
            "max_gap_alpha": float(alpha_values[int(np.argmax(gaps))]),
        }

    atomic_save_json(summary, output_dir / "subgroup_alpha_sweep_summary.json")

    log(f"\nComplete in {time.time() - t0:.1f}s")
    log(f"Results: {sweep_dir}")


if __name__ == "__main__":
    main()
