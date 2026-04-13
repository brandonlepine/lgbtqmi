#!/usr/bin/env python3
"""Steering validation: logit margin analysis + random vector controls.

Adds two analyses to the Stage 3 steering results without modifying
existing scripts:
  1. Margin stratification — are corrections concentrated among low-margin items?
  2. Random control — does a norm-matched random vector produce the same effect?

Usage
-----
# Margin analysis only (no model needed)
python scripts/run_steering_validation.py --margin_only \\
    --localization_dir results/sae_localization/llama-3.1-8b/2026-04-12/ \\
    --steering_dir results/sae_steering/llama-3.1-8b/2026-04-13/ \\
    --analysis_dir results/sae_analysis/2026-04-13-max1200-logits/ \\
    --sae_layer 14

# Random control only (needs model + SAE)
python scripts/run_steering_validation.py --random_only \\
    --model_path models/llama-3.1-8b \\
    --sae_source fnlp/Llama3_1-8B-Base-LXR-8x \\
    --sae_layer 14 --sae_expansion 8 \\
    --steering_dir results/sae_steering/llama-3.1-8b/2026-04-13/ \\
    --analysis_dir results/sae_analysis/2026-04-13-max1200-logits/ \\
    --localization_dir results/sae_localization/llama-3.1-8b/2026-04-12/ \\
    --categories disability,so

# Both analyses
python scripts/run_steering_validation.py \\
    --model_path models/llama-3.1-8b \\
    --sae_source fnlp/Llama3_1-8B-Base-LXR-8x \\
    --sae_layer 14 --sae_expansion 8 \\
    --steering_dir results/sae_steering/llama-3.1-8b/2026-04-13/ \\
    --analysis_dir results/sae_analysis/2026-04-13-max1200-logits/ \\
    --localization_dir results/sae_localization/llama-3.1-8b/2026-04-12/ \\
    --categories all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import atomic_save_json, ensure_dir
from src.utils.logging import log


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Steering validation: margin analysis + random vector controls"
    )

    # Model (optional — not needed for --margin_only)
    p.add_argument("--model_path", default=None)
    p.add_argument("--model_id", default="llama-3.1-8b")
    p.add_argument("--device", default="mps")

    # SAE (optional — not needed for --margin_only)
    p.add_argument("--sae_source", default=None)
    p.add_argument("--sae_layer", type=int, required=True)
    p.add_argument("--sae_expansion", type=int, default=8)

    # Directories
    p.add_argument("--steering_dir", required=True,
                   help="Stage 3 output dir with experiments/ subdirectory")
    p.add_argument("--analysis_dir", required=True,
                   help="Stage 2 output dir with features/ parquets")
    p.add_argument("--localization_dir", default=None,
                   help="Stage 1 dir for margin computation from activation .npz files")
    p.add_argument("--output_dir", default=None,
                   help="Output dir (default: <steering_dir>/validation/)")

    # Scope
    p.add_argument("--categories", default="all")
    p.add_argument("--max_items", type=int, default=None)
    p.add_argument("--n_trials", type=int, default=10,
                   help="Number of random-vector trials")

    # Modes
    p.add_argument("--margin_only", action="store_true",
                   help="Run margin analysis only (no model needed)")
    p.add_argument("--random_only", action="store_true",
                   help="Run random control only (needs model)")
    p.add_argument("--skip_figures", action="store_true")

    return p.parse_args()


def _load_feature_results(
    steering_dir: Path, categories: list[str],
) -> dict[str, dict]:
    """Load Experiment A and B JSON results from Stage 3."""
    exp_dir = steering_dir / "experiments"
    results: dict[str, dict] = {}

    for cat in categories:
        entry: dict = {}

        a_path = exp_dir / f"experiment_A_{cat}.json"
        if a_path.exists():
            with open(a_path) as f:
                a = json.load(f)
            entry["optimal_alpha"] = a.get("optimal_alpha", -10)
            entry["correction_rate"] = a.get("optimal_rates", {}).get("correction_rate", 0)
            entry["per_alpha"] = a.get("per_alpha", {})
            entry["feature_indices"] = a.get("feature_indices", [])
            entry["n_features"] = a.get("n_features", 0)

        b_path = exp_dir / f"experiment_B_{cat}.json"
        if b_path.exists():
            with open(b_path) as f:
                b = json.load(f)
            entry["optimal_alpha_b"] = b.get("optimal_alpha", 10)
            entry["corruption_rate"] = b.get("optimal_rates", {}).get("corruption_rate", 0)

        if entry:
            results[cat] = entry

    return results


def _load_merged_items(
    proc_dir: Path, loc_dir: Path | None, categories: list[str],
) -> dict[str, list[dict]]:
    """Load processed items merged with Stage-1 behavioral metadata."""
    import numpy as np
    all_items: dict[str, list[dict]] = {}

    for cat in categories:
        # Processed items
        files = sorted(proc_dir.glob(f"stimuli_{cat}_*.json"))
        if not files:
            continue
        with open(files[-1]) as f:
            proc = json.load(f)

        # Stage-1 metadata
        s1_by_idx: dict[int, dict] = {}
        if loc_dir:
            act_dir = loc_dir / "activations" / cat
            if act_dir.is_dir():
                for npz_path in sorted(act_dir.glob("item_*.npz")):
                    try:
                        data = np.load(npz_path, allow_pickle=True)
                        raw = data["metadata_json"]
                        meta = json.loads(raw.item() if raw.shape == () else str(raw))
                        s1_by_idx[meta.get("item_idx", -1)] = meta
                    except Exception:
                        continue

        merged = []
        for item in proc:
            m = dict(item)
            s1 = s1_by_idx.get(item.get("item_idx", -1), {})
            m["model_answer"] = s1.get("model_answer", "")
            m["model_answer_role"] = s1.get("model_answer_role", "")
            m["is_stereotyped_response"] = s1.get("is_stereotyped_response", False)
            merged.append(m)
        all_items[cat] = merged

    return all_items


def main() -> None:
    import numpy as np
    import pandas as pd

    t0 = time.time()
    args = parse_args()
    layer = args.sae_layer

    steering_dir = Path(args.steering_dir)
    analysis_dir = Path(args.analysis_dir)
    loc_dir = Path(args.localization_dir) if args.localization_dir else None

    output_dir = Path(args.output_dir) if args.output_dir else steering_dir / "validation"
    ensure_dir(output_dir)
    log(f"Output: {output_dir}")

    # Categories
    from src.data.bbq_loader import parse_categories
    categories = parse_categories(args.categories)

    # Filter to categories that have Stage 3 results
    exp_dir = steering_dir / "experiments"
    categories = [
        c for c in categories
        if (exp_dir / f"experiment_A_{c}.json").exists()
        or (exp_dir / f"experiment_A_{c}_sweep.parquet").exists()
    ]
    log(f"Categories with steering results: {categories}")

    if not categories:
        log("ERROR: no categories found with steering experiment results")
        sys.exit(1)

    # Load Stage 3 feature results
    feature_results = _load_feature_results(steering_dir, categories)

    # ================================================================
    # Analysis 1: Margin stratification
    # ================================================================
    margin_summary: dict = {}
    margins_df = None
    sweep_dfs: dict[str, pd.DataFrame] = {}

    if not args.random_only:
        log("\n=== Analysis 1: Logit margin stratification ===")
        from src.sae_localization.margin_analysis import run_margin_analysis

        if loc_dir is None:
            log("WARNING: --localization_dir not set; margins will be derived from sweep logits")

        margin_summary = run_margin_analysis(
            localization_dir=loc_dir or Path("."),
            steering_dir=steering_dir,
            categories=categories,
            output_dir=output_dir,
        )

        # Load margins_df and sweep parquets for figures
        margins_path = output_dir / "margin_analysis" / "margins_per_item.parquet"
        if margins_path.exists():
            margins_df = pd.read_parquet(margins_path)

        for cat in categories:
            sp = exp_dir / f"experiment_A_{cat}_sweep.parquet"
            if sp.exists():
                sweep_dfs[cat] = pd.read_parquet(sp)

    # ================================================================
    # Analysis 2: Random vector control
    # ================================================================
    random_summary: dict | None = None
    random_df: pd.DataFrame | None = None

    if not args.margin_only:
        log("\n=== Analysis 2: Random vector control ===")

        if not args.model_path:
            log("ERROR: --model_path required for random control")
            sys.exit(1)
        if not args.sae_source:
            log("ERROR: --sae_source required for random control")
            sys.exit(1)

        log("Loading model ...")
        from src.models.wrapper import ModelWrapper
        wrapper = ModelWrapper.from_pretrained(args.model_path, device=args.device)

        log("Loading SAE ...")
        from src.sae_localization.sae_wrapper import SAEWrapper
        sae = SAEWrapper(args.sae_source, layer=layer,
                         expansion=args.sae_expansion, device=args.device)

        from src.sae_localization.steering import SAESteerer, get_feature_set, load_significant_features
        from src.sae_localization.random_control import (
            run_random_trials, build_random_summary, compute_actual_steering_norm,
        )

        steerer = SAESteerer(wrapper, sae, layer)

        # Load items
        proc_dir = PROJECT_ROOT / "data" / "processed"
        all_items = _load_merged_items(proc_dir, loc_dir, categories)

        # Load features
        per_cat_df = load_significant_features(analysis_dir, layer, "per_category")

        from src.extraction.activations import format_prompt

        random_out = ensure_dir(output_dir / "random_control")
        all_random: list[pd.DataFrame] = []

        alpha_values = [-50, -40, -30, -20, -15, -10, -5, -2, -1]

        for cat in categories:
            feat = feature_results.get(cat, {})
            feature_indices = feat.get("feature_indices", [])

            if not feature_indices:
                # Try loading from per_cat_df
                feature_indices = get_feature_set(per_cat_df, cat, direction="pro_bias")

            if not feature_indices:
                log(f"  Skipping {cat}: no feature indices")
                continue

            items = all_items.get(cat, [])
            if not items:
                log(f"  Skipping {cat}: no items")
                continue

            # Experiment A: stereotyped items, negative alpha
            stereo_items = [it for it in items if it.get("model_answer_role") == "stereotyped_target"]
            if args.max_items:
                stereo_items = stereo_items[:args.max_items]

            if stereo_items:
                log(f"\n  Random control — {cat} (Exp A): {len(stereo_items)} stereo items")
                opt_alpha = feat.get("optimal_alpha", -10)
                target_norm = compute_actual_steering_norm(
                    sae, feature_indices, opt_alpha,
                )

                rdf = run_random_trials(
                    steerer=steerer,
                    items=stereo_items,
                    target_norm=target_norm,
                    alpha_values=alpha_values,
                    prompt_formatter=format_prompt,
                    category=cat,
                    sae=sae,
                    feature_indices=feature_indices,
                    n_trials=args.n_trials,
                    output_dir=random_out,
                    experiment="A",
                    save_per_item_trial=0,
                )
                all_random.append(rdf)

            # Experiment B: non-stereo items, positive alpha
            non_stereo = [it for it in items if it.get("model_answer_role") == "non_stereotyped"]
            if args.max_items:
                non_stereo = non_stereo[:args.max_items]

            if non_stereo:
                log(f"  Random control — {cat} (Exp B): {len(non_stereo)} non-stereo items")
                amplify_alphas = [1, 2, 5, 10, 15, 20, 30, 40, 50]

                rdf_b = run_random_trials(
                    steerer=steerer,
                    items=non_stereo,
                    target_norm=target_norm,
                    alpha_values=amplify_alphas,
                    prompt_formatter=format_prompt,
                    category=cat,
                    sae=sae,
                    feature_indices=feature_indices,
                    n_trials=args.n_trials,
                    output_dir=random_out,
                    experiment="B",
                )
                all_random.append(rdf_b)

        if all_random:
            random_df = pd.concat(all_random, ignore_index=True)
            random_df.to_parquet(random_out / "random_trials.parquet", index=False)

            random_summary = build_random_summary(random_df, feature_results)
            atomic_save_json(random_summary, random_out / "random_summary.json")
            log(f"  Random control complete: {len(random_summary.get('per_category', {}))} categories")
        else:
            log("  No random control results produced")

    # ================================================================
    # Figures
    # ================================================================
    if not args.skip_figures:
        log("\n=== Generating validation figures ===")
        from src.sae_localization.validation_figures import generate_validation_figures

        # If margin_summary wasn't computed this run, try loading from disk
        if not margin_summary:
            ms_path = output_dir / "margin_analysis" / "margin_summary.json"
            if ms_path.exists():
                with open(ms_path) as f:
                    margin_summary = json.load(f)

        if random_summary is None:
            rs_path = output_dir / "random_control" / "random_summary.json"
            if rs_path.exists():
                with open(rs_path) as f:
                    random_summary = json.load(f)

        generate_validation_figures(
            margin_summary=margin_summary or {},
            random_summary=random_summary,
            margins_df=margins_df,
            sweep_dfs=sweep_dfs or None,
            feature_results=feature_results,
            output_dir=output_dir,
        )

    total = time.time() - t0
    log(f"\nValidation complete in {total:.1f}s")
    log(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
