#!/usr/bin/env python3
"""Stage 3: SAE Feature Steering & Causal Validation Pipeline.

Tests whether bias-associated SAE features are causally operative via:
  A — Suppress bias (dampen pro-bias features)
  B — Elicit bias (amplify pro-bias features)
  C — Anti-bias feature validation
  D — Cross-dataset transfer (CrowS-Pairs)
  E — Side-effect testing (MMLU / MedQA)

Usage
-----
python scripts/run_sae_steering.py \\
    --model_path models/llama-3.1-8b \\
    --model_id llama-3.1-8b \\
    --device mps \\
    --sae_source fnlp/Llama3_1-8B-Base-LXR-8x \\
    --sae_layer 14 \\
    --sae_expansion 8 \\
    --analysis_dir results/sae_analysis/llama-3.1-8b/2026-04-12/ \\
    --categories all
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
        description="Stage 3: SAE Feature Steering & Causal Validation"
    )
    p.add_argument("--model_path", required=True)
    p.add_argument("--model_id", required=True)
    p.add_argument("--device", default="mps")

    p.add_argument("--sae_source", required=True)
    p.add_argument("--sae_layer", type=int, required=True)
    p.add_argument("--sae_expansion", type=int, default=8)
    p.add_argument(
        "--analysis_dir", required=True,
        help="Stage 2 output dir with features/ parquets",
    )
    p.add_argument("--localization_dir", default=None,
                   help="Stage 1 output dir (for re-loading item metadata)")

    p.add_argument("--categories", default="all")
    p.add_argument("--max_items", type=int, default=None)
    p.add_argument("--output_dir", default=None)

    p.add_argument(
        "--alpha_values", default=None,
        help="Comma-separated alpha values for dampening sweep "
        "(default: -50,-40,-30,-20,-15,-10,-5,-2,-1)",
    )
    p.add_argument(
        "--amplify_alphas", default=None,
        help="Comma-separated alpha values for amplifying sweep "
        "(default: 1,2,5,10,15,20,30,40,50)",
    )

    p.add_argument(
        "--experiments", default="A,B",
        help="Comma-separated experiments to run (A,B,C,D,E)",
    )
    p.add_argument("--crows_pairs_path", default=None)
    p.add_argument("--mmlu_path", default=None)
    p.add_argument("--medqa_path", default=None)

    p.add_argument("--skip_figures", action="store_true")
    p.add_argument("--skip_individual", action="store_true",
                   help="Skip individual feature testing (Option 3)")

    return p.parse_args()


def _load_items_from_stage1(
    loc_dir: Path, categories: list[str],
) -> dict[str, list[dict]]:
    """Load item metadata from Stage-1 activation .npz files."""
    import numpy as np

    items_by_cat: dict[str, list[dict]] = {}
    act_dir = loc_dir / "activations"
    if not act_dir.is_dir():
        log(f"  WARNING: no activations dir at {act_dir}")
        return items_by_cat

    for cat in categories:
        cat_dir = act_dir / cat
        if not cat_dir.is_dir():
            continue
        items = []
        for npz_path in sorted(cat_dir.glob("item_*.npz")):
            try:
                data = np.load(npz_path, allow_pickle=True)
                raw = data["metadata_json"]
                meta_str = raw.item() if raw.shape == () else str(raw)
                meta = json.loads(meta_str)
                items.append(meta)
            except Exception:
                continue
        if items:
            items_by_cat[cat] = items
            log(f"  Loaded {len(items)} items for {cat}")
    return items_by_cat


def _load_processed_items(categories: list[str]) -> dict[str, list[dict]]:
    """Load processed stimuli from data/processed/ (for prompt formatting)."""
    items_by_cat: dict[str, list[dict]] = {}
    proc_dir = PROJECT_ROOT / "data" / "processed"
    if not proc_dir.is_dir():
        return items_by_cat

    for cat in categories:
        files = sorted(proc_dir.glob(f"stimuli_{cat}_*.json"))
        if not files:
            continue
        try:
            with open(files[-1]) as f:
                stimuli = json.load(f)
            items_by_cat[cat] = stimuli
            log(f"  Loaded {len(stimuli)} processed items for {cat}")
        except Exception as exc:
            log(f"  WARNING: failed to load processed items for {cat}: {exc}")
    return items_by_cat


def main() -> None:
    try:
        import numpy as np  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "ERROR: numpy is required for SAE steering. Install with: pip install numpy"
        ) from exc
    try:
        import pandas as pd  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "ERROR: pandas is required for SAE steering outputs. Install with: pip install pandas pyarrow"
        ) from exc

    t0 = time.time()
    args = parse_args()

    experiments = [x.strip().upper() for x in args.experiments.split(",")]
    log(f"Experiments to run: {experiments}")

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = (
            PROJECT_ROOT / "results" / "sae_steering"
            / args.model_id / date.today().isoformat()
        )
    ensure_dir(output_dir)
    log(f"Output: {output_dir}")

    # Parse categories
    from src.data.bbq_loader import parse_categories
    categories = parse_categories(args.categories)

    # Alpha grids
    from src.sae_localization.steering import DAMPEN_ALPHAS, AMPLIFY_ALPHAS
    dampen_alphas = (
        [float(x) for x in args.alpha_values.split(",")]
        if args.alpha_values
        else DAMPEN_ALPHAS
    )
    amplify_alphas = (
        [float(x) for x in args.amplify_alphas.split(",")]
        if args.amplify_alphas
        else AMPLIFY_ALPHAS
    )

    # ---- Load model ----
    log("Loading model ...")
    from src.models.wrapper import ModelWrapper
    wrapper = ModelWrapper.from_pretrained(args.model_path, device=args.device)

    # ---- Load SAE ----
    log("Loading SAE ...")
    from src.sae_localization.sae_wrapper import SAEWrapper
    sae = SAEWrapper(
        args.sae_source,
        layer=args.sae_layer,
        expansion=args.sae_expansion,
        device=args.device,
    )

    # ---- Load Stage-2 features ----
    from src.sae_localization.steering import (
        SAESteerer, load_significant_features, get_feature_set,
    )
    analysis_dir = Path(args.analysis_dir)
    layer = args.sae_layer

    per_cat_df = load_significant_features(analysis_dir, layer, "per_category")
    per_sub_df = load_significant_features(analysis_dir, layer, "per_subcategory")
    log(f"Loaded {len(per_cat_df)} per-category and {len(per_sub_df)} per-subcategory sig features")
    if per_cat_df.empty and per_sub_df.empty:
        raise SystemExit(
            "ERROR: No significant feature parquet files found (or no significant features) for "
            f"layer {layer} under {analysis_dir}/features/. "
            "Make sure you ran Stage 2 (`scripts/run_sae_analysis.py`) with matching --sae_layer/--sae_expansion."
        )

    # ---- Create steerer ----
    steerer = SAESteerer(wrapper, sae, layer)

    # ---- Load items ----
    # Processed items have prompt fields (context, question, answers, answer_roles).
    # Stage-1 metadata has model behavioral responses (model_answer_role, is_stereotyped).
    # We merge: processed items for prompt formatting, Stage-1 for partitioning.
    loc_dir = Path(args.localization_dir) if args.localization_dir else None
    processed_items = _load_processed_items(categories)

    stage1_items: dict[str, list[dict]] = {}
    if loc_dir:
        stage1_items = _load_items_from_stage1(loc_dir, categories)

    # Build merged items: processed stimuli annotated with Stage-1 behavioral data
    all_items: dict[str, list[dict]] = {}
    for cat in categories:
        proc = processed_items.get(cat, [])
        s1 = stage1_items.get(cat, [])

        if not proc:
            continue

        # Index Stage-1 metadata by item_idx
        s1_by_idx: dict[int, dict] = {}
        for m in s1:
            idx = m.get("item_idx", -1)
            if idx >= 0:
                s1_by_idx[idx] = m

        merged = []
        for item in proc:
            idx = item.get("item_idx", -1)
            s1_meta = s1_by_idx.get(idx, {})
            # Annotate with behavioral data from Stage-1
            item_merged = dict(item)
            item_merged["model_answer"] = s1_meta.get("model_answer", "")
            item_merged["model_answer_role"] = s1_meta.get("model_answer_role", "")
            item_merged["is_stereotyped_response"] = s1_meta.get("is_stereotyped_response", False)
            item_merged["is_correct"] = s1_meta.get("is_correct", False)
            merged.append(item_merged)

        n_annotated = sum(1 for m in merged if m.get("model_answer_role"))
        all_items[cat] = merged
        log(f"  {cat}: {len(merged)} items, {n_annotated} with Stage-1 behavioral data")

    if not all_items:
        raise SystemExit(
            "ERROR: No processed items loaded for any category from data/processed/. "
            "Run: python scripts/prepare_stimuli.py --categories all --bbq_data_dir datasets/bbq/data"
        )

    # Filter to categories that have both items AND features
    valid_cats = [
        c for c in categories
        if c in all_items and len(get_feature_set(per_cat_df, c, direction="pro_bias")) > 0
    ]
    log(f"Valid categories (items + features): {valid_cats}")

    # Prompt formatter
    from src.extraction.activations import format_prompt

    # ---- Partition items by model response type (from Stage-1 behavioral data) ----
    def _partition_items(items: list[dict], max_items: int | None = None):
        stereo = [it for it in items if it.get("model_answer_role") == "stereotyped_target"]
        non_stereo = [it for it in items if it.get("model_answer_role") == "non_stereotyped"]
        if max_items:
            stereo = stereo[:max_items]
            non_stereo = non_stereo[:max_items]
        return stereo, non_stereo

    # ---- Run experiments ----
    from src.sae_localization.experiments import (
        experiment_a_suppress,
        experiment_b_elicit,
        experiment_c_antibias,
        experiment_d_crows_pairs,
        experiment_e_side_effects,
        experiment_cross_subgroup_transfer,
    )

    exp_a_all: dict[str, dict] = {}
    exp_b_all: dict[str, dict] = {}
    exp_c_all: dict[str, dict] = {}
    optimal_alphas: dict[str, float] = {}
    indiv_results: dict[str, pd.DataFrame] = {}
    transfer_results: dict[str, dict] = {}

    for cat in valid_cats:
        log(f"\n{'='*50}")
        log(f"Category: {cat}")
        log(f"{'='*50}")

        items = all_items[cat]
        stereo_items, non_stereo_items = _partition_items(items, args.max_items)
        log(f"  {len(stereo_items)} stereotyped, {len(non_stereo_items)} non-stereotyped")

        pro_bias_features = get_feature_set(per_cat_df, cat, direction="pro_bias")
        anti_bias_features = get_feature_set(per_cat_df, cat, direction="anti_bias")
        log(f"  {len(pro_bias_features)} pro-bias, {len(anti_bias_features)} anti-bias features")

        cat_out = ensure_dir(output_dir / "experiments")

        # Experiment A
        if "A" in experiments and stereo_items and pro_bias_features:
            result_a = experiment_a_suppress(
                steerer, stereo_items, pro_bias_features, dampen_alphas,
                format_prompt, cat, cat_out,
            )
            exp_a_all[cat] = result_a
            optimal_alphas[cat] = result_a.get("optimal_alpha", -10)

            # Individual feature testing (Option 3)
            if not args.skip_individual and len(pro_bias_features) > 1:
                opt_alpha = optimal_alphas[cat]
                # Pre-compute baseline answers for individual testing
                for it in stereo_items:
                    prompt = format_prompt(it)
                    baseline = steerer.evaluate_baseline(prompt)
                    it["_baseline_answer"] = baseline["model_answer"]

                indiv_df = steerer.test_individual_features(
                    stereo_items, pro_bias_features, opt_alpha, format_prompt,
                )
                indiv_results[cat] = indiv_df
                indiv_df.to_parquet(cat_out / f"individual_features_{cat}.parquet", index=False)

        # Experiment B
        if "B" in experiments and non_stereo_items and pro_bias_features:
            result_b = experiment_b_elicit(
                steerer, non_stereo_items, pro_bias_features, amplify_alphas,
                format_prompt, cat, cat_out,
            )
            exp_b_all[cat] = result_b

        # Experiment C
        if "C" in experiments and anti_bias_features:
            opt_alpha = optimal_alphas.get(cat, -10)
            result_c = experiment_c_antibias(
                steerer, stereo_items, non_stereo_items, anti_bias_features,
                abs(opt_alpha), format_prompt, cat, cat_out,
            )
            exp_c_all[cat] = result_c

        # Memory cleanup
        if args.device == "mps":
            import torch
            torch.mps.empty_cache()

    # Experiment D: CrowS-Pairs
    exp_d: dict[str, Any] = {}
    if "D" in experiments:
        crows_path = Path(args.crows_pairs_path) if args.crows_pairs_path else (
            PROJECT_ROOT / "data" / "raw" / "crows_pairs.csv"
        )
        if crows_path.exists():
            from src.data.crows_pairs_loader import load_crows_pairs_as_stimuli
            crows_items = load_crows_pairs_as_stimuli(crows_path, max_items=args.max_items)
            log(f"\nLoaded {len(crows_items)} CrowS-Pairs items")

            feature_map = {
                cat: get_feature_set(per_cat_df, cat, direction="pro_bias")
                for cat in valid_cats
            }
            exp_d = experiment_d_crows_pairs(
                steerer, crows_items, feature_map, optimal_alphas,
                ensure_dir(output_dir / "experiments"),
            )
        else:
            log(f"  CrowS-Pairs not found at {crows_path}, skipping Experiment D")

    # Experiment E: Side effects
    exp_e: dict[str, Any] = {}
    if "E" in experiments:
        log("\nExperiment E: side effects")
        # Build composite steering vector from all pro-bias features across categories
        import torch
        all_pro_features = []
        for cat in valid_cats:
            all_pro_features.extend(get_feature_set(per_cat_df, cat, direction="pro_bias"))
        all_pro_features = list(set(all_pro_features))

        if all_pro_features:
            mean_alpha = sum(optimal_alphas.values()) / max(len(optimal_alphas), 1)
            composite_vec = steerer.get_composite_steering(all_pro_features, mean_alpha)

            # Load MMLU / MedQA if paths provided
            mmlu_items = None
            medqa_items = None

            if args.mmlu_path and Path(args.mmlu_path).is_dir():
                log("  Loading MMLU items ...")
                # TODO: implement MMLU loading from standard format
                log("  MMLU loading not yet implemented — skipping")

            if args.medqa_path and Path(args.medqa_path).is_dir():
                log("  Loading MedQA items ...")
                # TODO: implement MedQA loading
                log("  MedQA loading not yet implemented — skipping")

            if mmlu_items or medqa_items:
                exp_e = experiment_e_side_effects(
                    steerer, composite_vec, mmlu_items, medqa_items,
                    ensure_dir(output_dir / "experiments"),
                )

    # ---- Cross-subgroup transfer (for Figure 28) ----
    for cat in valid_cats:
        if not per_sub_df.empty:
            cat_subs = per_sub_df[
                (per_sub_df["category"] == cat) & per_sub_df["is_significant"]
            ]["subcategory"].unique()
            if len(cat_subs) >= 2:
                features_by_sub = {
                    sub: get_feature_set(per_sub_df, cat, sub, "pro_bias")
                    for sub in cat_subs
                }
                features_by_sub = {k: v for k, v in features_by_sub.items() if v}

                items_by_sub: dict[str, list[dict]] = {}
                for it in all_items.get(cat, []):
                    for sg in it.get("stereotyped_groups", []):
                        items_by_sub.setdefault(sg, []).append(it)
                if args.max_items:
                    items_by_sub = {
                        k: v[:args.max_items] for k, v in items_by_sub.items()
                    }

                if len(features_by_sub) >= 2 and len(items_by_sub) >= 2:
                    opt_alpha = optimal_alphas.get(cat, -10)
                    tr = experiment_cross_subgroup_transfer(
                        steerer, items_by_sub, features_by_sub,
                        opt_alpha, format_prompt, cat,
                        ensure_dir(output_dir / "experiments"),
                    )
                    transfer_results[cat] = tr

    # ---- Figures ----
    if not args.skip_figures:
        from src.sae_localization.steering_figures import generate_all_steering_figures

        summary: dict[str, Any] = {
            "model_id": args.model_id,
            "sae_layer": args.sae_layer,
            "sae_expansion": args.sae_expansion,
            "optimal_alphas": optimal_alphas,
            "experiment_A": {"per_category": exp_a_all},
            "experiment_B": {"per_category": exp_b_all},
            "experiment_C": {"per_category": exp_c_all},
            "experiment_D": exp_d,
            "experiment_E": exp_e,
        }

        generate_all_steering_figures(
            summary=summary,
            exp_a_all=exp_a_all,
            exp_b_all=exp_b_all,
            exp_d=exp_d if exp_d else None,
            exp_e=exp_e if exp_e else None,
            indiv_results=indiv_results if indiv_results else None,
            transfer_results=transfer_results if transfer_results else None,
            output_dir=output_dir,
        )

    # ---- Save summary ----
    summary = {
        "model_id": args.model_id,
        "sae_layer": args.sae_layer,
        "sae_expansion": args.sae_expansion,
        "optimal_alphas": {k: float(v) for k, v in optimal_alphas.items()},
        "experiment_A": {
            "per_category": {
                cat: {
                    "n_items": r.get("n_items", 0),
                    "correction_rate": r.get("optimal_rates", {}).get("correction_rate", 0),
                    "degeneration_rate": r.get("optimal_rates", {}).get("degeneration_rate", 0),
                    "unknown_rate": r.get("optimal_rates", {}).get("unknown_rate", 0),
                }
                for cat, r in exp_a_all.items()
            },
        },
        "experiment_B": {
            "per_category": {
                cat: {
                    "n_items": r.get("n_items", 0),
                    "corruption_rate": r.get("optimal_rates", {}).get("corruption_rate", 0),
                    "degeneration_rate": r.get("optimal_rates", {}).get("degeneration_rate", 0),
                }
                for cat, r in exp_b_all.items()
            },
        },
        "experiment_D": {
            k: v for k, v in exp_d.items()
            if k not in ("per_bias_type",)  # keep summary compact
        } if exp_d else {},
        "experiment_E": exp_e,
    }
    atomic_save_json(summary, output_dir / "sae_steering_summary.json")

    total = time.time() - t0
    log(f"\nPipeline complete in {total:.1f}s")
    log(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
