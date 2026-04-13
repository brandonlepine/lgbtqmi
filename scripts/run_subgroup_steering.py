#!/usr/bin/env python3
"""Subgroup-specific stepwise steering evaluation.

For each subgroup, incrementally adds features from the ranked list and
sweeps alpha to find the optimal (k, alpha, injection_layer) triple.

Usage
-----
python scripts/run_subgroup_steering.py \\
    --model_path models/llama-3.1-8b \\
    --model_id llama-3.1-8b \\
    --device mps \\
    --sae_source fnlp/Llama3_1-8B-Base-LXR-8x \\
    --sae_expansion 8 \\
    --ranked_features results/steering_features/llama-3.1-8b/ranked_features_by_subgroup.json \\
    --localization_dir results/sae_localization/llama-3.1-8b/2026-04-12/ \\
    --categories so,disability \\
    --alpha_values "-80,-60,-40,-20,-10,-5,5,10,20,40,60,80"
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
        description="Subgroup-specific stepwise steering evaluation"
    )
    p.add_argument("--model_path", required=True)
    p.add_argument("--model_id", default="llama-3.1-8b")
    p.add_argument("--device", default="mps")

    p.add_argument("--sae_source", required=True)
    p.add_argument("--sae_expansion", type=int, default=8)

    p.add_argument("--ranked_features", required=True,
                   help="Path to ranked_features_by_subgroup.json from rank_subgroup_features.py")
    p.add_argument("--localization_dir", default=None,
                   help="Stage 1 dir (for behavioral metadata)")

    p.add_argument("--categories", default=None,
                   help="Comma-separated categories (default: all in ranked features)")
    p.add_argument("--subgroups", default=None,
                   help="Comma-separated subgroups within category (default: all)")
    p.add_argument("--max_items", type=int, default=None)
    p.add_argument("--output_dir", default=None)

    p.add_argument("--alpha_values", default="-80,-60,-40,-20,-10,-5,5,10,20,40,60,80")
    p.add_argument("--k_steps", default="1,2,3,5,8,13,21")
    p.add_argument("--skip_figures", action="store_true")

    return p.parse_args()


def _load_merged_items(
    proc_dir: Path, loc_dir: Path | None, cat: str,
) -> list[dict]:
    """Load processed items merged with Stage-1 metadata for one category."""
    import numpy as np

    files = sorted(proc_dir.glob(f"stimuli_{cat}_*.json"))
    if not files:
        return []
    with open(files[-1]) as f:
        proc = json.load(f)

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
    return merged


def main() -> None:
    import numpy as np
    import torch

    t0 = time.time()
    args = parse_args()

    alpha_values = [float(x) for x in args.alpha_values.split(",")]
    k_steps = [int(x) for x in args.k_steps.split(",")]

    # Load ranked features
    with open(args.ranked_features) as f:
        ranked = json.load(f)

    categories = list(ranked.keys())
    if args.categories:
        requested = [c.strip() for c in args.categories.split(",")]
        categories = [c for c in categories if c in requested]

    output_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "results" / "subgroup_steering"
        / args.model_id / date.today().isoformat()
    )
    ensure_dir(output_dir)
    log(f"Output: {output_dir}")

    # Load model
    log("Loading model ...")
    from src.models.wrapper import ModelWrapper
    wrapper = ModelWrapper.from_pretrained(args.model_path, device=args.device)

    # Determine which SAE layers we need
    needed_layers: set[int] = set()
    for cat in categories:
        for sub, dirs in ranked.get(cat, {}).items():
            for f in dirs.get("pro_bias", []) + dirs.get("anti_bias", []):
                needed_layers.add(f["layer"])

    log(f"Loading SAEs for layers: {sorted(needed_layers)}")
    from src.sae_localization.sae_wrapper import SAEWrapper
    sae_cache: dict[int, SAEWrapper] = {}
    for layer in sorted(needed_layers):
        sae_cache[layer] = SAEWrapper(
            args.sae_source, layer=layer,
            expansion=args.sae_expansion, device=args.device,
        )

    from src.extraction.activations import format_prompt
    from src.sae_localization.subgroup_steering import (
        run_stepwise_sweep,
        build_steering_manifest,
        build_subgroup_steering_vector,
        fig_stepwise_correction,
        fig_optimal_k_distribution,
        fig_alpha_vs_k_heatmap,
        fig_margin_conditioned_correction,
    )

    loc_dir = Path(args.localization_dir) if args.localization_dir else None
    proc_dir = PROJECT_ROOT / "data" / "processed"

    all_optimals: list[dict] = {}
    all_results: dict[str, dict[str, dict]] = {}  # cat → sub → result
    all_manifests: list[dict] = []

    for cat in categories:
        subs_data = ranked.get(cat, {})
        sub_names = sorted(subs_data.keys())
        if args.subgroups:
            requested_subs = [s.strip() for s in args.subgroups.split(",")]
            sub_names = [s for s in sub_names if s in requested_subs]

        if not sub_names:
            continue

        log(f"\n{'='*50}")
        log(f"Category: {cat} — subgroups: {sub_names}")
        log(f"{'='*50}")

        # Load items
        items = _load_merged_items(proc_dir, loc_dir, cat)
        n_annotated = sum(1 for m in items if m.get("model_answer_role"))
        log(f"  {len(items)} items, {n_annotated} with behavioral annotation")

        cat_out = ensure_dir(output_dir / "stepwise")
        cat_results: dict[str, dict] = {}

        for sub in sub_names:
            pro_feats = subs_data[sub].get("pro_bias", [])
            if not pro_feats:
                log(f"  Skipping {sub}: no pro-bias features")
                continue

            # Filter items targeting this subgroup (stereotyped only)
            sub_items = [
                it for it in items
                if it.get("model_answer_role") == "stereotyped_target"
                and sub in it.get("stereotyped_groups", [])
            ]
            if args.max_items:
                sub_items = sub_items[:args.max_items]

            if len(sub_items) < 3:
                log(f"  Skipping {sub}: only {len(sub_items)} stereotyped items")
                continue

            result = run_stepwise_sweep(
                wrapper=wrapper,
                sae_cache=sae_cache,
                feature_list=pro_feats,
                items=sub_items,
                alpha_values=alpha_values,
                k_steps=k_steps,
                prompt_formatter=format_prompt,
                subgroup=sub,
                category=cat,
                output_dir=cat_out,
            )
            cat_results[sub] = result

            # Save optimal steering vector
            opt = result.get("optimal", {})
            if opt and opt.get("k"):
                vec, inj = build_subgroup_steering_vector(
                    pro_feats, sae_cache, opt["k"], opt["alpha"],
                    device=args.device, dtype=wrapper.model.dtype,
                )
                vec_dir = ensure_dir(output_dir / "steering_vectors")
                np.savez(
                    vec_dir / f"{cat}_{sub}.npz",
                    vector=vec.float().cpu().numpy(),
                    injection_layer=inj,
                    alpha=opt["alpha"],
                    k=opt["k"],
                )

                manifest = build_steering_manifest(opt, float(vec.float().norm()))
                all_manifests.append(manifest)

                log(f"  Optimal for {sub}: k={opt['k']}, alpha={opt['alpha']}, "
                    f"layer={opt.get('injection_layer')}, "
                    f"correction={opt.get('correction_rate', 0):.3f}")

        all_results[cat] = cat_results

    # Save optimal configs & manifests
    atomic_save_json(
        {cat: {sub: r.get("optimal", {}) for sub, r in subs.items()}
         for cat, subs in all_results.items()},
        output_dir / "optimal_configs.json",
    )
    atomic_save_json(all_manifests, output_dir / "steering_manifests.json")

    # Save full stepwise results
    atomic_save_json(
        {cat: {sub: r.get("grid", []) for sub, r in subs.items()}
         for cat, subs in all_results.items()},
        output_dir / "stepwise_results.json",
    )

    # Figures
    if not args.skip_figures:
        fig_dir = ensure_dir(output_dir / "figures")
        log("Generating figures ...")

        all_opt_list = [m for m in all_manifests if m.get("optimal_k")]
        fig_optimal_k_distribution(all_opt_list, fig_dir)

        for cat, cat_results in all_results.items():
            fig_stepwise_correction(cat_results, cat, fig_dir)
            fig_alpha_vs_k_heatmap(cat_results, cat, fig_dir)

        fig_margin_conditioned_correction(all_results, fig_dir)

    total = time.time() - t0
    log(f"\nComplete in {total:.1f}s")
    log(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
