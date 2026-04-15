#!/usr/bin/env python3
"""Subgroup-specific stepwise steering evaluation with joint (k, alpha) optimization.

For each subgroup, incrementally adds features from the ranked list and
sweeps alpha to find the optimal (k, alpha, injection_layer) triple that
maximizes steering efficiency eta = RCR_1.0 / ||v||_2.

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
from typing import Any

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
    p.add_argument("--skip_exacerbation", action="store_true")

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


def _phase1_prune_alphas(
    wrapper: Any,
    sae_cache: dict[int, Any],
    feature_list: list[dict],
    items: list[dict],
    alpha_values: list[float],
    prompt_formatter: Any,
    category: str,
    subgroup: str,
    output_dir: Path,
) -> list[float]:
    """Phase 1: run k=1 across all alphas to prune non-viable ones.

    Viable = RCR_1.0 > 0 AND corruption < 0.05 AND degeneration < 0.05.
    If fewer than 3 are viable, keep all.
    """
    from src.metrics.bias_metrics import compute_margin, compute_rcr
    from src.sae_localization.steering import SAESteerer
    from src.sae_localization.subgroup_steering import build_subgroup_steering_vector

    viable: list[float] = []
    log(f"    Phase 1 (alpha pruning) for {category}/{subgroup}: k=1 across {len(alpha_values)} alphas")

    for alpha in alpha_values:
        ckpt_path = output_dir / f"{category}_{subgroup}_phase1_a{alpha}.json"
        if ckpt_path.exists():
            with open(ckpt_path) as f:
                rec = json.load(f)
            if rec.get("viable", False):
                viable.append(alpha)
            continue

        vec, inj_layer = build_subgroup_steering_vector(
            feature_list, sae_cache, k=1, alpha=alpha,
            device=wrapper.device, dtype=wrapper.model.dtype,
        )
        first_sae = sae_cache.get(inj_layer, next(iter(sae_cache.values())))
        steerer = SAESteerer(wrapper, first_sae, inj_layer)

        item_results: list[dict] = []
        n_degen = 0
        n_corrupted = 0
        for item in items:
            prompt = prompt_formatter(item)
            baseline = steerer.evaluate_baseline(prompt)
            result = steerer.steer_and_evaluate(prompt, vec)

            orig = baseline["model_answer"]
            steered_ans = result["model_answer"]
            orig_role = item.get("answer_roles", {}).get(orig, "unknown")
            steered_role = item.get("answer_roles", {}).get(steered_ans, "unknown")

            if result["degenerated"]:
                n_degen += 1

            corrected = (
                orig_role == "stereotyped_target"
                and steered_role in ("non_stereotyped", "unknown")
            )
            corrupted = (
                orig_role == "non_stereotyped"
                and steered_role == "stereotyped_target"
            )
            if corrupted:
                n_corrupted += 1

            logits_b = baseline.get("answer_logits", {})
            try:
                logits_b_float = {lk: float(lv) for lk, lv in logits_b.items()}
                margin = compute_margin(logits_b_float, orig) if orig in logits_b_float else 0.0
            except (ValueError, TypeError):
                margin = 0.0

            item_results.append({"corrected": corrected, "margin": margin})

        n = max(len(items), 1)
        degen_rate = n_degen / n
        corrupt_rate = n_corrupted / n
        rcr_result = compute_rcr(item_results, tau=1.0)
        is_viable = rcr_result["rcr"] > 0 and corrupt_rate < 0.05 and degen_rate < 0.05

        rec = {
            "alpha": alpha,
            "rcr_1.0": rcr_result["rcr"],
            "corruption_rate": corrupt_rate,
            "degeneration_rate": degen_rate,
            "viable": is_viable,
        }
        atomic_save_json(rec, ckpt_path)

        if is_viable:
            viable.append(alpha)
        log(f"      alpha={alpha}: RCR={rcr_result['rcr']:.3f} "
            f"corrupt={corrupt_rate:.3f} degen={degen_rate:.3f} "
            f"{'VIABLE' if is_viable else 'pruned'}")

    if len(viable) == 0:
        log(f"    Phase 1: no viable alphas found, keeping all {len(alpha_values)}")
        return alpha_values

    log(f"    Phase 1: {len(viable)}/{len(alpha_values)} alphas viable: {viable}")
    return viable


def _run_exacerbation(
    wrapper: Any,
    sae_cache: dict[int, Any],
    feature_list: list[dict],
    items: list[dict],
    best_k: int,
    best_alpha: float,
    prompt_formatter: Any,
    category: str,
    subgroup: str,
) -> dict[str, Any]:
    """Run exacerbation test: flip alpha sign and evaluate on non-stereotyped items."""
    from src.metrics.bias_metrics import compute_all_metrics, compute_margin
    from src.sae_localization.steering import SAESteerer
    from src.sae_localization.subgroup_steering import build_subgroup_steering_vector

    # Build exacerbation vector (flipped alpha sign)
    exac_vec, inj_layer = build_subgroup_steering_vector(
        feature_list, sae_cache, best_k, -best_alpha,
        device=wrapper.device, dtype=wrapper.model.dtype,
    )

    first_sae = sae_cache.get(inj_layer, next(iter(sae_cache.values())))
    steerer = SAESteerer(wrapper, first_sae, inj_layer)

    # Evaluate on ALL items (not just stereotyped) to test corruption
    exac_results: list[dict] = []
    for item in items:
        prompt = prompt_formatter(item)
        baseline = steerer.evaluate_baseline(prompt)
        result = steerer.steer_and_evaluate(prompt, exac_vec)

        orig = baseline["model_answer"]
        steered_ans = result["model_answer"]
        orig_role = item.get("answer_roles", {}).get(orig, "unknown")
        steered_role = item.get("answer_roles", {}).get(steered_ans, "unknown")

        corrected = (
            orig_role == "stereotyped_target"
            and steered_role in ("non_stereotyped", "unknown")
        )
        corrupted = (
            orig_role in ("non_stereotyped", "unknown")
            and steered_role == "stereotyped_target"
        )

        logits_b = baseline.get("answer_logits", {})
        logits_s = result.get("answer_logits", {})
        try:
            logits_b_float = {lk: float(lv) for lk, lv in logits_b.items()}
            margin = compute_margin(logits_b_float, orig) if orig in logits_b_float else 0.0
        except (ValueError, TypeError):
            logits_b_float = {}
            margin = 0.0
        try:
            logits_s_float = {lk: float(lv) for lk, lv in logits_s.items()}
        except (ValueError, TypeError):
            logits_s_float = {}

        answer_roles = item.get("answer_roles", {})
        stereo_opt = ""
        for letter, role in answer_roles.items():
            if role == "stereotyped_target":
                stereo_opt = letter
                break

        exac_results.append({
            "corrected": corrected,
            "corrupted": corrupted,
            "margin": margin,
            "logit_baseline": logits_b_float,
            "logit_steered": logits_s_float,
            "stereotyped_option": stereo_opt,
        })

    exac_metrics = compute_all_metrics(exac_results)
    return {
        "corruption_rate": exac_metrics.get("raw_corruption_rate", 0),
        "mean_logit_shift": exac_metrics.get("logit_shift", {}).get("mean_shift", 0),
        "n_items": len(exac_results),
        "metrics": exac_metrics,
    }


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

    all_results: dict[str, dict[str, dict]] = {}  # cat → sub → result
    all_manifests: list[dict] = []
    exacerbation_data: dict[str, dict[str, dict]] = {}  # cat → sub → exac result

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
        cat_exac: dict[str, dict] = {}

        for sub in sub_names:
            pro_feats = subs_data[sub].get("pro_bias", [])
            if not pro_feats:
                log(f"  Skipping {sub}: no pro-bias features")
                continue

            # Filter items targeting this subgroup — include ALL behavioral
            # outcomes (stereotyped, non-stereotyped, unknown) so the sweep
            # can measure both correction AND corruption.
            sub_items = [
                it for it in items
                if sub in it.get("stereotyped_groups", [])
                and it.get("model_answer_role")  # must have behavioral annotation
            ]
            if args.max_items:
                sub_items = sub_items[:args.max_items]

            n_stereo = sum(1 for it in sub_items
                           if it.get("model_answer_role") == "stereotyped_target")
            if n_stereo < 3:
                log(f"  Skipping {sub}: only {n_stereo} stereotyped items "
                    f"(of {len(sub_items)} total)")
                continue

            log(f"  {sub}: {len(sub_items)} items "
                f"({n_stereo} stereotyped, {len(sub_items) - n_stereo} non-stereo/unknown)")

            # Phase 1: alpha pruning
            viable_alphas = _phase1_prune_alphas(
                wrapper, sae_cache, pro_feats, sub_items,
                alpha_values, format_prompt, cat, sub, cat_out,
            )

            # Phase 2: full k x alpha grid with pruned alphas
            result = run_stepwise_sweep(
                wrapper=wrapper,
                sae_cache=sae_cache,
                feature_list=pro_feats,
                items=sub_items,
                alpha_values=viable_alphas,
                k_steps=k_steps,
                prompt_formatter=format_prompt,
                subgroup=sub,
                category=cat,
                output_dir=cat_out,
            )
            cat_results[sub] = result

            # Save optimal steering vector and per-item results
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

                # Build manifest with full schema
                manifest = _build_full_manifest(
                    opt, float(vec.float().norm()),
                    pro_feats[:opt["k"]], viable_alphas,
                )
                all_manifests.append(manifest)

                # Save per-item parquet for optimal config
                per_item_records = result.get("per_item_records", [])
                if per_item_records:
                    try:
                        import pandas as pd_lib
                        per_item_dir = ensure_dir(output_dir / "per_item")
                        # Flatten logit dicts to JSON strings for parquet compat
                        for rec in per_item_records:
                            rec["logit_baseline"] = json.dumps(rec.get("logit_baseline", {}))
                            rec["logit_steered"] = json.dumps(rec.get("logit_steered", {}))
                        df = pd_lib.DataFrame(per_item_records)
                        df.to_parquet(
                            per_item_dir / f"{cat}_{sub}_optimal.parquet",
                            index=False,
                        )
                        log(f"  Saved per-item parquet: {cat}_{sub}_optimal.parquet")
                    except ImportError:
                        log("  WARNING: pandas not available, skipping per-item parquet")

                log(f"  Optimal for {sub}: k={opt['k']}, alpha={opt['alpha']}, "
                    f"layer={opt.get('injection_layer')}, "
                    f"eta={opt.get('eta', 0):.3f}, ||v||={opt.get('vector_norm', 0):.3f}")

                # Run exacerbation test
                if not args.skip_exacerbation:
                    log(f"  Running exacerbation test for {sub} ...")
                    # Use ALL items (not just stereotyped) for exacerbation
                    all_sub_items = [
                        it for it in items
                        if sub in it.get("stereotyped_groups", [])
                    ]
                    if args.max_items:
                        all_sub_items = all_sub_items[:args.max_items]
                    exac = _run_exacerbation(
                        wrapper, sae_cache, pro_feats, all_sub_items,
                        opt["k"], opt["alpha"], format_prompt, cat, sub,
                    )
                    cat_exac[sub] = exac
                    manifest["exacerbation"] = exac
                    log(f"    Exacerbation: corruption={exac['corruption_rate']:.3f}, "
                        f"logit_shift={exac['mean_logit_shift']:.3f}")

        all_results[cat] = cat_results
        exacerbation_data[cat] = cat_exac

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

    # Save exacerbation results
    if exacerbation_data:
        atomic_save_json(exacerbation_data, output_dir / "exacerbation_results.json")

    # Figures
    if not args.skip_figures:
        fig_dir = ensure_dir(output_dir / "figures")
        log("Generating figures ...")

        all_opt_list = [m for m in all_manifests if m.get("optimal_k")]
        fig_optimal_k_distribution(all_opt_list, fig_dir)

        for cat, cat_results in all_results.items():
            fig_stepwise_correction(cat_results, cat, fig_dir)
            fig_alpha_vs_k_heatmap(cat_results, cat, fig_dir)
            fig_pareto_frontier(cat_results, cat, fig_dir)
            fig_marginal_analysis(cat_results, cat, fig_dir)

        fig_margin_conditioned_correction(all_results, fig_dir)

        if exacerbation_data:
            fig_exacerbation_asymmetry(all_results, exacerbation_data, fig_dir)

    total = time.time() - t0
    log(f"\nComplete in {total:.1f}s")
    log(f"Results: {output_dir}")


# ---------------------------------------------------------------------------
# Manifest builder (full schema)
# ---------------------------------------------------------------------------


def _build_full_manifest(
    optimal: dict[str, Any],
    vec_norm: float,
    features_used: list[dict],
    viable_alphas: list[float],
) -> dict[str, Any]:
    """Build the per-subgroup steering manifest with full schema."""
    return {
        "subgroup": optimal.get("subgroup", ""),
        "category": optimal.get("category", ""),
        "optimal_k": optimal.get("k", 0),
        "optimal_alpha": optimal.get("alpha", 0),
        "injection_layer": optimal.get("injection_layer", 0),
        "steering_efficiency_eta": round(optimal.get("eta", 0), 4),
        "steering_vector_norm": round(vec_norm, 4),
        "features": [
            {
                "feature_idx": f["feature_idx"],
                "layer": f["layer"],
                "cohens_d": f.get("cohens_d", 0),
            }
            for f in features_used
        ],
        "metrics": optimal.get("metrics", {}),
        "margin_bins": optimal.get("margin_bins", {}),
        "exacerbation": None,
        "phase1_viable_alphas": viable_alphas,
        "medqa_matched_delta": None,
        "medqa_within_cat_mismatched_delta": None,
        "medqa_cross_cat_mismatched_delta": None,
        "medqa_nodemo_delta": None,
        "mmlu_delta": None,
    }


# ---------------------------------------------------------------------------
# New figures
# ---------------------------------------------------------------------------


def fig_pareto_frontier(
    results: dict[str, dict[str, Any]],
    category: str,
    output_dir: Path,
) -> None:
    """Pareto frontier: RCR_1.0 vs ||v||_2 for each subgroup."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.visualization.style import (
        CATEGORY_LABELS, DPI, WONG_PALETTE, apply_style,
    )
    apply_style()

    subs = sorted(results.keys())
    if not subs:
        return

    n = len(subs)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False)

    for idx, sub in enumerate(subs):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        data = results[sub]
        grid = data.get("grid", [])
        opt = data.get("optimal", {})

        if not grid:
            ax.set_visible(False)
            continue

        # Extract data
        norms = [r.get("vector_norm", 0) for r in grid]
        rcrs = [r.get("metrics", {}).get("rcr_1.0", {}).get("rcr", 0) for r in grid]
        ks = [r["k"] for r in grid]

        # Color by k
        unique_ks = sorted(set(ks))
        k_to_norm = {k: i / max(len(unique_ks) - 1, 1) for i, k in enumerate(unique_ks)}
        colors = [plt.cm.viridis(k_to_norm[k]) for k in ks]

        sc = ax.scatter(norms, rcrs, c=ks, cmap="viridis", s=20, alpha=0.7, zorder=2)

        # Connect constant-alpha lines
        alphas_in_grid = sorted(set(r["alpha"] for r in grid))
        for alpha in alphas_in_grid:
            pts = [(r.get("vector_norm", 0), r.get("metrics", {}).get("rcr_1.0", {}).get("rcr", 0))
                   for r in grid if r["alpha"] == alpha]
            pts.sort()
            if len(pts) > 1:
                ax.plot([p[0] for p in pts], [p[1] for p in pts],
                        "-", color="gray", linewidth=0.5, alpha=0.3, zorder=1)

        # Mark optimum
        if opt:
            opt_norm = opt.get("vector_norm", 0)
            opt_rcr = opt.get("metrics", {}).get("rcr_1.0", {}).get("rcr", 0)
            ax.scatter([opt_norm], [opt_rcr], marker="*", s=150, c="red",
                       edgecolors="black", linewidths=0.5, zorder=3)
            eta_val = opt.get("eta", 0)
            ax.annotate(f"eta*={eta_val:.2f}", (opt_norm, opt_rcr),
                        fontsize=6, xytext=(5, 5), textcoords="offset points")

        ax.set_xlabel("||v||_2", fontsize=8)
        ax.set_ylabel("RCR_1.0", fontsize=8)
        ax.set_title(sub, fontsize=9)

        # Panel label
        panel_label = chr(65 + idx)  # A, B, C, ...
        ax.text(0.02, 0.95, panel_label, transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="top")

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    cat_label = CATEGORY_LABELS.get(category, category)
    fig.suptitle(f"Pareto frontier — {cat_label}", fontsize=11)
    fig.tight_layout()

    path = output_dir / f"fig_pareto_frontier_{category}.png"
    fig.savefig(str(path), dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(str(path.with_suffix(".pdf")), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log(f"    Saved fig_pareto_frontier_{category}")


def fig_marginal_analysis(
    results: dict[str, dict[str, Any]],
    category: str,
    output_dir: Path,
) -> None:
    """Marginal analysis: RCR_1.0(k) and ||v||_2(k) at optimal alpha."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.visualization.style import (
        BLUE, CATEGORY_LABELS, DPI, apply_style,
    )
    ORANGE = "#E69F00"
    apply_style()

    subs = sorted(results.keys())
    if not subs:
        return

    n = len(subs)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False)

    for idx, sub in enumerate(subs):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        data = results[sub]
        grid = data.get("grid", [])
        opt = data.get("optimal", {})

        if not grid or not opt:
            ax.set_visible(False)
            continue

        opt_alpha = opt.get("alpha", 0)
        opt_k = opt.get("k", 0)

        # Filter grid to optimal alpha
        at_alpha = [r for r in grid if r["alpha"] == opt_alpha]
        at_alpha.sort(key=lambda r: r["k"])

        if not at_alpha:
            ax.set_visible(False)
            continue

        ks = [r["k"] for r in at_alpha]
        rcrs = [r.get("metrics", {}).get("rcr_1.0", {}).get("rcr", 0) for r in at_alpha]
        norms = [r.get("vector_norm", 0) for r in at_alpha]

        ax.plot(ks, rcrs, "o-", color=BLUE, label="RCR_1.0", markersize=5)
        ax.set_ylabel("RCR_1.0", fontsize=8, color=BLUE)

        ax2 = ax.twinx()
        ax2.plot(ks, norms, "s--", color=ORANGE, label="||v||_2", markersize=4)
        ax2.set_ylabel("||v||_2", fontsize=8, color=ORANGE)

        # Mark optimal k
        ax.axvline(x=opt_k, color="gray", linestyle=":", alpha=0.7)
        ax.annotate(f"k*={opt_k}", (opt_k, max(rcrs) * 0.9),
                    fontsize=7, ha="right")

        ax.set_xlabel("k (features)", fontsize=8)
        ax.set_title(f"{sub} (alpha={opt_alpha})", fontsize=9)

        # Panel label
        panel_label = chr(65 + idx)
        ax.text(0.02, 0.95, panel_label, transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="top")

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    cat_label = CATEGORY_LABELS.get(category, category)
    fig.suptitle(f"Marginal analysis at optimal alpha — {cat_label}", fontsize=11)
    fig.tight_layout()

    path = output_dir / f"fig_marginal_analysis_{category}.png"
    fig.savefig(str(path), dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(str(path.with_suffix(".pdf")), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log(f"    Saved fig_marginal_analysis_{category}")


def fig_exacerbation_asymmetry(
    all_results: dict[str, dict[str, dict]],
    exacerbation_data: dict[str, dict[str, dict]],
    output_dir: Path,
) -> None:
    """Exacerbation asymmetry: paired bars of debiasing RCR vs exacerbation corruption."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from src.visualization.style import (
        BLUE, CATEGORY_LABELS, DPI, apply_style,
    )
    VERMILLION = "#D55E00"
    apply_style()

    # Collect data across all categories
    labels: list[str] = []
    rcr_vals: list[float] = []
    exac_vals: list[float] = []
    cat_breaks: list[int] = []  # indices where categories change

    for cat in sorted(all_results.keys()):
        cat_label = CATEGORY_LABELS.get(cat, cat)
        cat_subs = sorted(all_results[cat].keys())
        if not cat_subs:
            continue
        cat_breaks.append(len(labels))
        for sub in cat_subs:
            opt = all_results[cat][sub].get("optimal", {})
            rcr = opt.get("metrics", {}).get("rcr_1.0", {}).get("rcr", 0)
            exac = exacerbation_data.get(cat, {}).get(sub, {})
            corr = exac.get("corruption_rate", 0)
            labels.append(f"{sub}\n({cat_label})")
            rcr_vals.append(rcr)
            exac_vals.append(corr)

    if not labels:
        return

    n = len(labels)
    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, n * 1.2), 5))

    bars1 = ax.bar(x - width / 2, rcr_vals, width, color=BLUE, label="Debiasing RCR_1.0", alpha=0.8)
    bars2 = ax.bar(x + width / 2, exac_vals, width, color=VERMILLION, label="Exacerbation corruption", alpha=0.8)

    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.2f}", ha="center", va="bottom", fontsize=6)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.2f}", ha="center", va="bottom", fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Rate")
    ax.set_title("Exacerbation asymmetry across subgroups")
    ax.legend(fontsize=8)

    fig.tight_layout()
    path = output_dir / "fig_exacerbation_asymmetry.png"
    fig.savefig(str(path), dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(str(path.with_suffix(".pdf")), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log("    Saved fig_exacerbation_asymmetry")


if __name__ == "__main__":
    main()
