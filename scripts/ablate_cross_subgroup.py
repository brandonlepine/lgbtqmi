#!/usr/bin/env python3
"""Cross-subgroup ablation: ablate one subgroup's direction, measure effect on all.

For each category, creates a (subgroups x subgroups) effect matrix.

Usage:
    python scripts/ablate_cross_subgroup.py \
        --run_dir results/runs/llama2-13b-hf/2026-04-11/ \
        --model_path /workspace/lgbtqmi/models/llama2-13b \
        --device cuda \
        --categories so,race,religion,physical_appearance,gi,age
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.analysis.bias_scores import compute_bias_score
from src.data.bbq_loader import CATEGORY_MAP, parse_categories
from src.extraction.activations import format_prompt
from src.interventions.direction_ablation import apply_direction_ablation, remove_hooks
from src.utils.answers import best_choice_from_logits
from src.utils.io import atomic_save_json, ensure_dir
from src.utils.logging import log

CATEGORY_LABELS = {k: v for k, v in {
    "so": "Sexual Orientation", "gi": "Gender Identity",
    "race": "Race/Ethnicity", "religion": "Religion",
    "disability": "Disability", "physical_appearance": "Physical Appearance",
    "age": "Age",
}.items()}


def _load_model(model_path: str, device: str):
    try:
        from src.models.wrapper import ModelWrapper
        w = ModelWrapper.from_pretrained(model_path, device=device)
        return w.model, w.tokenizer, w.n_layers, w.hidden_dim, w.get_layer
    except (ImportError, AttributeError):
        pass
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if not getattr(tok, "is_fast", False):
        raise RuntimeError(
            "Tokenizer is not fast; offset mapping is required by the broader pipeline. "
            "Use a model with a fast tokenizer or ensure the fast tokenizer is available."
        )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float32 if device == "cpu" else torch.float16
    m = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype).to(device)
    m.eval()
    cfg = m.config
    n_l, h_d = cfg.num_hidden_layers, cfg.hidden_size
    inner = None
    for ap in ["model.layers", "transformer.h", "gpt_neox.layers"]:
        obj = m
        for a in ap.split("."):
            obj = getattr(obj, a, None)
            if obj is None: break
        if obj is not None:
            inner = obj; break
    if inner is None: raise RuntimeError("Cannot find decoder layers")
    return m, tok, n_l, h_d, lambda idx: inner[idx]


def _run_eval(model, tokenizer, items, device, condition="ambig"):
    preds = []
    for item in items:
        inp = tokenizer(format_prompt(item), return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inp)
        best, _ = best_choice_from_logits(out.logits[0, -1, :], tokenizer)
        preds.append(best or "")
    return compute_bias_score(items, preds, condition), preds


def _per_subgroup_bias(items, preds, condition="ambig"):
    """Compute bias per stereotyped subgroup."""
    sg_items: dict[str, tuple[list, list]] = {}
    for item, pred in zip(items, preds):
        if item.get("context_condition") != condition:
            continue
        groups = item.get("stereotyped_groups", [])
        sg = groups[0].lower() if groups else "unknown"
        sg_items.setdefault(sg, ([], []))
        sg_items[sg][0].append(item)
        sg_items[sg][1].append(pred)

    result: dict[str, float | None] = {}
    for sg, (sg_it, sg_pr) in sg_items.items():
        scores = compute_bias_score(sg_it, sg_pr, condition)
        result[sg] = scores["bias_score"] if scores["n_non_unknown"] > 0 else None
    return result


def _load_stimuli(run_dir, cat, max_items):
    files = sorted((run_dir / "stimuli").glob(f"stimuli_{cat}_*.json"))
    if not files:
        files = sorted(Path("data/processed").glob(f"stimuli_{cat}_*.json"))
    if not files:
        return None
    with open(files[-1]) as f:
        items = json.load(f)
    return items[:max_items] if max_items else items


def _load_subgroup_dirs(run_dir, cat):
    path = run_dir / "analysis" / "subgroup_directions.npz"
    if not path.exists():
        return {}
    data = np.load(path, allow_pickle=True)
    prefix = f"subgroup_{cat}_"
    return {
        k[len(prefix):]: data[k]
        for k in data.files if k.startswith(prefix)
    }


def _validate_hook(model, tokenizer, get_layer_fn, direction, layer, device, item, hidden_dim):
    inp = tokenizer(format_prompt(item), return_tensors="pt").to(device)
    with torch.no_grad():
        bl = model(**inp).logits[0, -1, :].clone()
    hooks = apply_direction_ablation(
        model,
        get_layer_fn,
        direction,
        [layer],
        device,
        hidden_dim=hidden_dim,
        alpha=1.0,
    )
    with torch.no_grad():
        hk = model(**inp).logits[0, -1, :]
    remove_hooks(hooks)
    return float((bl - hk).abs().max().item())


def main():
    parser = argparse.ArgumentParser(description="Cross-subgroup ablation.")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--target_layer", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=14.0)
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--categories", type=str, default="so,race,religion,physical_appearance,gi,age")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    analysis_dir = ensure_dir(run_dir / "analysis")
    model_id = args.model_id or run_dir.parent.name
    categories = parse_categories(args.categories)

    log(f"{'='*60}")
    log("CROSS-SUBGROUP ABLATION")
    log(f"{'='*60}")
    log(f"Model: {args.model_path}")
    log(f"Alpha: {args.alpha}")

    # Load model
    model, tokenizer, n_layers, hidden_dim, get_layer_fn = _load_model(args.model_path, args.device)
    target_layer = args.target_layer if args.target_layer is not None else n_layers // 2
    log(f"Target layer: {target_layer}")

    # Resume
    output_path = analysis_dir / "subgroup_ablation_results.json"
    if output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)
    else:
        existing = {"metadata": {"model_id": model_id, "target_layer": target_layer,
                                  "alpha": args.alpha}, "categories": {}}

    hook_validated = False

    for cat in categories:
        log(f"\n{'='*60}")
        log(f"Category: {cat}")
        log(f"{'='*60}")

        stimuli = _load_stimuli(run_dir, cat, args.max_items)
        if not stimuli:
            log("  No stimuli, skipping")
            continue

        sg_dirs = _load_subgroup_dirs(run_dir, cat)
        if len(sg_dirs) < 2:
            log(f"  <2 subgroup directions, skipping")
            continue

        sg_names = sorted(sg_dirs.keys())
        log(f"  Subgroups: {sg_names}")
        log(f"  Items: {len(stimuli)}")

        # Skip if already done
        if cat in existing["categories"]:
            done_conds = set(existing["categories"][cat].get("conditions", {}).keys())
            needed = {"baseline"} | {f"ablate_{sg}" for sg in sg_names}
            if needed <= done_conds:
                log("  Already complete, skipping")
                continue

        # Validate hook once
        if not hook_validated:
            test_dir = next(iter(sg_dirs.values()))[target_layer]
            diff = _validate_hook(model, tokenizer, get_layer_fn, test_dir,
                                  target_layer, args.device, stimuli[0], hidden_dim)
            log(f"  Hook validation: diff={diff:.4f}")
            if diff < 1e-6:
                raise RuntimeError("Hook had no effect!")
            hook_validated = True

        cat_results: dict[str, dict] = existing["categories"].get(cat, {})
        cat_conditions = cat_results.setdefault("conditions", {})
        cat_results["subgroups"] = sg_names
        cat_results["n_items"] = len(stimuli)

        # Baseline
        if "baseline" not in cat_conditions:
            log("  Running baseline...")
            scores, preds = _run_eval(model, tokenizer, stimuli, args.device)
            sg_bias = _per_subgroup_bias(stimuli, preds)
            cat_conditions["baseline"] = {
                "overall_bias": scores["bias_score"],
                "subgroup_bias": sg_bias,
            }
            log(f"  Baseline: overall={scores['bias_score']:.3f}")
            for sg, b in sorted(sg_bias.items()):
                log(f"    {sg}: {b}")

        # Ablate each subgroup direction
        for sg_name in sg_names:
            cond_key = f"ablate_{sg_name}"
            if cond_key in cat_conditions:
                log(f"  {cond_key}: already done, skipping")
                continue

            log(f"  Ablating: {sg_name}")
            t0 = time.time()
            d = sg_dirs[sg_name][target_layer]
            d = d / max(np.linalg.norm(d), 1e-10)
            hooks = apply_direction_ablation(
                model, get_layer_fn, d, [target_layer], args.device,
                hidden_dim=hidden_dim, alpha=args.alpha,
            )
            scores, preds = _run_eval(model, tokenizer, stimuli, args.device)
            sg_bias = _per_subgroup_bias(stimuli, preds)
            remove_hooks(hooks)

            cat_conditions[cond_key] = {
                "overall_bias": scores["bias_score"],
                "subgroup_bias": sg_bias,
            }
            log(f"    overall={scores['bias_score']:.3f} ({time.time()-t0:.1f}s)")
            for sg, b in sorted(sg_bias.items()):
                log(f"    {sg}: {b}")

            # Save incrementally
            existing["categories"][cat] = cat_results
            atomic_save_json(existing, output_path)

        # Also run family-level ablation if fragmentation data exists
        frag_path = analysis_dir / "subgroup_fragmentation.json"
        if frag_path.exists():
            with open(frag_path) as f:
                frag_data = json.load(f)
            families = frag_data.get("categories", {}).get(cat, {}).get("families", {})
            for fam_id, fam_members in families.items():
                cond_key = f"ablate_family_{fam_id}"
                if cond_key in cat_conditions:
                    continue
                # Average family direction
                fam_dirs = [sg_dirs[m] for m in fam_members if m in sg_dirs]
                if not fam_dirs:
                    continue
                fam_avg = np.stack(fam_dirs, axis=0).mean(axis=0)
                fam_d = fam_avg[target_layer]
                fam_d = fam_d / max(np.linalg.norm(fam_d), 1e-10)

                log(f"  Ablating family {fam_id} ({fam_members})...")
                hooks = apply_direction_ablation(
                    model, get_layer_fn, fam_d, [target_layer], args.device,
                    hidden_dim=hidden_dim, alpha=args.alpha,
                )
                scores, preds = _run_eval(model, tokenizer, stimuli, args.device)
                sg_bias = _per_subgroup_bias(stimuli, preds)
                remove_hooks(hooks)

                cat_conditions[cond_key] = {
                    "overall_bias": scores["bias_score"],
                    "subgroup_bias": sg_bias,
                    "family_members": fam_members,
                }
                existing["categories"][cat] = cat_results
                atomic_save_json(existing, output_path)
                log(f"    overall={scores['bias_score']:.3f}")

        if args.device == "mps":
            torch.mps.empty_cache()

    log(f"\nAll done -> {output_path}")


if __name__ == "__main__":
    main()
