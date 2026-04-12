#!/usr/bin/env python3
"""Causal intervention using pairwise shared directions.

For each high-cosine pair, ablates the shared direction and specific residuals,
then measures BBQ bias across all categories.

Usage:
    python scripts/ablate_pairwise_shared.py \
        --run_dir results/runs/llama2-13b-hf/2026-04-11/ \
        --model_path /workspace/lgbtqmi/models/llama2-13b \
        --model_id llama2-13b-hf \
        --device cuda \
        --target_layer 20 \
        --alpha 14.0 \
        --max_pairs 6
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
from src.data.bbq_loader import parse_categories
from src.extraction.activations import format_prompt
from src.interventions.direction_ablation import apply_direction_ablation, remove_hooks
from src.utils.answers import best_choice_from_logits
from src.utils.io import atomic_save_json, ensure_dir
from src.utils.logging import log

CATEGORY_LABELS: dict[str, str] = {
    "so": "Sexual Orientation", "gi": "Gender Identity",
    "race": "Race/Ethnicity", "religion": "Religion",
    "disability": "Disability", "physical_appearance": "Physical Appearance",
    "age": "Age",
}


def _lbl(cat: str) -> str:
    return CATEGORY_LABELS.get(cat, cat)


# ===== Model loading (same as ablate_meso_clusters) ========================

def _load_model(model_path: str, device: str):
    try:
        from src.models.wrapper import ModelWrapper
        wrapper = ModelWrapper.from_pretrained(model_path, device=device)
        return (wrapper.model, wrapper.tokenizer, wrapper.n_layers,
                wrapper.hidden_dim, wrapper.get_layer)
    except (ImportError, AttributeError):
        pass

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(
            "Tokenizer is not fast; offset mapping is required by the broader pipeline. "
            "Use a model with a fast tokenizer or ensure the fast tokenizer is available."
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float32 if device == "cpu" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype).to(device)
    model.eval()
    config = model.config
    n_layers = config.num_hidden_layers
    hidden_dim = config.hidden_size

    inner = None
    for attr_path in ["model.layers", "transformer.h", "gpt_neox.layers"]:
        obj = model
        for attr in attr_path.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            inner = obj
            break
    if inner is None:
        raise RuntimeError("Cannot find decoder layers")

    def get_layer_fn(idx):
        return inner[idx]

    return model, tokenizer, n_layers, hidden_dim, get_layer_fn


def _run_eval(model, tokenizer, items: list[dict], device: str,
              condition: str = "ambig") -> tuple[dict, list[str]]:
    predictions: list[str] = []
    for item in items:
        prompt = format_prompt(item)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
        best, _ = best_choice_from_logits(logits, tokenizer)
        predictions.append(best or "")
    return compute_bias_score(items, predictions, condition), predictions


def _eval_all_cats(model, tokenizer, cat_stimuli: dict[str, list[dict]],
                   device: str) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for cat in sorted(cat_stimuli.keys()):
        scores, _ = _run_eval(model, tokenizer, cat_stimuli[cat], device)
        results[_lbl(cat)] = {
            "ambig_bias": scores["bias_score"],
            "n_non_unknown": scores["n_non_unknown"],
            "n_total": scores["n_total"],
        }
    return results


def _validate_hook(model, tokenizer, get_layer_fn, direction, layer,
                   device, test_item, hidden_dim) -> float:
    prompt = format_prompt(test_item)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        bl = model(**inputs).logits[0, -1, :].clone()
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
        hk = model(**inputs).logits[0, -1, :]
    remove_hooks(hooks)
    return float((bl - hk).abs().max().item())


def _load_stimuli(run_dir: Path, categories: list[str],
                  max_items: int | None) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for cat in categories:
        files = sorted((run_dir / "stimuli").glob(f"stimuli_{cat}_*.json"))
        if not files:
            files = sorted(Path("data/processed").glob(f"stimuli_{cat}_*.json"))
        if files:
            with open(files[-1]) as f:
                items = json.load(f)
            if max_items:
                items = items[:max_items]
            out[cat] = items
    return out


def _load_pairwise_data(run_dir: Path):
    npz_path = run_dir / "analysis" / "pairwise_decomposition.npz"
    json_path = run_dir / "analysis" / "pairwise_decomposition.json"
    if not npz_path.exists() or not json_path.exists():
        raise FileNotFoundError("Run extract_pairwise_shared.py first.")
    arrays = np.load(npz_path, allow_pickle=True)
    with open(json_path) as f:
        summary = json.load(f)
    return arrays, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Pairwise shared-direction ablation.")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--target_layer", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=14.0)
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--max_pairs", type=int, default=None,
                        help="Limit number of pairs to ablate (highest |cosine| first)")
    parser.add_argument("--categories", type=str,
                        default="so,gi,race,religion,disability,physical_appearance,age")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    analysis_dir = ensure_dir(run_dir / "analysis")
    model_id = args.model_id or run_dir.parent.name
    categories = parse_categories(args.categories)

    log(f"{'='*60}")
    log("PAIRWISE SHARED-DIRECTION ABLATION")
    log(f"{'='*60}")
    log(f"Model: {args.model_path}")
    log(f"Alpha: {args.alpha}")

    # Load pairwise data
    arrays, pw_summary = _load_pairwise_data(run_dir)
    pairs_info = pw_summary["pairs"]
    triangles_info = pw_summary.get("triangles", {})

    # Sort pairs by |cosine|
    sorted_pairs = sorted(pairs_info.keys(),
                          key=lambda k: abs(pairs_info[k]["cosine_mid"]), reverse=True)
    if args.max_pairs:
        sorted_pairs = sorted_pairs[:args.max_pairs]
    log(f"Pairs to ablate: {len(sorted_pairs)}")

    # Load stimuli
    log("\nLoading stimuli...")
    cat_stimuli = _load_stimuli(run_dir, categories, args.max_items)
    if not cat_stimuli:
        raise RuntimeError("No stimuli found.")

    # Load model
    log("\nLoading model...")
    model, tokenizer, n_layers, hidden_dim, get_layer_fn = _load_model(
        args.model_path, args.device
    )
    target_layer = args.target_layer if args.target_layer is not None else n_layers // 2
    log(f"Target layer: {target_layer}, hidden_dim: {hidden_dim}")

    # Resume
    output_path = analysis_dir / "pairwise_ablation_results.json"
    if output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)
        done = set(existing.get("conditions", {}).keys())
        log(f"Resuming: {len(done)} conditions already done")
    else:
        existing = {
            "metadata": {
                "model_id": model_id,
                "target_layer": target_layer,
                "alpha": args.alpha,
                "n_layers": n_layers,
                "hidden_dim": hidden_dim,
                "categories": [str(c) for c in categories],
            },
            "conditions": {},
        }
        done = set()

    # Build condition list (priority ordered)
    conditions: list[tuple[str, np.ndarray | None]] = []
    conditions.append(("baseline", None))

    # Phase 1: shared ablation for top pairs
    for pk in sorted_pairs:
        key = f"pair_{pk}_shared"
        arr_key = f"pair_{pk}_shared"
        if arr_key in arrays.files:
            conditions.append((f"ablate_shared_{pk}", arrays[arr_key][target_layer]))

    # Phase 2: triangle 3-way (SO_GI_Religion)
    tri_key = "so_gi_religion"
    tri_3way_key = f"triangle_{tri_key}_shared_3way"
    if tri_3way_key in arrays.files:
        conditions.append((f"ablate_3way_{tri_key}", arrays[tri_3way_key][target_layer]))
        for sub in ["so_gi_only", "gi_religion_only", "so_religion_only"]:
            arr_k = f"triangle_{tri_key}_{sub}"
            if arr_k in arrays.files:
                conditions.append((f"ablate_{sub}_{tri_key}", arrays[arr_k][target_layer]))

    # Phase 3: specific residuals for top 4 pairs
    for pk in sorted_pairs[:4]:
        for suffix in ["a_specific", "b_specific"]:
            arr_key = f"pair_{pk}_{suffix}"
            if arr_key in arrays.files:
                cat_name = pairs_info[pk]["cat_a"] if suffix == "a_specific" else pairs_info[pk]["cat_b"]
                conditions.append((f"ablate_{suffix}_{pk}", arrays[arr_key][target_layer]))

    log(f"\nTotal conditions: {len(conditions)}, already done: {len(done)}")

    # Hook validation
    test_cat = next(iter(cat_stimuli.keys()))
    test_item = cat_stimuli[test_cat][0]
    first_dir = next(
        (d for _, d in conditions if d is not None), None
    )
    if first_dir is not None:
        diff = _validate_hook(
            model,
            tokenizer,
            get_layer_fn,
            first_dir,
            target_layer,
            args.device,
            test_item,
            hidden_dim,
        )
        log(f"Hook validation: max logit diff = {diff:.4f}")
        if diff < 1e-6:
            raise RuntimeError("Hook had no effect!")
        log("Hook validation passed.")

    # Run conditions
    for ci, (cond_name, direction) in enumerate(conditions):
        if cond_name in done:
            log(f"[{ci+1}/{len(conditions)}] {cond_name}: SKIPPED")
            continue

        log(f"\n[{ci+1}/{len(conditions)}] {cond_name}")
        t0 = time.time()

        hooks = []
        if direction is not None:
            d = direction / max(np.linalg.norm(direction), 1e-10)
            hooks = apply_direction_ablation(
                model, get_layer_fn, d, [target_layer], args.device,
                hidden_dim=hidden_dim, alpha=args.alpha,
            )

        results = _eval_all_cats(model, tokenizer, cat_stimuli, args.device)
        remove_hooks(hooks)

        elapsed = time.time() - t0
        for cl, r in sorted(results.items()):
            log(f"  {cl:>22s}: bias={r['ambig_bias']:.3f}")
        log(f"  ({elapsed:.1f}s)")

        existing["conditions"][cond_name] = results
        atomic_save_json(existing, output_path)

        if args.device == "mps":
            torch.mps.empty_cache()

    log(f"\nAll conditions complete -> {output_path}")


if __name__ == "__main__":
    main()
