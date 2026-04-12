#!/usr/bin/env python3
"""Causal intervention using meso-level cluster directions.

Reads meso_directions.npz from compute_meso_directions.py and runs new ablation
conditions that target cluster-level directions. Does NOT re-run any conditions
from the existing causal_ablation_hierarchy.py.

Usage:
    python scripts/ablate_meso_clusters.py \
        --run_dir results/runs/llama2-13b-hf/2026-04-11/ \
        --model_path /workspace/lgbtqmi/models/llama2-13b \
        --model_id llama2-13b-hf \
        --device cuda \
        --target_layer 20 \
        --alpha 14.0
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.analysis.bias_scores import compute_bias_score, bias_score_by_subgroup
from src.data.bbq_loader import parse_categories
from src.extraction.activations import format_prompt
from src.interventions.direction_ablation import (
    apply_direction_ablation,
    remove_hooks,
)
from src.utils.answers import best_choice_from_logits
from src.utils.io import atomic_save_json, ensure_dir
from src.utils.logging import ProgressLogger, log

# Cluster definitions (must match compute_meso_directions.py)
CLUSTERS: dict[str, list[str]] = {
    "lgbtq": ["so", "gi"],
    "social_group": ["race", "religion"],
    "bodily_physical": ["physical_appearance", "disability", "age"],
}

CATEGORY_LABELS: dict[str, str] = {
    "so": "Sexual Orientation",
    "gi": "Gender Identity",
    "race": "Race/Ethnicity",
    "religion": "Religion",
    "disability": "Disability",
    "physical_appearance": "Physical Appearance",
    "age": "Age",
}


# ===== Model loading (same pattern as existing causal_ablation_hierarchy) ====

def _load_model(model_path: str, device: str):
    """Load model. Returns (model, tokenizer, n_layers, hidden_dim, get_layer_fn)."""
    try:
        from src.models.wrapper import ModelWrapper
        wrapper = ModelWrapper.from_pretrained(model_path, device=device)
        log(f"Loaded via ModelWrapper: {type(wrapper.model).__name__}")
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
    log(f"Loaded: {type(model).__name__}, {n_layers} layers, dim={hidden_dim}")

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


def _run_behavioral_eval(
    model, tokenizer, items: list[dict], device: str,
    condition: str = "ambig",
) -> tuple[dict, list[str]]:
    """Run model on stimuli and compute bias scores. Returns (scores, predictions)."""
    predictions: list[str] = []
    for item in items:
        prompt = format_prompt(item)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
        best_letter, _ = best_choice_from_logits(logits, tokenizer)
        predictions.append(best_letter or "")

    scores = compute_bias_score(items, predictions, condition)
    return scores, predictions


def _validate_hook(
    model, tokenizer, get_layer_fn, direction: np.ndarray,
    layer_idx: int, device: str, test_item: dict, hidden_dim: int,
) -> float:
    """Validate that ablation hook actually changes model output. Returns max logit diff."""
    prompt = format_prompt(test_item)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Baseline
    with torch.no_grad():
        baseline_out = model(**inputs)
    baseline_logits = baseline_out.logits[0, -1, :].clone()

    # With hook
    hooks = apply_direction_ablation(
        model,
        get_layer_fn,
        direction,
        [layer_idx],
        device,
        hidden_dim=hidden_dim,
    )
    with torch.no_grad():
        hooked_out = model(**inputs)
    hooked_logits = hooked_out.logits[0, -1, :]
    remove_hooks(hooks)

    diff = (baseline_logits - hooked_logits).abs().max().item()
    return diff


def _load_stimuli(run_dir: Path, categories: list[str], max_items: int | None = None) -> dict[str, list[dict]]:
    """Load stimuli per category."""
    cat_stimuli: dict[str, list[dict]] = {}
    for cat in categories:
        stimuli_files = sorted((run_dir / "stimuli").glob(f"stimuli_{cat}_*.json"))
        if not stimuli_files:
            stimuli_files = sorted(Path("data/processed").glob(f"stimuli_{cat}_*.json"))
        if stimuli_files:
            with open(stimuli_files[-1]) as f:
                items = json.load(f)
            if max_items:
                items = items[:max_items]
            cat_stimuli[cat] = items
            log(f"  {cat}: {len(items)} items")
        else:
            log(f"  {cat}: no stimuli found, skipping")
    return cat_stimuli


def _load_meso_directions(run_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load cluster directions and within-cluster residuals."""
    path = run_dir / "analysis" / "meso_directions.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/compute_meso_directions.py first."
        )
    data = np.load(path, allow_pickle=True)

    cluster_dirs: dict[str, np.ndarray] = {}
    within_residuals: dict[str, np.ndarray] = {}
    for key in data.files:
        if key.startswith("_"):
            continue
        if key.endswith("_direction"):
            name = key[:-len("_direction")]
            cluster_dirs[name] = data[key]
        elif key.startswith("within_cluster_"):
            cat = key[len("within_cluster_"):]
            within_residuals[cat] = data[key]

    return cluster_dirs, within_residuals


def _eval_all_categories(
    model, tokenizer, cat_stimuli: dict[str, list[dict]], device: str,
) -> dict[str, dict]:
    """Evaluate all categories and return per-category results."""
    results: dict[str, dict] = {}
    for cat in sorted(cat_stimuli.keys()):
        items = cat_stimuli[cat]
        scores, preds = _run_behavioral_eval(model, tokenizer, items, device)
        # Per-subgroup bias
        group_bias = bias_score_by_subgroup(items, preds, "ambig")
        group_bias_flat = {k: v["bias_score"] for k, v in group_bias.items()}

        # Disambig accuracy sanity check
        disambig_scores, _ = _run_behavioral_eval(model, tokenizer, items, device, "disambig")

        results[CATEGORY_LABELS.get(cat, cat)] = {
            "ambig_bias": scores["bias_score"],
            "n_non_unknown": scores["n_non_unknown"],
            "n_total": scores["n_total"],
            "disambig_n_total": disambig_scores["n_total"],
            "group_bias": group_bias_flat,
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Meso-level cluster ablation experiment.")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--target_layer", type=int, default=None,
                        help="Layer for ablation (default: n_layers // 2)")
    parser.add_argument("--alpha", type=float, default=14.0,
                        help="Ablation strength (direction is projected out, alpha scales)")
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--categories", type=str, default="so,gi,race,religion,disability,physical_appearance,age")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    analysis_dir = ensure_dir(run_dir / "analysis")
    model_id = args.model_id or run_dir.parent.name
    categories = parse_categories(args.categories)

    log(f"{'='*60}")
    log(f"MESO-LEVEL CLUSTER ABLATION")
    log(f"{'='*60}")
    log(f"Model: {args.model_path} ({model_id})")
    log(f"Device: {args.device}")
    log(f"Alpha: {args.alpha}")

    # Load meso directions
    cluster_dirs, within_residuals = _load_meso_directions(run_dir)
    log(f"Cluster directions: {sorted(cluster_dirs.keys())}")
    log(f"Within-cluster residuals: {sorted(within_residuals.keys())}")

    # Load stimuli
    log("\nLoading stimuli...")
    cat_stimuli = _load_stimuli(run_dir, categories, args.max_items)
    if not cat_stimuli:
        raise RuntimeError(
            "No stimuli found for any requested category. "
            "Expected files under run_dir/stimuli like stimuli_so_YYYY-MM-DD*.json. "
            "Make sure you ran scripts/run_extraction_pipeline.py for this run_dir first, "
            "and that --categories is a valid list (or 'all')."
        )

    # Load model
    log("\nLoading model...")
    model, tokenizer, n_layers, hidden_dim, get_layer_fn = _load_model(
        args.model_path, args.device
    )
    target_layer = args.target_layer if args.target_layer is not None else n_layers // 2
    log(f"Target layer: {target_layer} (of {n_layers})")

    # Resume: load existing results
    output_path = analysis_dir / "meso_ablation_results.json"
    if output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)
        completed_conditions = set(existing.get("conditions", {}).keys())
        log(f"Resuming: {len(completed_conditions)} conditions already done")
    else:
        existing = {
            "metadata": {
                "model_id": model_id,
                "model_path": args.model_path,
                "target_layer": target_layer,
                "alpha": args.alpha,
                "n_layers": n_layers,
                "hidden_dim": hidden_dim,
                "categories": categories,
                "n_items_per_category": {cat: len(items) for cat, items in cat_stimuli.items()},
            },
            "conditions": {},
        }
        completed_conditions = set()

    # Define all conditions
    conditions: list[tuple[str, np.ndarray | None, float]] = []
    conditions.append(("baseline", None, 0.0))

    # Cluster ablations (positive and negative alpha)
    for cname in sorted(cluster_dirs.keys()):
        conditions.append((f"ablate_{cname}", cluster_dirs[cname][target_layer], args.alpha))
        conditions.append((f"ablate_{cname}_neg", cluster_dirs[cname][target_layer], -args.alpha))

    # Within-cluster residual ablations
    for cat in sorted(within_residuals.keys()):
        cluster_for_cat = None
        for cname, members in CLUSTERS.items():
            if cat in members:
                cluster_for_cat = cname
                break
        label = f"ablate_{cat}_within_{cluster_for_cat}" if cluster_for_cat else f"ablate_{cat}_within"
        conditions.append((label, within_residuals[cat][target_layer], args.alpha))

    log(f"\nTotal conditions: {len(conditions)}")
    log(f"Already completed: {len(completed_conditions)}")

    # Validate hook on first available test item
    test_cat = next(iter(cat_stimuli.keys()))
    if not cat_stimuli[test_cat]:
        raise RuntimeError(f"No items loaded for category '{test_cat}'.")
    test_item = cat_stimuli[test_cat][0]
    test_dir = next(iter(cluster_dirs.values()))[target_layer]
    logit_diff = _validate_hook(model, tokenizer, get_layer_fn, test_dir,
                                target_layer, args.device, test_item, hidden_dim)
    log(f"\nHook validation: max logit diff = {logit_diff:.4f}")
    if logit_diff < 1e-6:
        raise RuntimeError("Hook had no effect! Check architecture compatibility.")
    log("Hook validation passed.")

    # Run conditions
    for cond_idx, (cond_name, direction, alpha) in enumerate(conditions):
        if cond_name in completed_conditions:
            log(f"\n[{cond_idx+1}/{len(conditions)}] {cond_name}: SKIPPED (already done)")
            continue

        log(f"\n{'='*60}")
        log(f"[{cond_idx+1}/{len(conditions)}] Condition: {cond_name}")
        log(f"{'='*60}")
        t0 = time.time()

        # Register hook (or not for baseline)
        hooks = []
        if direction is not None:
            d = direction / max(np.linalg.norm(direction), 1e-8)
            hooks = apply_direction_ablation(
                model,
                get_layer_fn,
                d,
                [target_layer],
                args.device,
                hidden_dim=hidden_dim,
                alpha=float(alpha),
            )

        # Evaluate all categories
        results = _eval_all_categories(model, tokenizer, cat_stimuli, args.device)

        # Clean up hooks
        remove_hooks(hooks)

        elapsed = time.time() - t0
        log(f"  Completed in {elapsed:.1f}s")

        for cat_label, r in sorted(results.items()):
            log(f"    {cat_label:>22s}: bias={r['ambig_bias']:.3f} "
                f"(n_non_unk={r['n_non_unknown']}/{r['n_total']})")

        # Save incrementally
        existing["conditions"][cond_name] = results
        atomic_save_json(existing, output_path)
        log(f"  Saved -> {output_path}")

        # Memory management
        if args.device == "mps":
            torch.mps.empty_cache()

    # Final summary
    log(f"\n{'='*60}")
    log("ALL CONDITIONS COMPLETE")
    log(f"{'='*60}")
    log(f"Results -> {output_path}")

    n_conditions = len(existing["conditions"])
    log(f"Total conditions saved: {n_conditions}")


if __name__ == "__main__":
    main()
