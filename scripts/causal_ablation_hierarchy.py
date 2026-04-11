#!/usr/bin/env python3
"""Causal interventions: shared/specific direction ablation and RLHF-target head ablation.

Generates figures 17-19: ablation grouped bars, RLHF replication, cross-category effects.

Usage:
    python scripts/causal_ablation_hierarchy.py \
        --run_dir results/runs/llama2-13b/2026-04-10 \
        --model_path models/llama2-13b \
        --device cuda

    # With base vs chat comparison
    python scripts/causal_ablation_hierarchy.py \
        --base_run_dir results/runs/llama2-13b/2026-04-10 \
        --chat_run_dir results/runs/llama2-13b-chat/2026-04-10 \
        --model_path models/llama2-13b \
        --device cuda
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.analysis.bias_scores import compute_bias_score, predictions_from_metadata
from src.analysis.directions import load_activations
from src.analysis.geometry import shared_component_analysis
from src.data.bbq_loader import CATEGORY_MAP, parse_categories
from src.extraction.activations import format_prompt
from src.interventions.direction_ablation import (
    apply_direction_ablation,
    apply_multi_direction_ablation,
    remove_hooks,
)
from src.interventions.head_ablation import (
    apply_head_ablation,
    identify_rlhf_targets,
    remove_hooks as remove_head_hooks,
)
from src.utils.answers import best_choice_from_logits
from src.utils.io import atomic_save_json, ensure_dir
from src.utils.logging import ProgressLogger, log
from src.visualization.heatmaps import plot_cosine_heatmap
from src.visualization.summary import plot_ablation_grouped_bars


def _load_model(model_path: str, device: str):
    """Load model — same pattern as extraction pipeline."""
    try:
        from src.models.wrapper import ModelWrapper
        wrapper = ModelWrapper.from_pretrained(model_path, device=device)
        return (wrapper.model, wrapper.tokenizer, wrapper.n_layers,
                wrapper.hidden_dim, wrapper.get_layer,
                wrapper.get_o_proj,
                getattr(wrapper, "n_heads", None),
                getattr(wrapper, "head_dim", None))
    except (ImportError, AttributeError):
        pass

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float32 if device == "cpu" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype).to(device)
    model.eval()

    config = model.config
    n_layers = config.num_hidden_layers
    hidden_dim = config.hidden_size
    n_heads = config.num_attention_heads
    head_dim = hidden_dim // n_heads

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

    def get_o_proj_fn(idx: int):
        layer = inner[idx]
        attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None) or getattr(layer, "attention", None)
        if attn is None:
            raise RuntimeError(f"Cannot locate attention module on layer {idx}; implement ModelWrapper")
        o_proj = getattr(attn, "o_proj", None) or getattr(attn, "out_proj", None) or getattr(attn, "c_proj", None)
        if o_proj is None:
            raise RuntimeError(f"Cannot locate o_proj on layer {idx}; implement ModelWrapper")
        return o_proj

    return model, tokenizer, n_layers, hidden_dim, get_layer_fn, get_o_proj_fn, n_heads, head_dim


def run_behavioral_eval(
    model, tokenizer, stimuli_items: list[dict], device: str,
    condition: str = "ambig",
) -> tuple[dict, list[str]]:
    """Run model on stimuli and compute bias scores.

    Returns:
        (score_dict, predictions)
    """
    predictions: list[str] = []
    for item in stimuli_items:
        prompt = format_prompt(item)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits[0, -1, :]
        best_letter, _ = best_choice_from_logits(logits, tokenizer)
        predictions.append(best_letter or "")

    return compute_bias_score(stimuli_items, predictions, condition), predictions


def _count_changed(
    items: list[dict],
    preds_a: list[str],
    preds_b: list[str],
    *,
    condition: str = "ambig",
) -> int:
    changed = 0
    for it, a, b in zip(items, preds_a, preds_b):
        if it.get("context_condition") != condition:
            continue
        if a != b:
            changed += 1
    return changed


def _count_changed_non_unknown(
    items: list[dict],
    preds_a: list[str],
    preds_b: list[str],
    *,
    condition: str = "ambig",
) -> int:
    """Count changed predictions among ambig items where both are non-unknown."""
    changed = 0
    for it, a, b in zip(items, preds_a, preds_b):
        if it.get("context_condition") != condition:
            continue
        roles = it.get("answer_roles", {})
        ra = roles.get(a, "unknown")
        rb = roles.get(b, "unknown")
        if ra == "unknown" or rb == "unknown":
            continue
        if a != b:
            changed += 1
    return changed


def _transition_counts(
    items: list[dict],
    preds_a: list[str],
    preds_b: list[str],
    *,
    condition: str = "ambig",
) -> dict[str, int]:
    """Count role transitions (stereo/non/unknown) baseline->ablated on ambig items."""
    counts: dict[str, int] = {}
    for it, a, b in zip(items, preds_a, preds_b):
        if it.get("context_condition") != condition:
            continue
        roles = it.get("answer_roles", {})
        ra = roles.get(a, "unknown")
        rb = roles.get(b, "unknown")
        key = f"{ra}->{rb}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Causal ablation experiments.")
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--base_run_dir", type=str, default=None)
    parser.add_argument("--chat_run_dir", type=str, default=None)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--chat_model_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--categories", type=str, default="all")
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--model_id", type=str, default=None)
    args = parser.parse_args()

    base_dir = Path(args.base_run_dir or args.run_dir)
    chat_dir = Path(args.chat_run_dir) if args.chat_run_dir else None
    categories = parse_categories(args.categories)
    fig_dir = ensure_dir(base_dir / "figures")
    analysis_dir = ensure_dir(base_dir / "analysis")
    model_id = args.model_id or base_dir.parent.name

    log(f"Causal ablation analysis for {model_id}")
    log(f"Categories: {categories}")
    log(f"Device: {args.device}")

    # Load directions
    directions_path = analysis_dir / "directions.npz"
    if not directions_path.exists():
        log("ERROR: Run compute_directions.py first")
        return

    dir_data = np.load(directions_path, allow_pickle=True)
    cat_directions: dict[str, np.ndarray] = {}
    for cat in categories:
        key = f"direction_{cat}"
        if key in dir_data.files:
            cat_directions[cat] = dir_data[key]

    n_layers = list(cat_directions.values())[0].shape[0]
    mid_layer = n_layers // 2

    # Compute shared component per layer (more correct than using one mid-layer vector everywhere)
    shared_dirs_by_layer: dict[int, np.ndarray] = {}
    for layer in range(n_layers):
        sca = shared_component_analysis(cat_directions, layer)
        shared_dirs_by_layer[layer] = sca["shared_direction"]
    log(f"Shared direction extracted per-layer (representative layer={mid_layer})")

    # Load stimuli per category
    cat_stimuli: dict[str, list[dict]] = {}
    for cat in categories:
        stimuli_files = sorted((base_dir / "stimuli").glob(f"stimuli_{cat}_*.json"))
        if not stimuli_files:
            stimuli_files = sorted(Path("data/processed").glob(f"stimuli_{cat}_*.json"))
        if stimuli_files:
            with open(stimuli_files[-1]) as f:
                items = json.load(f)
            if args.max_items:
                items = items[:args.max_items]
            cat_stimuli[cat] = items

    # Load base model
    log("\nLoading base model...")
    model, tokenizer, n_layers_m, hidden_dim, get_layer_fn, get_o_proj_fn, n_heads, head_dim = \
        _load_model(args.model_path, args.device)

    all_layers = list(range(n_layers_m))
    ablation_results: dict[str, dict[str, float]] = {}

    for cat in categories:
        if cat not in cat_stimuli:
            continue
        items = cat_stimuli[cat]
        log(f"\n--- Category: {cat} ({len(items)} items) ---")

        # Baseline
        log("  Baseline...")
        baseline, preds_base = run_behavioral_eval(model, tokenizer, items, args.device)
        log(f"  Baseline bias: {baseline['bias_score']:.3f}")
        log(f"  Baseline n_non_unknown={baseline['n_non_unknown']}/{baseline['n_total']}")

        # Intervention 5a: Ablate shared direction
        log("  Ablating shared direction...")
        dirs_per_layer = {l: [shared_dirs_by_layer[l]] for l in all_layers}
        hooks = apply_multi_direction_ablation(model, get_layer_fn, dirs_per_layer, args.device)
        shared_result, preds_shared = run_behavioral_eval(model, tokenizer, items, args.device)
        remove_hooks(hooks)
        log(f"  Shared ablation bias: {shared_result['bias_score']:.3f}")
        log(
            f"  Shared ablation n_non_unknown={shared_result['n_non_unknown']}/{shared_result['n_total']} "
            f"changed_ambig={_count_changed(items, preds_base, preds_shared)} "
            f"changed_non_unknown_ambig={_count_changed_non_unknown(items, preds_base, preds_shared)}"
        )
        log(f"  Shared transitions: {_transition_counts(items, preds_base, preds_shared)}")

        # Intervention 5c: Ablate category-specific direction
        if cat in cat_directions:
            # Category-specific per layer = cat_dir[layer] - proj onto shared_dir[layer]
            specific_dirs_by_layer: dict[int, np.ndarray] = {}
            for l in all_layers:
                cd = cat_directions[cat][l]
                sd = shared_dirs_by_layer[l]
                proj = float(np.dot(cd, sd)) * sd
                sp = cd - proj
                nrm = float(np.linalg.norm(sp))
                if nrm > 1e-8:
                    sp = sp / nrm
                specific_dirs_by_layer[l] = sp.astype(np.float32)

            log("  Ablating category-specific direction...")
            dirs_per_layer = {l: [specific_dirs_by_layer[l]] for l in all_layers}
            hooks = apply_multi_direction_ablation(model, get_layer_fn, dirs_per_layer, args.device)
            specific_result, preds_specific = run_behavioral_eval(model, tokenizer, items, args.device)
            remove_hooks(hooks)
            log(f"  Specific ablation bias: {specific_result['bias_score']:.3f}")
            log(
                f"  Specific ablation n_non_unknown={specific_result['n_non_unknown']}/{specific_result['n_total']} "
                f"changed_ambig={_count_changed(items, preds_base, preds_specific)} "
                f"changed_non_unknown_ambig={_count_changed_non_unknown(items, preds_base, preds_specific)}"
            )
            log(f"  Specific transitions: {_transition_counts(items, preds_base, preds_specific)}")

            # Ablate both
            log("  Ablating both directions...")
            dirs_per_layer = {l: [shared_dirs_by_layer[l], specific_dirs_by_layer[l]] for l in all_layers}
            hooks = apply_multi_direction_ablation(model, get_layer_fn, dirs_per_layer, args.device)
            both_result, preds_both = run_behavioral_eval(model, tokenizer, items, args.device)
            remove_hooks(hooks)
            log(f"  Both ablation bias: {both_result['bias_score']:.3f}")
            log(
                f"  Both ablation n_non_unknown={both_result['n_non_unknown']}/{both_result['n_total']} "
                f"changed_ambig={_count_changed(items, preds_base, preds_both)} "
                f"changed_non_unknown_ambig={_count_changed_non_unknown(items, preds_base, preds_both)}"
            )
            log(f"  Both transitions: {_transition_counts(items, preds_base, preds_both)}")
        else:
            specific_result = baseline
            both_result = baseline

        ablation_results[cat] = {
            "baseline": baseline["bias_score"],
            "ablate_shared": shared_result["bias_score"],
            "ablate_specific": specific_result["bias_score"],
            "ablate_both": both_result["bias_score"],
            "n_items": baseline["n_total"],
            "baseline_n_non_unknown": baseline["n_non_unknown"],
            "shared_n_non_unknown": shared_result["n_non_unknown"],
            "specific_n_non_unknown": specific_result.get("n_non_unknown", baseline["n_non_unknown"]),
            "both_n_non_unknown": both_result.get("n_non_unknown", baseline["n_non_unknown"]),
            "baseline_counts": {
                "n_stereo": baseline["n_stereo"],
                "n_non_stereo": baseline["n_non_stereo"],
                "n_unknown": baseline["n_unknown"],
            },
            "shared_counts": {
                "n_stereo": shared_result["n_stereo"],
                "n_non_stereo": shared_result["n_non_stereo"],
                "n_unknown": shared_result["n_unknown"],
            },
            "specific_counts": {
                "n_stereo": specific_result.get("n_stereo", baseline["n_stereo"]),
                "n_non_stereo": specific_result.get("n_non_stereo", baseline["n_non_stereo"]),
                "n_unknown": specific_result.get("n_unknown", baseline["n_unknown"]),
            },
            "both_counts": {
                "n_stereo": both_result.get("n_stereo", baseline["n_stereo"]),
                "n_non_stereo": both_result.get("n_non_stereo", baseline["n_non_stereo"]),
                "n_unknown": both_result.get("n_unknown", baseline["n_unknown"]),
            },
        }

        # Memory management
        if args.device == "mps":
            torch.mps.empty_cache()

    # ===== Fig 17: Shared vs specific ablation =====
    log("\n--- Fig 17: Shared vs specific ablation ---")
    plot_ablation_grouped_bars(
        ablation_results,
        path=str(fig_dir / "fig_17_shared_vs_specific_ablation.png"),
        title=f"Ablation effects on bias ({model_id})",
    )
    log(f"  Saved fig_17")

    # ===== Intervention 5b: RLHF-target head ablation =====
    probe_matrices_path = analysis_dir / "probe_matrices.npz"
    if probe_matrices_path.exists() and chat_dir:
        log("\n--- RLHF-target head ablation ---")
        probe_data = np.load(probe_matrices_path)

        if "base_stereo" in probe_data and "chat_stereo" in probe_data:
            targets = identify_rlhf_targets(
                probe_data["base_stereo"], probe_data["chat_stereo"],
                top_k=20, min_base_acc=0.55,
            )

            if targets and head_dim:
                hooks = apply_head_ablation(get_o_proj_fn, targets, head_dim)
                rlhf_results: dict[str, dict[str, float]] = {}

                for cat in categories:
                    if cat not in cat_stimuli:
                        continue
                    items = cat_stimuli[cat]
                    result, _ = run_behavioral_eval(model, tokenizer, items, args.device)
                    rlhf_results[cat] = {
                        "base_baseline": ablation_results.get(cat, {}).get("baseline", 0),
                        "base_ablated": result["bias_score"],
                    }
                    log(f"  {cat}: RLHF-head ablation bias = {result['bias_score']:.3f}")

                remove_head_hooks(hooks)

                # Get chat model baselines if available
                if args.chat_model_path:
                    log("  Loading chat model for baseline comparison...")
                    chat_model, chat_tok, *_ = _load_model(args.chat_model_path, args.device)
                    for cat in categories:
                        if cat not in cat_stimuli:
                            continue
                        items = cat_stimuli[cat]
                        chat_result, _ = run_behavioral_eval(
                            chat_model, chat_tok, items, args.device
                        )
                        rlhf_results[cat]["chat_baseline"] = chat_result["bias_score"]
                    del chat_model

                # Fig 18: RLHF replication
                if rlhf_results:
                    log("\n--- Fig 18: RLHF replication ---")
                    plot_ablation_grouped_bars(
                        rlhf_results,
                        path=str(fig_dir / "fig_18_rlhf_replication.png"),
                        conditions=["base_baseline", "base_ablated", "chat_baseline"],
                        condition_labels=["Base baseline", "Base + head ablation", "Chat baseline"],
                        condition_colors=["#D55E00", "#E69F00", "#0072B2"],
                        title=f"RLHF replication via head ablation ({model_id})",
                    )
                    log(f"  Saved fig_18")

    # ===== Fig 19: Cross-category ablation effects =====
    log("\n--- Fig 19: Cross-category ablation heatmap ---")
    # Build matrix: rows = ablated direction, cols = measured category
    available = sorted(ablation_results.keys())
    n = len(available)
    if n >= 2:
        cross_ablation = np.zeros((n + 1, n), dtype=np.float32)  # +1 for shared row

        # Shared row (already computed)
        for j, cat in enumerate(available):
            cross_ablation[0, j] = (
                ablation_results[cat]["ablate_shared"] - ablation_results[cat]["baseline"]
            )

        # Per-category ablation rows
        for i, ablate_cat in enumerate(available):
            if ablate_cat not in cat_directions:
                continue
            ablate_dir = cat_directions[ablate_cat][mid_layer]
            hooks = apply_direction_ablation(model, get_layer_fn, ablate_dir, all_layers, args.device)

            for j, measure_cat in enumerate(available):
                if measure_cat not in cat_stimuli:
                    continue
                items = cat_stimuli[measure_cat]
                result, _ = run_behavioral_eval(model, tokenizer, items, args.device)
                cross_ablation[i + 1, j] = result["bias_score"] - ablation_results[measure_cat]["baseline"]

            remove_hooks(hooks)
            log(f"  Ablated {ablate_cat}: effects computed")

        row_names = ["shared"] + available
        vmax = max(abs(cross_ablation.min()), abs(cross_ablation.max()), 0.05)

        from src.visualization.style import apply_style, save_fig, CATEGORY_LABELS, TITLE_SIZE
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        apply_style()

        fig, ax = plt.subplots(figsize=(max(8, n * 1.2), max(6, (n + 1) * 0.8)))
        im = ax.imshow(cross_ablation, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n + 1))
        ax.set_xticklabels([CATEGORY_LABELS.get(c, c) for c in available], rotation=45, ha="right")
        ax.set_yticklabels([CATEGORY_LABELS.get(c, c) for c in row_names])
        ax.set_xlabel("Measured category")
        ax.set_ylabel("Ablated direction")
        ax.set_title(f"Cross-category ablation effects ({model_id})")
        for i in range(n + 1):
            for j in range(n):
                val = cross_ablation[i, j]
                color = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)
        fig.colorbar(im, ax=ax, shrink=0.8, label="Bias change from baseline")
        save_fig(fig, str(fig_dir / "fig_19_cross_category_ablation_effects.png"))
        log(f"  Saved fig_19")

    # Save all results
    save_data = {
        "model_id": model_id,
        "ablation_results": ablation_results,
        "mid_layer": mid_layer,
    }
    atomic_save_json(save_data, analysis_dir / "ablation_results.json")
    log(f"\nResults saved to {analysis_dir / 'ablation_results.json'}")
    log("Done!")


if __name__ == "__main__":
    main()
