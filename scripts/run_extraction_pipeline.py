#!/usr/bin/env python3
"""Orchestrator: run prepare_stimuli + extract_activations for multiple categories.

Usage:
    # All categories for one model
    python scripts/run_extraction_pipeline.py \
        --model_path /workspace/lgbtqmi/models/llama2-13b \
        --model_id llama2-13b \
        --device cuda \
        --categories so,gi,race,religion,disability,physical_appearance,age

    # Just new categories
    python scripts/run_extraction_pipeline.py \
        --model_path models/llama2-13b \
        --model_id llama2-13b \
        --device cuda \
        --categories race,religion,disability

    # Quick test
    python scripts/run_extraction_pipeline.py \
        --model_path models/llama2-13b \
        --model_id llama2-13b \
        --device cuda \
        --categories so \
        --max_items 20
"""

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.data.bbq_loader import CATEGORY_MAP, load_and_standardize, parse_categories
from src.data.crows_pairs_loader import load_crows_pairs_as_stimuli, validate_crows_pairs_csv
from src.extraction.activations import run_extraction
from src.utils.io import atomic_save_json, ensure_dir
from src.utils.logging import log


def _load_model(model_path: str, device: str) -> tuple:
    """Load model once for all categories.

    Returns (model, tokenizer, n_layers, hidden_dim, get_layer_fn, get_o_proj_fn).
    """
    # Try ModelWrapper first
    try:
        from src.models.wrapper import ModelWrapper

        wrapper = ModelWrapper.from_pretrained(model_path, device=device)
        log(f"Loaded via ModelWrapper: {type(wrapper.model).__name__}")
        log(f"  Layers: {wrapper.n_layers}, Hidden dim: {wrapper.hidden_dim}")
        return (
            wrapper.model,
            wrapper.tokenizer,
            wrapper.n_layers,
            wrapper.hidden_dim,
            wrapper.get_layer,
            wrapper.get_o_proj,
        )
    except (ImportError, AttributeError) as e:
        log(f"ModelWrapper not available ({e}), falling back to direct loading")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(
            "Tokenizer is not fast; offset mapping is required for extraction. "
            "Use a model with a fast tokenizer or ensure the fast tokenizer is available."
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float32 if device == "cpu" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=dtype
    ).to(device)
    model.eval()

    config = model.config
    n_layers = config.num_hidden_layers
    hidden_dim = config.hidden_size
    log(f"Loaded: {type(model).__name__}, {n_layers} layers, dim={hidden_dim}")

    # Find decoder layers
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
        raise RuntimeError(
            f"Cannot find decoder layers for {type(model).__name__}. "
            f"Implement ModelWrapper for this architecture."
        )

    # Validate hook output
    test_input = tokenizer("test", return_tensors="pt").to(device)
    test_hs = {}

    def test_hook(module, args, output):
        test_hs["out"] = output[0] if isinstance(output, tuple) else output

    h = inner[0].register_forward_hook(test_hook)
    with torch.no_grad():
        model(**test_input)
    h.remove()

    if test_hs["out"].shape[-1] != hidden_dim:
        raise RuntimeError(
            f"Hook output[0] dim={test_hs['out'].shape[-1]} != {hidden_dim}. "
            f"Need ModelWrapper for this architecture."
        )

    def get_layer_fn(idx: int):
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

    return model, tokenizer, n_layers, hidden_dim, get_layer_fn, get_o_proj_fn


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run stimulus preparation and activation extraction for multiple BBQ categories."
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to HF model")
    parser.add_argument("--model_id", type=str, required=True, help="Model identifier")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cuda, mps, cpu")
    parser.add_argument(
        "--categories",
        type=str,
        default="all",
        help="Comma-separated category short names or 'all'",
    )
    parser.add_argument(
        "--bbq_data_dir",
        type=str,
        default="datasets/bbq/data",
        help="Path to BBQ JSONL files",
    )
    parser.add_argument("--max_items", type=int, default=None, help="Max items per category (for testing)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for answer shuffling")
    parser.add_argument("--run_date", type=str, default=None, help="Override run date")
    parser.add_argument(
        "--crows_pairs_path",
        type=str,
        default=None,
        help="Optional: path to CrowS-Pairs CSV. If set, will prepare stimuli + extract activations under activations/crows_pairs/.",
    )
    parser.add_argument(
        "--crows_max_items",
        type=int,
        default=None,
        help="Optional: max items for CrowS-Pairs (default: --max_items, else all).",
    )

    args = parser.parse_args()
    run_date = args.run_date or date.today().isoformat()
    categories = parse_categories(args.categories)

    log(f"{'='*60}")
    log(f"MULTI-CATEGORY EXTRACTION PIPELINE")
    log(f"{'='*60}")
    log(f"Model: {args.model_path} ({args.model_id})")
    log(f"Device: {args.device}")
    log(f"Categories: {categories}")
    log(f"Run date: {run_date}")
    if args.max_items:
        log(f"Max items per category: {args.max_items}")

    # Set up output paths
    run_base = Path("results") / "runs" / args.model_id / run_date
    stimuli_dir = run_base / "stimuli"
    activations_base = run_base / "activations"
    ensure_dir(stimuli_dir)

    # Step 1: Load model once
    log(f"\n{'='*60}")
    log("STEP 1: Loading model")
    log(f"{'='*60}")
    t0 = time.time()
    model, tokenizer, n_layers, hidden_dim, get_layer_fn, get_o_proj_fn = _load_model(
        args.model_path, args.device
    )
    log(f"Model loaded in {time.time() - t0:.1f}s")

    # Step 2: Process each category
    pipeline_summary: dict[str, dict] = {}

    for cat_idx, cat in enumerate(categories):
        bbq_name = CATEGORY_MAP[cat]
        log(f"\n{'='*60}")
        log(f"CATEGORY {cat_idx + 1}/{len(categories)}: {cat} ({bbq_name})")
        log(f"{'='*60}")

        # 2a: Prepare stimuli
        log("Preparing stimuli...")
        items = load_and_standardize(cat, args.bbq_data_dir, seed=args.seed)
        if args.max_items is not None:
            items = items[: args.max_items]
            log(f"Truncated to {len(items)} items")

        # Save stimuli
        stimuli_path = stimuli_dir / f"stimuli_{cat}_{run_date}.json"
        atomic_save_json(items, stimuli_path)
        log(f"Stimuli saved -> {stimuli_path}")

        # 2b: Extract activations
        output_dir = str(activations_base / cat)
        ensure_dir(output_dir)

        log("Extracting activations...")
        t1 = time.time()
        summary = run_extraction(
            items=items,
            model=model,
            tokenizer=tokenizer,
            device=args.device,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            get_layer_fn=get_layer_fn,
            get_o_proj_fn=get_o_proj_fn,
            output_dir=output_dir,
            max_items=args.max_items,
        )
        elapsed = time.time() - t1

        summary["category"] = cat
        summary["bbq_name"] = bbq_name
        summary["elapsed_seconds"] = round(elapsed, 1)
        summary["stimuli_path"] = str(stimuli_path)
        summary["activations_dir"] = output_dir
        pipeline_summary[cat] = summary

        log(f"Category {cat} done in {elapsed:.1f}s")

        # Free MPS memory between categories
        if args.device == "mps":
            torch.mps.empty_cache()

    # Step 2c: Optional CrowS-Pairs
    if args.crows_pairs_path:
        crows_path = Path(args.crows_pairs_path)
        if not crows_path.exists():
            raise FileNotFoundError(f"CrowS-Pairs CSV not found: {crows_path}")

        log(f"\n{'='*60}")
        log("CROWS-PAIRS")
        log(f"{'='*60}")
        schema = validate_crows_pairs_csv(crows_path)
        log(f"  CrowS-Pairs rows: {schema['n_rows']}")

        crows_n = args.crows_max_items if args.crows_max_items is not None else args.max_items
        stimuli_path = stimuli_dir / f"stimuli_crows_pairs_{run_date}.json"

        if stimuli_path.exists():
            log(f"Stimuli exists -> {stimuli_path} (reusing)")
            with open(stimuli_path) as f:
                crows_items = json.load(f)
        else:
            log("Preparing CrowS-Pairs stimuli...")
            crows_items = load_crows_pairs_as_stimuli(crows_path, max_items=crows_n)
            atomic_save_json(crows_items, stimuli_path)
            log(f"Stimuli saved -> {stimuli_path}")

        output_dir = str(activations_base / "crows_pairs")
        ensure_dir(output_dir)

        log("Extracting CrowS-Pairs activations...")
        t1 = time.time()
        summary = run_extraction(
            items=crows_items,
            model=model,
            tokenizer=tokenizer,
            device=args.device,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            get_layer_fn=get_layer_fn,
            get_o_proj_fn=get_o_proj_fn,
            output_dir=output_dir,
            max_items=crows_n,
        )
        elapsed = time.time() - t1
        summary["category"] = "crows_pairs"
        summary["elapsed_seconds"] = round(elapsed, 1)
        summary["stimuli_path"] = str(stimuli_path)
        summary["activations_dir"] = output_dir
        summary["schema"] = schema
        pipeline_summary["crows_pairs"] = summary

        log(f"CrowS-Pairs done in {elapsed:.1f}s")
        if args.device == "mps":
            torch.mps.empty_cache()

    # Step 3: Final summary
    log(f"\n{'='*60}")
    log("PIPELINE SUMMARY")
    log(f"{'='*60}")
    total_items = 0
    total_extracted = 0
    for cat, info in pipeline_summary.items():
        log(
            f"  {cat:>20s}: {info['n_total']:>5d} items, "
            f"{info['n_extracted']} extracted, {info['n_skipped']} skipped, "
            f"{info['elapsed_seconds']:.1f}s"
        )
        total_items += info["n_total"]
        total_extracted += info["n_extracted"]

    log(f"  {'TOTAL':>20s}: {total_items} items, {total_extracted} extracted")

    # Save pipeline summary
    pipeline_summary["_meta"] = {
        "model_id": args.model_id,
        "model_path": args.model_path,
        "device": args.device,
        "categories": categories,
        "run_date": run_date,
        "max_items": args.max_items,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
    }
    summary_path = run_base / "pipeline_summary.json"
    atomic_save_json(pipeline_summary, summary_path)
    log(f"\nPipeline summary -> {summary_path}")


if __name__ == "__main__":
    main()
