#!/usr/bin/env python3
"""SAE layer localization pipeline: extract + analyze.

Usage:
    # Full run
    python scripts/run_sae_localization.py \
        --model_path models/llama2-13b --model_id llama2-13b --device mps

    # Quick test
    python scripts/run_sae_localization.py \
        --model_path models/llama2-13b --model_id llama2-13b --device mps \
        --max_items 20

    # Single category
    python scripts/run_sae_localization.py \
        --model_path models/llama2-13b --model_id llama2-13b --device mps \
        --categories so
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.data.bbq_loader import CATEGORY_MAP, parse_categories
from src.sae_localization.analyze import run_analysis
from src.sae_localization.extract import run_extraction
from src.utils.io import ensure_dir
from src.utils.logging import log


def _load_model(model_path: str, device: str):
    """Load model via ModelWrapper or direct HF."""
    try:
        from src.models.wrapper import ModelWrapper
        wrapper = ModelWrapper.from_pretrained(model_path, device=device)
        log(f"Loaded via ModelWrapper: {type(wrapper.model).__name__}")
        log(f"  Layers: {wrapper.n_layers}, dim: {wrapper.hidden_dim}")
        return wrapper.model, wrapper.tokenizer, wrapper.n_layers, wrapper.hidden_dim, wrapper.get_layer
    except (ImportError, AttributeError) as e:
        log(f"ModelWrapper unavailable ({e}), falling back to direct loading")

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

    # Validate hook output
    test_inp = tokenizer("test", return_tensors="pt").to(device)
    test_hs = {}
    def _locate_hidden(output):
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, tuple):
            for x in output:
                if isinstance(x, torch.Tensor) and x.dim() == 3 and x.shape[-1] == hidden_dim:
                    return x
            return output[0] if output and isinstance(output[0], torch.Tensor) else None
        return None
    def _th(module, args, output):
        test_hs["out"] = _locate_hidden(output)
    h = inner[0].register_forward_hook(_th)
    with torch.no_grad():
        model(**test_inp)
    h.remove()
    if test_hs.get("out") is None:
        raise RuntimeError("Could not locate hidden state tensor in layer hook output for this architecture.")
    if test_hs["out"].shape[-1] != hidden_dim:
        raise RuntimeError(f"Hook output dim mismatch: {test_hs['out'].shape[-1]} vs {hidden_dim}")

    def get_layer_fn(idx):
        return inner[idx]

    return model, tokenizer, n_layers, hidden_dim, get_layer_fn


def _load_stimuli(cat: str) -> list[dict] | None:
    """Load processed stimuli for a category."""
    files = sorted(Path("data/processed").glob(f"stimuli_{cat}_*.json"))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)

def _available_categories(categories: list[str]) -> tuple[list[str], list[str]]:
    """Return (available, missing) based on presence of processed stimuli JSON files."""
    avail: list[str] = []
    missing: list[str] = []
    for cat in categories:
        files = sorted(Path("data/processed").glob(f"stimuli_{cat}_*.json"))
        (avail if files else missing).append(cat)
    return avail, missing


def main() -> None:
    parser = argparse.ArgumentParser(description="SAE layer localization pipeline.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--categories", type=str, default="all",
                        help="Comma-separated category short names or 'all'")
    parser.add_argument("--max_items", type=int, default=None,
                        help="Max items per category (for testing)")
    parser.add_argument("--run_date", type=str, default=None)
    parser.add_argument("--skip_extraction", action="store_true",
                        help="Skip extraction, only run analysis on existing data")
    args = parser.parse_args()

    run_date = args.run_date or date.today().isoformat()
    categories = parse_categories(args.categories)
    run_dir = Path("results") / "sae_localization" / args.model_id / run_date
    ensure_dir(run_dir)

    log(f"{'='*60}")
    log(f"SAE LAYER LOCALIZATION PIPELINE")
    log(f"{'='*60}")
    log(f"Model: {args.model_path} ({args.model_id})")
    log(f"Device: {args.device}")
    log(f"Categories: {categories}")
    log(f"Run dir: {run_dir}")
    if args.max_items:
        log(f"Max items per category: {args.max_items}")

    if not args.skip_extraction:
        available, missing = _available_categories(categories)
        if missing:
            log(f"\nNOTE: Missing processed stimuli for: {missing}")
            log("Run this first (no model load):")
            log("  python scripts/prepare_stimuli.py --categories all")
        if not available:
            raise SystemExit(
                "ERROR: No processed stimuli found for any requested category under data/processed/. "
                "Run: python scripts/prepare_stimuli.py --categories all"
            )
        categories = available

        # Load model once
        log(f"\n--- Loading model ---")
        t0 = time.time()
        model, tokenizer, n_layers, hidden_dim, get_layer_fn = _load_model(
            args.model_path, args.device
        )
        log(f"Model loaded in {time.time() - t0:.1f}s")

        # Extract per category
        for cat in categories:
            log(f"\n{'='*60}")
            log(f"EXTRACTING: {cat} ({CATEGORY_MAP.get(cat, cat)})")
            log(f"{'='*60}")

            stimuli = _load_stimuli(cat)
            if stimuli is None:
                log(f"  No stimuli for {cat}. Run: python scripts/prepare_stimuli.py --categories {cat}")
                continue

            act_dir = run_dir / "activations" / cat
            ensure_dir(act_dir)

            t1 = time.time()
            result = run_extraction(
                items=stimuli,
                model=model,
                tokenizer=tokenizer,
                get_layer_fn=get_layer_fn,
                n_layers=n_layers,
                hidden_dim=hidden_dim,
                device=args.device,
                output_dir=act_dir,
                max_items=args.max_items,
            )
            log(f"  Done in {time.time() - t1:.1f}s: {result}")

            if args.device == "mps":
                torch.mps.empty_cache()

        # Free model memory before analysis
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    else:
        act_base = run_dir / "activations"
        if not act_base.exists():
            raise SystemExit(
                f"ERROR: --skip_extraction was set, but no activations directory exists at {act_base}. "
                "Either run without --skip_extraction to generate activations, or point --run_date at an existing extraction."
            )

    # Run analysis
    log(f"\n{'='*60}")
    log(f"RUNNING ANALYSIS")
    log(f"{'='*60}")
    recommendation = run_analysis(run_dir, categories)

    if recommendation:
        overall = recommendation.get("overall_recommended_range", [])
        log(f"\n{'='*60}")
        log(f"RESULT: Recommended SAE injection layers: {overall}")
        log(f"{'='*60}")
        for cat, info in recommendation.get("per_category", {}).items():
            log(f"  {cat:>22s}: diverge={info['divergence_layer']}, "
                f"peak_stereo={info['peak_derivative_layer_stereotyped']}, "
                f"range={info['recommended_range']}, "
                f"n_stereo={info['n_stereotyped']}")

    log("\nPipeline complete!")


if __name__ == "__main__":
    main()
