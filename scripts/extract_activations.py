#!/usr/bin/env python3
"""Extract hidden-state activations from a model for a set of BBQ stimuli.

Usage:
    python scripts/extract_activations.py \
        --model_path models/llama2-13b \
        --model_id llama2-13b \
        --stimuli_json data/processed/stimuli_so_2026-04-10.json \
        --category so \
        --device cuda

    # Quick test with 20 items
    python scripts/extract_activations.py \
        --model_path models/llama2-13b \
        --model_id llama2-13b \
        --stimuli_json data/processed/stimuli_so_2026-04-10.json \
        --category so \
        --device cuda \
        --max_items 20
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.extraction.activations import run_extraction
from src.utils.io import ensure_dir
from src.utils.logging import log


def _load_model_and_tokenizer(
    model_path: str, device: str
) -> tuple:
    """Load model and tokenizer via ModelWrapper if available, else direct HF loading.

    Returns: (model, tokenizer, n_layers, hidden_dim, get_layer_fn, get_o_proj_fn)
    """
    # Try ModelWrapper first
    try:
        from src.models.wrapper import ModelWrapper

        wrapper = ModelWrapper.from_pretrained(model_path, device=device)
        log(f"Loaded model via ModelWrapper: {model_path}")
        log(f"  Architecture: {type(wrapper.model).__name__}")
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
        log(f"ModelWrapper not available ({e}), using direct HF loading")

    # Direct HuggingFace loading
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(
            "Tokenizer is not fast; offset mapping is required for extraction. "
            "Use a model with a fast tokenizer or ensure the fast tokenizer is available."
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float32 if device == "cpu" else torch.float16
    log(f"Loading model from {model_path} (dtype={dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=dtype
    ).to(device)
    model.eval()

    # Detect architecture for layer access
    config = model.config
    n_layers = config.num_hidden_layers
    hidden_dim = config.hidden_size
    log(f"  Architecture: {type(model).__name__}")
    log(f"  Layers: {n_layers}, Hidden dim: {hidden_dim}")

    # Build get_layer_fn by detecting the decoder layer path
    # Common patterns: model.model.layers, model.transformer.h, model.gpt_neox.layers
    inner = None
    for attr_path in ["model.layers", "transformer.h", "gpt_neox.layers"]:
        obj = model
        for attr in attr_path.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            inner = obj
            log(f"  Decoder layers found at: model.{attr_path}")
            break

    if inner is None:
        raise RuntimeError(
            f"Cannot find decoder layers for {type(model).__name__}. "
            f"Implement ModelWrapper in src/models/wrapper.py for this architecture."
        )

    def get_layer_fn(idx: int):
        return inner[idx]

    # Attempt to locate attention o_proj modules (best-effort)
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
        description="Extract activations from a model for BBQ stimuli."
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to HF model")
    parser.add_argument("--model_id", type=str, required=True, help="Model identifier for output paths")
    parser.add_argument("--stimuli_json", type=str, required=True, help="Path to processed stimuli JSON")
    parser.add_argument("--category", type=str, required=True, help="Category short name (for output path)")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cuda, mps, cpu")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--max_items", type=int, default=None, help="Max items to extract (for testing)")
    parser.add_argument("--run_date", type=str, default=None, help="Date string for run directory")

    args = parser.parse_args()
    run_date = args.run_date or date.today().isoformat()

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = str(
            Path("results") / "runs" / args.model_id / run_date / "activations" / args.category
        )
    ensure_dir(output_dir)

    log(f"Model: {args.model_path} ({args.model_id})")
    log(f"Stimuli: {args.stimuli_json}")
    log(f"Category: {args.category}")
    log(f"Device: {args.device}")
    log(f"Output: {output_dir}")
    if args.max_items:
        log(f"Max items: {args.max_items}")

    # Load stimuli
    with open(args.stimuli_json) as f:
        items = json.load(f)
    log(f"Loaded {len(items)} stimuli items")

    # Load model
    model, tokenizer, n_layers, hidden_dim, get_layer_fn, get_o_proj_fn = _load_model_and_tokenizer(
        args.model_path, args.device
    )

    # Run extraction
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

    # Save extraction summary
    summary["model_id"] = args.model_id
    summary["model_path"] = args.model_path
    summary["category"] = args.category
    summary["device"] = args.device
    summary["output_dir"] = output_dir
    summary_path = Path(output_dir) / "extraction_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"Summary -> {summary_path}")


if __name__ == "__main__":
    main()
