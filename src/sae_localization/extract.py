"""SAE localization: extract last-token hidden states at every layer.

Records model answer, stereotyping segment, and per-layer normalized hidden states.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.extraction.activations import format_prompt
from src.utils.answers import best_choice_from_logits
from src.utils.io import atomic_save_npz, ensure_dir
from src.utils.logging import ProgressLogger, log


def _locate_hidden(output: Any, hidden_dim: int) -> torch.Tensor:
    """Locate (batch, seq, hidden_dim) hidden state tensor in layer hook output."""
    if isinstance(output, torch.Tensor):
        if output.dim() == 3 and output.shape[-1] == hidden_dim:
            return output
        raise ValueError(f"Unexpected tensor output shape: {tuple(output.shape)}")

    if isinstance(output, tuple):
        for x in output:
            if isinstance(x, torch.Tensor) and x.dim() == 3 and x.shape[-1] == hidden_dim:
                return x
        if output and isinstance(output[0], torch.Tensor):
            raise ValueError(
                "Could not locate hidden state in output tuple; "
                f"output[0] shape={tuple(output[0].shape)} hidden_dim={hidden_dim}"
            )
        raise ValueError("Could not locate hidden state in output tuple")

    raise ValueError(f"Unexpected output type: {type(output)}")


def extract_item(
    item: dict[str, Any],
    model: Any,
    tokenizer: Any,
    get_layer_fn: Any,
    n_layers: int,
    hidden_dim: int,
    device: str,
) -> dict[str, Any]:
    """Run a single item through the model and capture all-layer last-token hidden states.

    Returns dict with hidden_states (normalised), raw norms, answer info, metadata.
    """
    prompt = format_prompt(item)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    seq_len = inputs["input_ids"].shape[1]
    last_pos = seq_len - 1

    # Register hooks on every layer
    hidden_states: dict[int, torch.Tensor] = {}
    hook_counts: dict[int, int] = {}

    def _make_hook(layer_idx: int):
        def hook_fn(module: Any, args: Any, output: Any) -> None:
            hook_counts[layer_idx] = hook_counts.get(layer_idx, 0) + 1
            h = _locate_hidden(output, hidden_dim)
            # Grab last-token hidden state
            hidden_states[layer_idx] = h[0, last_pos, :].detach().cpu().float()
        return hook_fn

    hooks = []
    for i in range(n_layers):
        layer = get_layer_fn(i)
        hooks.append(layer.register_forward_hook(_make_hook(i)))

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # Clean up hooks
    for h in hooks:
        h.remove()

    missing = [i for i in range(n_layers) if hook_counts.get(i, 0) == 0]
    if missing:
        raise RuntimeError(
            f"Residual hooks did not fire for layers: {missing}. "
            "Architecture likely unsupported without ModelWrapper adjustments."
        )

    # Stack hidden states: (n_layers, hidden_dim)
    hs = torch.stack([hidden_states[i] for i in range(n_layers)], dim=0).numpy()

    # Compute raw norms and normalise
    raw_norms = np.linalg.norm(hs, axis=1).astype(np.float32)  # (n_layers,)
    safe_norms = np.maximum(raw_norms, 1e-8)[:, None]
    hs_normed = (hs / safe_norms).astype(np.float16)

    # Extract answer from logits
    logits = outputs.logits[0, last_pos, :]
    best_letter, letter_logits = best_choice_from_logits(logits, tokenizer)
    model_answer = best_letter or ""
    model_answer_role = item.get("answer_roles", {}).get(model_answer, "unknown")
    is_correct = model_answer == item.get("correct_letter", "")
    is_stereotyped = model_answer_role == "stereotyped_target"

    return {
        "hidden_states": hs_normed,
        "hidden_states_raw_norms": raw_norms,
        "model_answer": model_answer,
        "model_answer_role": model_answer_role,
        "is_correct": is_correct,
        "is_stereotyped_response": is_stereotyped,
        "answer_logits": letter_logits,
        "context_condition": item.get("context_condition", ""),
        "category": item.get("category", ""),
        "stereotyped_groups": item.get("stereotyped_groups", []),
        "item_idx": item.get("item_idx", -1),
    }


def run_extraction(
    items: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    get_layer_fn: Any,
    n_layers: int,
    hidden_dim: int,
    device: str,
    output_dir: str | Path,
    max_items: int | None = None,
) -> dict[str, int]:
    """Extract all-layer last-token hidden states for a list of items.

    Saves one .npz per item. Resume-safe: skips items with existing output.
    """
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    if max_items is not None:
        items = items[:max_items]

    n_items = len(items)
    progress = ProgressLogger(n_items, prefix="[extract]")
    n_extracted = 0
    n_skipped = 0

    # Segment counters
    counts = {"stereotyped": 0, "non_stereotyped": 0, "unknown_selected": 0}

    for item in items:
        idx = item.get("item_idx", 0)
        out_path = output_dir / f"item_{idx:04d}.npz"

        if out_path.exists():
            progress.skip()
            n_skipped += 1
            continue

        result = extract_item(
            item, model, tokenizer, get_layer_fn, n_layers, hidden_dim, device,
        )

        # Validation on first extracted item
        if n_extracted == 0:
            hs = result["hidden_states"]
            log(f"  Validation: hidden_states shape = {hs.shape}, "
                f"expected ({n_layers}, {hidden_dim})")
            norms_check = np.linalg.norm(hs.astype(np.float32), axis=1)
            log(f"  Norm range after normalisation: [{norms_check.min():.4f}, {norms_check.max():.4f}]")
            log(f"  Model answer: {result['model_answer']} "
                f"(role={result['model_answer_role']}, correct={result['is_correct']})")
            log(f"  Logits: {result['answer_logits']}")

        # Save atomically
        atomic_save_npz(
            out_path,
            hidden_states=result["hidden_states"],
            hidden_states_raw_norms=result["hidden_states_raw_norms"],
            metadata_json=np.array(
                json.dumps(
                    {
                        "model_answer": result["model_answer"],
                        "model_answer_role": result["model_answer_role"],
                        "is_correct": result["is_correct"],
                        "is_stereotyped_response": result["is_stereotyped_response"],
                        "answer_logits": result["answer_logits"],
                        "context_condition": result["context_condition"],
                        "category": result["category"],
                        "stereotyped_groups": result["stereotyped_groups"],
                        "item_idx": result["item_idx"],
                    },
                    ensure_ascii=False,
                )
            ),
        )

        # Track segments
        role = result["model_answer_role"]
        if role == "stereotyped_target":
            counts["stereotyped"] += 1
        elif role == "non_stereotyped":
            counts["non_stereotyped"] += 1
        else:
            counts["unknown_selected"] += 1

        n_extracted += 1
        if n_extracted % 50 == 0 or n_extracted == 1:
            progress.count = n_skipped + n_extracted
            progress.step(
                f"ans={result['model_answer']} role={result['model_answer_role']}"
            )

        # Memory management
        if device == "mps" and n_extracted % 50 == 0:
            torch.mps.empty_cache()
        elif device.startswith("cuda") and n_extracted % 100 == 0:
            torch.cuda.empty_cache()

    log(f"Extraction complete: {n_extracted} new, {n_skipped} skipped")
    log(f"Segment counts: {counts}")
    return {"n_extracted": n_extracted, "n_skipped": n_skipped, "counts": counts}
