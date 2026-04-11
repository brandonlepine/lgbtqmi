"""Answer-letter extraction utilities (BBQ A/B/C)."""

from __future__ import annotations

from typing import Any

import torch


def _single_token_ids(tokenizer: Any, s: str) -> list[int]:
    ids = tokenizer.encode(s, add_special_tokens=False)
    if len(ids) == 1:
        return [int(ids[0])]
    return []


def letter_token_ids(tokenizer: Any, letter: str) -> list[int]:
    """Return token ids to consider for a letter ('A','B','C').

    Many tokenizers represent a standalone choice as either "A" or " A".
    We consider both when they are single tokens.
    """
    letter = letter.upper().strip()
    ids = set(_single_token_ids(tokenizer, letter))
    ids |= set(_single_token_ids(tokenizer, f" {letter}"))
    return sorted(ids)


def best_choice_from_logits(
    logits: torch.Tensor,
    tokenizer: Any,
    letters: tuple[str, ...] = ("A", "B", "C"),
) -> tuple[str, dict[str, float]]:
    """Pick the best A/B/C letter from vocab logits at the decision position."""
    if logits.dim() != 1:
        raise ValueError(f"Expected 1D vocab logits, got shape {tuple(logits.shape)}")

    best_letter = ""
    best_val = -float("inf")
    per_letter: dict[str, float] = {}

    for L in letters:
        ids = letter_token_ids(tokenizer, L)
        if not ids:
            per_letter[L] = -float("inf")
            continue
        vals = [float(logits[i].detach().cpu()) for i in ids]
        v = max(vals)
        per_letter[L] = v
        if v > best_val:
            best_val = v
            best_letter = L

    return best_letter, per_letter

