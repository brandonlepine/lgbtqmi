"""CrowS-Pairs loader and standardization to the repo's stimuli format."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_REQUIRED_COLS = {"sent_more", "sent_less"}
_OPTIONAL_COLS = {"bias_type", "stereo_antistereo", "stereo_type", "id"}


def _read_csv_rows(path: str | Path) -> tuple[list[dict[str, str]], list[str]]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CrowS-Pairs CSV has no header row")
        rows = [r for r in reader]
        return rows, list(reader.fieldnames)


def validate_crows_pairs_csv(path: str | Path) -> dict[str, Any]:
    """Validate file structure and return a schema summary."""
    rows, cols = _read_csv_rows(path)
    missing = sorted(_REQUIRED_COLS - set(cols))
    if missing:
        raise ValueError(
            f"CrowS-Pairs CSV missing required columns: {missing}. "
            f"Found columns: {cols}"
        )

    bias_types = {}
    for r in rows:
        bt = (r.get("bias_type") or "").strip()
        if bt:
            bias_types[bt] = bias_types.get(bt, 0) + 1

    return {
        "path": str(path),
        "n_rows": len(rows),
        "columns": cols,
        "bias_type_counts": bias_types,
    }


def load_crows_pairs_as_stimuli(
    path: str | Path,
    *,
    max_items: int | None = None,
) -> list[dict[str, Any]]:
    """Convert CrowS-Pairs pairs into BBQ-like multiple-choice stimuli.

    We map:
    - A: sent_more  -> stereotyped_target
    - B: sent_less  -> non_stereotyped
    - C: unknown    -> placeholder

    All items are marked context_condition='ambig' for compatibility with bias scoring.
    """
    rows, cols = _read_csv_rows(path)
    if max_items is not None:
        rows = rows[:max_items]

    items: list[dict[str, Any]] = []
    for i, r in enumerate(rows):
        sent_more = (r.get("sent_more") or "").strip()
        sent_less = (r.get("sent_less") or "").strip()
        if not sent_more or not sent_less:
            continue

        bias_type = (r.get("bias_type") or "").strip() or "unknown"
        stereo_type = (r.get("stereo_antistereo") or "").strip() or ""
        rid = (r.get("id") or "").strip()

        items.append(
            {
                "item_idx": i,
                "example_id": rid or i,
                "category": "CrowS-Pairs",
                "context": "",
                "question": "Which sentence is more stereotypical?",
                "answers": {
                    "A": sent_more,
                    "B": sent_less,
                    "C": "Not enough information",
                },
                "correct_letter": "A",  # for bookkeeping only; not used as “gold” here
                "context_condition": "ambig",
                "question_polarity": "nonneg",
                "alignment": "ambiguous",
                "stereotyped_groups": [bias_type],
                "answer_roles": {
                    "A": "stereotyped_target",
                    "B": "non_stereotyped",
                    "C": "unknown",
                },
                # Tags align with answer text (so identity-subspan extraction can work).
                "answer_role_tags": {
                    "A": bias_type,
                    "B": bias_type,
                    "C": "unknown",
                },
                "identity_role_tags": [bias_type],
                "subcategory": stereo_type or "None",
            }
        )

    return items

