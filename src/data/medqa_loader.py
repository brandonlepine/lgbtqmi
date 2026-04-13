"""MedQA loader utilities.

Goal: support the RunPod workflow where MedQA is downloaded via `hf download`
into a local directory (often parquet shards), and then consumed by Stage 3
steering (`scripts/run_sae_steering.py --experiments E --medqa_path ...`).

Supported local formats:
- Parquet shards with either:
  - columns: question, choices, answer  (choices is list-like; answer is index or letter)
  - columns: question, A, B, C, D, (optional E), answer or answer_idx
- JSONL with the same schemas

Output items are normalized to:
  { "prompt": str, "answer": "A"|...|"E", "letters": tuple[str,...], "mentions_demographic": bool }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def _normalize_answer(ans: Any) -> str:
    if ans is None:
        return ""
    if isinstance(ans, int):
        return {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}.get(ans, "")
    s = str(ans).strip()
    if s in {"0", "1", "2", "3", "4"}:
        return {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}.get(int(s), "")
    s = s.upper()
    return s if s in {"A", "B", "C", "D", "E"} else ""


def _format_medqa_prompt(question: str, options: dict[str, str]) -> str:
    """Minimal stable formatting; model answers with a single letter."""
    lines = [str(question).strip()]
    for k in sorted(options.keys()):
        lines.append(f"{k}. {options[k]}")
    lines.append("Answer:")
    return "\n".join(lines)


def _mentions_demographic(text: str) -> bool:
    # Conservative heuristic; used only for reporting side-effects.
    t = text.lower()
    keywords = [
        "trans", "transgender", "gay", "lesbian", "bisexual", "pansexual",
        "black", "white", "asian", "latino", "hispanic", "arab",
        "muslim", "jewish", "christian", "catholic",
        "disabled", "autistic", "blind", "deaf",
        "old", "elderly", "pregnant",
    ]
    return any(k in t for k in keywords)


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _iter_medqa_from_objects(objs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for obj in objs:
        q = obj.get("question", "") or obj.get("prompt", "")
        if not q:
            continue

        options: dict[str, str] = {}
        if "options" in obj and isinstance(obj["options"], dict) and obj["options"]:
            # Common MedQA JSONL schema: {"question": ..., "options": {"A":..., ...}, "answer_idx": "E"}
            options = {str(k).strip().upper(): str(v) for k, v in obj["options"].items()}
        elif "choices" in obj and isinstance(obj["choices"], (list, tuple)) and len(obj["choices"]) >= 4:
            # HF-style: choices is list-like, answer is index or letter.
            choices = [str(x) for x in obj["choices"]]
            letters = ["A", "B", "C", "D", "E"]
            options = {letters[i]: choices[i] for i in range(min(len(choices), len(letters)))}
        else:
            for k in ["A", "B", "C", "D", "E"]:
                v = obj.get(k, "")
                if v:
                    options[k] = str(v)

        ans = _normalize_answer(obj.get("answer", obj.get("label", "")))
        if not ans:
            # Some datasets use answer_idx
            ans = _normalize_answer(obj.get("answer_idx", None))
        # Require at least A-D; E is optional.
        if not (options.get("A") and options.get("B") and options.get("C") and options.get("D") and ans):
            continue
        if ans not in options:
            # If answer points to a missing option key, skip (schema mismatch).
            continue

        prompt = _format_medqa_prompt(str(q), options)
        letters = tuple(sorted(options.keys()))
        items.append(
            {
                "prompt": prompt,
                "answer": ans,
                "letters": letters,
                "mentions_demographic": _mentions_demographic(prompt),
            }
        )
    return items


def _load_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Reading MedQA parquet requires pandas + pyarrow. Install with: pip install pandas pyarrow"
        ) from exc

    df = pd.read_parquet(path)
    cols = set(df.columns)
    if not (("question" in cols) or ("prompt" in cols)):
        raise ValueError(f"Unsupported MedQA parquet schema in {path}; columns={sorted(cols)}")

    objs = df.to_dict(orient="records")
    return _iter_medqa_from_objects(objs)


def load_medqa_items(path: str | Path, *, max_items: int | None = None) -> list[dict[str, Any]]:
    p = Path(path)
    items: list[dict[str, Any]] = []

    if p.is_file():
        if p.suffix.lower() == ".parquet":
            items.extend(_load_parquet(p))
        elif p.suffix.lower() == ".jsonl":
            items.extend(_iter_medqa_from_objects(_iter_jsonl(p)))
        elif p.suffix.lower() == ".json":
            data = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                raise ValueError(f"Unsupported JSON structure in {p} (expected list)")
            items.extend(_iter_medqa_from_objects(data))
        else:
            raise ValueError(f"Unsupported MedQA file type: {p.suffix}")
    elif p.is_dir():
        parquet_files = sorted(p.rglob("*.parquet"))
        jsonl_files = sorted(p.rglob("*.jsonl"))
        json_files = sorted(p.rglob("*.json"))

        for f in parquet_files:
            items.extend(_load_parquet(f))
            if max_items is not None and len(items) >= max_items:
                break
        if (max_items is None) or (len(items) < max_items):
            # Prefer canonical split files when present (avoid aux jsonl like metamap phrases).
            preferred = []
            for name in ["test.jsonl", "dev.jsonl", "train.jsonl", "US_qbank.jsonl"]:
                fp = p / name
                if fp.exists() and fp.is_file():
                    preferred.append(fp)
            preferred_set = {x.resolve() for x in preferred}
            ordered_jsonl = preferred + [f for f in jsonl_files if f.resolve() not in preferred_set]

            for f in ordered_jsonl:
                items.extend(_iter_medqa_from_objects(_iter_jsonl(f)))
                if max_items is not None and len(items) >= max_items:
                    break
        if (max_items is None) or (len(items) < max_items):
            for f in json_files:
                data = json.loads(f.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    items.extend(_iter_medqa_from_objects(data))
                if max_items is not None and len(items) >= max_items:
                    break
    else:
        raise FileNotFoundError(f"MedQA path not found: {p}")

    if max_items is not None:
        items = items[:max_items]
    return items

