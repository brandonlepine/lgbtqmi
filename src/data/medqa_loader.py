"""MedQA loader utilities.

Goal: support the RunPod workflow where MedQA is downloaded via `hf download`
into a local directory (often parquet shards), and then consumed by Stage 3
steering (`scripts/run_sae_steering.py --experiments E --medqa_path ...`).

Supported local formats:
- Parquet shards with either:
  - columns: question, choices, answer  (choices is list-like; answer is index or letter)
  - columns: question, A, B, C, D, answer
- JSONL with the same schemas

Output items are normalized to:
  { "prompt": str, "answer": "A"|"B"|"C"|"D", "mentions_demographic": bool }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def _normalize_answer(ans: Any) -> str:
    if ans is None:
        return ""
    if isinstance(ans, int):
        return {0: "A", 1: "B", 2: "C", 3: "D"}.get(ans, "")
    s = str(ans).strip()
    if s in {"0", "1", "2", "3"}:
        return {0: "A", 1: "B", 2: "C", 3: "D"}.get(int(s), "")
    s = s.upper()
    return s if s in {"A", "B", "C", "D"} else ""


def _format_medqa_prompt(question: str, A: str, B: str, C: str, D: str) -> str:
    # Minimal stable formatting; model answers with a single letter.
    return (
        f"{question}\n"
        f"A. {A}\n"
        f"B. {B}\n"
        f"C. {C}\n"
        f"D. {D}\n"
        "Answer:"
    )


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

        if "choices" in obj and isinstance(obj["choices"], (list, tuple)) and len(obj["choices"]) >= 4:
            A, B, C, D = obj["choices"][:4]
        else:
            A, B, C, D = obj.get("A", ""), obj.get("B", ""), obj.get("C", ""), obj.get("D", "")

        ans = _normalize_answer(obj.get("answer", obj.get("label", "")))
        if not ans:
            # Some datasets use answer_idx
            ans = _normalize_answer(obj.get("answer_idx", None))
        if not (A and B and C and D and ans):
            continue

        prompt = _format_medqa_prompt(str(q), str(A), str(B), str(C), str(D))
        items.append(
            {
                "prompt": prompt,
                "answer": ans,
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
            for f in jsonl_files:
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

