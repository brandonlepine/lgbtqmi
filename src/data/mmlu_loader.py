"""MMLU loader utilities.

Supports common local formats:
- Directory of per-subject CSV files (often named ``*_test.csv``) with either:
  - header row: question,A,B,C,D,answer
  - or no header: 6 columns in that same order
- JSONL files containing either:
  - {"question":..., "choices":[A,B,C,D], "answer":0-3, "subject":...}
  - or {"question":..., "A":..., "B":..., "C":..., "D":..., "answer":"A"/...}
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


def _normalize_answer(ans: Any) -> str:
    """Normalize answer to an A/B/C/D letter."""
    if ans is None:
        return ""
    if isinstance(ans, int):
        return {0: "A", 1: "B", 2: "C", 3: "D"}.get(ans, "")
    s = str(ans).strip()
    if s in {"0", "1", "2", "3"}:
        return {0: "A", 1: "B", 2: "C", 3: "D"}.get(int(s), "")
    s = s.upper()
    return s if s in {"A", "B", "C", "D"} else ""


def _iter_csv_items(path: Path, subject: str) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return []

    # Detect header
    header = [c.strip().lower() for c in rows[0]]
    has_header = "question" in header and "answer" in header
    start = 1 if has_header else 0

    items: list[dict[str, Any]] = []
    for r in rows[start:]:
        if len(r) < 6:
            continue
        q, a, b, c, d, ans = r[:6]
        ans_l = _normalize_answer(ans)
        if not ans_l:
            continue
        items.append(
            {
                "subject": subject,
                "few_shot_text": "",
                "question": q,
                "A": a,
                "B": b,
                "C": c,
                "D": d,
                "answer": ans_l,
            }
        )
    return items


def _iter_jsonl_items(path: Path, subject: str | None) -> Iterable[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            subj = subject or obj.get("subject", "other")
            q = obj.get("question", "")
            if "choices" in obj and isinstance(obj["choices"], list) and len(obj["choices"]) >= 4:
                A, B, C, D = obj["choices"][:4]
            else:
                A, B, C, D = obj.get("A", ""), obj.get("B", ""), obj.get("C", ""), obj.get("D", "")
            ans_l = _normalize_answer(obj.get("answer", ""))
            if not q or not ans_l:
                continue
            items.append(
                {
                    "subject": subj,
                    "few_shot_text": obj.get("few_shot_text", ""),
                    "question": q,
                    "A": A,
                    "B": B,
                    "C": C,
                    "D": D,
                    "answer": ans_l,
                }
            )
    return items


def load_mmlu_items(path: str | Path, *, max_items: int | None = None) -> list[dict[str, Any]]:
    """Load MMLU items from a directory or file."""
    p = Path(path)
    items: list[dict[str, Any]] = []

    if p.is_file():
        if p.suffix.lower() == ".csv":
            subject = p.stem.replace("_test", "").replace("_dev", "")
            items.extend(_iter_csv_items(p, subject))
        elif p.suffix.lower() in {".jsonl", ".json"}:
            if p.suffix.lower() == ".json":
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    tmp = []
                    for obj in data:
                        tmp.append(json.dumps(obj))
                    # Treat as jsonl in-memory
                    for line in tmp:
                        obj = json.loads(line)
                        subj = obj.get("subject", "other")
                        q = obj.get("question", "")
                        if "choices" in obj and isinstance(obj["choices"], list) and len(obj["choices"]) >= 4:
                            A, B, C, D = obj["choices"][:4]
                        else:
                            A, B, C, D = obj.get("A", ""), obj.get("B", ""), obj.get("C", ""), obj.get("D", "")
                        ans_l = _normalize_answer(obj.get("answer", ""))
                        if not q or not ans_l:
                            continue
                        items.append(
                            {
                                "subject": subj,
                                "few_shot_text": obj.get("few_shot_text", ""),
                                "question": q,
                                "A": A,
                                "B": B,
                                "C": C,
                                "D": D,
                                "answer": ans_l,
                            }
                        )
                else:
                    raise ValueError(f"Unsupported JSON structure in {p} (expected list)")
            else:
                subject = p.stem
                items.extend(_iter_jsonl_items(p, subject))
        else:
            raise ValueError(f"Unsupported MMLU file type: {p.suffix}")
    elif p.is_dir():
        csv_files = sorted(p.rglob("*_test.csv")) or sorted(p.rglob("*.csv"))
        jsonl_files = sorted(p.rglob("*.jsonl"))
        for f in csv_files:
            subject = f.stem.replace("_test", "").replace("_dev", "")
            items.extend(_iter_csv_items(f, subject))
            if max_items is not None and len(items) >= max_items:
                break
        if (max_items is None) or (len(items) < max_items):
            for f in jsonl_files:
                items.extend(_iter_jsonl_items(f, None))
                if max_items is not None and len(items) >= max_items:
                    break
    else:
        raise FileNotFoundError(f"MMLU path not found: {p}")

    if max_items is not None:
        items = items[:max_items]
    return items

