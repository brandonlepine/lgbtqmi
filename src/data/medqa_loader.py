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
  {
    "prompt": str,
    "answer": "A"|...|"E",
    "letters": tuple[str,...],
    "demographic_tags": list[str],  # tags aligned to BBQ subgroup names when possible
    "mentions_demographic": bool,   # bool(demographic_tags)
  }
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


def _demographic_tags(text: str, *, mode: str = "narrow") -> list[str]:
    """Extract demographic tags from a prompt.

    - mode="narrow": identity-category mentions (religion, SO/GI, race/ethnicity, etc.)
    - mode="broad": also includes age + binary sex terms, which will tag many MedQA items.

    Tags are chosen to align with BBQ subgroup names where possible (e.g., "F", "M", "old", "nonOld",
    "Muslim", "Jewish", "gay", "lesbian", ...).
    """
    import re

    t = text.lower()
    tags: list[str] = []

    def has(pat: str) -> bool:
        return re.search(pat, t) is not None

    # Sexual orientation (BBQ SO subgroups are lower-case)
    if has(r"\b(bisexual)\b"):
        tags.append("bisexual")
    if has(r"\b(gay)\b"):
        tags.append("gay")
    if has(r"\b(lesbian)\b"):
        tags.append("lesbian")
    if has(r"\b(pansexual)\b"):
        tags.append("pansexual")
    if has(r"\b(straight|heterosexual)\b"):
        tags.append("straight")

    # Gender identity (mix of subgroup labels appears in BBQ)
    if has(r"\b(transgender women|trans woman|transgender woman)\b"):
        tags.append("Transgender women")
    if has(r"\b(transgender men|trans man|transgender man)\b"):
        tags.append("transgender men")
    if has(r"\b(transgender|nonbinary|non-binary|cisgender)\b"):
        tags.append("trans")

    # Religion (capitalize to match BBQ subgroup labels)
    if has(r"\b(atheist|atheism)\b"):
        tags.append("Atheist")
    if has(r"\b(catholic)\b"):
        tags.append("Catholic")
    if has(r"\b(christian)\b"):
        tags.append("Christian")
    if has(r"\b(jewish)\b"):
        tags.append("Jewish")
    if has(r"\b(mormon)\b"):
        tags.append("Mormon")
    if has(r"\b(muslim|islam)\b"):
        tags.append("Muslim")
    if has(r"\b(orthodox)\b"):
        tags.append("Orthodox")
    if has(r"\b(hindu)\b"):
        tags.append("Hindu")

    # Race/ethnicity (prefer multi-word labels)
    if has(r"\b(african american)\b"):
        tags.append("African American")
    if has(r"\b(native american)\b"):
        tags.append("Native American")
    if has(r"\b(middle eastern)\b"):
        tags.append("Middle Eastern")
    if has(r"\b(latino)\b"):
        tags.append("Latino")
    if has(r"\b(hispanic)\b"):
        tags.append("Hispanic")
    if has(r"\b(arab)\b"):
        tags.append("Arab")

    # Disability/physical difference
    if has(r"\b(autistic)\b"):
        tags.append("autistic people")
    if has(r"\b(blind)\b"):
        tags.append("people with blindness or low-vision")
    if has(r"\b(deaf)\b"):
        tags.append("D/deaf")
    if has(r"\b(down['’]s syndrome)\b"):
        tags.append("Down's syndrome")
    if has(r"\b(wheelchair)\b"):
        tags.append("physically disabled")

    if mode == "broad":
        # Binary sex markers -> GI subgroups F/M
        if has(r"\b(female|woman|girl)\b"):
            tags.append("F")
        if has(r"\b(male|man|boy)\b"):
            tags.append("M")

        # Age markers -> age subgroups old/nonOld
        m = re.search(r"\b(\d{1,3})-year-old\b", t)
        if m:
            try:
                age = int(m.group(1))
                tags.append("old" if age >= 60 else "nonOld")
            except Exception:
                pass
        if has(r"\b(elderly|geriatric)\b"):
            tags.append("old")

    # Deduplicate while keeping order
    seen: set[str] = set()
    out: list[str] = []
    for x in tags:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _iter_medqa_from_objects(
    objs: Iterable[dict[str, Any]],
    *,
    demographic_mode: str = "narrow",
) -> list[dict[str, Any]]:
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
        demo_tags = _demographic_tags(prompt, mode=demographic_mode)
        items.append(
            {
                "prompt": prompt,
                "answer": ans,
                "letters": letters,
                "demographic_tags": demo_tags,
                "mentions_demographic": bool(demo_tags),
            }
        )
    return items


def _load_parquet(path: Path, *, demographic_mode: str = "narrow") -> list[dict[str, Any]]:
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
    return _iter_medqa_from_objects(objs, demographic_mode=demographic_mode)


def load_medqa_items(
    path: str | Path,
    *,
    max_items: int | None = None,
    demographic_mode: str = "narrow",
) -> list[dict[str, Any]]:
    p = Path(path)
    items: list[dict[str, Any]] = []

    if p.is_file():
        if p.suffix.lower() == ".parquet":
            items.extend(_load_parquet(p, demographic_mode=demographic_mode))
        elif p.suffix.lower() == ".jsonl":
            items.extend(_iter_medqa_from_objects(_iter_jsonl(p), demographic_mode=demographic_mode))
        elif p.suffix.lower() == ".json":
            data = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                raise ValueError(f"Unsupported JSON structure in {p} (expected list)")
            items.extend(_iter_medqa_from_objects(data, demographic_mode=demographic_mode))
        else:
            raise ValueError(f"Unsupported MedQA file type: {p.suffix}")
    elif p.is_dir():
        parquet_files = sorted(p.rglob("*.parquet"))
        jsonl_files = sorted(p.rglob("*.jsonl"))
        json_files = sorted(p.rglob("*.json"))

        for f in parquet_files:
            items.extend(_load_parquet(f, demographic_mode=demographic_mode))
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
                items.extend(_iter_medqa_from_objects(_iter_jsonl(f), demographic_mode=demographic_mode))
                if max_items is not None and len(items) >= max_items:
                    break
        if (max_items is None) or (len(items) < max_items):
            for f in json_files:
                data = json.loads(f.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    items.extend(_iter_medqa_from_objects(data, demographic_mode=demographic_mode))
                if max_items is not None and len(items) >= max_items:
                    break
    else:
        raise FileNotFoundError(f"MedQA path not found: {p}")

    if max_items is not None:
        items = items[:max_items]
    return items

