#!/usr/bin/env python3
"""Generalization evaluation: MedQA + MMLU with subgroup-specific steering vectors.

Tests matched/mismatched/no-demographic conditions and bias exacerbation.

Usage
-----
python scripts/evaluate_generalization.py \\
    --model_path models/llama-3.1-8b \\
    --model_id llama-3.1-8b \\
    --device mps \\
    --sae_source fnlp/Llama3_1-8B-Base-LXR-8x \\
    --sae_expansion 8 \\
    --steering_dir results/subgroup_steering/llama-3.1-8b/2026-04-13/ \\
    --medqa_path datasets/medqa/ \\
    --mmlu_path datasets/mmlu/ \\
    --categories so,disability
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import atomic_save_json, ensure_dir
from src.utils.logging import log

try:
    import pandas as pd
except ImportError:
    pd = None


# ---------------------------------------------------------------------------
# Demographic classification — audit-based + regex fallback
# ---------------------------------------------------------------------------

# Pre-validated audit CSV from a prior project maps each MedQA item to
# demographic attributes using careful heuristics.  We use it as ground
# truth and fall back to regex only for items not covered.
_DEFAULT_AUDIT_PATH = PROJECT_ROOT / "data" / "raw" / "medqa_demographic_audit.csv"

# Audit label → BBQ subgroup name mappings
_AUDIT_RACE_MAP: dict[str, str] = {
    "Black/African American": "African American",
    "Asian": "Asian",
    "Hispanic/Latino": "Hispanic",
    "White/Caucasian": "White",
    "Middle Eastern": "Middle Eastern",
    "Native American": "Native American",
    "Amish": "Amish",
    "Ashkenazi Jewish": "Jewish",
}
_AUDIT_DISABILITY_MAP: dict[str, str] = {
    "blind_deaf": "blind",
    "intellectual": "autistic",
    "mental_health": "mentally-ill",
    "wheelchair": "disabled",
}
_AUDIT_RELIGION_MAP: dict[str, str] = {
    "Hindu": "Hindu",
    "Christian": "Christian",
    "Jewish_religious": "Jewish",
    "Jehovah_Witness": "Christian",
    "religious_general": None,
}


def load_medqa_audit(
    audit_path: Path | None = None, split: str | None = None,
) -> dict[str, list[str]]:
    """Load the pre-validated demographic audit CSV.

    Returns dict keyed by question-text prefix (first 100 chars, lowered)
    → list of BBQ subgroup names.  Using text prefix as key avoids
    reliance on positional indices that depend on file load order.

    Parameters
    ----------
    split : str | None
        If given, restrict to that split.  If ``None``, load all splits
        (recommended — gives 11 k items instead of 1.3 k).
    """
    import ast as _ast

    path = audit_path or _DEFAULT_AUDIT_PATH
    if not path.exists():
        log(f"  Audit CSV not found at {path}; will use regex fallback")
        return {}

    import pandas as _pd
    df = _pd.read_csv(path)
    if split is not None:
        df = df[df["split"] == split]

    index: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        # Key by question preview prefix (lowered, stripped) — unique enough
        preview = str(row.get("question_preview", "")).strip().lower()[:100]
        if not preview:
            continue
        subgroups: list[str] = []

        # Sexual orientation
        so_raw = str(row.get("so_cues", ""))
        if "gay" in so_raw:
            subgroups.append("gay")
        if "bisexual" in so_raw:
            subgroups.append("bisexual")
        if "partner_same_sex" in so_raw:
            subgroups.append("gay")

        # Gender identity
        if row.get("has_gender_identity_cue"):
            subgroups.append("trans")

        # Race — values may be bare strings ("Black/African American")
        # or list literals (["Black/African American"])
        races_raw = str(row.get("races", ""))
        if races_raw not in ("", "nan", "[]"):
            # Try parsing as list literal first; fall back to treating as
            # pipe-delimited bare string.
            race_tokens: list[str] = []
            try:
                parsed = _ast.literal_eval(races_raw)
                if isinstance(parsed, list):
                    for r in parsed:
                        race_tokens.extend(str(r).split("|"))
                else:
                    race_tokens = str(parsed).split("|")
            except (ValueError, SyntaxError):
                race_tokens = races_raw.split("|")

            for token in race_tokens:
                mapped = _AUDIT_RACE_MAP.get(token.strip())
                if mapped:
                    subgroups.append(mapped)

        # Religion
        relig_raw = str(row.get("religion_cues", ""))
        if relig_raw not in ("", "nan", "[]"):
            for token in relig_raw.split("|"):
                mapped = _AUDIT_RELIGION_MAP.get(token.strip())
                if mapped:
                    subgroups.append(mapped)

        # Disability
        dis_raw = str(row.get("disability_cues", ""))
        if dis_raw not in ("", "nan", "[]"):
            for token in dis_raw.split("|"):
                mapped = _AUDIT_DISABILITY_MAP.get(token.strip())
                if mapped:
                    subgroups.append(mapped)

        # Pregnancy / physical appearance
        preg_raw = str(row.get("pregnancy_cues", ""))
        if "pregnant" in preg_raw:
            subgroups.append("pregnant")

        index[preview] = sorted(set(subgroups))

    n_with = sum(1 for v in index.values() if v)
    log(f"  Loaded MedQA audit: {len(index)} items, {n_with} with subgroup labels")
    return index


# Regex fallback for items not in audit (e.g. different split or dataset).
# Tuned for MedQA: avoids "man/woman" (routine), "blind" (study design),
# "trans" (pharmacology), "black" (clinical findings).
SUBGROUP_PATTERNS: dict[str, list[str]] = {
    "gay": [r"\bgay\b", r"\bhomosexual\b"],
    "lesbian": [r"\blesbian\b"],
    "bisexual": [r"\bbisexual\b"],
    "pansexual": [r"\bpansexual\b"],
    "trans": [r"\btransgender\b", r"\btranssexual\b", r"\bgender dysphoria\b"],
    "African American": [r"\bafrican.american\b"],
    "Asian": [r"\basian\b"],
    "Hispanic": [r"\bhispanic\b", r"\blatino\b", r"\blatina\b"],
    "Muslim": [r"\bmuslim\b", r"\bislam\b"],
    "Jewish": [r"\bjewish\b", r"\bashkenazi\b"],
    "Christian": [r"\bchristian\b"],
    "Hindu": [r"\bhindu\b"],
    "disabled": [r"\bphysically disabled\b", r"\bwheelchair\b"],
    "mentally-ill": [r"\bmental illness\b", r"\bschizophren\w+\b", r"\bbipolar disorder\b"],
    "autistic": [r"\bautis\w+\b", r"\basperger\b"],
    "blind": [r"\blegally blind\b", r"\bvisually impaired\b"],
    "obese": [r"\bobese\b", r"\bobesity\b"],
    "pregnant": [r"\bpregnant\b", r"\bpregnancy\b", r"\bgestational\b"],
}


def classify_demographic(
    text: str, subgroups: list[str],
) -> list[str]:
    """Regex fallback: return which subgroups are mentioned in text."""
    t = text.lower()
    matched = []
    for sub in subgroups:
        patterns = SUBGROUP_PATTERNS.get(sub, [])
        for pat in patterns:
            if re.search(pat, t):
                matched.append(sub)
                break
    return matched


# ---------------------------------------------------------------------------
# Evaluation core
# ---------------------------------------------------------------------------


def evaluate_items(
    steerer: Any,
    items: list[dict[str, Any]],
    steering_vec: torch.Tensor,
    letters: tuple[str, ...] = ("A", "B", "C", "D"),
) -> list[dict[str, Any]]:
    """Run baseline + steered evaluation on all items.

    Returns list of per-item result dicts.
    """
    results = []
    for i, item in enumerate(items):
        prompt = item.get("prompt", "")
        correct = item.get("answer", "")

        baseline = steerer.evaluate_baseline_mcq(prompt, letters=letters)
        steered = steerer.steer_and_evaluate(prompt, steering_vec, letters=letters)

        b_correct = int(baseline["model_answer"] == correct)
        s_correct = int(steered["model_answer"] == correct)
        flipped = baseline["model_answer"] != steered["model_answer"]

        results.append({
            "item_idx": item.get("item_idx", i),
            "correct": correct,
            "baseline_answer": baseline["model_answer"],
            "steered_answer": steered["model_answer"],
            "baseline_correct": b_correct,
            "steered_correct": s_correct,
            "flipped": int(flipped),
            "subject": item.get("subject", ""),
            "demographic_subgroups": item.get("demographic_subgroups", []),
        })

        if (i + 1) % 50 == 0:
            log(f"      [{i + 1}/{len(items)}]")

    return results


def _summarise(results: list[dict]) -> dict[str, Any]:
    """Compute accuracy metrics from per-item results."""
    n = len(results)
    if n == 0:
        return {"n": 0}
    b = sum(r["baseline_correct"] for r in results)
    s = sum(r["steered_correct"] for r in results)
    f = sum(r["flipped"] for r in results)
    return {
        "n": n,
        "accuracy_baseline": round(b / n, 4),
        "accuracy_steered": round(s / n, 4),
        "delta": round((s - b) / n, 4),
        "flip_rate": round(f / n, 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generalization evaluation with subgroup-specific steering"
    )
    p.add_argument("--model_path", required=True)
    p.add_argument("--model_id", default="llama-3.1-8b")
    p.add_argument("--device", default="mps")

    p.add_argument("--sae_source", required=True)
    p.add_argument("--sae_expansion", type=int, default=8)

    p.add_argument("--steering_dir", required=True,
                   help="Dir from run_subgroup_steering.py with steering_vectors/ and manifests")
    p.add_argument("--medqa_path", default=None)
    p.add_argument("--mmlu_path", default=None)
    p.add_argument("--categories", default=None)
    p.add_argument("--max_items", type=int, default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--skip_figures", action="store_true")
    p.add_argument("--exacerbation", action="store_true",
                   help="Also test with flipped alpha (bias amplification)")

    return p.parse_args()


def main() -> None:
    t0 = time.time()
    args = parse_args()

    steering_dir = Path(args.steering_dir)
    vec_dir = steering_dir / "steering_vectors"
    if not vec_dir.is_dir():
        log(f"ERROR: no steering_vectors/ dir at {vec_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "results" / "generalization"
        / args.model_id / date.today().isoformat()
    )
    ensure_dir(output_dir)

    # Load manifests
    manifest_path = steering_dir / "steering_manifests.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifests = json.load(f)
    else:
        manifests = []
    log(f"Loaded {len(manifests)} steering manifests")

    # Load steering vectors
    vectors: dict[str, dict[str, Any]] = {}  # "cat_sub" → {vector, layer, alpha}
    for npz_path in sorted(vec_dir.glob("*.npz")):
        data = np.load(npz_path, allow_pickle=True)
        key = npz_path.stem  # e.g. "so_gay"
        vectors[key] = {
            "vector": torch.from_numpy(data["vector"]),
            "injection_layer": int(data["injection_layer"]),
            "alpha": float(data["alpha"]),
            "k": int(data["k"]),
        }
    log(f"Loaded {len(vectors)} steering vectors")

    if args.categories:
        requested = [c.strip() for c in args.categories.split(",")]
        vectors = {k: v for k, v in vectors.items()
                   if any(k.startswith(c + "_") for c in requested)}

    # Load model
    log("Loading model ...")
    from src.models.wrapper import ModelWrapper
    wrapper = ModelWrapper.from_pretrained(args.model_path, device=args.device)

    # Load SAEs needed for steerer
    from src.sae_localization.sae_wrapper import SAEWrapper
    from src.sae_localization.steering import SAESteerer

    needed_layers = set(v["injection_layer"] for v in vectors.values())
    sae_cache: dict[int, SAEWrapper] = {}
    for layer in sorted(needed_layers):
        sae_cache[layer] = SAEWrapper(
            args.sae_source, layer=layer,
            expansion=args.sae_expansion, device=args.device,
        )

    # ---- Load evaluation datasets ----
    medqa_items = None
    mmlu_items = None

    if args.medqa_path:
        from src.data.medqa_loader import load_medqa_items
        medqa_items = load_medqa_items(args.medqa_path, max_items=args.max_items)
        log(f"Loaded {len(medqa_items)} MedQA items")

    if args.mmlu_path:
        from src.data.mmlu_loader import load_mmlu_items
        mmlu_items = load_mmlu_items(args.mmlu_path, max_items=args.max_items)
        log(f"Loaded {len(mmlu_items)} MMLU items")

    # Classify MedQA items by demographic content.
    # Prefer the pre-validated audit CSV; fall back to regex for items not covered.
    all_subgroups = list(SUBGROUP_PATTERNS.keys())
    audit_index = load_medqa_audit()  # returns {} if file not found
    n_from_audit = 0
    n_from_regex = 0

    if medqa_items:
        for i, item in enumerate(medqa_items):
            # Match to audit by question text prefix (first 100 chars, lowered)
            text = item.get("prompt", "") or item.get("question", "")
            # The prompt starts with "Question: ...", extract the question part
            q_text = text
            if q_text.startswith("Question:"):
                q_text = q_text[len("Question:"):].strip()
            key = q_text.strip().lower()[:100]

            if key in audit_index:
                item["demographic_subgroups"] = audit_index[key]
                n_from_audit += 1
            else:
                item["demographic_subgroups"] = classify_demographic(text, all_subgroups)
                n_from_regex += 1
            item["mentions_demographic"] = bool(item["demographic_subgroups"])

        n_demo = sum(1 for it in medqa_items if it["mentions_demographic"])
        log(f"  Demographic classification: {n_from_audit} from audit, "
            f"{n_from_regex} from regex, {n_demo} total with labels")

    # ---- Run evaluations per steering vector ----
    all_medqa: dict[str, dict[str, Any]] = {}
    all_mmlu: dict[str, dict[str, Any]] = {}

    for vec_key, vec_data in sorted(vectors.items()):
        parts = vec_key.split("_", 1)
        cat = parts[0] if len(parts) > 1 else vec_key
        sub = parts[1] if len(parts) > 1 else vec_key

        layer = vec_data["injection_layer"]
        sae = sae_cache.get(layer, next(iter(sae_cache.values())))
        steerer = SAESteerer(wrapper, sae, layer)

        vec = vec_data["vector"].to(dtype=wrapper.model.dtype, device=args.device)

        log(f"\n--- {vec_key} (layer={layer}, alpha={vec_data['alpha']}) ---")

        # ---- MedQA ----
        if medqa_items:
            medqa_out = ensure_dir(output_dir / "medqa")

            # Matched: items mentioning this subgroup
            matched = [it for it in medqa_items if sub in it.get("demographic_subgroups", [])]

            # Mismatched: items mentioning a different subgroup from same category
            # Find all subgroups for this category from vectors
            cat_subs = [k.split("_", 1)[1] for k in vectors if k.startswith(cat + "_")]
            mismatched = [
                it for it in medqa_items
                if any(s in it.get("demographic_subgroups", []) for s in cat_subs if s != sub)
                and sub not in it.get("demographic_subgroups", [])
            ]

            # No-demographic
            no_demo = [it for it in medqa_items if not it.get("mentions_demographic")]

            entry: dict[str, Any] = {"vector": vec_key, "category": cat, "subgroup": sub}

            if matched:
                log(f"    MedQA matched ({sub}): {len(matched)} items")
                res = evaluate_items(steerer, matched, vec, letters=("A", "B", "C", "D", "E"))
                entry["matched"] = _summarise(res)
            if mismatched:
                log(f"    MedQA mismatched: {len(mismatched)} items")
                res = evaluate_items(steerer, mismatched, vec, letters=("A", "B", "C", "D", "E"))
                entry["mismatched"] = _summarise(res)
            if no_demo:
                log(f"    MedQA no-demo: {len(no_demo[:200])} items")
                res = evaluate_items(steerer, no_demo[:200], vec, letters=("A", "B", "C", "D", "E"))
                entry["no_demographic"] = _summarise(res)

            # Exacerbation: flip alpha sign
            if args.exacerbation:
                neg_vec = -vec
                if matched:
                    log(f"    MedQA exacerbation (matched): {len(matched)} items")
                    res_ex = evaluate_items(steerer, matched, neg_vec, letters=("A", "B", "C", "D", "E"))
                    entry["exacerbation_matched"] = _summarise(res_ex)

            all_medqa[vec_key] = entry
            atomic_save_json(entry, medqa_out / f"{vec_key}.json")

        # ---- MMLU ----
        if mmlu_items:
            mmlu_out = ensure_dir(output_dir / "mmlu")
            log(f"    MMLU: {len(mmlu_items[:200])} items")
            res = evaluate_items(steerer, mmlu_items[:200], vec)
            mmlu_entry = _summarise(res)
            mmlu_entry["vector"] = vec_key

            # Per-subject breakdown
            per_subj: dict[str, dict[str, int]] = {}
            for r in res:
                s = r.get("subject", "other")
                per_subj.setdefault(s, {"b": 0, "s": 0, "n": 0, "f": 0})
                per_subj[s]["b"] += r["baseline_correct"]
                per_subj[s]["s"] += r["steered_correct"]
                per_subj[s]["n"] += 1
                per_subj[s]["f"] += r["flipped"]
            mmlu_entry["per_subject"] = {
                s: {
                    "accuracy_baseline": d["b"] / max(d["n"], 1),
                    "accuracy_steered": d["s"] / max(d["n"], 1),
                    "delta": (d["s"] - d["b"]) / max(d["n"], 1),
                    "flip_rate": d["f"] / max(d["n"], 1),
                    "n": d["n"],
                }
                for s, d in per_subj.items()
            }

            all_mmlu[vec_key] = mmlu_entry
            atomic_save_json(mmlu_entry, mmlu_out / f"{vec_key}.json")

        # Memory cleanup
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    # Save aggregated results
    if all_medqa:
        atomic_save_json(all_medqa, output_dir / "medqa_steering_results.json")
    if all_mmlu:
        atomic_save_json(all_mmlu, output_dir / "mmlu_steering_results.json")

    # Update manifests with generalization results
    for m in manifests:
        key = f"{m['category']}_{m['subgroup']}"
        if key in all_medqa:
            md = all_medqa[key]
            m["medqa_matched_accuracy_delta"] = md.get("matched", {}).get("delta")
            m["medqa_mismatched_accuracy_delta"] = md.get("mismatched", {}).get("delta")
            m["medqa_nodemo_accuracy_delta"] = md.get("no_demographic", {}).get("delta")
        if key in all_mmlu:
            m["mmlu_accuracy_delta"] = all_mmlu[key].get("delta")

    atomic_save_json(manifests, output_dir / "steering_manifests_with_gen.json")

    total = time.time() - t0
    log(f"\nComplete in {total:.1f}s")
    log(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
