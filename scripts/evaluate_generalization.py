#!/usr/bin/env python3
"""Generalization evaluation: MedQA + MMLU with subgroup-specific steering vectors.

Tests matched, within-category mismatched, cross-category mismatched,
and no-demographic conditions.  Runs both debiasing and exacerbation
directions by default.  Produces confidence-aware metrics and per-item
parquet outputs.

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

from src.metrics.bias_metrics import compute_all_metrics, compute_margin
from src.utils.io import atomic_save_json, ensure_dir
from src.utils.logging import log

try:
    import pandas as pd
except ImportError:
    pd = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.visualization.style import (
    BLUE, CATEGORY_COLORS, CATEGORY_LABELS, DPI, GRAY, GREEN, ORANGE,
    WONG_PALETTE, apply_style,
)
apply_style()

VERMILLION = "#D55E00"


def _save_both(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(str(path), dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(str(path.with_suffix(".pdf")), bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Demographic classification — audit-based + regex fallback
# ---------------------------------------------------------------------------

_DEFAULT_AUDIT_PATH = PROJECT_ROOT / "data" / "raw" / "medqa_demographic_audit.csv"

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
    -> list of BBQ subgroup names.
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
        preview = str(row.get("question_preview", "")).strip().lower()[:100]
        if not preview:
            continue
        subgroups: list[str] = []

        so_raw = str(row.get("so_cues", ""))
        if "gay" in so_raw:
            subgroups.append("gay")
        if "bisexual" in so_raw:
            subgroups.append("bisexual")
        if "partner_same_sex" in so_raw:
            subgroups.append("gay")

        if row.get("has_gender_identity_cue"):
            subgroups.append("trans")

        races_raw = str(row.get("races", ""))
        if races_raw not in ("", "nan", "[]"):
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

        relig_raw = str(row.get("religion_cues", ""))
        if relig_raw not in ("", "nan", "[]"):
            for token in relig_raw.split("|"):
                mapped = _AUDIT_RELIGION_MAP.get(token.strip())
                if mapped:
                    subgroups.append(mapped)

        dis_raw = str(row.get("disability_cues", ""))
        if dis_raw not in ("", "nan", "[]"):
            for token in dis_raw.split("|"):
                mapped = _AUDIT_DISABILITY_MAP.get(token.strip())
                if mapped:
                    subgroups.append(mapped)

        preg_raw = str(row.get("pregnancy_cues", ""))
        if "pregnant" in preg_raw:
            subgroups.append("pregnant")

        index[preview] = sorted(set(subgroups))

    n_with = sum(1 for v in index.values() if v)
    log(f"  Loaded MedQA audit: {len(index)} items, {n_with} with subgroup labels")
    return index


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

    Returns list of per-item result dicts with fields needed for
    compute_all_metrics().
    """
    results = []
    for i, item in enumerate(items):
        prompt = item.get("prompt", "")
        correct = item.get("answer", "")
        # Use per-item letters if available, otherwise fall back to caller's default
        item_letters = tuple(item["letters"]) if "letters" in item else letters

        baseline = steerer.evaluate_baseline_mcq(prompt, letters=item_letters)
        steered = steerer.steer_and_evaluate(prompt, steering_vec, letters=item_letters)

        b_ans = baseline["model_answer"]
        s_ans = steered["model_answer"]
        b_correct = int(b_ans == correct)
        s_correct = int(s_ans == correct)
        flipped = b_ans != s_ans

        logits_b = {k: float(v) for k, v in baseline.get("answer_logits", {}).items()}
        logits_s = {k: float(v) for k, v in steered.get("answer_logits", {}).items()}

        margin = compute_margin(logits_b, b_ans) if b_ans in logits_b else 0.0

        # Determine stereotyped option (if available in item)
        stereo_opt = ""
        answer_roles = item.get("answer_roles", {})
        for letter, role in answer_roles.items():
            if role == "stereotyped_target":
                stereo_opt = letter
                break

        # Corrected: baseline was wrong (or stereotyped), steered is correct
        corrected = bool(not b_correct and s_correct)
        corrupted = bool(b_correct and not s_correct)

        results.append({
            "item_idx": item.get("item_idx", i),
            "correct": correct,
            "baseline_answer": b_ans,
            "steered_answer": s_ans,
            "baseline_correct": b_correct,
            "steered_correct": s_correct,
            "flipped": int(flipped),
            "margin": margin,
            "logit_baseline": logits_b,
            "logit_steered": logits_s,
            "stereotyped_option": stereo_opt,
            "corrected": corrected,
            "corrupted": corrupted,
            "subject": item.get("subject", ""),
            "demographic_subgroups": item.get("demographic_subgroups", []),
        })

        if (i + 1) % 50 == 0:
            log(f"      [{i + 1}/{len(items)}]")

    return results


def _summarise(results: list[dict]) -> dict[str, Any]:
    """Compute accuracy metrics + confidence-aware metrics from per-item results."""
    n = len(results)
    if n == 0:
        return {"n": 0}
    b = sum(r["baseline_correct"] for r in results)
    s = sum(r["steered_correct"] for r in results)
    f = sum(r["flipped"] for r in results)

    summary = {
        "n": n,
        "accuracy_baseline": round(b / n, 4),
        "accuracy_steered": round(s / n, 4),
        "delta": round((s - b) / n, 4),
        "flip_rate": round(f / n, 4),
    }

    # Add confidence-aware metrics
    metrics = compute_all_metrics(results)
    summary["metrics"] = metrics

    return summary


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_medqa_matched_vs_mismatched(
    all_medqa: dict[str, dict[str, Any]],
    vectors: dict[str, Any],
    output_dir: Path,
) -> None:
    """Per-category grouped bars: accuracy delta by condition."""
    # Group by category
    by_cat: dict[str, list[tuple[str, dict]]] = {}
    for vec_key, entry in all_medqa.items():
        cat = entry.get("category", "")
        by_cat.setdefault(cat, []).append((entry.get("subgroup", vec_key), entry))

    cats = sorted(by_cat.keys())
    if not cats:
        return

    n = len(cats)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    conditions = ["matched", "within_cat_mismatched", "cross_cat_mismatched", "no_demographic"]
    cond_labels = ["Matched", "Within-cat\nmismatch", "Cross-cat\nmismatch", "No demo"]
    cond_colors = [BLUE, ORANGE, GREEN, GRAY]

    for idx, cat in enumerate(cats):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        subs = sorted(by_cat[cat], key=lambda x: x[0])
        sub_names = [s for s, _ in subs]
        n_subs = len(sub_names)
        n_conds = len(conditions)
        width = 0.8 / n_conds
        x = np.arange(n_subs)

        for ci, (cond, clabel, ccolor) in enumerate(zip(conditions, cond_labels, cond_colors)):
            vals = []
            for _, entry in subs:
                d = entry.get(cond, {}).get("delta", 0)
                vals.append(d)
            bars = ax.bar(x + ci * width - 0.4 + width / 2, vals, width,
                          color=ccolor, label=clabel if idx == 0 else "", alpha=0.8)
            for bar, v in zip(bars, vals):
                if v != 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, v,
                            f"{v:.2f}", ha="center", va="bottom" if v > 0 else "top",
                            fontsize=5)

        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(sub_names, fontsize=8, rotation=30, ha="right")
        ax.set_ylabel("Accuracy delta", fontsize=8)
        cat_label = CATEGORY_LABELS.get(cat, cat)
        ax.set_title(cat_label, fontsize=9)

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.legend(loc="upper center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("MedQA accuracy delta by condition", fontsize=11, y=1.05)
    _save_both(fig, output_dir / "fig_medqa_matched_vs_mismatched.png")
    log("    Saved fig_medqa_matched_vs_mismatched")


def fig_medqa_exacerbation(
    all_medqa: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    """Paired bars: debiasing vs exacerbation delta."""
    entries = [(e.get("subgroup", k), e) for k, e in all_medqa.items()]
    entries.sort()

    labels = [s for s, _ in entries]
    debias_vals = [e.get("matched", {}).get("delta", 0) for _, e in entries]
    exac_vals = [e.get("exacerbation_matched", {}).get("delta", 0) for _, e in entries]

    if not labels:
        return

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 5))
    ax.bar(x - width / 2, debias_vals, width, color=BLUE, label="Debiasing", alpha=0.8)
    ax.bar(x + width / 2, exac_vals, width, color=VERMILLION, label="Exacerbation", alpha=0.8)

    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy delta")
    ax.set_title("MedQA debiasing vs exacerbation")
    ax.legend(fontsize=8)

    _save_both(fig, output_dir / "fig_medqa_exacerbation.png")
    log("    Saved fig_medqa_exacerbation")


def fig_side_effect_heatmap(
    all_medqa: dict[str, dict[str, Any]],
    all_mmlu: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    """Heatmap: steering vectors vs knowledge domain accuracy deltas."""
    vec_keys = sorted(set(list(all_medqa.keys()) + list(all_mmlu.keys())))
    if not vec_keys:
        return

    # Columns: MedQA no-demo, MMLU overall, MMLU STEM, MMLU humanities, MMLU social science
    col_labels = ["MedQA\nno-demo", "MMLU\noverall", "MMLU\nSTEM", "MMLU\nHumanities", "MMLU\nSocial Sci"]
    mat = np.full((len(vec_keys), len(col_labels)), np.nan)

    for i, vk in enumerate(vec_keys):
        medqa = all_medqa.get(vk, {})
        mmlu = all_mmlu.get(vk, {})

        mat[i, 0] = medqa.get("no_demographic", {}).get("delta", np.nan)
        mat[i, 1] = mmlu.get("delta", np.nan)

        per_subj = mmlu.get("per_subject", {})
        stem_subjs = [s for s in per_subj if any(kw in s.lower() for kw in
                      ["math", "physics", "chemistry", "biology", "computer", "engineering"])]
        hum_subjs = [s for s in per_subj if any(kw in s.lower() for kw in
                     ["history", "philosophy", "literature", "law"])]
        soc_subjs = [s for s in per_subj if any(kw in s.lower() for kw in
                     ["sociology", "psychology", "economics", "politics", "geography"])]

        for col_idx, subjs in [(2, stem_subjs), (3, hum_subjs), (4, soc_subjs)]:
            if subjs:
                deltas = [per_subj[s].get("delta", 0) for s in subjs]
                mat[i, col_idx] = float(np.mean(deltas))

    vmax = max(abs(np.nanmin(mat)), abs(np.nanmax(mat)), 0.05)
    fig, ax = plt.subplots(figsize=(max(7, len(col_labels) * 1.5),
                                     max(5, len(vec_keys) * 0.4)))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticks(range(len(vec_keys)))
    ax.set_yticklabels(vec_keys, fontsize=7)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                color = "white" if abs(mat[i, j]) > vmax * 0.6 else "black"
                ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center",
                        fontsize=6, color=color)

    fig.colorbar(im, ax=ax, label="Accuracy delta", shrink=0.8)
    ax.set_title("Side effects of steering on knowledge benchmarks")

    _save_both(fig, output_dir / "fig_side_effect_heatmap.png")
    log("    Saved fig_side_effect_heatmap")


def fig_debiasing_vs_exacerbation_asymmetry(
    all_medqa: dict[str, dict[str, Any]],
    manifests: list[dict],
    output_dir: Path,
) -> None:
    """Scatter: BBQ RCR_1.0 vs MedQA exacerbation accuracy drop."""
    x_vals, y_vals, labels, cats = [], [], [], []

    manifest_by_key: dict[str, dict] = {}
    for m in manifests:
        key = f"{m.get('category', '')}_{m.get('subgroup', '')}"
        manifest_by_key[key] = m

    for vec_key, entry in all_medqa.items():
        m = manifest_by_key.get(vec_key, {})
        rcr = m.get("metrics", {}).get("rcr_1.0", {}).get("rcr", 0)
        exac_delta = entry.get("exacerbation_matched", {}).get("delta", None)
        if exac_delta is None:
            continue

        x_vals.append(rcr)
        y_vals.append(exac_delta)
        labels.append(vec_key)
        cats.append(entry.get("category", ""))

    if not x_vals:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for i in range(len(x_vals)):
        color = CATEGORY_COLORS.get(cats[i], GRAY)
        ax.scatter(x_vals[i], y_vals[i], c=color, s=40, alpha=0.8)
        ax.annotate(labels[i].split("_", 1)[-1], (x_vals[i], y_vals[i]),
                    fontsize=6, xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel("BBQ RCR_1.0 (debiasing benefit)", fontsize=9)
    ax.set_ylabel("MedQA accuracy delta under exacerbation", fontsize=9)
    ax.set_title("Debiasing benefit vs exacerbation vulnerability")
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)

    _save_both(fig, output_dir / "fig_debiasing_vs_exacerbation_asymmetry.png")
    log("    Saved fig_debiasing_vs_exacerbation_asymmetry")


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
    vectors: dict[str, dict[str, Any]] = {}
    for npz_path in sorted(vec_dir.glob("*.npz")):
        data = np.load(npz_path, allow_pickle=True)
        key = npz_path.stem
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

    # Classify MedQA items by demographic content
    all_subgroups = list(SUBGROUP_PATTERNS.keys())
    audit_index = load_medqa_audit()
    n_from_audit = 0
    n_from_regex = 0

    if medqa_items:
        for i, item in enumerate(medqa_items):
            text = item.get("prompt", "") or item.get("question", "")
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
    all_per_item: list[dict] = []  # 7D: per-item records

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

            # 7B: Within-category mismatched
            cat_subs = [k.split("_", 1)[1] for k in vectors if k.startswith(cat + "_")]
            within_mismatched = [
                it for it in medqa_items
                if any(s in it.get("demographic_subgroups", []) for s in cat_subs if s != sub)
                and sub not in it.get("demographic_subgroups", [])
            ]

            # 7B: Cross-category mismatched
            other_cat_subs = [k.split("_", 1)[1] for k in vectors if not k.startswith(cat + "_")]
            cross_mismatched = [
                it for it in medqa_items
                if any(s in it.get("demographic_subgroups", []) for s in other_cat_subs)
                and not any(s in it.get("demographic_subgroups", []) for s in cat_subs)
            ]

            # No-demographic
            no_demo = [it for it in medqa_items if not it.get("mentions_demographic")]
            # 7C: Use --max_items instead of hardcoded [:200]
            if args.max_items:
                no_demo = no_demo[:args.max_items]

            entry: dict[str, Any] = {
                "vector": vec_key, "category": cat, "subgroup": sub,
            }

            if matched:
                log(f"    MedQA matched ({sub}): {len(matched)} items")
                res = evaluate_items(steerer, matched, vec, letters=("A", "B", "C", "D", "E"))
                entry["matched"] = _summarise(res)
                for r in res:
                    r["condition"] = "matched"
                    r["steering_vector_key"] = vec_key
                    r["category"] = cat
                    r["subgroup"] = sub
                all_per_item.extend(res)

            if within_mismatched:
                log(f"    MedQA within-cat mismatched: {len(within_mismatched)} items")
                res = evaluate_items(steerer, within_mismatched, vec, letters=("A", "B", "C", "D", "E"))
                entry["within_cat_mismatched"] = _summarise(res)
                for r in res:
                    r["condition"] = "within_cat_mismatched"
                    r["steering_vector_key"] = vec_key
                    r["category"] = cat
                    r["subgroup"] = sub
                all_per_item.extend(res)

            if cross_mismatched:
                log(f"    MedQA cross-cat mismatched: {len(cross_mismatched)} items")
                res = evaluate_items(steerer, cross_mismatched, vec, letters=("A", "B", "C", "D", "E"))
                entry["cross_cat_mismatched"] = _summarise(res)
                for r in res:
                    r["condition"] = "cross_cat_mismatched"
                    r["steering_vector_key"] = vec_key
                    r["category"] = cat
                    r["subgroup"] = sub
                all_per_item.extend(res)

            if no_demo:
                log(f"    MedQA no-demo: {len(no_demo)} items")
                res = evaluate_items(steerer, no_demo, vec, letters=("A", "B", "C", "D", "E"))
                entry["no_demographic"] = _summarise(res)

            # 7A: Exacerbation always runs (both directions)
            neg_vec = -vec
            if matched:
                log(f"    MedQA exacerbation (matched): {len(matched)} items")
                res_ex = evaluate_items(steerer, matched, neg_vec, letters=("A", "B", "C", "D", "E"))
                entry["exacerbation_matched"] = _summarise(res_ex)

            if within_mismatched:
                log(f"    MedQA exacerbation (within-cat mismatch): {len(within_mismatched)} items")
                res_ex = evaluate_items(steerer, within_mismatched, neg_vec, letters=("A", "B", "C", "D", "E"))
                entry["exacerbation_within_cat_mismatched"] = _summarise(res_ex)

            if cross_mismatched:
                log(f"    MedQA exacerbation (cross-cat mismatch): {len(cross_mismatched)} items")
                res_ex = evaluate_items(steerer, cross_mismatched, neg_vec, letters=("A", "B", "C", "D", "E"))
                entry["exacerbation_cross_cat_mismatched"] = _summarise(res_ex)

            if no_demo:
                log(f"    MedQA exacerbation (no-demo): {len(no_demo)} items")
                res_ex = evaluate_items(steerer, no_demo, neg_vec, letters=("A", "B", "C", "D", "E"))
                entry["exacerbation_no_demographic"] = _summarise(res_ex)

            all_medqa[vec_key] = entry
            atomic_save_json(entry, medqa_out / f"{vec_key}.json")

        # ---- MMLU ----
        if mmlu_items:
            mmlu_out = ensure_dir(output_dir / "mmlu")
            # 7C: Use --max_items instead of hardcoded [:200]
            items_to_eval = mmlu_items
            if args.max_items:
                items_to_eval = mmlu_items[:args.max_items]

            log(f"    MMLU: {len(items_to_eval)} items")
            res = evaluate_items(steerer, items_to_eval, vec)
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

            # Find worst subject
            if per_subj:
                worst = min(per_subj.items(), key=lambda x: (x[1]["s"] - x[1]["b"]) / max(x[1]["n"], 1))
                mmlu_entry["worst_subject"] = worst[0]
                mmlu_entry["worst_subject_delta"] = (worst[1]["s"] - worst[1]["b"]) / max(worst[1]["n"], 1)

            # 7A: MMLU exacerbation
            log(f"    MMLU exacerbation: {len(items_to_eval)} items")
            res_ex = evaluate_items(steerer, items_to_eval, -vec)
            mmlu_entry["exacerbation"] = _summarise(res_ex)

            all_mmlu[vec_key] = mmlu_entry
            atomic_save_json(mmlu_entry, mmlu_out / f"{vec_key}.json")

        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # Save aggregated results
    if all_medqa:
        atomic_save_json(all_medqa, output_dir / "medqa_steering_results.json")
    if all_mmlu:
        atomic_save_json(all_mmlu, output_dir / "mmlu_steering_results.json")

    # 7D: Save per-item parquet
    if all_per_item and pd is not None:
        per_item_dir = ensure_dir(output_dir / "per_item")
        for r in all_per_item:
            r["logit_baseline"] = json.dumps(r.get("logit_baseline", {}))
            r["logit_steered"] = json.dumps(r.get("logit_steered", {}))
            r["demographic_subgroups"] = json.dumps(r.get("demographic_subgroups", []))
        df = pd.DataFrame(all_per_item)
        df.to_parquet(per_item_dir / "medqa_per_item.parquet", index=False)
        log(f"Saved {len(all_per_item)} per-item MedQA records")

    # 7F: Update manifests with generalization results
    for m in manifests:
        key = f"{m['category']}_{m['subgroup']}"
        if key in all_medqa:
            md = all_medqa[key]
            m["medqa_matched_delta"] = md.get("matched", {}).get("delta")
            m["medqa_within_cat_mismatched_delta"] = md.get("within_cat_mismatched", {}).get("delta")
            m["medqa_cross_cat_mismatched_delta"] = md.get("cross_cat_mismatched", {}).get("delta")
            m["medqa_nodemo_delta"] = md.get("no_demographic", {}).get("delta")
            m["medqa_exacerbation_matched_delta"] = md.get("exacerbation_matched", {}).get("delta")
        if key in all_mmlu:
            m["mmlu_delta"] = all_mmlu[key].get("delta")
            m["mmlu_worst_subject"] = all_mmlu[key].get("worst_subject")
            m["mmlu_worst_subject_delta"] = all_mmlu[key].get("worst_subject_delta")

    atomic_save_json(manifests, output_dir / "steering_manifests_with_gen.json")

    # 7G: Figures
    if not args.skip_figures:
        fig_out = ensure_dir(output_dir / "figures")
        log("\nGenerating figures ...")

        if all_medqa:
            fig_medqa_matched_vs_mismatched(all_medqa, vectors, fig_out)
            fig_medqa_exacerbation(all_medqa, fig_out)
            fig_debiasing_vs_exacerbation_asymmetry(all_medqa, manifests, fig_out)

        if all_medqa or all_mmlu:
            fig_side_effect_heatmap(all_medqa, all_mmlu, fig_out)

    total = time.time() - t0
    log(f"\nComplete in {total:.1f}s")
    log(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
