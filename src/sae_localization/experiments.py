"""Steering experiments A–E for causal validation of SAE bias features.

Each experiment function returns results and saves incrementally.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch

from src.utils.io import atomic_save_json, ensure_dir
from src.utils.logging import log

try:
    import pandas as pd
except ImportError:
    pd = None


def _require_pandas():
    global pd
    if pd is not None:
        return
    try:
        import pandas as _pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required for SAE steering experiments (parquet I/O + analysis). "
            "Install with: pip install pandas pyarrow"
        ) from exc
    pd = _pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


CROWS_TO_BBQ_CATEGORY = {
    "race-color": "race",
    "gender": "gi",
    "socioeconomic": None,
    "nationality": None,
    "religion": "religion",
    "age": "age",
    "sexual-orientation": "so",
    "physical-appearance": "physical_appearance",
    "disability": "disability",
}


def _compute_rates(df: "pd.DataFrame", direction: str = "suppress") -> dict[str, Any]:
    """Compute correction/corruption/degeneration rates from sweep results."""
    n = len(df)
    if n == 0:
        return {"n_items": 0}

    n_degen = int(df["degenerated"].sum())
    n_flipped = int(df["flipped"].sum())

    if direction == "suppress":
        # Experiment A: items were stereotyped, count switches to non-stereo or unknown
        n_corrected = int(
            df.loc[df["flipped"], "steered_role"].isin(["non_stereotyped", "unknown"]).sum()
        )
        n_to_unknown = int(
            df.loc[df["flipped"], "steered_role"].eq("unknown").sum()
        )
        return {
            "n_items": n,
            "correction_rate": n_corrected / n,
            "degeneration_rate": n_degen / n,
            "unknown_rate": n_to_unknown / n,
            "flip_rate": n_flipped / n,
            "n_corrected": n_corrected,
            "n_degenerated": n_degen,
        }
    else:
        # Experiment B: items were non-stereotyped, count switches to stereo
        n_corrupted = int(
            df.loc[df["flipped"], "steered_role"].eq("stereotyped_target").sum()
        )
        return {
            "n_items": n,
            "corruption_rate": n_corrupted / n,
            "degeneration_rate": n_degen / n,
            "flip_rate": n_flipped / n,
            "n_corrupted": n_corrupted,
            "n_degenerated": n_degen,
        }


def _select_optimal_alpha(
    sweep_df: "pd.DataFrame", direction: str = "suppress", max_degen: float = 0.05,
) -> float:
    """Select the alpha with highest correction/corruption rate under degeneration constraint."""
    _require_pandas()
    best_alpha = 0.0
    best_rate = -1.0

    for alpha in sweep_df["alpha"].unique():
        sub = sweep_df[sweep_df["alpha"] == alpha]
        degen_rate = sub["degenerated"].mean()
        if degen_rate > max_degen:
            continue

        if direction == "suppress":
            flipped_roles = sub.loc[sub["flipped"], "steered_role"]
            rate = flipped_roles.isin(["non_stereotyped", "unknown"]).sum() / max(len(sub), 1)
        else:
            flipped_roles = sub.loc[sub["flipped"], "steered_role"]
            rate = flipped_roles.eq("stereotyped_target").sum() / max(len(sub), 1)

        if rate > best_rate:
            best_rate = rate
            best_alpha = alpha

    return float(best_alpha)


# ---------------------------------------------------------------------------
# Experiment A: Suppress bias
# ---------------------------------------------------------------------------


def experiment_a_suppress(
    steerer: Any,
    items: list[dict[str, Any]],
    feature_indices: list[int],
    alpha_values: list[float],
    prompt_formatter: Callable,
    category: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Experiment A: dampen pro-bias features on stereotyped-response items.

    Parameters
    ----------
    items : list
        Items where model originally gave stereotyped response.
    feature_indices : list[int]
        Pro-bias feature indices.
    alpha_values : list[float]
        Negative alpha values for dampening.
    """
    _require_pandas()
    out_path = output_dir / f"experiment_A_{category}.json"

    log(f"  Experiment A (suppress): {category} — {len(items)} items, "
        f"{len(feature_indices)} features")

    if not items or not feature_indices:
        result = {"category": category, "n_items": 0, "skipped": True}
        atomic_save_json(result, out_path)
        return result

    sweep_df = steerer.run_alpha_sweep(
        items, feature_indices, alpha_values, prompt_formatter,
    )

    # Save raw sweep
    sweep_path = output_dir / f"experiment_A_{category}_sweep.parquet"
    sweep_df.to_parquet(sweep_path, index=False)

    # Compute rates per alpha
    per_alpha = {}
    for alpha in alpha_values:
        sub = sweep_df[sweep_df["alpha"] == alpha]
        per_alpha[str(alpha)] = _compute_rates(sub, "suppress")

    optimal_alpha = _select_optimal_alpha(sweep_df, "suppress")
    optimal_sub = sweep_df[sweep_df["alpha"] == optimal_alpha]
    optimal_rates = _compute_rates(optimal_sub, "suppress")

    result = {
        "category": category,
        "experiment": "A",
        "n_items": len(items),
        "n_features": len(feature_indices),
        "feature_indices": feature_indices[:50],  # truncate for JSON
        "optimal_alpha": optimal_alpha,
        "optimal_rates": optimal_rates,
        "per_alpha": per_alpha,
    }
    atomic_save_json(result, out_path)
    log(f"    Optimal alpha={optimal_alpha}: correction={optimal_rates.get('correction_rate', 0):.3f} "
        f"degen={optimal_rates.get('degeneration_rate', 0):.3f}")
    return result


# ---------------------------------------------------------------------------
# Experiment B: Elicit bias
# ---------------------------------------------------------------------------


def experiment_b_elicit(
    steerer: Any,
    items: list[dict[str, Any]],
    feature_indices: list[int],
    alpha_values: list[float],
    prompt_formatter: Callable,
    category: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Experiment B: amplify pro-bias features on non-stereotyped-response items."""
    _require_pandas()
    out_path = output_dir / f"experiment_B_{category}.json"

    log(f"  Experiment B (elicit): {category} — {len(items)} items, "
        f"{len(feature_indices)} features")

    if not items or not feature_indices:
        result = {"category": category, "n_items": 0, "skipped": True}
        atomic_save_json(result, out_path)
        return result

    sweep_df = steerer.run_alpha_sweep(
        items, feature_indices, alpha_values, prompt_formatter,
    )

    sweep_path = output_dir / f"experiment_B_{category}_sweep.parquet"
    sweep_df.to_parquet(sweep_path, index=False)

    per_alpha = {}
    for alpha in alpha_values:
        sub = sweep_df[sweep_df["alpha"] == alpha]
        per_alpha[str(alpha)] = _compute_rates(sub, "amplify")

    optimal_alpha = _select_optimal_alpha(sweep_df, "amplify")
    optimal_sub = sweep_df[sweep_df["alpha"] == optimal_alpha]
    optimal_rates = _compute_rates(optimal_sub, "amplify")

    result = {
        "category": category,
        "experiment": "B",
        "n_items": len(items),
        "n_features": len(feature_indices),
        "optimal_alpha": optimal_alpha,
        "optimal_rates": optimal_rates,
        "per_alpha": per_alpha,
    }
    atomic_save_json(result, out_path)
    log(f"    Optimal alpha={optimal_alpha}: corruption={optimal_rates.get('corruption_rate', 0):.3f} "
        f"degen={optimal_rates.get('degeneration_rate', 0):.3f}")
    return result


# ---------------------------------------------------------------------------
# Experiment C: Anti-bias feature validation
# ---------------------------------------------------------------------------


def experiment_c_antibias(
    steerer: Any,
    stereo_items: list[dict[str, Any]],
    non_stereo_items: list[dict[str, Any]],
    anti_bias_features: list[int],
    alpha: float,
    prompt_formatter: Callable,
    category: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Experiment C: amplify anti-bias on stereo items (C1), dampen on non-stereo (C2)."""
    _require_pandas()
    out_path = output_dir / f"experiment_C_{category}.json"

    log(f"  Experiment C (anti-bias): {category} — "
        f"{len(stereo_items)} stereo, {len(non_stereo_items)} non-stereo, "
        f"{len(anti_bias_features)} anti-bias features")

    result: dict[str, Any] = {
        "category": category, "experiment": "C",
        "n_anti_bias_features": len(anti_bias_features),
    }

    if not anti_bias_features:
        result["skipped"] = True
        atomic_save_json(result, out_path)
        return result

    # C1: Amplify anti-bias on stereotyped items → expect correction
    if stereo_items:
        c1_df = steerer.run_alpha_sweep(
            stereo_items, anti_bias_features, [alpha], prompt_formatter,
        )
        result["C1_amplify_antibias"] = _compute_rates(c1_df, "suppress")
        result["C1_amplify_antibias"]["n_items"] = len(stereo_items)

    # C2: Dampen anti-bias on non-stereo items → expect corruption
    if non_stereo_items:
        c2_df = steerer.run_alpha_sweep(
            non_stereo_items, anti_bias_features, [-abs(alpha)], prompt_formatter,
        )
        result["C2_dampen_antibias"] = _compute_rates(c2_df, "amplify")
        result["C2_dampen_antibias"]["n_items"] = len(non_stereo_items)

    atomic_save_json(result, out_path)
    return result


# ---------------------------------------------------------------------------
# Experiment D: CrowS-Pairs transfer
# ---------------------------------------------------------------------------


def experiment_d_crows_pairs(
    steerer: Any,
    crows_items: list[dict[str, Any]],
    feature_map: dict[str, list[int]],
    alpha_map: dict[str, float],
    output_dir: Path,
) -> dict[str, Any]:
    """Experiment D: apply steering to CrowS-Pairs items.

    Parameters
    ----------
    crows_items : list
        CrowS-Pairs items from crows_pairs_loader.
    feature_map : dict
        Maps BBQ category short name → pro-bias feature indices.
    alpha_map : dict
        Maps BBQ category short name → optimal alpha from Experiment A.
    """
    _require_pandas()
    out_path = output_dir / "experiment_D_crows_pairs.json"

    log(f"  Experiment D (CrowS-Pairs): {len(crows_items)} items")

    records: list[dict[str, Any]] = []

    for i, item in enumerate(crows_items):
        bias_type = (item.get("stereotyped_groups") or ["unknown"])[0]
        bbq_cat = CROWS_TO_BBQ_CATEGORY.get(bias_type)

        if bbq_cat is None or bbq_cat not in feature_map:
            continue

        features = feature_map[bbq_cat]
        alpha = alpha_map.get(bbq_cat, -10)

        if not features:
            continue

        sent_more = item["answers"]["A"]  # stereotyped
        sent_less = item["answers"]["B"]  # non-stereotyped

        # Baseline log-probs
        lp_more_orig = steerer.compute_log_prob(sent_more)
        lp_less_orig = steerer.compute_log_prob(sent_less)

        # Steered log-probs
        vec = steerer.get_composite_steering(features, alpha)
        lp_more_steered = steerer.compute_log_prob_steered(sent_more, vec)
        lp_less_steered = steerer.compute_log_prob_steered(sent_less, vec)

        prefers_stereo_orig = lp_more_orig > lp_less_orig
        prefers_stereo_steered = lp_more_steered > lp_less_steered
        flipped = prefers_stereo_orig != prefers_stereo_steered

        records.append({
            "item_idx": item.get("item_idx", i),
            "bias_type": bias_type,
            "bbq_category": bbq_cat,
            "lp_more_orig": lp_more_orig,
            "lp_less_orig": lp_less_orig,
            "lp_more_steered": lp_more_steered,
            "lp_less_steered": lp_less_steered,
            "prefers_stereo_orig": prefers_stereo_orig,
            "prefers_stereo_steered": prefers_stereo_steered,
            "flipped": flipped,
        })

        if (i + 1) % 50 == 0:
            log(f"    [{i + 1}/{len(crows_items)}]")

    if not records:
        result = {"n_items": 0, "skipped": True}
        atomic_save_json(result, out_path)
        return result

    df = pd.DataFrame(records)
    df.to_parquet(output_dir / "experiment_D_crows_pairs.parquet", index=False)

    stereo_orig = float(df["prefers_stereo_orig"].mean())
    stereo_steered = float(df["prefers_stereo_steered"].mean())
    flip_rate = float(df["flipped"].mean())

    # Per bias_type breakdown
    per_type: dict[str, dict[str, Any]] = {}
    for bt in df["bias_type"].unique():
        sub = df[df["bias_type"] == bt]
        per_type[bt] = {
            "n_items": len(sub),
            "stereo_rate_orig": float(sub["prefers_stereo_orig"].mean()),
            "stereo_rate_steered": float(sub["prefers_stereo_steered"].mean()),
            "flip_rate": float(sub["flipped"].mean()),
        }

    result = {
        "experiment": "D",
        "n_items": len(df),
        "stereotype_preference_rate_original": stereo_orig,
        "stereotype_preference_rate_steered": stereo_steered,
        "flip_rate": flip_rate,
        "per_bias_type": per_type,
    }
    atomic_save_json(result, out_path)
    log(f"    CrowS-Pairs: stereo_pref {stereo_orig:.3f} → {stereo_steered:.3f}, "
        f"flip_rate={flip_rate:.3f}")
    return result


# ---------------------------------------------------------------------------
# Experiment E: Side-effect testing (MMLU / MedQA)
# ---------------------------------------------------------------------------

MMLU_FEW_SHOT_TEMPLATE = """The following are multiple choice questions (with answers) about {subject}.

{examples}
{question}
A. {A}
B. {B}
C. {C}
D. {D}
Answer:"""


def _format_mmlu_item(item: dict[str, Any]) -> str:
    """Format an MMLU item as a few-shot prompt."""
    return MMLU_FEW_SHOT_TEMPLATE.format(
        subject=item.get("subject", "general knowledge"),
        examples=item.get("few_shot_text", ""),
        question=item.get("question", ""),
        A=item.get("A", ""),
        B=item.get("B", ""),
        C=item.get("C", ""),
        D=item.get("D", ""),
    )


def experiment_e_side_effects(
    steerer: Any,
    steering_vec: torch.Tensor,
    mmlu_items: list[dict[str, Any]] | None,
    medqa_items: list[dict[str, Any]] | None,
    output_dir: Path,
) -> dict[str, Any]:
    """Experiment E: measure side effects on MMLU and MedQA.

    Parameters
    ----------
    steering_vec : torch.Tensor
        Pre-computed composite steering vector (from Experiment A optimal alpha).
    mmlu_items / medqa_items : list | None
        Pre-loaded items with expected fields.
    """
    _require_pandas()
    from src.utils.answers import best_choice_from_logits

    result: dict[str, Any] = {"experiment": "E"}

    # E1: MMLU
    if mmlu_items:
        log(f"  Experiment E1 (MMLU): {len(mmlu_items)} items")
        orig_correct = 0
        steered_correct = 0
        per_subject: dict[str, dict[str, int]] = {}

        for i, item in enumerate(mmlu_items):
            prompt = _format_mmlu_item(item)
            correct = item.get("answer", "")
            subject = item.get("subject", "other")

            baseline = steerer.evaluate_baseline_mcq(prompt, letters=("A", "B", "C", "D"))
            steered = steerer.steer_and_evaluate(prompt, steering_vec, letters=("A", "B", "C", "D"))

            b_correct = int(baseline["model_answer"] == correct)
            s_correct = int(steered["model_answer"] == correct)
            orig_correct += b_correct
            steered_correct += s_correct

            per_subject.setdefault(subject, {"orig": 0, "steered": 0, "total": 0})
            per_subject[subject]["orig"] += b_correct
            per_subject[subject]["steered"] += s_correct
            per_subject[subject]["total"] += 1

            if (i + 1) % 50 == 0:
                log(f"    MMLU [{i + 1}/{len(mmlu_items)}]")

        n = len(mmlu_items)
        result["mmlu"] = {
            "n_items": n,
            "accuracy_original": orig_correct / max(n, 1),
            "accuracy_steered": steered_correct / max(n, 1),
            "delta": (steered_correct - orig_correct) / max(n, 1),
            "per_subject": {
                s: {
                    "accuracy_original": d["orig"] / max(d["total"], 1),
                    "accuracy_steered": d["steered"] / max(d["total"], 1),
                    "delta": (d["steered"] - d["orig"]) / max(d["total"], 1),
                    "n_items": d["total"],
                }
                for s, d in per_subject.items()
            },
        }

    # E2: MedQA
    if medqa_items:
        log(f"  Experiment E2 (MedQA): {len(medqa_items)} items")
        orig_correct = 0
        steered_correct = 0
        demo_orig = 0
        demo_steered = 0
        n_demo = 0

        for i, item in enumerate(medqa_items):
            prompt = item.get("prompt", "")
            correct = item.get("answer", "")
            is_demo = item.get("mentions_demographic", False)
            letters = tuple(item.get("letters") or ("A", "B", "C", "D"))

            baseline = steerer.evaluate_baseline_mcq(prompt, letters=letters)
            steered = steerer.steer_and_evaluate(prompt, steering_vec, letters=letters)

            b_correct = int(baseline["model_answer"] == correct)
            s_correct = int(steered["model_answer"] == correct)
            orig_correct += b_correct
            steered_correct += s_correct

            if is_demo:
                demo_orig += b_correct
                demo_steered += s_correct
                n_demo += 1

            if (i + 1) % 50 == 0:
                log(f"    MedQA [{i + 1}/{len(medqa_items)}]")

        n = len(medqa_items)
        result["medqa"] = {
            "n_items": n,
            "accuracy_original": orig_correct / max(n, 1),
            "accuracy_steered": steered_correct / max(n, 1),
            "delta": (steered_correct - orig_correct) / max(n, 1),
            "n_demographic": n_demo,
            "demographic_accuracy_original": demo_orig / max(n_demo, 1),
            "demographic_accuracy_steered": demo_steered / max(n_demo, 1),
            "demographic_delta": (demo_steered - demo_orig) / max(n_demo, 1),
        }

    atomic_save_json(result, output_dir / "experiment_E_side_effects.json")
    return result


# ---------------------------------------------------------------------------
# Cross-subgroup transfer (for Figure 28)
# ---------------------------------------------------------------------------


def experiment_cross_subgroup_transfer(
    steerer: Any,
    items_by_subgroup: dict[str, list[dict[str, Any]]],
    features_by_subgroup: dict[str, list[int]],
    alpha: float,
    prompt_formatter: Callable,
    category: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Test steering with features from subgroup X on items targeting subgroup Y.

    Returns a matrix: source_subgroup x target_subgroup → correction/corruption rate.
    """
    _require_pandas()
    out_path = output_dir / f"cross_subgroup_transfer_{category}.json"

    log(f"  Cross-subgroup transfer: {category}")

    sources = sorted(features_by_subgroup.keys())
    targets = sorted(items_by_subgroup.keys())

    matrix: dict[str, dict[str, Any]] = {}

    for source in sources:
        features = features_by_subgroup[source]
        if not features:
            continue

        matrix[source] = {}
        for target in targets:
            target_items = items_by_subgroup[target]
            if not target_items:
                matrix[source][target] = {"n_items": 0}
                continue

            # Run steering
            vec = steerer.get_composite_steering(features, alpha)
            n_flipped = 0
            n_items = len(target_items)

            for item in target_items:
                prompt = prompt_formatter(item)
                baseline = steerer.evaluate_baseline(prompt)
                result = steerer.steer_and_evaluate(prompt, vec)

                if result["model_answer"] != baseline["model_answer"]:
                    n_flipped += 1

            matrix[source][target] = {
                "n_items": n_items,
                "flip_rate": n_flipped / max(n_items, 1),
                "n_flipped": n_flipped,
            }

            log(f"    {source} → {target}: {n_flipped}/{n_items} flipped")

    result = {
        "category": category,
        "alpha": alpha,
        "sources": sources,
        "targets": targets,
        "matrix": matrix,
    }
    atomic_save_json(result, out_path)
    return result
