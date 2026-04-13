"""Logit margin stratification for steering validation.

Computes the logit margin (gap between top-1 and top-2 answer logits) for
each item, bins items by margin, and reports correction/corruption rates
per bin to determine whether steering effects are concentrated among
low-confidence items.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

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
    import pandas as _pd
    pd = _pd


# ---------------------------------------------------------------------------
# Margin bins
# ---------------------------------------------------------------------------

MARGIN_BINS = [
    ("near_indifferent", 0.0, 0.5),
    ("low_confidence",   0.5, 1.0),
    ("moderate",         1.0, 2.0),
    ("confident",        2.0, 5.0),
    ("very_confident",   5.0, float("inf")),
]


def _bin_margin(margin: float) -> str:
    for name, lo, hi in MARGIN_BINS:
        if lo <= margin < hi:
            return name
    return MARGIN_BINS[-1][0]


# ---------------------------------------------------------------------------
# Step 1: Compute margins from Stage 1 or sweep data
# ---------------------------------------------------------------------------


def compute_margins_from_stage1(
    localization_dir: Path,
    categories: list[str],
) -> "pd.DataFrame":
    """Load answer_logits from Stage-1 .npz files and compute margins.

    Returns DataFrame with columns: item_idx, category, model_answer,
    model_answer_role, margin, margin_bin, is_stereotyped, logits_A/B/C.
    """
    _require_pandas()
    records: list[dict[str, Any]] = []
    act_dir = localization_dir / "activations"

    for cat in categories:
        cat_dir = act_dir / cat
        if not cat_dir.is_dir():
            continue
        for npz_path in sorted(cat_dir.glob("item_*.npz")):
            try:
                data = np.load(npz_path, allow_pickle=True)
                raw = data["metadata_json"]
                meta_str = raw.item() if raw.shape == () else str(raw)
                meta = json.loads(meta_str)
            except Exception:
                continue

            logits = meta.get("answer_logits", {})
            model_answer = meta.get("model_answer", "")

            if not logits or not model_answer:
                continue

            # Ensure logits are numeric
            try:
                logit_vals = {k: float(v) for k, v in logits.items()}
            except (ValueError, TypeError):
                continue

            # Margin = logit[top] - max(logit[others])
            top_val = logit_vals.get(model_answer, 0.0)
            other_vals = [v for k, v in logit_vals.items() if k != model_answer]
            if not other_vals:
                continue
            margin = top_val - max(other_vals)

            records.append({
                "item_idx": meta.get("item_idx", -1),
                "category": cat,
                "model_answer": model_answer,
                "model_answer_role": meta.get("model_answer_role", ""),
                "is_stereotyped": meta.get("is_stereotyped_response", False),
                "margin": margin,
                "margin_bin": _bin_margin(margin),
                "logits_A": logit_vals.get("A", 0.0),
                "logits_B": logit_vals.get("B", 0.0),
                "logits_C": logit_vals.get("C", 0.0),
            })

    df = pd.DataFrame(records)
    log(f"  Computed margins for {len(df)} items across "
        f"{df['category'].nunique()} categories")
    return df


def compute_margins_from_sweep(
    sweep_df: "pd.DataFrame",
) -> "pd.DataFrame":
    """Extract margins from sweep parquet original_logits.

    Returns DataFrame with: item_idx, category, margin, margin_bin,
    original_answer, original_role.
    """
    _require_pandas()
    records: list[dict[str, Any]] = []

    for _, row in sweep_df.iterrows():
        logits = row.get("original_logits", {})
        answer = row.get("original_answer", "")
        if not logits or not answer:
            continue

        try:
            logit_vals = {k: float(v) for k, v in logits.items()}
        except (ValueError, TypeError):
            continue

        top_val = logit_vals.get(answer, 0.0)
        other_vals = [v for k, v in logit_vals.items() if k != answer]
        if not other_vals:
            continue
        margin = top_val - max(other_vals)

        records.append({
            "item_idx": row.get("item_idx", -1),
            "category": row.get("category", ""),
            "original_answer": answer,
            "original_role": row.get("original_role", ""),
            "margin": margin,
            "margin_bin": _bin_margin(margin),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Step 2–5: Join margins with steering results & compute per-bin stats
# ---------------------------------------------------------------------------


def stratify_corrections(
    margins_df: "pd.DataFrame",
    sweep_df: "pd.DataFrame",
    alpha: float,
    experiment: str = "A",
) -> "pd.DataFrame":
    """Join margins with steering results at a specific alpha and compute per-bin rates.

    Parameters
    ----------
    margins_df : DataFrame
        From compute_margins_from_stage1 or compute_margins_from_sweep.
    sweep_df : DataFrame
        Experiment A or B sweep parquet.
    alpha : float
        The alpha value to analyse.
    experiment : str
        "A" (correction) or "B" (corruption).

    Returns DataFrame with columns: category, margin_bin, n_items, n_affected,
    rate, fraction_of_total.
    """
    _require_pandas()

    # Filter sweep to target alpha
    at_alpha = sweep_df[sweep_df["alpha"] == alpha].copy()
    if at_alpha.empty:
        return pd.DataFrame()

    # Compute margins from sweep data if not available from Stage 1
    if margins_df.empty:
        margins_df = compute_margins_from_sweep(at_alpha)

    # Merge on item_idx + category
    margin_lookup = margins_df.set_index(["item_idx", "category"])[
        ["margin", "margin_bin"]
    ]
    at_alpha = at_alpha.copy()
    at_alpha["_key"] = list(zip(at_alpha["item_idx"], at_alpha["category"]))

    margins_dict = {}
    for (idx, cat), row in margin_lookup.iterrows():
        margins_dict[(idx, cat)] = (row["margin"], row["margin_bin"])

    at_alpha["margin"] = at_alpha["_key"].map(lambda k: margins_dict.get(k, (np.nan, "unknown"))[0])
    at_alpha["margin_bin"] = at_alpha["_key"].map(lambda k: margins_dict.get(k, (np.nan, "unknown"))[1])
    at_alpha = at_alpha.dropna(subset=["margin"])

    # Determine what counts as "affected"
    if experiment == "A":
        # Correction: flipped from stereotyped to non-stereo or unknown
        at_alpha["affected"] = at_alpha["flipped"] & at_alpha["steered_role"].isin(
            ["non_stereotyped", "unknown"]
        )
    else:
        # Corruption: flipped from non-stereo to stereotyped
        at_alpha["affected"] = at_alpha["flipped"] & (
            at_alpha["steered_role"] == "stereotyped_target"
        )

    # Per-bin stats
    results: list[dict[str, Any]] = []
    for cat in sorted(at_alpha["category"].unique()):
        cat_df = at_alpha[at_alpha["category"] == cat]
        total_affected = int(cat_df["affected"].sum())

        for bin_name, _, _ in MARGIN_BINS:
            bin_df = cat_df[cat_df["margin_bin"] == bin_name]
            n = len(bin_df)
            if n == 0:
                continue
            n_aff = int(bin_df["affected"].sum())
            results.append({
                "category": cat,
                "margin_bin": bin_name,
                "n_items": n,
                "n_affected": n_aff,
                "rate": n_aff / n,
                "fraction_of_total": n_aff / max(total_affected, 1),
            })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------


def build_margin_summary(
    margins_df: "pd.DataFrame",
    correction_df: "pd.DataFrame",
    corruption_df: "pd.DataFrame",
) -> dict[str, Any]:
    """Build the margin_summary.json structure."""
    _require_pandas()
    summary: dict[str, Any] = {"per_category": {}}

    for cat in sorted(margins_df["category"].unique()):
        cat_margins = margins_df[margins_df["category"] == cat]

        # Margin distribution
        dist = {}
        total = len(cat_margins)
        for bin_name, _, _ in MARGIN_BINS:
            n = int((cat_margins["margin_bin"] == bin_name).sum())
            dist[bin_name] = {"n": n, "frac": round(n / max(total, 1), 3)}

        # Correction by bin
        corr_by_bin = {}
        cat_corr = correction_df[correction_df["category"] == cat] if not correction_df.empty else pd.DataFrame()
        for _, row in cat_corr.iterrows():
            corr_by_bin[row["margin_bin"]] = {
                "correction_rate": round(row["rate"], 4),
                "fraction_of_corrections": round(row["fraction_of_total"], 4),
                "n_items": int(row["n_items"]),
            }

        # Median margins for stereotyped items
        stereo = cat_margins[cat_margins["is_stereotyped"] == True]  # noqa: E712
        non_stereo = cat_margins[
            (cat_margins["is_stereotyped"] == False)  # noqa: E712
            & (cat_margins["model_answer_role"] != "unknown")
        ]

        entry: dict[str, Any] = {
            "margin_distribution": dist,
            "correction_by_bin": corr_by_bin,
            "median_margin_all_stereotyped": round(float(stereo["margin"].median()), 3) if not stereo.empty else None,
            "median_margin_non_stereotyped": round(float(non_stereo["margin"].median()), 3) if not non_stereo.empty else None,
        }

        # Corruption by bin
        if not corruption_df.empty:
            cat_corrupt = corruption_df[corruption_df["category"] == cat]
            corrupt_by_bin = {}
            for _, row in cat_corrupt.iterrows():
                corrupt_by_bin[row["margin_bin"]] = {
                    "corruption_rate": round(row["rate"], 4),
                    "fraction_of_corruptions": round(row["fraction_of_total"], 4),
                    "n_items": int(row["n_items"]),
                }
            entry["corruption_by_bin"] = corrupt_by_bin

        summary["per_category"][cat] = entry

    return summary


# ---------------------------------------------------------------------------
# Full margin analysis pipeline
# ---------------------------------------------------------------------------


def run_margin_analysis(
    localization_dir: Path,
    steering_dir: Path,
    categories: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    """Run full margin stratification analysis.

    Parameters
    ----------
    localization_dir : Path
        Stage 1 dir with activations/.
    steering_dir : Path
        Stage 3 dir with experiments/ containing sweep parquets.
    categories : list[str]
        Category short names.
    output_dir : Path
        Where to save results.
    """
    _require_pandas()
    out = ensure_dir(output_dir / "margin_analysis")
    exp_dir = steering_dir / "experiments"

    # Step 1: compute margins from Stage 1
    margins_df = compute_margins_from_stage1(localization_dir, categories)
    if margins_df.empty:
        log("  WARNING: no Stage-1 margins; will derive from sweep logits")

    margins_df.to_parquet(out / "margins_per_item.parquet", index=False)

    # Step 2-5: per-experiment stratification
    all_correction = []
    all_corruption = []

    for cat in categories:
        # Experiment A (correction)
        sweep_path = exp_dir / f"experiment_A_{cat}_sweep.parquet"
        result_path = exp_dir / f"experiment_A_{cat}.json"
        if sweep_path.exists() and result_path.exists():
            sweep_a = pd.read_parquet(sweep_path)
            with open(result_path) as f:
                result_a = json.load(f)
            opt_alpha = result_a.get("optimal_alpha", -10)

            # If no Stage-1 margins, derive from sweep
            cat_margins = margins_df[margins_df["category"] == cat]
            if cat_margins.empty:
                cat_margins = compute_margins_from_sweep(
                    sweep_a[sweep_a["alpha"] == opt_alpha]
                )

            corr = stratify_corrections(cat_margins, sweep_a, opt_alpha, "A")
            if not corr.empty:
                all_correction.append(corr)

        # Experiment B (corruption)
        sweep_path_b = exp_dir / f"experiment_B_{cat}_sweep.parquet"
        result_path_b = exp_dir / f"experiment_B_{cat}.json"
        if sweep_path_b.exists() and result_path_b.exists():
            sweep_b = pd.read_parquet(sweep_path_b)
            with open(result_path_b) as f:
                result_b = json.load(f)
            opt_alpha_b = result_b.get("optimal_alpha", 10)

            cat_margins_b = margins_df[margins_df["category"] == cat]
            if cat_margins_b.empty:
                cat_margins_b = compute_margins_from_sweep(
                    sweep_b[sweep_b["alpha"] == opt_alpha_b]
                )

            corrupt = stratify_corrections(cat_margins_b, sweep_b, opt_alpha_b, "B")
            if not corrupt.empty:
                all_corruption.append(corrupt)

    correction_df = pd.concat(all_correction, ignore_index=True) if all_correction else pd.DataFrame()
    corruption_df = pd.concat(all_corruption, ignore_index=True) if all_corruption else pd.DataFrame()

    if not correction_df.empty:
        correction_df.to_parquet(out / "correction_by_margin.parquet", index=False)
    if not corruption_df.empty:
        corruption_df.to_parquet(out / "corruption_by_margin.parquet", index=False)

    # Build summary
    summary = build_margin_summary(margins_df, correction_df, corruption_df)
    atomic_save_json(summary, out / "margin_summary.json")

    n_cats = len(summary.get("per_category", {}))
    log(f"  Margin analysis complete: {n_cats} categories")

    return summary
