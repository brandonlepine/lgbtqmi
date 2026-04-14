"""Random vector control for steering validation.

Runs N trials of random-direction perturbations matched in L2 norm to the
actual bias steering vector, establishing a baseline for non-specific
perturbation effects.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from scipy.stats import ttest_1samp

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


N_TRIALS_DEFAULT = 10


# ---------------------------------------------------------------------------
# Random vector generation
# ---------------------------------------------------------------------------


def make_random_steering_vector(
    hidden_dim: int,
    target_norm: float,
    seed: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate a unit-random vector scaled to match *target_norm*.

    Uses ``torch.manual_seed(seed)`` for reproducibility.
    """
    torch.manual_seed(seed)
    v = torch.randn(hidden_dim, dtype=torch.float32)
    v = v / v.norm()
    v = v * target_norm
    return v.to(dtype=dtype, device=device)


def compute_actual_steering_norm(
    sae: Any,
    feature_indices: list[int],
    alpha: float,
    scale_by_sqrt_n: bool = True,
) -> float:
    """Compute the L2 norm of the actual composite steering vector."""
    n = len(feature_indices)
    if n == 0:
        return 0.0
    eff_alpha = alpha / math.sqrt(n) if scale_by_sqrt_n else alpha
    vec = torch.zeros(sae.hidden_dim, dtype=torch.float32)
    for fidx in feature_indices:
        d = torch.from_numpy(sae.get_feature_direction(fidx))
        vec += eff_alpha * d
    return float(vec.norm())


# ---------------------------------------------------------------------------
# Run random trials
# ---------------------------------------------------------------------------


def run_random_trials(
    steerer: Any,
    items: list[dict[str, Any]],
    target_norm: float,
    alpha_values: list[float],
    prompt_formatter: Callable,
    category: str,
    sae: Any,
    feature_indices: list[int],
    n_trials: int = N_TRIALS_DEFAULT,
    output_dir: Path | None = None,
    experiment: str = "A",
    save_per_item_trial: int = 0,
    scale_by_sqrt_n: bool = True,
) -> "pd.DataFrame":
    """Run random-direction steering across multiple trials and alpha values.

    Parameters
    ----------
    steerer : SAESteerer
    items : list
        Items to steer (stereotyped for correction, non-stereo for corruption).
    target_norm : float
        Norm of the actual steering vector at the optimal alpha.
    alpha_values : list[float]
        Alpha grid (negative for correction, positive for corruption).
    prompt_formatter : callable
    category : str
    sae : SAEWrapper
    feature_indices : list[int]
        Actual bias features (for computing norm at each alpha).
    n_trials : int
    output_dir : Path | None
    experiment : str
        "A" (correction) or "B" (corruption).
    save_per_item_trial : int
        Which trial index to save per-item results for (-1 = none).

    Returns DataFrame with columns: category, trial, alpha, correction_rate or
    corruption_rate, degeneration_rate, vec_norm, n_items.
    """
    _require_pandas()
    hidden_dim = sae.hidden_dim
    device = steerer.wrapper.device
    model_dtype = steerer.wrapper.model.dtype

    records: list[dict[str, Any]] = []
    per_item_records: list[dict[str, Any]] = []

    for trial in range(n_trials):
        seed = 42 + trial

        for alpha in alpha_values:
            # Checkpoint
            if output_dir:
                ckpt = output_dir / f"{category}_{experiment}_trial{trial}_alpha{alpha}.json"
                if ckpt.exists():
                    with open(ckpt) as f:
                        rec = json.load(f)
                    records.append(rec)
                    continue

            # Compute actual norm at this alpha
            actual_norm = compute_actual_steering_norm(
                sae, feature_indices, alpha, scale_by_sqrt_n,
            )

            # Generate random vector
            rand_vec = make_random_steering_vector(
                hidden_dim, actual_norm, seed, device=device, dtype=model_dtype,
            )

            n_flipped = 0
            n_degen = 0
            n_items = len(items)

            for i, item in enumerate(items):
                prompt = prompt_formatter(item)
                baseline = steerer.evaluate_baseline(prompt)
                result = steerer.steer_and_evaluate(prompt, rand_vec)

                flipped = result["model_answer"] != baseline["model_answer"]
                if flipped:
                    n_flipped += 1
                if result["degenerated"]:
                    n_degen += 1

                if trial == save_per_item_trial:
                    per_item_records.append({
                        "item_idx": item.get("item_idx", i),
                        "category": category,
                        "trial": trial,
                        "alpha": alpha,
                        "original_answer": baseline["model_answer"],
                        "steered_answer": result["model_answer"],
                        "flipped": flipped,
                        "degenerated": result["degenerated"],
                    })

            rec = {
                "category": category,
                "trial": trial,
                "alpha": alpha,
                "n_items": n_items,
                "n_flipped": n_flipped,
                "flip_rate": n_flipped / max(n_items, 1),
                "degeneration_rate": n_degen / max(n_items, 1),
                "vec_norm": actual_norm,
                "experiment": experiment,
            }
            records.append(rec)

            if output_dir:
                atomic_save_json(rec, ckpt)

            log(f"    trial={trial} alpha={alpha}: "
                f"flip={n_flipped}/{n_items} ({rec['flip_rate']:.3f})")

        # Memory cleanup
        # Only clear the cache for the active backend.
        dev = str(device).lower()
        if dev == "mps":
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        elif dev.startswith("cuda"):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Save per-item results for the designated trial
    if per_item_records and output_dir:
        pi_df = pd.DataFrame(per_item_records)
        pi_path = ensure_dir(output_dir / "per_item_random")
        pi_df.to_parquet(pi_path / f"{category}_trial_{save_per_item_trial}.parquet", index=False)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


def build_random_summary(
    random_df: "pd.DataFrame",
    feature_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build random_summary.json with attributable effects and t-tests.

    Parameters
    ----------
    random_df : DataFrame
        Output of run_random_trials across all categories.
    feature_results : dict
        Maps category → {"optimal_alpha": float, "correction_rate": float,
        "corruption_rate": float, ...} from Stage 3 experiment JSON files.
    """
    _require_pandas()
    summary: dict[str, Any] = {
        "n_trials": int(random_df["trial"].nunique()) if not random_df.empty else 0,
        "per_category": {},
    }

    if random_df.empty:
        return summary

    for cat in sorted(random_df["category"].unique()):
        cat_df = random_df[random_df["category"] == cat]
        feat = feature_results.get(cat, {})

        opt_alpha = feat.get("optimal_alpha", 0)
        feat_correction = feat.get("correction_rate", 0)
        feat_corruption = feat.get("corruption_rate", 0)

        entry: dict[str, Any] = {
            "optimal_alpha": opt_alpha,
        }

        # Correction (Experiment A — negative alphas)
        corr_df = cat_df[(cat_df["experiment"] == "A") & (cat_df["alpha"] == opt_alpha)]
        if not corr_df.empty:
            trials = corr_df["flip_rate"].values
            mean_r = float(np.mean(trials))
            std_r = float(np.std(trials, ddof=1)) if len(trials) > 1 else 0.0
            attr = feat_correction - mean_r

            t_stat, p_val = (0.0, 1.0)
            if len(trials) >= 3 and std_r > 1e-10:
                t_stat, p_val = ttest_1samp(trials, feat_correction)
                # One-sided: feature > random
                p_val = float(p_val / 2) if t_stat < 0 else float(1 - p_val / 2)
                t_stat = float(t_stat)

            entry.update({
                "vec_norm_at_optimal": float(corr_df["vec_norm"].iloc[0]),
                "feature_correction": feat_correction,
                "random_correction_mean": round(mean_r, 4),
                "random_correction_std": round(std_r, 4),
                "attributable_correction": round(attr, 4),
                "correction_ttest_t": round(t_stat, 3),
                "correction_ttest_p": round(p_val, 5),
            })

        # Corruption (Experiment B — positive alphas)
        opt_alpha_b = feat.get("optimal_alpha_b", abs(opt_alpha))
        corrupt_df = cat_df[(cat_df["experiment"] == "B") & (cat_df["alpha"] == opt_alpha_b)]
        if not corrupt_df.empty:
            trials = corrupt_df["flip_rate"].values
            mean_r = float(np.mean(trials))
            std_r = float(np.std(trials, ddof=1)) if len(trials) > 1 else 0.0
            attr = feat_corruption - mean_r

            t_stat, p_val = (0.0, 1.0)
            if len(trials) >= 3 and std_r > 1e-10:
                t_stat, p_val = ttest_1samp(trials, feat_corruption)
                p_val = float(p_val / 2) if t_stat < 0 else float(1 - p_val / 2)
                t_stat = float(t_stat)

            entry.update({
                "feature_corruption": feat_corruption,
                "random_corruption_mean": round(mean_r, 4),
                "random_corruption_std": round(std_r, 4),
                "attributable_corruption": round(attr, 4),
                "corruption_ttest_t": round(t_stat, 3),
                "corruption_ttest_p": round(p_val, 5),
            })

        # Per-alpha random curve (for overlay figure)
        per_alpha: dict[str, dict[str, float]] = {}
        for alpha in sorted(cat_df["alpha"].unique()):
            a_df = cat_df[cat_df["alpha"] == alpha]
            per_alpha[str(alpha)] = {
                "random_flip_mean": round(float(a_df["flip_rate"].mean()), 4),
                "random_flip_std": round(float(a_df["flip_rate"].std()), 4),
                "n_trials": int(a_df["trial"].nunique()),
            }
        entry["per_alpha_random"] = per_alpha

        summary["per_category"][cat] = entry

    return summary
