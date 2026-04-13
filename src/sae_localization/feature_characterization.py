"""Feature characterization: interpretability reports for bias-associated SAE features.

For each significant feature, produces:
- Top activating BBQ items
- Activation distribution (stereotyped vs non-stereotyped)
- Logit attribution (if model loaded)
- Co-activation analysis
- Subgroup breakdown
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import scipy.sparse as sp

from src.utils.io import atomic_save_json, ensure_dir
from src.utils.logging import log

# Optional dependency (only needed if you run characterization)
try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover
    pd = None  # type: ignore


def _require_pandas() -> None:
    global pd  # noqa: PLW0603
    if pd is not None:
        return
    try:
        import pandas as _pd  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pandas is required for SAE feature characterization. Install with: pip install pandas"
        ) from exc
    pd = _pd  # type: ignore

# Max features to characterise per layer
MAX_FEATURES = 50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_stimuli_text(
    data_dir: Path, categories: list[str]
) -> dict[int, dict[str, Any]]:
    """Load raw BBQ stimuli text, indexed by item_idx."""
    items: dict[int, dict[str, Any]] = {}
    try:
        from src.data.bbq_loader import find_bbq_file, load_bbq_items, CATEGORY_MAP
    except ImportError:
        log("  WARNING: cannot import bbq_loader; stimulus text unavailable")
        return items

    for cat_short in categories:
        try:
            path = find_bbq_file(cat_short, data_dir)
            bbq_items = load_bbq_items(path)
            for it in bbq_items:
                idx = it.get("item_idx", -1)
                items[idx] = it
        except Exception as exc:
            log(f"  WARNING: failed to load stimuli for {cat_short}: {exc}")
    return items


def _load_processed_stimuli(
    processed_dir: Path, categories: list[str]
) -> dict[int, dict[str, Any]]:
    """Load standardized processed stimuli JSON, indexed by item_idx.

    This is the preferred source for interpretability text because it matches the
    exact prompt content used in extraction (shuffled answers, etc.).
    """
    items: dict[int, dict[str, Any]] = {}
    for cat in categories:
        files = sorted(processed_dir.glob(f"stimuli_{cat}_*.json"))
        if not files:
            continue
        try:
            with open(files[-1]) as f:
                stimuli = json.load(f)
            for it in stimuli:
                try:
                    idx = int(it.get("item_idx", -1))
                except Exception:
                    continue
                if idx >= 0:
                    items[idx] = it
        except Exception as exc:
            log(f"  WARNING: failed to load processed stimuli for {cat}: {exc}")
    return items


def _top_activating_items(
    feature_idx: int,
    sparse_mat: sp.csr_matrix,
    meta_list: list[dict[str, Any]],
    stimuli: dict[int, dict[str, Any]],
    stage1_act_base: Path | None = None,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Find the items where this feature activates most strongly."""
    col = sparse_mat[:, feature_idx].toarray().ravel()
    top_indices = np.argsort(-col)[:top_k]

    results = []
    for idx in top_indices:
        if col[idx] <= 0:
            break
        meta = meta_list[idx]
        item_idx = meta.get("item_idx", -1)
        stim = stimuli.get(item_idx, {})
        model_answer = meta.get("model_answer", "") or ""

        # Fallback: load model_answer from Stage-1 item metadata if available.
        if (not model_answer) and stage1_act_base is not None:
            cat_short = meta.get("category_short", "")
            if cat_short:
                p = stage1_act_base / str(cat_short) / f"item_{int(item_idx):04d}.npz"
                try:
                    import numpy as _np
                    data = _np.load(p, allow_pickle=True)
                    raw = data.get("metadata_json", None)
                    if raw is not None:
                        meta_str = raw.item() if getattr(raw, "shape", None) == () else str(raw)
                        j = json.loads(meta_str)
                        model_answer = j.get("model_answer", "") or model_answer
                except Exception:
                    pass

        results.append({
            "item_idx": item_idx,
            "category": meta.get("category", ""),
            "context": stim.get("context", ""),
            "question": stim.get("question", ""),
            "answers": stim.get("answers", {}),
            "model_answer": model_answer,
            "model_answer_role": meta.get("model_answer_role", ""),
            "is_stereotyped_response": meta.get("is_stereotyped_response", False),
            "activation_value": float(col[idx]),
        })
    return results


def _subgroup_breakdown(
    feature_idx: int,
    sparse_mat: sp.csr_matrix,
    groups_lists: list[list[str]],
) -> dict[str, float]:
    """Mean activation of a feature broken down by stereotyped_groups."""
    col = sparse_mat[:, feature_idx].toarray().ravel()
    group_vals: dict[str, list[float]] = {}
    for i, gl in enumerate(groups_lists):
        for g in gl:
            group_vals.setdefault(g, []).append(col[i])
    return {g: float(np.mean(vals)) for g, vals in sorted(group_vals.items())}


def _co_activation(
    feature_idx: int,
    sparse_mat: sp.csr_matrix,
    significant_features: set[int],
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Find features most correlated with the target feature."""
    target_col = sparse_mat[:, feature_idx].toarray().ravel()
    if target_col.std() < 1e-10:
        return []

    results = []
    for other in significant_features:
        if other == feature_idx:
            continue
        other_col = sparse_mat[:, other].toarray().ravel()
        if other_col.std() < 1e-10:
            continue
        corr = float(np.corrcoef(target_col, other_col)[0, 1])
        results.append({"feature_idx": int(other), "correlation": corr})

    results.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    return results[:top_k]


def _logit_attribution(
    feature_idx: int,
    sae: Any,
    lm_head_weight: np.ndarray,
    tokenizer: Any,
    top_k: int = 10,
) -> dict[str, list[dict[str, Any]]]:
    """Compute which output tokens this feature promotes/suppresses.

    logit_effect = W_dec[feature_idx] @ lm_head.weight.T
    """
    direction = sae.get_feature_direction(feature_idx)  # unit norm
    # Get raw decoder direction (not normalised) for logit attribution
    raw_dec = sae._W_dec[feature_idx].detach().float().cpu().numpy()

    logit_effect = raw_dec @ lm_head_weight.T  # (vocab_size,)

    top_pos = np.argsort(-logit_effect)[:top_k]
    top_neg = np.argsort(logit_effect)[:top_k]

    promoted = [
        {
            "token": tokenizer.decode([int(t)]),
            "token_id": int(t),
            "logit_change": float(logit_effect[t]),
        }
        for t in top_pos
    ]
    suppressed = [
        {
            "token": tokenizer.decode([int(t)]),
            "token_id": int(t),
            "logit_change": float(logit_effect[t]),
        }
        for t in top_neg
    ]
    return {"promoted": promoted, "suppressed": suppressed}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_feature_characterization(
    cat_data: dict[str, dict[str, Any]],
    feature_results: dict[str, pd.DataFrame],
    sae: Any,
    target_layer: int,
    output_dir: Path,
    data_dir: Optional[Path] = None,
    localization_dir: Optional[Path] = None,
    model: Any = None,
    tokenizer: Any = None,
) -> list[dict[str, Any]]:
    """Characterize top bias-associated features.

    Parameters
    ----------
    cat_data : dict
        Per-category feature activations from feature_discovery.
    feature_results : dict
        Granularity → DataFrame from run_differential_analysis.
    sae : SAEWrapper
    target_layer : int
    output_dir : Path
    data_dir : Path | None
        BBQ data directory for loading stimulus text.
    model : Any | None
        Language model (for logit attribution).  If None, logit
        attribution is skipped.
    tokenizer : Any | None
        Tokenizer (for logit attribution).

    Returns
    -------
    list of per-feature characterisation dicts.
    """
    reports_dir = ensure_dir(output_dir / "feature_reports")
    categories = list(cat_data.keys())

    # Load stimuli text (prefer processed stimuli JSON; fall back to raw BBQ JSONL if provided)
    stimuli: dict[int, dict[str, Any]] = {}
    processed_dir = Path("data") / "processed"
    if processed_dir.is_dir():
        stimuli = _load_processed_stimuli(processed_dir, categories)
        if stimuli:
            log(f"  Loaded {len(stimuli)} processed stimulus texts from {processed_dir}")
    if not stimuli and data_dir is not None:
        stimuli = _load_stimuli_text(data_dir, categories)
        log(f"  Loaded {len(stimuli)} raw BBQ stimulus texts from {data_dir}")

    stage1_act_base = None
    if localization_dir is not None:
        base = Path(localization_dir) / "activations"
        if base.is_dir():
            stage1_act_base = base

    # Pool all activations and metadata
    all_sparse: list[sp.csr_matrix] = []
    all_meta: list[dict[str, Any]] = []
    all_groups: list[list[str]] = []

    for cat, d in cat_data.items():
        mat = d["activations_sparse"]
        n_items = mat.shape[0]
        for i in range(n_items):
            all_meta.append({
                "item_idx": int(d["item_indices"][i]),
                "is_stereotyped_response": bool(d["is_stereotyped"][i]),
                "model_answer_role": str(d["model_answer_roles"][i]),
                "context_condition": str(d["context_conditions"][i]),
                "category": str(d["categories_arr"][i]),
                "category_short": str(cat),
                "model_answer": "",  # not in saved metadata
            })
            all_groups.append(
                d["stereotyped_groups"][i]
                if isinstance(d["stereotyped_groups"], list)
                else []
            )
        all_sparse.append(mat)

    if not all_sparse:
        log("  No data for characterization")
        return []

    pooled_mat = sp.vstack(all_sparse, format="csr")
    log(f"  Pooled matrix: {pooled_mat.shape[0]} items x {pooled_mat.shape[1]} features")

    # Select top features by |Cohen's d| from pooled results
    pooled_df = feature_results.get("pooled", pd.DataFrame())
    per_cat_df = feature_results.get("per_category", pd.DataFrame())

    if pooled_df.empty and per_cat_df.empty:
        log("  No significant features to characterize")
        return []

    # Merge all significant features, rank by max |d|
    sig_frames = []
    for key, df in feature_results.items():
        if df.empty:
            continue
        sig = df[df["is_significant"]].copy()
        if not sig.empty:
            sig_frames.append(sig)

    if not sig_frames:
        log("  No significant features to characterize")
        return []

    all_sig = pd.concat(sig_frames, ignore_index=True)
    sig_features = set(all_sig["feature_idx"].unique())

    # Rank by max |d| across all analyses
    max_d = (
        all_sig.groupby("feature_idx")["cohens_d"]
        .apply(lambda x: x.abs().max())
        .sort_values(ascending=False)
    )
    top_features = list(max_d.index[:MAX_FEATURES])
    log(f"  Characterizing top {len(top_features)} features ...")

    # Prepare logit attribution resources
    lm_head_weight = None
    if model is not None and tokenizer is not None:
        try:
            import torch
            lm_head = model.lm_head if hasattr(model, "lm_head") else None
            if lm_head is not None:
                lm_head_weight = lm_head.weight.detach().float().cpu().numpy()
                log(f"  LM head weight: {lm_head_weight.shape}")
        except Exception as exc:
            log(f"  WARNING: cannot access lm_head: {exc}")

    # Determine which categories each feature is significant in
    cat_sig_map: dict[int, list[str]] = {}
    sub_sig_map: dict[int, list[str]] = {}
    for _, row in all_sig.iterrows():
        fid = int(row["feature_idx"])
        if row["granularity"] == "per_category":
            cat_sig_map.setdefault(fid, []).append(row["category"])
        elif row["granularity"] == "per_subcategory":
            sub_sig_map.setdefault(fid, []).append(row["subcategory"])

    # Characterize each feature
    reports: list[dict[str, Any]] = []
    for rank, fid in enumerate(top_features):
        fid = int(fid)
        log(f"    [{rank + 1}/{len(top_features)}] L{target_layer}_F{fid}")

        feat_dir = ensure_dir(reports_dir / f"feature_L{target_layer}_F{fid}")

        # Basic info from all_sig
        feat_rows = all_sig[all_sig["feature_idx"] == fid]
        best_row = feat_rows.loc[feat_rows["cohens_d"].abs().idxmax()]

        report: dict[str, Any] = {
            "feature_idx": fid,
            "layer": target_layer,
            "feature_label": f"L{target_layer}_F{fid}",
            "cohens_d": float(best_row["cohens_d"]),
            "direction": str(best_row["direction"]),
            "categories_significant_in": cat_sig_map.get(fid, []),
            "subcategories_significant_in": sub_sig_map.get(fid, []),
        }

        # Top activating items
        top_items = _top_activating_items(
            fid, pooled_mat, all_meta, stimuli, stage1_act_base=stage1_act_base
        )
        report["top_activating_items"] = top_items
        atomic_save_json(top_items, feat_dir / "top_activating_items.json")

        # Subgroup breakdown
        subgroup_means = _subgroup_breakdown(fid, pooled_mat, all_groups)
        report["subgroup_mean_activations"] = subgroup_means
        atomic_save_json(
            subgroup_means, feat_dir / "subgroup_breakdown.json"
        )

        # Co-activation
        coact = _co_activation(fid, pooled_mat, sig_features)
        report["co_activated_features"] = [c["feature_idx"] for c in coact[:3]]
        atomic_save_json(coact, feat_dir / "co_activation.json")

        # Logit attribution
        if lm_head_weight is not None and tokenizer is not None:
            logit_attr = _logit_attribution(
                fid, sae, lm_head_weight, tokenizer
            )
            report["top_promoted_tokens"] = logit_attr["promoted"][:5]
            report["top_suppressed_tokens"] = logit_attr["suppressed"][:5]
            atomic_save_json(logit_attr, feat_dir / "logit_attribution.json")
        else:
            report["top_promoted_tokens"] = []
            report["top_suppressed_tokens"] = []

        # Activation distribution data (for figures)
        col = pooled_mat[:, fid].toarray().ravel()
        stereo_mask = np.array(
            [m["is_stereotyped_response"] for m in all_meta]
        )
        role_mask = np.array(
            [m["model_answer_role"] for m in all_meta]
        )
        non_stereo_mask = (~stereo_mask) & (role_mask != "unknown")

        report["activation_dist"] = {
            "stereotyped_values": col[stereo_mask].tolist(),
            "non_stereotyped_values": col[non_stereo_mask].tolist(),
        }

        # Per-category mean activation (for profile figures)
        cat_means: dict[str, dict[str, float]] = {}
        cat_arr = np.array([m["category"] for m in all_meta])
        for cat in sorted(set(cat_arr)):
            cat_mask = cat_arr == cat
            s_mask = cat_mask & stereo_mask
            ns_mask = cat_mask & non_stereo_mask
            cat_means[cat] = {
                "stereotyped": float(col[s_mask].mean()) if s_mask.any() else 0.0,
                "non_stereotyped": float(col[ns_mask].mean()) if ns_mask.any() else 0.0,
            }
        report["per_category_means"] = cat_means

        reports.append(report)

    # Save combined reports
    combined_path = output_dir / f"feature_characterizations_layer_{target_layer}.json"
    atomic_save_json(reports, combined_path)
    log(f"  Saved {len(reports)} feature characterizations")

    return reports
