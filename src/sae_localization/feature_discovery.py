"""SAE feature discovery: identify bias-associated SAE features.

Loads Stage-1 activation .npz files, passes them through a pre-trained SAE
encoder, and runs differential activation analysis at three granularity
levels: pooled, per-category, and per-subcategory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import scipy.sparse as sp
from scipy.stats import mannwhitneyu

from src.utils.io import atomic_save_json, ensure_dir
from src.utils.logging import ProgressLogger, log


try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover
    pd = None  # type: ignore


def _require_pandas() -> None:
    """Ensure pandas is importable; raise an actionable error if not."""
    global pd  # noqa: PLW0603
    if pd is not None:
        return
    try:
        import pandas as _pd  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pandas is required for SAE feature discovery outputs (DataFrames / parquet). "
            "Install with: pip install pandas"
        ) from exc
    pd = _pd  # type: ignore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FDR_THRESHOLD = 0.05
COHENS_D_THRESHOLD = 0.3
MIN_FIRING_RATE = 0.05
MIN_SUBGROUP_SIZE = 15


# ---------------------------------------------------------------------------
# Step 1 — Collect SAE latent activations
# ---------------------------------------------------------------------------


def _load_item_npz(path: Path) -> dict[str, Any] | None:
    """Load a single Stage-1 item .npz and return dict with arrays + metadata."""
    try:
        data = np.load(path, allow_pickle=True)
    except Exception as exc:
        log(f"  WARNING: failed to load {path}: {exc}")
        return None

    hs = data["hidden_states"].astype(np.float32)  # (n_layers, hidden_dim)
    norms = data["hidden_states_raw_norms"]  # (n_layers,)

    raw = data["metadata_json"]
    meta_str = raw.item() if raw.shape == () else str(raw)
    meta = json.loads(meta_str)

    return {"hidden_states": hs, "raw_norms": norms, "meta": meta}


def collect_sae_activations(
    activations_dir: Path,
    sae: Any,
    target_layer: int,
    categories: list[str],
    output_dir: Path,
    max_items: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Encode Stage-1 hidden states through the SAE and save sparse results.

    Parameters
    ----------
    activations_dir : Path
        Root directory with per-category subdirs of item_XXXX.npz files.
    sae : SAEWrapper
        Pre-trained SAE for the target layer.
    target_layer : int
        Layer index to extract from saved hidden states.
    categories : list[str]
        Category short names to process.
    output_dir : Path
        Where to save feature activation .npz files.
    max_items : int | None
        Limit items per category (for testing).

    Returns
    -------
    dict mapping category → dict with keys:
        activations_sparse (csr_matrix), item_indices, is_stereotyped,
        model_answer_roles, context_conditions, stereotyped_groups,
        categories_arr.
    """
    import torch

    ensure_dir(output_dir)
    all_results: dict[str, dict[str, Any]] = {}

    for cat in categories:
        cat_dir = activations_dir / cat
        if not cat_dir.is_dir():
            log(f"  Skipping category '{cat}': no activations directory")
            continue

        npz_files = sorted(cat_dir.glob("item_*.npz"))
        if max_items is not None:
            npz_files = npz_files[:max_items]
        if not npz_files:
            log(f"  Skipping category '{cat}': no .npz files")
            continue

        log(f"  Encoding {len(npz_files)} items for category '{cat}' "
            f"(layer {target_layer}) ...")

        rows_data: list[np.ndarray] = []  # sparse rows
        item_indices: list[int] = []
        is_stereotyped: list[bool] = []
        answer_roles: list[str] = []
        context_conds: list[str] = []
        stereo_groups: list[list[str]] = []
        cat_labels: list[str] = []

        progress = ProgressLogger(len(npz_files), prefix=f"  [{cat}]")
        for npz_path in npz_files:
            item = _load_item_npz(npz_path)
            if item is None:
                progress.skip(reason="load failed")
                continue

            meta = item["meta"]
            hs_layer = item["hidden_states"][target_layer]  # (hidden_dim,)
            norm = item["raw_norms"][target_layer]

            # De-normalise: recover raw activation magnitude
            raw_activation = hs_layer * norm

            # Encode through SAE
            with torch.no_grad():
                act_tensor = torch.from_numpy(raw_activation).to(sae.device)
                feat_acts = sae.encode(act_tensor).float().cpu().numpy()

            rows_data.append(feat_acts)
            item_indices.append(meta.get("item_idx", -1))
            is_stereotyped.append(bool(meta.get("is_stereotyped_response", False)))
            answer_roles.append(meta.get("model_answer_role", "unknown"))
            context_conds.append(meta.get("context_condition", ""))
            stereo_groups.append(meta.get("stereotyped_groups", []))
            cat_labels.append(meta.get("category", cat))

            if len(rows_data) % 100 == 0:
                progress.count = len(rows_data)
                progress.step()

        if not rows_data:
            log(f"  No valid items for category '{cat}'")
            continue

        # Stack and convert to sparse
        dense = np.vstack(rows_data)  # (n_items, n_features)
        sparse_mat = sp.csr_matrix(dense)

        n_items = sparse_mat.shape[0]
        n_active_mean = sparse_mat.nnz / max(n_items, 1)
        log(f"  {cat}: {n_items} items, mean {n_active_mean:.1f} active "
            f"features/item, sparse density={sparse_mat.nnz / max(1, np.prod(sparse_mat.shape)):.6f}")

        # Save sparse
        cat_out = ensure_dir(output_dir / cat)
        save_path = cat_out / f"layer_{target_layer}.npz"
        sp.save_npz(save_path, sparse_mat)

        # Save metadata sidecar
        meta_path = cat_out / f"layer_{target_layer}_meta.json"
        atomic_save_json(
            {
                "item_indices": item_indices,
                "is_stereotyped": is_stereotyped,
                "model_answer_roles": answer_roles,
                "context_conditions": context_conds,
                "stereotyped_groups": stereo_groups,
                "categories": cat_labels,
                "n_items": n_items,
                "n_features": int(sparse_mat.shape[1]),
                "layer": target_layer,
            },
            meta_path,
        )

        all_results[cat] = {
            "activations_sparse": sparse_mat,
            "item_indices": np.array(item_indices),
            "is_stereotyped": np.array(is_stereotyped),
            "model_answer_roles": np.array(answer_roles),
            "context_conditions": np.array(context_conds),
            "stereotyped_groups": stereo_groups,
            "categories_arr": np.array(cat_labels),
        }

    return all_results


def load_sae_activations(
    feature_dir: Path,
    target_layer: int,
    categories: list[str],
) -> dict[str, dict[str, Any]]:
    """Load previously-saved SAE feature activations from disk.

    Mirror of what :func:`collect_sae_activations` returns.
    """
    results: dict[str, dict[str, Any]] = {}
    for cat in categories:
        sparse_path = feature_dir / cat / f"layer_{target_layer}.npz"
        meta_path = feature_dir / cat / f"layer_{target_layer}_meta.json"
        if not sparse_path.exists() or not meta_path.exists():
            log(
                f"  Skipping {cat}: no cached SAE feature_activations for layer {target_layer} yet "
                f"(will encode after SAE loads)"
            )
            continue

        sparse_mat = sp.load_npz(sparse_path)
        with open(meta_path) as f:
            meta = json.load(f)

        results[cat] = {
            "activations_sparse": sparse_mat,
            "item_indices": np.array(meta["item_indices"]),
            "is_stereotyped": np.array(meta["is_stereotyped"]),
            "model_answer_roles": np.array(meta["model_answer_roles"]),
            "context_conditions": np.array(meta["context_conditions"]),
            "stereotyped_groups": meta["stereotyped_groups"],
            "categories_arr": np.array(meta["categories"]),
        }
    return results


# ---------------------------------------------------------------------------
# Step 2 — Differential feature analysis
# ---------------------------------------------------------------------------


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction.  Returns FDR-adjusted p-values."""
    n = len(p_values)
    if n == 0:
        return np.array([])
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    adjusted = np.empty(n, dtype=np.float64)
    cummin = 1.0
    for i in range(n - 1, -1, -1):
        val = sorted_p[i] * n / (i + 1)
        cummin = min(cummin, val)
        adjusted[sorted_idx[i]] = min(cummin, 1.0)
    return adjusted


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d with pooled standard deviation.  Handles zero-variance."""
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return 0.0
    m_a, m_b = a.mean(), b.mean()
    v_a, v_b = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((n_a - 1) * v_a + (n_b - 1) * v_b) / (n_a + n_b - 2))
    if pooled < 1e-12:
        return 0.0
    return float((m_a - m_b) / pooled)


def _differential_test_features(
    sparse_mat: sp.csr_matrix,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
) -> pd.DataFrame:
    """Run Mann-Whitney U test for each feature between two groups.

    Parameters
    ----------
    sparse_mat : csr_matrix  (n_items, n_features)
    mask_a, mask_b : boolean arrays (n_items,)

    Returns DataFrame with per-feature test results (no FDR yet).
    """
    dense_a = sparse_mat[mask_a].toarray()  # (n_a, n_features)
    dense_b = sparse_mat[mask_b].toarray()  # (n_b, n_features)
    n_a, n_b = dense_a.shape[0], dense_b.shape[0]
    n_features = sparse_mat.shape[1]

    if n_a < 2 or n_b < 2:
        return pd.DataFrame()

    # Pre-filter: only test features with nonzero in at least one group
    nnz_a = np.diff(sparse_mat[mask_a].tocsc().indptr)
    nnz_b = np.diff(sparse_mat[mask_b].tocsc().indptr)
    firing_a = nnz_a / n_a
    firing_b = nnz_b / n_b
    test_mask = (firing_a > 0) | (firing_b > 0)
    test_indices = np.where(test_mask)[0]

    if len(test_indices) == 0:
        return pd.DataFrame()

    log(f"    Testing {len(test_indices)} features with nonzero activations "
        f"(n_a={n_a}, n_b={n_b})")

    results: list[dict[str, Any]] = []
    for j in test_indices:
        col_a = dense_a[:, j]
        col_b = dense_b[:, j]

        mean_a = float(col_a.mean())
        mean_b = float(col_b.mean())

        d = _cohens_d(col_a, col_b)

        # Mann-Whitney U
        try:
            _, p = mannwhitneyu(col_a, col_b, alternative="two-sided")
        except ValueError:
            p = 1.0

        results.append({
            "feature_idx": int(j),
            "mean_stereotyped": mean_a,
            "mean_non_stereotyped": mean_b,
            "cohens_d": d,
            "p_value": float(p),
            "firing_rate_stereotyped": float(firing_a[j]),
            "firing_rate_non_stereotyped": float(firing_b[j]),
            "n_stereotyped": n_a,
            "n_non_stereotyped": n_b,
        })

    return pd.DataFrame(results)


def _apply_significance(df: pd.DataFrame) -> pd.DataFrame:
    """Apply FDR correction and significance thresholds.  Modifies in place."""
    if df.empty:
        return df
    df["p_value_fdr"] = _benjamini_hochberg(df["p_value"].values)
    df["is_significant"] = (
        (df["p_value_fdr"] < FDR_THRESHOLD)
        & (df["cohens_d"].abs() > COHENS_D_THRESHOLD)
        & (
            (df["firing_rate_stereotyped"] > MIN_FIRING_RATE)
            | (df["firing_rate_non_stereotyped"] > MIN_FIRING_RATE)
        )
    )
    df["direction"] = np.where(
        df["mean_stereotyped"] > df["mean_non_stereotyped"],
        "pro_bias",
        "anti_bias",
    )
    return df


# ---------------------------------------------------------------------------
# Public entry point: run_differential_analysis
# ---------------------------------------------------------------------------


def run_differential_analysis(
    cat_data: dict[str, dict[str, Any]],
    target_layer: int,
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Run differential feature analysis at all three granularity levels.

    Parameters
    ----------
    cat_data : dict
        Output of :func:`collect_sae_activations` or :func:`load_sae_activations`.
    target_layer : int
    output_dir : Path
        Where to save parquet results.

    Returns
    -------
    dict mapping granularity name → DataFrame.
    """
    _require_pandas()
    features_dir = ensure_dir(output_dir / "features")
    results: dict[str, pd.DataFrame] = {}

    # ---- Level 1: Pooled ----
    log("  Level 1: Pooled analysis (all categories) ...")
    pooled_df = _pooled_analysis(cat_data, target_layer)
    if not pooled_df.empty:
        path = features_dir / f"pooled_layer_{target_layer}.parquet"
        pooled_df.to_parquet(path, index=False)
        n_sig = pooled_df["is_significant"].sum()
        log(f"    Pooled: {n_sig} significant features "
            f"(of {len(pooled_df)} tested)")
    results["pooled"] = pooled_df

    # ---- Level 2: Per category ----
    log("  Level 2: Per-category analysis ...")
    per_cat_df = _per_category_analysis(cat_data, target_layer)
    if not per_cat_df.empty:
        path = features_dir / f"per_category_layer_{target_layer}.parquet"
        per_cat_df.to_parquet(path, index=False)
        for cat in per_cat_df["category"].unique():
            n_sig = per_cat_df.loc[
                (per_cat_df["category"] == cat) & per_cat_df["is_significant"]
            ].shape[0]
            log(f"    {cat}: {n_sig} significant features")
    results["per_category"] = per_cat_df

    # ---- Level 3: Per subcategory ----
    log("  Level 3: Per-subcategory analysis ...")
    per_sub_df = _per_subcategory_analysis(cat_data, target_layer)
    if not per_sub_df.empty:
        path = features_dir / f"per_subcategory_layer_{target_layer}.parquet"
        per_sub_df.to_parquet(path, index=False)
        for sub in per_sub_df["subcategory"].unique():
            n_sig = per_sub_df.loc[
                (per_sub_df["subcategory"] == sub) & per_sub_df["is_significant"]
            ].shape[0]
            log(f"    {sub}: {n_sig} significant features")
    results["per_subcategory"] = per_sub_df

    # ---- Feature overlap analysis ----
    log("  Computing feature overlap ...")
    overlap = compute_feature_overlap(results, target_layer)
    overlap_path = output_dir / f"feature_overlap_layer_{target_layer}.json"
    atomic_save_json(overlap, overlap_path)

    return results


# ---------------------------------------------------------------------------
# Level 1: Pooled
# ---------------------------------------------------------------------------


def _pooled_analysis(
    cat_data: dict[str, dict[str, Any]], target_layer: int
) -> pd.DataFrame:
    """Differential analysis pooled across all categories."""
    all_sparse: list[sp.csr_matrix] = []
    all_stereo: list[np.ndarray] = []
    all_roles: list[np.ndarray] = []

    for cat, d in cat_data.items():
        all_sparse.append(d["activations_sparse"])
        all_stereo.append(d["is_stereotyped"])
        all_roles.append(d["model_answer_roles"])

    if not all_sparse:
        return pd.DataFrame()

    pooled_mat = sp.vstack(all_sparse, format="csr")
    pooled_stereo = np.concatenate(all_stereo)
    pooled_roles = np.concatenate(all_roles)

    mask_a = pooled_stereo.astype(bool)
    mask_b = (~pooled_stereo) & (pooled_roles != "unknown")

    df = _differential_test_features(pooled_mat, mask_a, mask_b)
    if df.empty:
        return df

    df = _apply_significance(df)
    df["layer"] = target_layer
    df["granularity"] = "pooled"
    df["category"] = "all"
    df["subcategory"] = "all"
    return df


# ---------------------------------------------------------------------------
# Level 2: Per category
# ---------------------------------------------------------------------------


def _per_category_analysis(
    cat_data: dict[str, dict[str, Any]], target_layer: int
) -> pd.DataFrame:
    """Differential analysis per BBQ category."""
    frames: list[pd.DataFrame] = []

    for cat, d in cat_data.items():
        mat = d["activations_sparse"]
        stereo = d["is_stereotyped"]
        roles = d["model_answer_roles"]

        mask_a = stereo.astype(bool)
        mask_b = (~stereo) & (roles != "unknown")

        df = _differential_test_features(mat, mask_a, mask_b)
        if df.empty:
            continue

        df = _apply_significance(df)
        df["layer"] = target_layer
        df["granularity"] = "per_category"
        df["category"] = cat
        df["subcategory"] = "all"
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Level 3: Per subcategory
# ---------------------------------------------------------------------------


def _per_subcategory_analysis(
    cat_data: dict[str, dict[str, Any]], target_layer: int
) -> pd.DataFrame:
    """Differential analysis per subgroup within each category."""
    frames: list[pd.DataFrame] = []

    for cat, d in cat_data.items():
        mat = d["activations_sparse"]
        stereo = d["is_stereotyped"]
        roles = d["model_answer_roles"]
        groups_lists = d["stereotyped_groups"]  # list of lists

        # Collect unique subgroups
        subgroup_counts: dict[str, int] = {}
        for gl in groups_lists:
            for g in gl:
                subgroup_counts[g] = subgroup_counts.get(g, 0) + 1

        for subgroup in sorted(subgroup_counts.keys()):
            # Items targeting this subgroup
            item_mask = np.array(
                [subgroup in gl for gl in groups_lists], dtype=bool
            )
            if item_mask.sum() < 5:
                continue

            sub_mat = mat[item_mask]
            sub_stereo = stereo[item_mask]
            sub_roles = roles[item_mask]

            mask_a = sub_stereo.astype(bool)
            mask_b = (~sub_stereo) & (sub_roles != "unknown")

            n_a = mask_a.sum()
            n_b = mask_b.sum()

            if n_a < MIN_SUBGROUP_SIZE or n_b < MIN_SUBGROUP_SIZE:
                log(f"    Skipping {cat}/{subgroup}: "
                    f"n_stereo={n_a}, n_non_stereo={n_b} "
                    f"(need >= {MIN_SUBGROUP_SIZE} each)")
                continue

            df = _differential_test_features(sub_mat, mask_a, mask_b)
            if df.empty:
                continue

            df = _apply_significance(df)
            df["layer"] = target_layer
            df["granularity"] = "per_subcategory"
            df["category"] = cat
            df["subcategory"] = subgroup
            frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Step 3 — Feature overlap analysis
# ---------------------------------------------------------------------------


def compute_feature_overlap(
    results: dict[str, pd.DataFrame],
    target_layer: int,
) -> dict[str, Any]:
    """Compute cross-category and subgroup overlap metrics."""
    overlap: dict[str, Any] = {"layer": target_layer}

    # --- Cross-category Jaccard ---
    per_cat_df = results.get("per_category", pd.DataFrame())
    if not per_cat_df.empty:
        sig_df = per_cat_df[per_cat_df["is_significant"]]
        cats = sorted(sig_df["category"].unique())
        cat_sets: dict[str, set[int]] = {
            c: set(sig_df.loc[sig_df["category"] == c, "feature_idx"].tolist())
            for c in cats
        }

        jaccard_matrix: dict[str, dict[str, float]] = {}
        for c1 in cats:
            jaccard_matrix[c1] = {}
            for c2 in cats:
                if c1 == c2:
                    jaccard_matrix[c1][c2] = len(cat_sets[c1])  # count on diagonal
                else:
                    inter = len(cat_sets[c1] & cat_sets[c2])
                    union = len(cat_sets[c1] | cat_sets[c2])
                    jaccard_matrix[c1][c2] = inter / union if union > 0 else 0.0

        overlap["cross_category_jaccard"] = jaccard_matrix
        overlap["significant_per_category"] = {
            c: len(s) for c, s in cat_sets.items()
        }

    # --- Subgroup specificity ---
    per_sub_df = results.get("per_subcategory", pd.DataFrame())
    if not per_sub_df.empty:
        sig_sub = per_sub_df[per_sub_df["is_significant"]]
        subgroup_specificity: dict[str, dict[str, Any]] = {}

        for cat in sig_sub["category"].unique():
            cat_sub = sig_sub[sig_sub["category"] == cat]
            subs = sorted(cat_sub["subcategory"].unique())
            sub_sets: dict[str, set[int]] = {
                s: set(cat_sub.loc[cat_sub["subcategory"] == s, "feature_idx"].tolist())
                for s in subs
            }

            pairwise_jaccard: dict[str, dict[str, float]] = {}
            asymmetric: dict[str, dict[str, float]] = {}
            for s1 in subs:
                pairwise_jaccard[s1] = {}
                asymmetric[s1] = {}
                for s2 in subs:
                    if s1 == s2:
                        pairwise_jaccard[s1][s2] = len(sub_sets[s1])
                    else:
                        inter = len(sub_sets[s1] & sub_sets[s2])
                        union = len(sub_sets[s1] | sub_sets[s2])
                        pairwise_jaccard[s1][s2] = inter / union if union else 0.0
                        # Fraction of s1's features also in s2
                        n1 = len(sub_sets[s1])
                        asymmetric[s1][s2] = inter / n1 if n1 else 0.0

            subgroup_specificity[cat] = {
                "jaccard": pairwise_jaccard,
                "asymmetric_overlap": asymmetric,
                "feature_counts": {s: len(sub_sets[s]) for s in subs},
            }

        overlap["subgroup_specificity"] = subgroup_specificity

    # --- Feature breadth ---
    pooled_df = results.get("pooled", pd.DataFrame())
    if not per_cat_df.empty:
        sig_per_cat = per_cat_df[per_cat_df["is_significant"]]
        breadth_counts: dict[int, int] = {}
        for fid in sig_per_cat["feature_idx"].unique():
            n_cats = sig_per_cat.loc[
                sig_per_cat["feature_idx"] == fid, "category"
            ].nunique()
            breadth_counts[int(fid)] = n_cats

        breadth_dist: dict[str, int] = {}
        for n_cats in range(1, 8):
            breadth_dist[str(n_cats)] = sum(
                1 for v in breadth_counts.values() if v == n_cats
            )

        overlap["feature_breadth"] = {
            "distribution": breadth_dist,
            "narrow_1_cat": sum(1 for v in breadth_counts.values() if v == 1),
            "broad_5plus_cats": sum(1 for v in breadth_counts.values() if v >= 5),
            "per_feature": {str(k): v for k, v in breadth_counts.items()},
        }

    return overlap


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run differential SAE feature analysis"
    )
    parser.add_argument(
        "--activations_dir",
        required=True,
        help="Directory with feature_activations/{cat}/layer_N.npz",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--categories", default="all")
    args = parser.parse_args()

    from src.data.bbq_loader import parse_categories

    cats = parse_categories(args.categories)
    data = load_sae_activations(
        Path(args.activations_dir), args.layer, cats
    )

    run_differential_analysis(data, args.layer, Path(args.output_dir))
