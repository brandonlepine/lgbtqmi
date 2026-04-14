"""Hybrid DIM-projection analysis: project SAE decoder columns onto DIM directions.

This module is OPTIONAL — it runs only when DIM direction files from the
main fragmentation pipeline exist at ``results/runs/<model_id>/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy.stats import spearmanr

from src.utils.io import atomic_save_json, ensure_dir
from src.utils.logging import log

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
            "pandas is required for SAE hybrid projection analysis. Install with: pip install pandas"
        ) from exc
    pd = _pd  # type: ignore

# ---------------------------------------------------------------------------
# Direction discovery
# ---------------------------------------------------------------------------


def find_dim_directions(
    run_dirs: list[Path],
) -> dict[str, np.ndarray]:
    """Scan run directories for DIM direction files (.npy or .npz).

    Supports both individual ``.npy`` files (e.g. ``direction_so.npy``)
    and bundled ``.npz`` archives (e.g. ``directions.npz`` from
    ``compute_directions.py`` with keys like ``direction_so``).

    Returns dict mapping a label (e.g. ``"so"``) to direction vector
    of shape ``(hidden_dim,)``.  Returns empty dict if nothing found.
    """
    directions: dict[str, np.ndarray] = {}

    npy_patterns = [
        "dim_direction_*.npy",
        "direction_*.npy",
        "*_direction.npy",
        "bias_direction_*.npy",
    ]

    for rd in run_dirs:
        if not rd.is_dir():
            continue

        # Scan for individual .npy files
        for pat in npy_patterns:
            for f in rd.glob(pat):
                stem = f.stem
                for prefix in ("dim_direction_", "direction_", "bias_direction_"):
                    if stem.startswith(prefix):
                        label = stem[len(prefix):]
                        break
                else:
                    label = stem.replace("_direction", "")

                try:
                    d = np.load(f).astype(np.float32)
                    if d.ndim == 1:
                        directions[label] = d
                        log(f"  Found DIM direction: {label} from {f}")
                except Exception as exc:
                    log(f"  WARNING: failed to load {f}: {exc}")

        # Scan for bundled .npz archives (e.g. analysis/directions.npz)
        for npz_path in rd.glob("**/directions.npz"):
            try:
                data = np.load(npz_path, allow_pickle=True)
                for key in data.files:
                    if key.startswith("_"):
                        continue  # skip metadata keys
                    arr = data[key]
                    if not isinstance(arr, np.ndarray) or arr.ndim != 1:
                        continue
                    # Extract label: "direction_so" -> "so", "subgroup_so_gay" -> "so_gay"
                    label = key
                    for prefix in ("direction_", "subgroup_"):
                        if key.startswith(prefix):
                            label = key[len(prefix):]
                            break
                    directions[label] = arr.astype(np.float32)
                    log(f"  Found DIM direction: {label} from {npz_path}")
            except Exception as exc:
                log(f"  WARNING: failed to load {npz_path}: {exc}")

    return directions


# ---------------------------------------------------------------------------
# Projection analysis
# ---------------------------------------------------------------------------


def run_hybrid_projection(
    sae: Any,
    dim_directions: dict[str, np.ndarray],
    feature_results: Optional[pd.DataFrame],
    target_layer: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Project SAE decoder columns onto DIM directions and cross-reference.

    Parameters
    ----------
    sae : SAEWrapper
    dim_directions : dict
        Label → direction vector (hidden_dim,).
    feature_results : DataFrame | None
        Pooled or per-category results from feature_discovery (needs
        ``feature_idx`` and ``cohens_d`` columns).
    target_layer : int
    output_dir : Path

    Returns
    -------
    dict with projection results and validation metrics.
    """
    _require_pandas()
    hybrid_dir = ensure_dir(output_dir / "hybrid_projection")
    decoder_matrix = sae.get_decoder_matrix()  # (n_features, hidden_dim)
    n_features = decoder_matrix.shape[0]

    summary: dict[str, Any] = {"layer": target_layer, "directions_analysed": []}

    for label, direction in dim_directions.items():
        # Ensure matching dimensions
        if direction.shape[0] != decoder_matrix.shape[1]:
            log(f"  Skipping direction '{label}': dim mismatch "
                f"({direction.shape[0]} vs {decoder_matrix.shape[1]})")
            continue

        # Normalise direction
        d_norm = np.linalg.norm(direction)
        if d_norm < 1e-8:
            log(f"  Skipping direction '{label}': near-zero norm")
            continue
        d_hat = direction / d_norm

        # Cosine similarity of each decoder column with DIM direction
        # decoder_matrix rows are already unit-normalised
        cos_scores = decoder_matrix @ d_hat  # (n_features,)

        # Rank by |cosine|
        abs_cos = np.abs(cos_scores)
        ranked = np.argsort(-abs_cos)

        top_aligned = ranked[:20]
        top_anti = ranked[-20:][::-1]  # least aligned

        # For top-aligned, get those with positive vs negative cosine
        pos_ranked = np.argsort(-cos_scores)[:20]
        neg_ranked = np.argsort(cos_scores)[:20]

        projection_result = {
            "direction_label": label,
            "top_20_aligned": [
                {
                    "feature_idx": int(idx),
                    "cosine": float(cos_scores[idx]),
                    "abs_cosine": float(abs_cos[idx]),
                }
                for idx in top_aligned
            ],
            "top_20_positive": [
                {"feature_idx": int(idx), "cosine": float(cos_scores[idx])}
                for idx in pos_ranked
            ],
            "top_20_negative": [
                {"feature_idx": int(idx), "cosine": float(cos_scores[idx])}
                for idx in neg_ranked
            ],
        }

        # Cross-reference with differential analysis
        if feature_results is not None and not feature_results.empty:
            # Build feature_idx → cohens_d map
            d_map = dict(
                zip(
                    feature_results["feature_idx"].values,
                    feature_results["cohens_d"].values,
                )
            )
            sig_set = set(
                feature_results.loc[
                    feature_results["is_significant"], "feature_idx"
                ].values
            )

            # Compute rank correlation: |cos with DIM| vs |Cohen's d|
            shared_features = [
                f for f in range(n_features) if f in d_map
            ]
            if len(shared_features) > 10:
                cos_vals = np.array([abs_cos[f] for f in shared_features])
                d_vals = np.array([abs(d_map[f]) for f in shared_features])
                rho, p_rho = spearmanr(cos_vals, d_vals)
                projection_result["spearman_r"] = float(rho)
                projection_result["spearman_p"] = float(p_rho)
                log(f"    {label}: Spearman r={rho:.3f} (p={p_rho:.4f})")

            # Categorise top-aligned features
            top_aligned_set = set(int(i) for i in top_aligned)
            n_both = len(top_aligned_set & sig_set)
            n_dim_only = len(top_aligned_set - sig_set)
            projection_result["top_aligned_also_significant"] = n_both
            projection_result["top_aligned_dim_only"] = n_dim_only

            # Full feature classification for scatter plot data
            scatter_data: list[dict[str, Any]] = []
            for f in shared_features:
                scatter_data.append({
                    "feature_idx": f,
                    "abs_cosine": float(abs_cos[f]),
                    "abs_cohens_d": float(abs(d_map[f])),
                    "in_sig_differential": f in sig_set,
                    "in_top_aligned": f in top_aligned_set,
                })
            scatter_path = hybrid_dir / f"scatter_{label}_layer_{target_layer}.json"
            atomic_save_json(scatter_data, scatter_path)

        # Save per-direction result
        dir_path = hybrid_dir / f"projection_{label}_layer_{target_layer}.json"
        atomic_save_json(projection_result, dir_path)

        summary["directions_analysed"].append(projection_result)

    # Save summary
    summary_path = hybrid_dir / f"hybrid_summary_layer_{target_layer}.json"
    atomic_save_json(summary, summary_path)

    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Project SAE decoder onto DIM directions"
    )
    parser.add_argument("--sae_source", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--expansion", type=int, default=8)
    parser.add_argument("--dim_dir", required=True, help="Path to DIM direction .npy files")
    parser.add_argument("--features_parquet", help="Pooled features parquet from Module 2")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    from src.sae_localization.sae_wrapper import SAEWrapper

    sae = SAEWrapper(
        args.sae_source, layer=args.layer, expansion=args.expansion, device=args.device
    )

    directions = find_dim_directions([Path(args.dim_dir)])
    if not directions:
        log("No DIM directions found. Nothing to do.")
    else:
        feat_df = None
        if args.features_parquet:
            feat_df = pd.read_parquet(args.features_parquet)
        run_hybrid_projection(sae, directions, feat_df, args.layer, Path(args.output_dir))
