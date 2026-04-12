#!/usr/bin/env python3
"""Extract pairwise shared directions via Gram-Schmidt decomposition.

Reads directions.npz from existing pipeline and decomposes each high-cosine
category pair into shared + category-specific components.  Also performs
3-way triangle decomposition for strongly connected cliques.

Usage:
    python scripts/extract_pairwise_shared.py \
        --run_dir results/runs/llama2-13b-hf/2026-04-11/ \
        --threshold 0.4
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.analysis.geometry import cosine_similarity_matrix, run_pca
from src.utils.io import atomic_save_json, atomic_save_npz, ensure_dir
from src.utils.logging import log
from src.visualization.style import (
    ANNOT_SIZE, CATEGORY_COLORS, CATEGORY_LABELS, LABEL_SIZE,
    TICK_SIZE, TITLE_SIZE, apply_style, label_panel, save_fig,
)

ALL_CATS = ["so", "gi", "race", "religion", "disability", "physical_appearance", "age"]


# ===== Helpers ==============================================================

def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / max(n, 1e-10)


def _lbl(cat: str) -> str:
    return CATEGORY_LABELS.get(cat, cat)


def _load_category_directions(run_dir: Path) -> dict[str, np.ndarray]:
    path = run_dir / "analysis" / "directions.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run compute_directions.py first.")
    data = np.load(path, allow_pickle=True)
    dirs: dict[str, np.ndarray] = {}
    for cat in ALL_CATS:
        key = f"direction_{cat}"
        if key in data.files:
            dirs[cat] = data[key]
    return dirs


# ===== Pairwise decomposition ==============================================

def decompose_pair(
    d_a: np.ndarray, d_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Gram-Schmidt pairwise decomposition for one layer.

    Returns (shared, a_specific, b_specific, a_shared_frac, b_shared_frac).
    All returned directions are unit-normalised (or zero if degenerate).
    """
    shared_raw = (d_a + d_b) / 2.0
    shared = _unit(shared_raw)

    a_proj = np.dot(d_a, shared)
    b_proj = np.dot(d_b, shared)

    a_spec_raw = d_a - a_proj * shared
    b_spec_raw = d_b - b_proj * shared

    a_spec_norm = np.linalg.norm(a_spec_raw)
    b_spec_norm = np.linalg.norm(b_spec_raw)

    a_spec = a_spec_raw / max(a_spec_norm, 1e-10)
    b_spec = b_spec_raw / max(b_spec_norm, 1e-10)

    a_shared_frac = float(a_proj ** 2)
    b_shared_frac = float(b_proj ** 2)

    return shared, a_spec, b_spec, a_shared_frac, b_shared_frac


def decompose_pair_all_layers(
    dir_a: np.ndarray, dir_b: np.ndarray
) -> dict:
    """Run pairwise decomposition across all layers."""
    n_layers = dir_a.shape[0]
    dim = dir_a.shape[1]

    shared = np.zeros_like(dir_a)
    a_spec = np.zeros_like(dir_a)
    b_spec = np.zeros_like(dir_a)
    a_shared_frac = np.zeros(n_layers, dtype=np.float32)
    b_shared_frac = np.zeros(n_layers, dtype=np.float32)
    recon_err_a = np.zeros(n_layers, dtype=np.float32)
    recon_err_b = np.zeros(n_layers, dtype=np.float32)
    ortho_check_a = np.zeros(n_layers, dtype=np.float32)
    ortho_check_b = np.zeros(n_layers, dtype=np.float32)

    for layer in range(n_layers):
        s, asp, bsp, asf, bsf = decompose_pair(dir_a[layer], dir_b[layer])
        shared[layer] = s
        a_spec[layer] = asp
        b_spec[layer] = bsp
        a_shared_frac[layer] = asf
        b_shared_frac[layer] = bsf

        # Reconstruction error
        a_proj_coeff = np.dot(dir_a[layer], s)
        a_spec_coeff = np.dot(dir_a[layer], asp) if np.linalg.norm(asp) > 1e-10 else 0.0
        recon_a = a_proj_coeff * s + a_spec_coeff * asp
        recon_err_a[layer] = float(np.linalg.norm(dir_a[layer] - recon_a))

        b_proj_coeff = np.dot(dir_b[layer], s)
        b_spec_coeff = np.dot(dir_b[layer], bsp) if np.linalg.norm(bsp) > 1e-10 else 0.0
        recon_b = b_proj_coeff * s + b_spec_coeff * bsp
        recon_err_b[layer] = float(np.linalg.norm(dir_b[layer] - recon_b))

        # Orthogonality check
        ortho_check_a[layer] = abs(float(np.dot(asp, s)))
        ortho_check_b[layer] = abs(float(np.dot(bsp, s)))

    return {
        "shared": shared.astype(np.float32),
        "a_specific": a_spec.astype(np.float32),
        "b_specific": b_spec.astype(np.float32),
        "a_shared_fraction": a_shared_frac,
        "b_shared_fraction": b_shared_frac,
        "reconstruction_error_a": recon_err_a,
        "reconstruction_error_b": recon_err_b,
        "ortho_check_a": ortho_check_a,
        "ortho_check_b": ortho_check_b,
    }


# ===== Triangle decomposition ==============================================

def decompose_triangle(
    d_a: np.ndarray, d_b: np.ndarray, d_c: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """3-way Gram-Schmidt decomposition for one layer.

    Returns dict with 3way_shared, ab_only, bc_only, ac_only directions and
    per-category variance fractions.
    """
    shared_3 = _unit((d_a + d_b + d_c) / 3.0)

    # Residuals after removing 3-way shared
    ra = d_a - np.dot(d_a, shared_3) * shared_3
    rb = d_b - np.dot(d_b, shared_3) * shared_3
    rc = d_c - np.dot(d_c, shared_3) * shared_3

    ab_only = _unit((ra + rb) / 2.0)
    bc_only = _unit((rb + rc) / 2.0)
    ac_only = _unit((ra + rc) / 2.0)

    return {
        "shared_3way": shared_3,
        "ab_only": ab_only,
        "bc_only": bc_only,
        "ac_only": ac_only,
    }


def decompose_triangle_all_layers(
    dir_a: np.ndarray, dir_b: np.ndarray, dir_c: np.ndarray,
    names: tuple[str, str, str],
) -> dict:
    n_layers = dir_a.shape[0]
    dim = dir_a.shape[1]

    shared_3 = np.zeros_like(dir_a)
    ab_only = np.zeros_like(dir_a)
    bc_only = np.zeros_like(dir_a)
    ac_only = np.zeros_like(dir_a)

    for layer in range(n_layers):
        result = decompose_triangle(dir_a[layer], dir_b[layer], dir_c[layer])
        shared_3[layer] = result["shared_3way"]
        ab_only[layer] = result["ab_only"]
        bc_only[layer] = result["bc_only"]
        ac_only[layer] = result["ac_only"]

    return {
        "shared_3way": shared_3.astype(np.float32),
        f"{names[0]}_{names[1]}_only": ab_only.astype(np.float32),
        f"{names[1]}_{names[2]}_only": bc_only.astype(np.float32),
        f"{names[0]}_{names[2]}_only": ac_only.astype(np.float32),
        "names": names,
    }


# ===== Figures ==============================================================

def plot_fig40(
    cat_dirs: dict[str, np.ndarray], mid_layer: int, path: str,
) -> None:
    """Fig 40: Pairwise cosine network graph."""
    apply_style()
    pca_result = run_pca(cat_dirs, mid_layer, n_components=min(5, len(cat_dirs)))
    loadings = pca_result["loadings"]
    names = pca_result["names"]
    var_ratios = pca_result["explained_variance_ratio"]

    sim, sim_names = cosine_similarity_matrix(cat_dirs, mid_layer)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Edges
    for i, na in enumerate(names):
        for j, nb in enumerate(names):
            if j <= i:
                continue
            si = sim_names.index(na)
            sj = sim_names.index(nb)
            cos = sim[si, sj]
            if abs(cos) < 0.3:
                continue
            xi, yi = loadings[i, 0], loadings[i, 1]
            xj, yj = loadings[j, 0], loadings[j, 1]
            color = "#D55E00" if cos > 0 else "#0072B2"
            width = abs(cos) * 4
            ax.plot([xi, xj], [yi, yj], color=color, linewidth=width, alpha=0.4, zorder=1)
            mx, my = (xi + xj) / 2, (yi + yj) / 2
            ax.text(mx, my, f"{cos:.2f}", fontsize=ANNOT_SIZE - 1, ha="center",
                    va="center", color=color, fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1))

    # Nodes
    for i, name in enumerate(names):
        color = CATEGORY_COLORS.get(name, "#999999")
        ax.scatter(loadings[i, 0], loadings[i, 1], c=color, s=200,
                   edgecolors="black", linewidths=1, zorder=5)
        ax.annotate(_lbl(name), (loadings[i, 0], loadings[i, 1]),
                    textcoords="offset points", xytext=(10, 8),
                    fontsize=TICK_SIZE, fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor=color, alpha=0.8, pad=2))

    xlabel = f"PC1 ({var_ratios[0]:.1%})" if len(var_ratios) > 0 else "PC1"
    ylabel = f"PC2 ({var_ratios[1]:.1%})" if len(var_ratios) > 1 else "PC2"
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.set_title(f"Pairwise cosine network (Layer {mid_layer})", fontsize=TITLE_SIZE)
    ax.axhline(0, color="gray", linewidth=0.3)
    ax.axvline(0, color="gray", linewidth=0.3)
    save_fig(fig, path)


def plot_fig41(
    pair_results: dict[str, dict], mid_layer: int, path: str,
) -> None:
    """Fig 41: Pairwise decomposition bars — shared vs specific fractions."""
    apply_style()
    # Sort pairs by absolute cosine (highest first)
    pairs = sorted(pair_results.keys(),
                   key=lambda k: abs(pair_results[k]["cosine_mid"]), reverse=True)

    fig, ax = plt.subplots(figsize=(max(12, len(pairs) * 1.8), 5))
    x_offset = 0
    tick_positions = []
    tick_labels = []

    for pair_key in pairs:
        pr = pair_results[pair_key]
        a_name, b_name = pair_key.split("_", 1)
        cos_val = pr["cosine_mid"]
        a_shared = pr["a_shared_fraction_mid"]
        b_shared = pr["b_shared_fraction_mid"]

        # A bar
        ax.bar(x_offset, a_shared, 0.35, color="#0072B2", edgecolor="black", linewidth=0.4)
        ax.bar(x_offset, 1 - a_shared, 0.35, bottom=a_shared, color="#E69F00",
               edgecolor="black", linewidth=0.4)
        # B bar
        ax.bar(x_offset + 0.4, b_shared, 0.35, color="#0072B2", edgecolor="black", linewidth=0.4)
        ax.bar(x_offset + 0.4, 1 - b_shared, 0.35, bottom=b_shared, color="#E69F00",
               edgecolor="black", linewidth=0.4)

        mid = x_offset + 0.2
        ax.text(mid, 1.02, f"cos={cos_val:.2f}", ha="center", fontsize=ANNOT_SIZE - 1)
        tick_positions.extend([x_offset, x_offset + 0.4])
        tick_labels.extend([_lbl(a_name), _lbl(b_name)])
        x_offset += 1.2

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=60, ha="right", fontsize=TICK_SIZE - 1)
    ax.set_ylabel("Fraction of direction variance", fontsize=LABEL_SIZE)
    ax.set_ylim(0, 1.15)
    ax.set_title("Pairwise decomposition: shared (blue) vs specific (orange)", fontsize=TITLE_SIZE)

    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#0072B2", label="Shared"),
                       Patch(color="#E69F00", label="Specific")],
              fontsize=TICK_SIZE, loc="upper right")
    save_fig(fig, path)


# ===== Main =================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract pairwise shared directions via Gram-Schmidt decomposition."
    )
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.4,
                        help="Minimum |cosine| to analyze a pair (default 0.4)")
    parser.add_argument("--model_id", type=str, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    analysis_dir = ensure_dir(run_dir / "analysis")
    fig_dir = ensure_dir(run_dir / "figures")
    model_id = args.model_id or run_dir.parent.name

    log(f"Extracting pairwise shared directions for {model_id}")
    log(f"Cosine threshold: {args.threshold}")

    cat_dirs = _load_category_directions(run_dir)
    available = sorted(cat_dirs.keys())
    n_layers = next(iter(cat_dirs.values())).shape[0]
    mid_layer = n_layers // 2
    log(f"Loaded {len(cat_dirs)} directions, {n_layers} layers, mid={mid_layer}")

    # Compute full cosine matrix at mid layer
    sim, sim_names = cosine_similarity_matrix(cat_dirs, mid_layer)
    log(f"\nCross-category cosines at layer {mid_layer}:")
    for i, na in enumerate(sim_names):
        for j, nb in enumerate(sim_names):
            if j > i:
                log(f"  {_lbl(na):>22s} ↔ {_lbl(nb):<22s}: {sim[i,j]:+.3f}")

    # Identify pairs above threshold
    pairs_to_analyze: list[tuple[str, str, float]] = []
    for i, na in enumerate(sim_names):
        for j, nb in enumerate(sim_names):
            if j <= i:
                continue
            cos = float(sim[i, j])
            if abs(cos) >= args.threshold:
                pairs_to_analyze.append((na, nb, cos))

    pairs_to_analyze.sort(key=lambda x: abs(x[2]), reverse=True)
    log(f"\nPairs above threshold ({args.threshold}): {len(pairs_to_analyze)}")
    for a, b, c in pairs_to_analyze:
        log(f"  {_lbl(a)} ↔ {_lbl(b)}: {c:+.3f}")

    # Decompose each pair
    log(f"\n--- Pairwise decomposition ---")
    save_arrays: dict[str, np.ndarray] = {}
    pair_summaries: dict[str, dict] = {}

    for cat_a, cat_b, cos_mid in pairs_to_analyze:
        pair_key = f"{cat_a}_{cat_b}"
        log(f"\n  Pair: {_lbl(cat_a)} ↔ {_lbl(cat_b)} (cos={cos_mid:+.3f})")

        result = decompose_pair_all_layers(cat_dirs[cat_a], cat_dirs[cat_b])

        # Save arrays
        save_arrays[f"pair_{pair_key}_shared"] = result["shared"]
        save_arrays[f"pair_{pair_key}_a_specific"] = result["a_specific"]
        save_arrays[f"pair_{pair_key}_b_specific"] = result["b_specific"]

        # Validation
        max_ortho_a = float(result["ortho_check_a"].max())
        max_ortho_b = float(result["ortho_check_b"].max())
        max_recon_a = float(result["reconstruction_error_a"].max())
        max_recon_b = float(result["reconstruction_error_b"].max())
        log(f"    Orthogonality: max|a_spec·shared|={max_ortho_a:.2e}, "
            f"max|b_spec·shared|={max_ortho_b:.2e}")
        log(f"    Reconstruction: max_err_a={max_recon_a:.6f}, max_err_b={max_recon_b:.6f}")
        log(f"    Shared fraction (mid): "
            f"{_lbl(cat_a)}={result['a_shared_fraction'][mid_layer]:.3f}, "
            f"{_lbl(cat_b)}={result['b_shared_fraction'][mid_layer]:.3f}")

        pair_summaries[pair_key] = {
            "cat_a": cat_a,
            "cat_b": cat_b,
            "cosine_mid": float(cos_mid),
            "a_shared_fraction_mid": float(result["a_shared_fraction"][mid_layer]),
            "b_shared_fraction_mid": float(result["b_shared_fraction"][mid_layer]),
            "max_reconstruction_error_a": max_recon_a,
            "max_reconstruction_error_b": max_recon_b,
            "max_ortho_violation_a": max_ortho_a,
            "max_ortho_violation_b": max_ortho_b,
        }

    # Triangle decomposition: SO ↔ GI ↔ Religion
    triangle_summaries: dict[str, dict] = {}
    triangle_cats = ("so", "gi", "religion")
    if all(c in cat_dirs for c in triangle_cats):
        tri_key = "_".join(triangle_cats)
        log(f"\n--- Triangle decomposition: {' ↔ '.join(_lbl(c) for c in triangle_cats)} ---")
        tri_result = decompose_triangle_all_layers(
            cat_dirs["so"], cat_dirs["gi"], cat_dirs["religion"], triangle_cats
        )
        for k, v in tri_result.items():
            if isinstance(v, np.ndarray):
                save_arrays[f"triangle_{tri_key}_{k}"] = v

        # Variance decomposition at mid layer
        shared_3 = tri_result["shared_3way"][mid_layer]
        so_gi_only = tri_result["so_gi_only"][mid_layer]
        gi_religion_only = tri_result["gi_religion_only"][mid_layer]
        so_religion_only = tri_result["so_religion_only"][mid_layer]

        tri_var_decomp = {}
        for cat in triangle_cats:
            v = cat_dirs[cat][mid_layer]
            total = float(np.dot(v, v))
            s3 = float(np.dot(v, shared_3) ** 2)
            # Project onto pairwise directions
            pw_fracs = {}
            for pw_name, pw_dir in [("so_gi_only", so_gi_only),
                                     ("gi_religion_only", gi_religion_only),
                                     ("so_religion_only", so_religion_only)]:
                pw_fracs[pw_name] = float(np.dot(v, pw_dir) ** 2)
            used = s3 + sum(pw_fracs.values())
            resid = max(total - used, 0.0)
            denom = max(total, 1e-12)
            tri_var_decomp[cat] = {
                "3way_shared": s3 / denom,
                **{k: fv / denom for k, fv in pw_fracs.items()},
                "specific": resid / denom,
            }
            log(f"  {_lbl(cat)}: 3way={s3/denom:.2f}  "
                + "  ".join(f"{k}={fv/denom:.2f}" for k, fv in pw_fracs.items())
                + f"  resid={resid/denom:.2f}")

        triangle_summaries[tri_key] = {
            "categories": list(triangle_cats),
            "variance_decomposition": tri_var_decomp,
        }

    # Also try GI ↔ PhysAppear ↔ Disability triangle
    tri2_cats = ("gi", "physical_appearance", "disability")
    if all(c in cat_dirs for c in tri2_cats):
        tri2_key = "_".join(tri2_cats)
        log(f"\n--- Triangle: {' ↔ '.join(_lbl(c) for c in tri2_cats)} ---")
        tri2_result = decompose_triangle_all_layers(
            cat_dirs[tri2_cats[0]], cat_dirs[tri2_cats[1]], cat_dirs[tri2_cats[2]], tri2_cats
        )
        for k, v in tri2_result.items():
            if isinstance(v, np.ndarray):
                save_arrays[f"triangle_{tri2_key}_{k}"] = v
        triangle_summaries[tri2_key] = {"categories": list(tri2_cats)}

    # Save
    log(f"\n--- Saving ---")
    npz_path = analysis_dir / "pairwise_decomposition.npz"
    atomic_save_npz(npz_path, **save_arrays)
    log(f"  Arrays -> {npz_path} ({len(save_arrays)} entries)")

    json_summary = {
        "model_id": model_id,
        "n_layers": n_layers,
        "mid_layer": mid_layer,
        "threshold": args.threshold,
        "n_pairs": len(pairs_to_analyze),
        "pairs": pair_summaries,
        "triangles": triangle_summaries,
    }
    json_path = analysis_dir / "pairwise_decomposition.json"
    atomic_save_json(json_summary, json_path)
    log(f"  Summary -> {json_path}")

    # Figures
    log(f"\n--- Figures ---")
    plot_fig40(cat_dirs, mid_layer, str(fig_dir / "fig_40_pairwise_cosine_network.png"))
    log("  Saved fig_40")

    plot_fig41(pair_summaries, mid_layer, str(fig_dir / "fig_41_pairwise_decomposition_bars.png"))
    log("  Saved fig_41")

    log("\nDone!")


if __name__ == "__main__":
    main()
