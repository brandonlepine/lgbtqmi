#!/usr/bin/env python3
"""Analyze pairwise ablation results and generate figures 42-49.

Reads pairwise_ablation_results.json and pairwise_decomposition.json.
Optionally reads meso_ablation_results.json and ablation_results.json
for comparison.

Usage:
    python scripts/analyze_pairwise_ablation.py \
        --run_dir results/runs/llama2-13b-hf/2026-04-11/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from src.analysis.geometry import cosine_similarity_matrix, run_pca
from src.utils.io import atomic_save_json, ensure_dir
from src.utils.logging import log
from src.visualization.style import (
    ANNOT_SIZE, CATEGORY_COLORS, CATEGORY_LABELS, LABEL_SIZE,
    TICK_SIZE, TITLE_SIZE, apply_style, label_panel, save_fig,
)

ALL_CATS = ["so", "gi", "race", "religion", "disability", "physical_appearance", "age"]


def _lbl(cat: str) -> str:
    return CATEGORY_LABELS.get(cat, cat)


def _get_bias(conditions: dict, cond: str, cat_label: str) -> float:
    return conditions.get(cond, {}).get(cat_label, {}).get("ambig_bias", 0.0)


# ===== Analysis functions ===================================================

def compute_pair_specificity(
    conditions: dict, cat_a: str, cat_b: str, cond_name: str,
) -> dict:
    baseline = conditions.get("baseline", {})
    in_effects: list[float] = []
    out_effects: list[float] = []
    for c in ALL_CATS:
        bl = baseline.get(_lbl(c), {}).get("ambig_bias", 0.0)
        ab = _get_bias(conditions, cond_name, _lbl(c))
        change = abs(ab - bl)
        if c in (cat_a, cat_b):
            in_effects.append(change)
        else:
            out_effects.append(change)
    in_m = float(np.mean(in_effects)) if in_effects else 0.0
    out_m = float(np.mean(out_effects)) if out_effects else 0.0
    total = in_m + out_m
    return {
        "specificity": in_m / total if total > 1e-10 else 0.5,
        "in_pair_effect": in_m,
        "out_pair_effect": out_m,
    }


# ===== Figures ==============================================================

def plot_fig42(conditions: dict, pairs_info: dict, fig_dir: str) -> None:
    """Fig 42: Grid of small panels showing ablation effect per pair."""
    apply_style()
    pair_keys = sorted(pairs_info.keys(),
                       key=lambda k: abs(pairs_info[k]["cosine_mid"]), reverse=True)
    n_pairs = len(pair_keys)
    ncols = min(4, n_pairs)
    nrows = (n_pairs + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.2), sharey=True)
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    baseline = conditions.get("baseline", {})

    for idx, pk in enumerate(pair_keys):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        pi = pairs_info[pk]
        cat_a, cat_b = pi["cat_a"], pi["cat_b"]
        cos = pi["cosine_mid"]
        cond_name = f"ablate_shared_{pk}"

        if cond_name not in conditions:
            ax.set_visible(False)
            continue

        spec = compute_pair_specificity(conditions, cat_a, cat_b, cond_name)
        changes = []
        colors = []
        for cat in ALL_CATS:
            bl = baseline.get(_lbl(cat), {}).get("ambig_bias", 0.0)
            ab = _get_bias(conditions, cond_name, _lbl(cat))
            changes.append(ab - bl)
            colors.append(CATEGORY_COLORS.get(cat, "#999999") if cat in (cat_a, cat_b) else "#cccccc")

        x = np.arange(len(ALL_CATS))
        ax.bar(x, changes, color=colors, edgecolor="black", linewidth=0.4)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels([_lbl(c)[:6] for c in ALL_CATS], rotation=45,
                           ha="right", fontsize=ANNOT_SIZE - 1)
        ax.set_title(f"{_lbl(cat_a)[:4]}↔{_lbl(cat_b)[:4]}\ncos={cos:+.2f} spec={spec['specificity']:.2f}",
                     fontsize=TICK_SIZE - 1)

    for idx in range(n_pairs, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    fig.suptitle("Pairwise shared-direction ablation effects", fontsize=TITLE_SIZE + 1, y=1.01)
    save_fig(fig, f"{fig_dir}/fig_42_pairwise_ablation_effects.png")


def plot_fig43(conditions: dict, pairs_info: dict, fig_dir: str) -> None:
    """Fig 43: Specificity vs |cosine| scatter."""
    apply_style()
    cosines: list[float] = []
    specificities: list[float] = []
    labels: list[str] = []

    for pk, pi in pairs_info.items():
        cond = f"ablate_shared_{pk}"
        if cond not in conditions:
            continue
        spec = compute_pair_specificity(conditions, pi["cat_a"], pi["cat_b"], cond)
        cosines.append(abs(pi["cosine_mid"]))
        specificities.append(spec["specificity"])
        labels.append(f"{_lbl(pi['cat_a'])[:3]}↔{_lbl(pi['cat_b'])[:3]}")

    if not cosines:
        log("  No pair ablations found, skipping fig_43")
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(cosines, specificities, c="#0072B2", s=80, edgecolors="black", linewidths=0.7)
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (cosines[i], specificities[i]), textcoords="offset points",
                    xytext=(6, 5), fontsize=ANNOT_SIZE)

    # Regression
    if len(cosines) > 2:
        cos_arr = np.array(cosines)
        spec_arr = np.array(specificities)
        coeffs = np.polyfit(cos_arr, spec_arr, 1)
        x_line = np.linspace(min(cosines), max(cosines), 50)
        ax.plot(x_line, np.polyval(coeffs, x_line), "r--", linewidth=1.5, alpha=0.6)
        # r²
        ss_res = np.sum((spec_arr - np.polyval(coeffs, cos_arr)) ** 2)
        ss_tot = np.sum((spec_arr - spec_arr.mean()) ** 2)
        r2 = 1 - ss_res / max(ss_tot, 1e-10)
        ax.text(0.05, 0.95, f"r²={r2:.2f}", transform=ax.transAxes,
                fontsize=TICK_SIZE, va="top")

    ax.axhline(0.5, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_xlabel("|Pair cosine|", fontsize=LABEL_SIZE)
    ax.set_ylabel("Intervention specificity", fontsize=LABEL_SIZE)
    ax.set_title("Geometric similarity predicts causal specificity", fontsize=TITLE_SIZE)
    save_fig(fig, f"{fig_dir}/fig_43_specificity_vs_cosine.png")


def plot_fig44(conditions: dict, pw_json: dict, fig_dir: str) -> None:
    """Fig 44: SO↔GI↔Religion triangle decomposition."""
    tri_key = "so_gi_religion"
    tri_info = pw_json.get("triangles", {}).get(tri_key)
    if not tri_info:
        log("  No triangle data for so_gi_religion, skipping fig_44")
        return

    apply_style()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(15, 6))

    # Panel A: Schematic triangle
    ax_a.set_xlim(-0.5, 3.5)
    ax_a.set_ylim(-0.5, 3.5)
    ax_a.axis("off")
    pts = {"so": (0.5, 0.3), "gi": (2.5, 0.3), "religion": (1.5, 2.8)}
    for cat, (x, y) in pts.items():
        ax_a.scatter(x, y, c=CATEGORY_COLORS.get(cat, "#999999"), s=300,
                     edgecolors="black", linewidths=1.5, zorder=10)
        ax_a.text(x, y - 0.35, _lbl(cat), ha="center", fontsize=TICK_SIZE, fontweight="bold")
    # Edges with cosine labels
    edges = [("so", "gi"), ("gi", "religion"), ("so", "religion")]
    pairs_info = pw_json.get("pairs", {})

    def _pair_cos(a: str, b: str) -> float:
        k1 = f"{a}_{b}"
        k2 = f"{b}_{a}"
        if k1 in pairs_info:
            return float(pairs_info[k1].get("cosine_mid", 0.0))
        if k2 in pairs_info:
            return float(pairs_info[k2].get("cosine_mid", 0.0))
        return 0.0

    for a, b in edges:
        xa, ya = pts[a]
        xb, yb = pts[b]
        ax_a.plot([xa, xb], [ya, yb], "k-", linewidth=1.5, alpha=0.4)
        mx, my = (xa + xb) / 2, (ya + yb) / 2
        cos = _pair_cos(a, b)
        ax_a.text(mx, my + 0.15, f"cos={cos:.2f}", ha="center",
                  fontsize=ANNOT_SIZE, fontweight="bold",
                  bbox=dict(facecolor="white", edgecolor="gray", alpha=0.8, pad=2))
    ax_a.text(1.5, 1.3, "3-way\nshared", ha="center", fontsize=ANNOT_SIZE,
              color="#D55E00", fontweight="bold")
    ax_a.set_title("Triangle structure", fontsize=TITLE_SIZE)
    label_panel(ax_a, "A", x=-0.05)

    # Panel B: Component × category heatmap
    components = [
        f"ablate_3way_{tri_key}",
        f"ablate_so_gi_only_{tri_key}",
        f"ablate_gi_religion_only_{tri_key}",
        f"ablate_so_religion_only_{tri_key}",
    ]
    comp_labels = ["3-way shared", "SO-GI only", "GI-Rel only", "SO-Rel only"]
    baseline = conditions.get("baseline", {})

    available_comps = [(c, l) for c, l in zip(components, comp_labels) if c in conditions]
    if available_comps:
        matrix = np.zeros((len(available_comps), len(ALL_CATS)), dtype=np.float32)
        for i, (cond, _) in enumerate(available_comps):
            for j, cat in enumerate(ALL_CATS):
                bl = baseline.get(_lbl(cat), {}).get("ambig_bias", 0.0)
                ab = _get_bias(conditions, cond, _lbl(cat))
                matrix[i, j] = ab - bl

        vmax = max(abs(matrix.min()), abs(matrix.max()), 0.01)
        im = ax_b.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax_b.set_xticks(range(len(ALL_CATS)))
        ax_b.set_yticks(range(len(available_comps)))
        ax_b.set_xticklabels([_lbl(c) for c in ALL_CATS], rotation=45,
                             ha="right", fontsize=TICK_SIZE - 1)
        ax_b.set_yticklabels([l for _, l in available_comps], fontsize=TICK_SIZE)
        for i in range(len(available_comps)):
            for j in range(len(ALL_CATS)):
                val = matrix[i, j]
                clr = "white" if abs(val) > vmax * 0.6 else "black"
                ax_b.text(j, i, f"{val:+.3f}", ha="center", va="center",
                          fontsize=ANNOT_SIZE - 1, color=clr)
        fig.colorbar(im, ax=ax_b, shrink=0.7, label="Bias change")
    ax_b.set_title("Triangle component ablation effects", fontsize=TITLE_SIZE)
    label_panel(ax_b, "B")

    fig.suptitle("SO ↔ GI ↔ Religion triangle decomposition",
                 fontsize=TITLE_SIZE + 1, y=1.02)
    save_fig(fig, f"{fig_dir}/fig_44_triangle_decomposition.png")


def plot_fig45(conditions: dict, pairs_info: dict, fig_dir: str) -> None:
    """Fig 45: Anti-correlated pair steering (both move toward zero)."""
    apply_style()
    anti_pairs = [(pk, pi) for pk, pi in pairs_info.items() if pi["cosine_mid"] < -0.4]
    anti_pairs.sort(key=lambda x: x[1]["cosine_mid"])

    if not anti_pairs:
        log("  No anti-correlated pairs, skipping fig_45")
        return

    n = len(anti_pairs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    baseline = conditions.get("baseline", {})
    for ax, (pk, pi) in zip(axes, anti_pairs):
        cond = f"ablate_shared_{pk}"
        cat_a, cat_b = pi["cat_a"], pi["cat_b"]
        cos = pi["cosine_mid"]

        if cond not in conditions:
            ax.set_visible(False)
            continue

        for i, cat in enumerate([cat_a, cat_b]):
            bl = baseline.get(_lbl(cat), {}).get("ambig_bias", 0.0)
            ab = _get_bias(conditions, cond, _lbl(cat))
            color = CATEGORY_COLORS.get(cat, "#999999")

            # Arrow from baseline to ablated
            ax.barh(i, bl, height=0.3, color=color, alpha=0.4, edgecolor="black", linewidth=0.5)
            ax.barh(i, ab, height=0.3, color=color, alpha=0.9, edgecolor="black", linewidth=0.5)
            ax.annotate("", xy=(ab, i), xytext=(bl, i),
                        arrowprops=dict(arrowstyle="->", color="black", linewidth=2))
            ax.text(max(abs(bl), abs(ab)) + 0.03, i,
                    f"{_lbl(cat)}\n{bl:+.3f}→{ab:+.3f}", va="center",
                    fontsize=ANNOT_SIZE)

        ax.axvline(0, color="gray", linewidth=1, linestyle="--")
        ax.set_yticks([0, 1])
        ax.set_yticklabels([_lbl(cat_a), _lbl(cat_b)])
        ax.set_xlabel("Ambig bias score", fontsize=LABEL_SIZE)
        ax.set_title(f"{_lbl(cat_a)[:4]}↔{_lbl(cat_b)[:4]} (cos={cos:+.2f})",
                     fontsize=TITLE_SIZE)

    fig.suptitle("Anti-correlated pair steering: both move toward zero",
                 fontsize=TITLE_SIZE + 1, y=1.02)
    save_fig(fig, f"{fig_dir}/fig_45_anticorrelated_pair_steering.png")


def plot_fig46(conditions: dict, pairs_info: dict,
               prev_ablation: dict | None, fig_dir: str) -> None:
    """Fig 46: Additivity test — shared + specific ≈ full direction."""
    apply_style()
    baseline = conditions.get("baseline", {})
    predicted: list[float] = []
    actual: list[float] = []
    labels: list[str] = []

    for pk, pi in pairs_info.items():
        cat_a, cat_b = pi["cat_a"], pi["cat_b"]
        shared_cond = f"ablate_shared_{pk}"
        a_spec_cond = f"ablate_a_specific_{pk}"

        if shared_cond not in conditions or a_spec_cond not in conditions:
            continue

        bl_a = baseline.get(_lbl(cat_a), {}).get("ambig_bias", 0.0)
        shared_effect = _get_bias(conditions, shared_cond, _lbl(cat_a)) - bl_a
        a_spec_effect = _get_bias(conditions, a_spec_cond, _lbl(cat_a)) - bl_a
        pred = shared_effect + a_spec_effect

        # Actual full-direction ablation from previous results
        if prev_ablation and "ablation_results" in prev_ablation:
            prev = prev_ablation["ablation_results"]
            if cat_a in prev:
                act = prev[cat_a].get("ablate_specific", bl_a) - bl_a
                predicted.append(pred)
                actual.append(act)
                labels.append(f"{_lbl(cat_a)[:3]}({_lbl(cat_b)[:3]})")

    if len(predicted) < 2:
        log("  Not enough data for additivity test, skipping fig_46")
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(predicted, actual, c="#0072B2", s=80, edgecolors="black", linewidths=0.7)
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (predicted[i], actual[i]), textcoords="offset points",
                    xytext=(6, 5), fontsize=ANNOT_SIZE)

    lims = [min(min(predicted), min(actual)) - 0.02,
            max(max(predicted), max(actual)) + 0.02]
    ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5, label="y = x (perfect additivity)")

    if len(predicted) > 2:
        r2 = float(np.corrcoef(predicted, actual)[0, 1] ** 2)
        ax.text(0.05, 0.95, f"r²={r2:.2f}", transform=ax.transAxes, fontsize=TICK_SIZE, va="top")

    ax.set_xlabel("Predicted (shared + specific)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Actual (full direction)", fontsize=LABEL_SIZE)
    ax.set_title("Additivity test: decomposition cleanness", fontsize=TITLE_SIZE)
    ax.legend(fontsize=TICK_SIZE - 1)
    save_fig(fig, f"{fig_dir}/fig_46_additivity_test.png")


def plot_fig47(
    pw_conditions: dict, meso_conditions: dict | None,
    pairs_info: dict, fig_dir: str,
) -> None:
    """Fig 47: Pairwise vs meso comparison."""
    apply_style()
    baseline = pw_conditions.get("baseline", {})

    cats = ALL_CATS
    display = [_lbl(c) for c in cats]

    # For each category, find best pairwise shared ablation effect
    pw_effects = []
    for cat in cats:
        best_change = 0.0
        for pk, pi in pairs_info.items():
            cond = f"ablate_shared_{pk}"
            if cond not in pw_conditions:
                continue
            if cat not in (pi["cat_a"], pi["cat_b"]):
                continue
            bl = baseline.get(_lbl(cat), {}).get("ambig_bias", 0.0)
            ab = _get_bias(pw_conditions, cond, _lbl(cat))
            change = ab - bl
            if abs(change) > abs(best_change):
                best_change = change
        pw_effects.append(best_change)

    levels: list[tuple[str, list[float], str]] = [
        ("Pairwise", pw_effects, "#0072B2"),
    ]

    if meso_conditions:
        from scripts.compute_meso_directions import CLUSTERS as MESO_CLUSTERS
        CAT_TO_CLUSTER = {}
        for cn, ms in MESO_CLUSTERS.items():
            for m in ms:
                CAT_TO_CLUSTER[m] = cn
        meso_bl = meso_conditions.get("baseline", {})
        meso_effs = []
        for cat in cats:
            cluster = CAT_TO_CLUSTER.get(cat)
            cond = f"ablate_{cluster}" if cluster else "baseline"
            bl = meso_bl.get(_lbl(cat), {}).get("ambig_bias", 0.0)
            ab = _get_bias(meso_conditions, cond, _lbl(cat))
            meso_effs.append(ab - bl)
        levels.append(("Meso cluster", meso_effs, "#E69F00"))

    n = len(cats)
    n_lev = len(levels)
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(10, n * 1.5), 5))
    x = np.arange(n)

    for i, (label, effects, color) in enumerate(levels):
        offset = (i - n_lev / 2 + 0.5) * w
        ax.bar(x + offset, effects, w, label=label, color=color,
               edgecolor="black", linewidth=0.4)

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(display, rotation=45, ha="right")
    ax.set_ylabel("Bias change from baseline", fontsize=LABEL_SIZE)
    ax.set_title("Pairwise vs meso-cluster ablation", fontsize=TITLE_SIZE)
    ax.legend(fontsize=TICK_SIZE)
    save_fig(fig, f"{fig_dir}/fig_47_decomposition_vs_meso.png")


def plot_fig48(conditions: dict, pairs_info: dict,
               pw_json: dict, run_dir: Path, fig_dir: str) -> None:
    """Fig 48: GI ↔ PhysAppear deep dive."""
    apply_style()
    pk = "gi_physical_appearance"
    if pk not in pairs_info:
        log("  GI↔PhysAppear pair not found, skipping fig_48")
        return

    pi = pairs_info[pk]
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: Cosine trajectory
    dirs_path = run_dir / "analysis" / "directions.npz"
    if dirs_path.exists():
        data = np.load(dirs_path, allow_pickle=True)
        gi_key = "direction_gi"
        pa_key = "direction_physical_appearance"
        if gi_key in data.files and pa_key in data.files:
            gi_dir = data[gi_key]
            pa_dir = data[pa_key]
            n_layers = gi_dir.shape[0]
            cosines = np.array([
                float(np.dot(gi_dir[l] / max(np.linalg.norm(gi_dir[l]), 1e-10),
                             pa_dir[l] / max(np.linalg.norm(pa_dir[l]), 1e-10)))
                for l in range(n_layers)
            ])
            ax_a.plot(cosines, color="#CC79A7", linewidth=2)
            ax_a.axhline(0, color="gray", linewidth=0.5, linestyle="--")
            ax_a.set_xlabel("Layer", fontsize=LABEL_SIZE)
            ax_a.set_ylabel("Cosine similarity", fontsize=LABEL_SIZE)
            ax_a.set_title("GI ↔ PhysAppear cosine across layers", fontsize=TITLE_SIZE)
    label_panel(ax_a, "A")

    # Panel B: Ablation of shared vs specific
    baseline = conditions.get("baseline", {})
    conds_to_plot = [
        (f"ablate_shared_{pk}", "Shared ablation"),
        (f"ablate_a_specific_{pk}", "GI-specific ablation"),
        (f"ablate_b_specific_{pk}", "PhysAppear-specific ablation"),
    ]
    available = [(c, l) for c, l in conds_to_plot if c in conditions]
    if available:
        focus_cats = ["gi", "physical_appearance"]
        x = np.arange(len(available))
        width = 0.35
        for i, fc in enumerate(focus_cats):
            vals = []
            for cond, _ in available:
                bl = baseline.get(_lbl(fc), {}).get("ambig_bias", 0.0)
                ab = _get_bias(conditions, cond, _lbl(fc))
                vals.append(ab - bl)
            ax_b.bar(x + i * width, vals, width, label=_lbl(fc),
                     color=CATEGORY_COLORS.get(fc, "#999999"),
                     edgecolor="black", linewidth=0.4)
        ax_b.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax_b.set_xticks(x + width / 2)
        ax_b.set_xticklabels([l for _, l in available], rotation=30, ha="right",
                             fontsize=TICK_SIZE - 1)
        ax_b.set_ylabel("Bias change", fontsize=LABEL_SIZE)
        ax_b.legend(fontsize=TICK_SIZE - 1)
    ax_b.set_title("Shared vs specific ablation", fontsize=TITLE_SIZE)
    label_panel(ax_b, "B")

    fig.suptitle(f"GI ↔ Physical Appearance deep dive (cos={pi['cosine_mid']:.2f})",
                 fontsize=TITLE_SIZE + 1, y=1.02)
    save_fig(fig, f"{fig_dir}/fig_48_gi_physappear_deep_dive.png")


def plot_fig49(
    conditions: dict, pairs_info: dict,
    run_dir: Path, fig_dir: str,
) -> None:
    """Fig 49: Publication-ready pairwise summary (4 panels)."""
    apply_style()
    fig = plt.figure(figsize=(17, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # Panel A: Network with causal specificity as edge width
    ax_a = fig.add_subplot(gs[0, 0])
    dirs_path = run_dir / "analysis" / "directions.npz"
    if dirs_path.exists():
        data = np.load(dirs_path, allow_pickle=True)
        cat_dirs = {}
        for c in ALL_CATS:
            k = f"direction_{c}"
            if k in data.files:
                cat_dirs[c] = data[k]
        if len(cat_dirs) >= 2:
            n_layers = next(iter(cat_dirs.values())).shape[0]
            mid = n_layers // 2
            pca_r = run_pca(cat_dirs, mid, n_components=min(5, len(cat_dirs)))
            loadings = pca_r["loadings"]
            names = pca_r["names"]

            for i, nm in enumerate(names):
                clr = CATEGORY_COLORS.get(nm, "#999999")
                ax_a.scatter(loadings[i, 0], loadings[i, 1], c=clr, s=120,
                             edgecolors="black", linewidths=0.8, zorder=5)
                ax_a.annotate(_lbl(nm)[:5], (loadings[i, 0], loadings[i, 1]),
                              textcoords="offset points", xytext=(6, 4),
                              fontsize=ANNOT_SIZE - 1)

            for pk, pi in pairs_info.items():
                cond = f"ablate_shared_{pk}"
                if cond not in conditions:
                    continue
                spec = compute_pair_specificity(conditions, pi["cat_a"], pi["cat_b"], cond)
                ca, cb = pi["cat_a"], pi["cat_b"]
                if ca in names and cb in names:
                    ia, ib = names.index(ca), names.index(cb)
                    cos = pi["cosine_mid"]
                    color = "#D55E00" if cos > 0 else "#0072B2"
                    ax_a.plot([loadings[ia, 0], loadings[ib, 0]],
                              [loadings[ia, 1], loadings[ib, 1]],
                              color=color, linewidth=spec["specificity"] * 5,
                              alpha=0.4)

    ax_a.set_title("Causal specificity network", fontsize=TITLE_SIZE - 1)
    ax_a.axhline(0, color="gray", linewidth=0.3)
    ax_a.axvline(0, color="gray", linewidth=0.3)
    label_panel(ax_a, "A")

    # Panel B: Specificity vs cosine
    ax_b = fig.add_subplot(gs[0, 1])
    cosines, specs, lbls = [], [], []
    for pk, pi in pairs_info.items():
        cond = f"ablate_shared_{pk}"
        if cond not in conditions:
            continue
        sp = compute_pair_specificity(conditions, pi["cat_a"], pi["cat_b"], cond)
        cosines.append(abs(pi["cosine_mid"]))
        specs.append(sp["specificity"])
        lbls.append(f"{_lbl(pi['cat_a'])[:3]}↔{_lbl(pi['cat_b'])[:3]}")

    if cosines:
        ax_b.scatter(cosines, specs, c="#0072B2", s=60, edgecolors="black", linewidths=0.6)
        for i, l in enumerate(lbls):
            ax_b.annotate(l, (cosines[i], specs[i]), textcoords="offset points",
                          xytext=(5, 4), fontsize=ANNOT_SIZE - 1)
        if len(cosines) > 2:
            c_arr = np.polyfit(cosines, specs, 1)
            xl = np.linspace(min(cosines), max(cosines), 50)
            ax_b.plot(xl, np.polyval(c_arr, xl), "r--", linewidth=1.2, alpha=0.5)
    ax_b.axhline(0.5, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax_b.set_xlabel("|Cosine|", fontsize=LABEL_SIZE - 1)
    ax_b.set_ylabel("Specificity", fontsize=LABEL_SIZE - 1)
    ax_b.set_title("Cosine predicts specificity", fontsize=TITLE_SIZE - 1)
    label_panel(ax_b, "B")

    # Panel C: Triangle heatmap (if available)
    ax_c = fig.add_subplot(gs[1, 0])
    tri_key = "so_gi_religion"
    components = [
        f"ablate_3way_{tri_key}", f"ablate_so_gi_only_{tri_key}",
        f"ablate_gi_religion_only_{tri_key}", f"ablate_so_religion_only_{tri_key}",
    ]
    comp_labels = ["3-way", "SO-GI", "GI-Rel", "SO-Rel"]
    baseline = conditions.get("baseline", {})
    avail = [(c, l) for c, l in zip(components, comp_labels) if c in conditions]
    if avail:
        tri_cats = ["so", "gi", "religion"]
        matrix = np.zeros((len(avail), len(tri_cats)), dtype=np.float32)
        for i, (cd, _) in enumerate(avail):
            for j, tc in enumerate(tri_cats):
                bl = baseline.get(_lbl(tc), {}).get("ambig_bias", 0.0)
                ab = _get_bias(conditions, cd, _lbl(tc))
                matrix[i, j] = ab - bl
        vmax = max(abs(matrix.min()), abs(matrix.max()), 0.01)
        im = ax_c.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax_c.set_xticks(range(len(tri_cats)))
        ax_c.set_yticks(range(len(avail)))
        ax_c.set_xticklabels([_lbl(c) for c in tri_cats], fontsize=TICK_SIZE - 1)
        ax_c.set_yticklabels([l for _, l in avail], fontsize=TICK_SIZE - 1)
        for i in range(len(avail)):
            for j in range(len(tri_cats)):
                val = matrix[i, j]
                clr = "white" if abs(val) > vmax * 0.6 else "black"
                ax_c.text(j, i, f"{val:+.3f}", ha="center", va="center",
                          fontsize=ANNOT_SIZE, color=clr)
        fig.colorbar(im, ax=ax_c, shrink=0.7)
    ax_c.set_title("Triangle decomposition", fontsize=TITLE_SIZE - 1)
    label_panel(ax_c, "C")

    # Panel D: Anti-correlated steering
    ax_d = fig.add_subplot(gs[1, 1])
    anti = [(pk, pi) for pk, pi in pairs_info.items() if pi["cosine_mid"] < -0.4]
    anti.sort(key=lambda x: x[1]["cosine_mid"])
    y_pos = 0
    for pk, pi in anti[:3]:
        cond = f"ablate_shared_{pk}"
        if cond not in conditions:
            continue
        for cat in [pi["cat_a"], pi["cat_b"]]:
            bl = baseline.get(_lbl(cat), {}).get("ambig_bias", 0.0)
            ab = _get_bias(conditions, cond, _lbl(cat))
            color = CATEGORY_COLORS.get(cat, "#999999")
            ax_d.barh(y_pos, bl, height=0.35, color=color, alpha=0.3,
                      edgecolor="black", linewidth=0.4)
            ax_d.barh(y_pos, ab, height=0.35, color=color, alpha=0.8,
                      edgecolor="black", linewidth=0.4)
            ax_d.annotate("", xy=(ab, y_pos), xytext=(bl, y_pos),
                          arrowprops=dict(arrowstyle="->", color="black", linewidth=1.5))
            ax_d.text(max(abs(bl), abs(ab)) + 0.02, y_pos,
                      f"{_lbl(cat)[:5]}", va="center", fontsize=ANNOT_SIZE - 1)
            y_pos += 1
        y_pos += 0.5
    ax_d.axvline(0, color="gray", linewidth=1, linestyle="--")
    ax_d.set_xlabel("Bias score", fontsize=LABEL_SIZE - 1)
    ax_d.set_title("Anti-correlated steering", fontsize=TITLE_SIZE - 1)
    label_panel(ax_d, "D")

    fig.suptitle("Pairwise decomposition: summary",
                 fontsize=TITLE_SIZE + 2, y=0.98)
    save_fig(fig, f"{fig_dir}/fig_49_pairwise_summary.png", tight=False)


# ===== Main =================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze pairwise ablation results.")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--model_id", type=str, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    fig_dir = str(ensure_dir(run_dir / "figures"))
    analysis_dir = ensure_dir(run_dir / "analysis")
    model_id = args.model_id or run_dir.parent.name

    # Load data
    abl_path = analysis_dir / "pairwise_ablation_results.json"
    if not abl_path.exists():
        raise FileNotFoundError(f"{abl_path} not found. Run ablate_pairwise_shared.py first.")
    with open(abl_path) as f:
        abl_data = json.load(f)
    conditions = abl_data.get("conditions", {})
    log(f"Loaded {len(conditions)} conditions")

    pw_json_path = analysis_dir / "pairwise_decomposition.json"
    with open(pw_json_path) as f:
        pw_json = json.load(f)
    pairs_info = pw_json.get("pairs", {})
    log(f"Loaded {len(pairs_info)} pair decompositions")

    # Optional: previous ablation for comparison
    prev_path = analysis_dir / "ablation_results.json"
    prev_ablation = None
    if prev_path.exists():
        with open(prev_path) as f:
            prev_ablation = json.load(f)
        log("Loaded previous ablation results")

    # Optional: meso results
    meso_path = analysis_dir / "meso_ablation_results.json"
    meso_conditions = None
    if meso_path.exists():
        with open(meso_path) as f:
            meso_conditions = json.load(f).get("conditions", {})
        log("Loaded meso ablation results")

    # ===== Analysis =====
    log("\n--- Pairwise specificity ---")
    spec_results: dict[str, dict] = {}
    for pk, pi in pairs_info.items():
        cond = f"ablate_shared_{pk}"
        if cond in conditions:
            spec = compute_pair_specificity(conditions, pi["cat_a"], pi["cat_b"], cond)
            spec_results[pk] = spec
            log(f"  {_lbl(pi['cat_a'])} ↔ {_lbl(pi['cat_b'])}: "
                f"spec={spec['specificity']:.3f}, in={spec['in_pair_effect']:.4f}, "
                f"out={spec['out_pair_effect']:.4f}")

    # ===== Figures =====
    log("\n--- Generating figures ---")

    plot_fig42(conditions, pairs_info, fig_dir)
    log("  Saved fig_42")

    plot_fig43(conditions, pairs_info, fig_dir)
    log("  Saved fig_43")

    plot_fig44(conditions, pw_json, fig_dir)
    log("  Saved fig_44")

    plot_fig45(conditions, pairs_info, fig_dir)
    log("  Saved fig_45")

    plot_fig46(conditions, pairs_info, prev_ablation, fig_dir)
    log("  Saved fig_46")

    plot_fig47(conditions, meso_conditions, pairs_info, fig_dir)
    log("  Saved fig_47")

    plot_fig48(conditions, pairs_info, pw_json, run_dir, fig_dir)
    log("  Saved fig_48")

    plot_fig49(conditions, pairs_info, run_dir, fig_dir)
    log("  Saved fig_49")

    # Save analysis
    analysis = {
        "model_id": model_id,
        "specificity_results": spec_results,
        "n_pairs": len(pairs_info),
        "n_conditions": len(conditions),
    }
    atomic_save_json(analysis, analysis_dir / "pairwise_analysis.json")
    log(f"\nAnalysis -> {analysis_dir / 'pairwise_analysis.json'}")
    log("Done!")


if __name__ == "__main__":
    main()
