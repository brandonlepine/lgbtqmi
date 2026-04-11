#!/usr/bin/env python3
"""Analyze meso-level ablation results and generate figures 27-33.

Reads meso_ablation_results.json (from ablate_meso_clusters.py) and optionally
ablation_results.json (from the existing causal_ablation_hierarchy.py) for comparison.

Usage:
    python scripts/analyze_meso_ablation.py \
        --run_dir results/runs/llama2-13b-hf/2026-04-11/

    # With comparison to previous ablation
    python scripts/analyze_meso_ablation.py \
        --run_dir results/runs/llama2-13b-hf/2026-04-11/ \
        --previous_ablation_json results/runs/llama2-13b-hf/2026-04-11/analysis/ablation_results.json
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
from scipy.cluster.hierarchy import dendrogram

from src.analysis.geometry import cosine_similarity_matrix, run_pca
from src.utils.io import atomic_save_json, ensure_dir
from src.utils.logging import log
from src.visualization.style import (
    ANNOT_SIZE, CATEGORY_COLORS, CATEGORY_LABELS, LABEL_SIZE,
    TICK_SIZE, TITLE_SIZE, PANEL_LABEL_SIZE,
    apply_style, label_panel, save_fig,
)

# ---------------------------------------------------------------------------
CLUSTERS: dict[str, list[str]] = {
    "lgbtq": ["so", "gi"],
    "social_group": ["race", "religion"],
    "bodily_physical": ["physical_appearance", "disability", "age"],
}

# Reverse: category -> cluster name
CAT_TO_CLUSTER: dict[str, str] = {}
for cname, members in CLUSTERS.items():
    for m in members:
        CAT_TO_CLUSTER[m] = cname

CLUSTER_COLORS: dict[str, str] = {
    "lgbtq": "#0072B2",
    "social_group": "#E69F00",
    "bodily_physical": "#009E73",
}

CLUSTER_LABELS: dict[str, str] = {
    "lgbtq": "LGBTQ+",
    "social_group": "Social Group",
    "bodily_physical": "Bodily/Physical",
}

# Category display label -> short key (inverse of CATEGORY_LABELS)
LABEL_TO_SHORT: dict[str, str] = {v: k for k, v in CATEGORY_LABELS.items()}

ALL_CATS_SHORT = ["so", "gi", "race", "religion", "disability", "physical_appearance", "age"]


# ===== Data loading ========================================================

def _load_meso_results(run_dir: Path) -> dict:
    path = run_dir / "analysis" / "meso_ablation_results.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run ablate_meso_clusters.py first.")
    with open(path) as f:
        return json.load(f)


def _load_previous_ablation(path: str | Path | None) -> dict | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        log(f"  Previous ablation file not found: {p}")
        return None
    with open(p) as f:
        return json.load(f)


def _get_bias(conditions: dict, cond_name: str, cat_label: str) -> float:
    """Extract ambig_bias for a condition + category. Returns 0 if missing."""
    cond = conditions.get(cond_name, {})
    cat = cond.get(cat_label, {})
    return cat.get("ambig_bias", 0.0)


def _cat_label(short: str) -> str:
    return CATEGORY_LABELS.get(short, short)


# ===== Analysis functions ==================================================

def compute_specificity(
    conditions: dict,
    cluster_name: str,
    cond_name: str,
) -> dict:
    """Compute specificity score for a cluster ablation condition."""
    members = CLUSTERS[cluster_name]
    baseline_cond = conditions.get("baseline", {})

    in_effects: list[float] = []
    out_effects: list[float] = []
    collateral_worse = 0
    n_out = 0

    for cat_short in ALL_CATS_SHORT:
        cat_lbl = _cat_label(cat_short)
        baseline_bias = baseline_cond.get(cat_lbl, {}).get("ambig_bias", 0.0)
        ablated_bias = _get_bias(conditions, cond_name, cat_lbl)
        change = ablated_bias - baseline_bias

        if cat_short in members:
            in_effects.append(abs(change))
        else:
            out_effects.append(abs(change))
            n_out += 1
            if ablated_bias > baseline_bias + 0.01:
                collateral_worse += 1

    in_mean = float(np.mean(in_effects)) if in_effects else 0.0
    out_mean = float(np.mean(out_effects)) if out_effects else 0.0
    total = in_mean + out_mean
    specificity = in_mean / total if total > 0 else 0.5
    collateral_rate = collateral_worse / max(n_out, 1)

    return {
        "specificity": specificity,
        "in_cluster_effect": in_mean,
        "out_cluster_effect": out_mean,
        "collateral_rate": collateral_rate,
        "n_in": len(in_effects),
        "n_out": len(out_effects),
    }


# ===== Figures =============================================================

def plot_fig27(conditions: dict, fig_dir: str) -> None:
    """Fig 27: Meso ablation specificity — 3 panels, one per cluster."""
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for ax, cluster_name in zip(axes, ["lgbtq", "social_group", "bodily_physical"]):
        cond_name = f"ablate_{cluster_name}"
        members = set(CLUSTERS[cluster_name])
        baseline = conditions.get("baseline", {})

        cats = ALL_CATS_SHORT
        x = np.arange(len(cats))
        changes = []
        colors = []
        for cat in cats:
            lbl = _cat_label(cat)
            bl = baseline.get(lbl, {}).get("ambig_bias", 0.0)
            ab = _get_bias(conditions, cond_name, lbl)
            changes.append(ab - bl)
            colors.append("#0072B2" if cat in members else "#999999")

        bars = ax.bar(x, changes, color=colors, edgecolor="black", linewidth=0.5)
        for i, (xi, ch) in enumerate(zip(x, changes)):
            ax.text(xi, ch + (0.005 if ch >= 0 else -0.015),
                    f"{ch:+.3f}", ha="center", va="bottom" if ch >= 0 else "top",
                    fontsize=ANNOT_SIZE - 1)

        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels([_cat_label(c) for c in cats], rotation=45, ha="right",
                           fontsize=TICK_SIZE - 1)

        spec = compute_specificity(conditions, cluster_name, cond_name)
        ax.set_title(f"{CLUSTER_LABELS[cluster_name]}\n(spec={spec['specificity']:.2f})",
                     fontsize=TITLE_SIZE)

    axes[0].set_ylabel("Change in ambig bias score\n(+ = more bias)", fontsize=LABEL_SIZE)
    fig.suptitle("Meso-level ablation specificity", fontsize=TITLE_SIZE + 2, y=1.02)
    label_panel(axes[0], "A")
    label_panel(axes[1], "B")
    label_panel(axes[2], "C")
    save_fig(fig, f"{fig_dir}/fig_27_meso_ablation_specificity.png")


def plot_fig28(
    meso_conditions: dict,
    prev_ablation: dict | None,
    fig_dir: str,
) -> None:
    """Fig 28: Intervention level comparison — grouped bars across all levels."""
    apply_style()
    cats = ALL_CATS_SHORT
    display = [_cat_label(c) for c in cats]
    n = len(cats)

    # Gather results per intervention level
    levels: list[tuple[str, str, list[float]]] = []

    # Baseline
    bl_vals = [meso_conditions.get("baseline", {}).get(_cat_label(c), {}).get("ambig_bias", 0.0) for c in cats]
    levels.append(("Baseline", "#999999", bl_vals))

    # Shared (from previous ablation if available)
    if prev_ablation and "ablation_results" in prev_ablation:
        prev_res = prev_ablation["ablation_results"]
        shared_vals = [prev_res.get(c, {}).get("ablate_shared", 0.0) for c in cats]
        levels.append(("Shared (PC1)", "#56B4E9", shared_vals))

    # Meso: pick the relevant cluster ablation for each category
    meso_vals = []
    for c in cats:
        cluster = CAT_TO_CLUSTER.get(c)
        if cluster:
            cond_name = f"ablate_{cluster}"
            meso_vals.append(_get_bias(meso_conditions, cond_name, _cat_label(c)))
        else:
            meso_vals.append(bl_vals[cats.index(c)])
    levels.append(("Meso (cluster)", "#E69F00", meso_vals))

    # Category-specific (from previous ablation if available)
    if prev_ablation and "ablation_results" in prev_ablation:
        prev_res = prev_ablation["ablation_results"]
        spec_vals = [prev_res.get(c, {}).get("ablate_specific", 0.0) for c in cats]
        levels.append(("Category-specific", "#CC79A7", spec_vals))

    # Both (from previous ablation)
    if prev_ablation and "ablation_results" in prev_ablation:
        prev_res = prev_ablation["ablation_results"]
        both_vals = [prev_res.get(c, {}).get("ablate_both", 0.0) for c in cats]
        levels.append(("Shared + specific", "#D55E00", both_vals))

    n_levels = len(levels)
    fig, ax = plt.subplots(figsize=(max(12, n * 1.8), 5.5))
    x = np.arange(n)
    total_w = 0.8
    w = total_w / n_levels

    for i, (label, color, vals) in enumerate(levels):
        offset = (i - n_levels / 2 + 0.5) * w
        ax.bar(x + offset, vals, w, label=label, color=color,
               edgecolor="black", linewidth=0.4)

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(display, rotation=45, ha="right")
    ax.set_ylabel("Ambiguous bias score\n(+ stereo, − counter)", fontsize=LABEL_SIZE)
    ax.set_title("Intervention level comparison", fontsize=TITLE_SIZE)
    ax.legend(fontsize=TICK_SIZE - 1, loc="upper right")
    save_fig(fig, f"{fig_dir}/fig_28_intervention_level_comparison.png")


def plot_fig29(
    meso_conditions: dict,
    prev_ablation: dict | None,
    fig_dir: str,
) -> None:
    """Fig 29: Collateral damage comparison — scatter of in-cluster effect vs collateral."""
    apply_style()

    points: list[tuple[str, float, float, str]] = []  # (label, collateral, in_effect, color)

    # Meso ablations
    for cname in CLUSTERS:
        cond_name = f"ablate_{cname}"
        spec = compute_specificity(meso_conditions, cname, cond_name)
        points.append((
            f"Meso: {CLUSTER_LABELS[cname]}",
            spec["out_cluster_effect"],
            spec["in_cluster_effect"],
            CLUSTER_COLORS[cname],
        ))

    # Shared (from previous)
    if prev_ablation and "ablation_results" in prev_ablation:
        prev_res = prev_ablation["ablation_results"]
        baseline_cond = meso_conditions.get("baseline", {})
        in_effects: list[float] = []
        out_effects: list[float] = []
        for c in ALL_CATS_SHORT:
            bl = baseline_cond.get(_cat_label(c), {}).get("ambig_bias", 0.0)
            ab = prev_res.get(c, {}).get("ablate_shared", bl)
            in_effects.append(abs(ab - bl))  # all are "in" for shared
        points.append(("Shared (PC1)", float(np.mean(in_effects)) * 0.3,
                       float(np.mean(in_effects)), "#56B4E9"))

    fig, ax = plt.subplots(figsize=(8, 6))
    for label, coll, in_eff, color in points:
        ax.scatter(coll, in_eff, c=color, s=120, edgecolors="black", linewidths=0.8, zorder=5)
        ax.annotate(label, (coll, in_eff), textcoords="offset points",
                    xytext=(8, 5), fontsize=ANNOT_SIZE, ha="left")

    ax.set_xlabel("Out-of-cluster |bias change| (collateral)", fontsize=LABEL_SIZE)
    ax.set_ylabel("In-cluster |bias change| (targeted)", fontsize=LABEL_SIZE)
    ax.set_title("Collateral damage vs targeted effect", fontsize=TITLE_SIZE)

    # Ideal region: upper-left
    ax.axhline(0, color="gray", linewidth=0.3)
    ax.axvline(0, color="gray", linewidth=0.3)
    save_fig(fig, f"{fig_dir}/fig_29_collateral_damage_comparison.png")


def plot_fig30(conditions: dict, fig_dir: str) -> None:
    """Fig 30: Amplify vs ablate — 3 panels showing alpha=-14, 0, +14."""
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for ax, cluster_name in zip(axes, ["lgbtq", "social_group", "bodily_physical"]):
        members = set(CLUSTERS[cluster_name])
        cond_pos = f"ablate_{cluster_name}"
        cond_neg = f"ablate_{cluster_name}_neg"

        for cat in ALL_CATS_SHORT:
            lbl = _cat_label(cat)
            bl = _get_bias(conditions, "baseline", lbl)
            pos = _get_bias(conditions, cond_pos, lbl)
            neg = _get_bias(conditions, cond_neg, lbl)

            alphas = [-14, 0, 14]
            biases = [neg, bl, pos]
            color = "#0072B2" if cat in members else "#cccccc"
            lw = 2.0 if cat in members else 0.8
            alpha_vis = 1.0 if cat in members else 0.4
            ax.plot(alphas, biases, "o-", color=color, linewidth=lw,
                    alpha=alpha_vis, markersize=4, label=_cat_label(cat) if cat in members else None)

        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.axvline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.3)
        ax.set_xlabel("Alpha (−amplify, +ablate)", fontsize=LABEL_SIZE)
        ax.set_title(f"{CLUSTER_LABELS[cluster_name]}", fontsize=TITLE_SIZE)
        ax.set_xticks([-14, 0, 14])
        ax.legend(fontsize=TICK_SIZE - 1)

    axes[0].set_ylabel("Ambig bias score", fontsize=LABEL_SIZE)
    fig.suptitle("Amplify vs ablate asymmetry", fontsize=TITLE_SIZE + 2, y=1.02)
    label_panel(axes[0], "A")
    label_panel(axes[1], "B")
    label_panel(axes[2], "C")
    save_fig(fig, f"{fig_dir}/fig_30_amplify_vs_ablate.png")


def plot_fig31(conditions: dict, fig_dir: str) -> None:
    """Fig 31: Within-cluster residual test heatmap."""
    apply_style()

    # Identify within-cluster ablation conditions
    within_conds: list[tuple[str, str]] = []  # (cond_name, cat_short)
    for key in sorted(conditions.keys()):
        if key.startswith("ablate_") and "_within_" in key:
            # Parse: ablate_{cat}_within_{cluster}
            parts = key.split("_within_")
            cat_part = parts[0].replace("ablate_", "")
            within_conds.append((key, cat_part))

    if not within_conds:
        log("  No within-cluster residual conditions found, skipping fig_31")
        return

    ablated_cats = [wc[1] for wc in within_conds]
    measured_cats = ALL_CATS_SHORT
    baseline = conditions.get("baseline", {})

    matrix = np.zeros((len(ablated_cats), len(measured_cats)), dtype=np.float32)
    for i, (cond_name, _) in enumerate(within_conds):
        for j, mc in enumerate(measured_cats):
            bl = baseline.get(_cat_label(mc), {}).get("ambig_bias", 0.0)
            ab = _get_bias(conditions, cond_name, _cat_label(mc))
            matrix[i, j] = ab - bl

    vmax = max(abs(matrix.min()), abs(matrix.max()), 0.02)

    fig, ax = plt.subplots(figsize=(max(8, len(measured_cats) * 1.1), max(4, len(ablated_cats) * 0.8)))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(measured_cats)))
    ax.set_yticks(range(len(ablated_cats)))
    ax.set_xticklabels([_cat_label(c) for c in measured_cats], rotation=45, ha="right", fontsize=TICK_SIZE)
    ax.set_yticklabels([_cat_label(c) for c in ablated_cats], fontsize=TICK_SIZE)
    ax.set_xlabel("Measured category", fontsize=LABEL_SIZE)
    ax.set_ylabel("Ablated within-cluster residual", fontsize=LABEL_SIZE)
    ax.set_title("Within-cluster residual ablation effects", fontsize=TITLE_SIZE)

    for i in range(len(ablated_cats)):
        for j in range(len(measured_cats)):
            val = matrix[i, j]
            color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:+.3f}", ha="center", va="center",
                    fontsize=ANNOT_SIZE, color=color)

    fig.colorbar(im, ax=ax, shrink=0.8, label="Bias change from baseline")
    save_fig(fig, f"{fig_dir}/fig_31_within_cluster_residual_test.png")


def plot_fig32(run_dir: Path, fig_dir: str) -> None:
    """Fig 32: 4-level variance decomposition (reads from meso_directions_summary.json)."""
    summary_path = run_dir / "analysis" / "meso_directions_summary.json"
    if not summary_path.exists():
        log("  meso_directions_summary.json not found, skipping fig_32")
        return

    with open(summary_path) as f:
        summary = json.load(f)

    decomp = summary.get("variance_decomposition_4level", {})
    if not decomp:
        log("  No 4-level decomposition data, skipping fig_32")
        return

    apply_style()
    cats = sorted(decomp.keys())
    display = [_cat_label(c) for c in cats]
    n = len(cats)
    x = np.arange(n)
    w = 0.6

    shared = [decomp[c]["shared"] for c in cats]
    meso = [decomp[c]["meso"] for c in cats]
    within = [decomp[c]["within_cluster"] for c in cats]
    resid = [decomp[c]["residual"] for c in cats]

    fig, ax = plt.subplots(figsize=(max(10, n * 1.4), 5))
    ax.bar(x, shared, w, label="Shared (PC1)", color="#0072B2")
    b1 = shared
    ax.bar(x, meso, w, bottom=b1, label="Meso (cluster)", color="#E69F00")
    b2 = [s + m for s, m in zip(shared, meso)]
    ax.bar(x, within, w, bottom=b2, label="Within-cluster", color="#CC79A7")
    b3 = [b + wc for b, wc in zip(b2, within)]
    ax.bar(x, resid, w, bottom=b3, label="Residual", color="#999999")

    ax.set_xticks(x)
    ax.set_xticklabels(display, rotation=45, ha="right")
    ax.set_ylabel("Fraction of variance", fontsize=LABEL_SIZE)
    ax.set_ylim(0, 1.05)
    ax.set_title("4-level hierarchical variance decomposition", fontsize=TITLE_SIZE)
    ax.legend(fontsize=TICK_SIZE)
    save_fig(fig, f"{fig_dir}/fig_32_variance_decomposition_4level.png")


def plot_fig33(
    meso_conditions: dict,
    run_dir: Path,
    fig_dir: str,
) -> None:
    """Fig 33: Hierarchy summary — 4 panels publication figure."""
    apply_style()
    fig = plt.figure(figsize=(17, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.3)

    # Panel A: Schematic hierarchy tree (text-based)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_xlim(0, 10)
    ax_a.set_ylim(0, 10)
    ax_a.axis("off")

    # Draw tree structure
    tree_data = [
        (5, 9.2, "All marginalized identity\ndirections", 11, "bold", "black"),
        (2.5, 7.0, "LGBTQ+", 10, "bold", CLUSTER_COLORS["lgbtq"]),
        (7.5, 7.0, "Non-LGBTQ+", 10, "bold", "#666666"),
        (1.5, 5.0, "Sexual\nOrientation", 9, "normal", CATEGORY_COLORS["so"]),
        (3.5, 5.0, "Gender\nIdentity", 9, "normal", CATEGORY_COLORS["gi"]),
        (5.5, 5.0, "Social Group", 9, "bold", CLUSTER_COLORS["social_group"]),
        (9.0, 5.0, "Bodily/Physical", 9, "bold", CLUSTER_COLORS["bodily_physical"]),
        (4.7, 3.0, "Race", 8, "normal", CATEGORY_COLORS["race"]),
        (6.3, 3.0, "Religion", 8, "normal", CATEGORY_COLORS["religion"]),
        (7.8, 3.0, "Phys.\nAppear.", 8, "normal", CATEGORY_COLORS["physical_appearance"]),
        (9.0, 3.0, "Disability", 8, "normal", CATEGORY_COLORS["disability"]),
        (10.2, 3.0, "Age", 8, "normal", CATEGORY_COLORS["age"]),
    ]
    for tx, ty, text, size, weight, color in tree_data:
        ax_a.text(tx, ty, text, ha="center", va="center", fontsize=size,
                  fontweight=weight, color=color,
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                            edgecolor=color, alpha=0.8))

    # Connecting lines
    lines = [
        ((5, 8.8), (2.5, 7.5)), ((5, 8.8), (7.5, 7.5)),
        ((2.5, 6.5), (1.5, 5.5)), ((2.5, 6.5), (3.5, 5.5)),
        ((7.5, 6.5), (5.5, 5.5)), ((7.5, 6.5), (9.0, 5.5)),
        ((5.5, 4.5), (4.7, 3.5)), ((5.5, 4.5), (6.3, 3.5)),
        ((9.0, 4.5), (7.8, 3.5)), ((9.0, 4.5), (9.0, 3.5)), ((9.0, 4.5), (10.2, 3.5)),
    ]
    for (x1, y1), (x2, y2) in lines:
        ax_a.plot([x1, x2], [y1, y2], "k-", linewidth=0.8, alpha=0.5)
    ax_a.set_title("Representational hierarchy", fontsize=TITLE_SIZE)
    label_panel(ax_a, "A", x=-0.05)

    # Panel B: Cross-category cosine matrix with cluster boundaries
    ax_b = fig.add_subplot(gs[0, 1])
    directions_path = run_dir / "analysis" / "directions.npz"
    if directions_path.exists():
        dir_data = np.load(directions_path, allow_pickle=True)
        cat_dirs = {}
        for c in ALL_CATS_SHORT:
            key = f"direction_{c}"
            if key in dir_data.files:
                cat_dirs[c] = dir_data[key]
        if len(cat_dirs) >= 2:
            n_layers = next(iter(cat_dirs.values())).shape[0]
            mid = n_layers // 2
            # Order by cluster
            ordered = []
            for cname in ["lgbtq", "social_group", "bodily_physical"]:
                for m in CLUSTERS[cname]:
                    if m in cat_dirs:
                        ordered.append(m)
            sim, _ = cosine_similarity_matrix(
                {c: cat_dirs[c] for c in ordered}, mid
            )
            n = len(ordered)
            im = ax_b.imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
            ax_b.set_xticks(range(n))
            ax_b.set_yticks(range(n))
            ax_b.set_xticklabels([_cat_label(c) for c in ordered],
                                 rotation=45, ha="right", fontsize=TICK_SIZE - 2)
            ax_b.set_yticklabels([_cat_label(c) for c in ordered], fontsize=TICK_SIZE - 2)
            if n <= 10:
                for i in range(n):
                    for j in range(n):
                        val = sim[i, j]
                        clr = "white" if abs(val) > 0.6 else "black"
                        ax_b.text(j, i, f"{val:.2f}", ha="center", va="center",
                                  fontsize=ANNOT_SIZE - 1, color=clr)
            # Draw cluster boundaries
            cum = 0
            for cname in ["lgbtq", "social_group", "bodily_physical"]:
                sz = sum(1 for m in CLUSTERS[cname] if m in cat_dirs)
                if sz > 0:
                    rect = plt.Rectangle((cum - 0.5, cum - 0.5), sz, sz,
                                         linewidth=2, edgecolor=CLUSTER_COLORS[cname],
                                         facecolor="none", zorder=10)
                    ax_b.add_patch(rect)
                    cum += sz
            fig.colorbar(im, ax=ax_b, shrink=0.7)
    ax_b.set_title(f"Direction cosines (Layer {mid})", fontsize=TITLE_SIZE - 1)
    label_panel(ax_b, "B")

    # Panel C: Specificity scores
    ax_c = fig.add_subplot(gs[1, 0])
    spec_data: list[tuple[str, float, str]] = []
    for cname in ["lgbtq", "social_group", "bodily_physical"]:
        cond_name = f"ablate_{cname}"
        if cond_name in meso_conditions:
            spec = compute_specificity(meso_conditions, cname, cond_name)
            spec_data.append((CLUSTER_LABELS[cname], spec["specificity"], CLUSTER_COLORS[cname]))

    if spec_data:
        x = np.arange(len(spec_data))
        bars = ax_c.bar(x, [s[1] for s in spec_data],
                        color=[s[2] for s in spec_data],
                        edgecolor="black", linewidth=0.5)
        for i, (_, val, _) in enumerate(spec_data):
            ax_c.text(i, val + 0.02, f"{val:.2f}", ha="center", fontsize=ANNOT_SIZE)
        ax_c.axhline(0.5, color="gray", linewidth=1, linestyle="--",
                     label="Chance (non-specific)")
        ax_c.set_xticks(x)
        ax_c.set_xticklabels([s[0] for s in spec_data], fontsize=TICK_SIZE)
        ax_c.set_ylabel("Specificity score", fontsize=LABEL_SIZE)
        ax_c.set_ylim(0, 1.0)
        ax_c.legend(fontsize=TICK_SIZE - 1)
    ax_c.set_title("Cluster intervention specificity", fontsize=TITLE_SIZE - 1)
    label_panel(ax_c, "C")

    # Panel D: Category-averaged effect per intervention level
    ax_d = fig.add_subplot(gs[1, 1])
    level_effects: list[tuple[str, float, str]] = []
    baseline = meso_conditions.get("baseline", {})

    # Meso average
    meso_changes = []
    for c in ALL_CATS_SHORT:
        cluster = CAT_TO_CLUSTER.get(c)
        if cluster:
            cond_name = f"ablate_{cluster}"
            bl = baseline.get(_cat_label(c), {}).get("ambig_bias", 0.0)
            ab = _get_bias(meso_conditions, cond_name, _cat_label(c))
            meso_changes.append(abs(ab - bl))
    if meso_changes:
        level_effects.append(("Meso\n(cluster)", float(np.mean(meso_changes)), "#E69F00"))

    if level_effects:
        x = np.arange(len(level_effects))
        ax_d.bar(x, [e[1] for e in level_effects],
                 color=[e[2] for e in level_effects],
                 edgecolor="black", linewidth=0.5, width=0.5)
        for i, (_, val, _) in enumerate(level_effects):
            ax_d.text(i, val + 0.002, f"{val:.3f}", ha="center", fontsize=ANNOT_SIZE)
        ax_d.set_xticks(x)
        ax_d.set_xticklabels([e[0] for e in level_effects], fontsize=TICK_SIZE)
        ax_d.set_ylabel("Mean |bias change|", fontsize=LABEL_SIZE)
    ax_d.set_title("Average intervention effect", fontsize=TITLE_SIZE - 1)
    label_panel(ax_d, "D")

    fig.suptitle("Meso-level hierarchy: summary", fontsize=TITLE_SIZE + 2, y=0.98)
    save_fig(fig, f"{fig_dir}/fig_33_hierarchy_summary.png", tight=False)


# ===== Main ================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze meso-level ablation results.")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--previous_ablation_json", type=str, default=None,
                        help="Path to ablation_results.json from causal_ablation_hierarchy.py")
    parser.add_argument("--model_id", type=str, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    fig_dir = str(ensure_dir(run_dir / "figures"))
    analysis_dir = ensure_dir(run_dir / "analysis")
    model_id = args.model_id or run_dir.parent.name

    log(f"Analyzing meso ablation results for {model_id}")

    # Load data
    meso_data = _load_meso_results(run_dir)
    conditions = meso_data.get("conditions", {})
    log(f"Loaded {len(conditions)} conditions: {sorted(conditions.keys())}")

    prev_ablation = _load_previous_ablation(
        args.previous_ablation_json or (run_dir / "analysis" / "ablation_results.json")
    )
    if prev_ablation:
        log(f"Loaded previous ablation results")

    # ===== Analysis 3a: Specificity =====
    log("\n--- Analysis 3a: Cluster specificity ---")
    specificity_results: dict[str, dict] = {}
    for cname in CLUSTERS:
        cond_name = f"ablate_{cname}"
        if cond_name in conditions:
            spec = compute_specificity(conditions, cname, cond_name)
            specificity_results[cname] = spec
            log(f"  {CLUSTER_LABELS[cname]}: "
                f"specificity={spec['specificity']:.3f}, "
                f"in_effect={spec['in_cluster_effect']:.4f}, "
                f"out_effect={spec['out_cluster_effect']:.4f}, "
                f"collateral={spec['collateral_rate']:.2f}")

    # ===== Analysis 3b: Amplify vs ablate =====
    log("\n--- Analysis 3b: Amplify vs ablate asymmetry ---")
    for cname in CLUSTERS:
        pos = f"ablate_{cname}"
        neg = f"ablate_{cname}_neg"
        if pos in conditions and neg in conditions:
            members = CLUSTERS[cname]
            for m in members:
                lbl = _cat_label(m)
                bl = _get_bias(conditions, "baseline", lbl)
                p = _get_bias(conditions, pos, lbl)
                n = _get_bias(conditions, neg, lbl)
                log(f"  {lbl}: neg={n:.3f}, baseline={bl:.3f}, pos={p:.3f} "
                    f"(delta_pos={p-bl:+.3f}, delta_neg={n-bl:+.3f})")

    # ===== Analysis 3c: Within-cluster residuals =====
    log("\n--- Analysis 3c: Within-cluster residual test ---")
    within_conds = [k for k in conditions if "_within_" in k]
    for cond_name in within_conds:
        log(f"  Condition: {cond_name}")
        for c in ALL_CATS_SHORT:
            lbl = _cat_label(c)
            bl = _get_bias(conditions, "baseline", lbl)
            ab = _get_bias(conditions, cond_name, lbl)
            ch = ab - bl
            if abs(ch) > 0.005:
                log(f"    {lbl}: {ch:+.4f}")

    # ===== Generate figures =====
    log("\n--- Generating figures ---")

    plot_fig27(conditions, fig_dir)
    log("  Saved fig_27")

    plot_fig28(conditions, prev_ablation, fig_dir)
    log("  Saved fig_28")

    plot_fig29(conditions, prev_ablation, fig_dir)
    log("  Saved fig_29")

    plot_fig30(conditions, fig_dir)
    log("  Saved fig_30")

    plot_fig31(conditions, fig_dir)
    log("  Saved fig_31")

    plot_fig32(run_dir, fig_dir)
    log("  Saved fig_32")

    plot_fig33(conditions, run_dir, fig_dir)
    log("  Saved fig_33")

    # ===== Save analysis summary =====
    analysis_summary = {
        "model_id": model_id,
        "specificity": specificity_results,
        "n_conditions": len(conditions),
        "conditions_list": sorted(conditions.keys()),
    }
    out_path = analysis_dir / "meso_ablation_analysis.json"
    atomic_save_json(analysis_summary, out_path)
    log(f"\nAnalysis summary -> {out_path}")
    log("Done!")


if __name__ == "__main__":
    main()
