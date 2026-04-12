#!/usr/bin/env python3
"""Analyze all subgroup-level results and generate figures 63-70.

Reads outputs from compute_subgroup_directions, analyze_subgroup_fragmentation,
train_subgroup_probes, and ablate_cross_subgroup.

Usage:
    python scripts/analyze_subgroup_results.py \
        --run_dir results/runs/llama2-13b-hf/2026-04-11/

    # With chat model comparison
    python scripts/analyze_subgroup_results.py \
        --run_dir results/runs/llama2-13b-hf/2026-04-11/ \
        --chat_run_dir results/runs/llama2-13b-chat-hf/2026-04-11/
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
from scipy import stats

from src.analysis.geometry import cosine_similarity_matrix, run_pca
from src.data.bbq_loader import CATEGORY_MAP
from src.utils.io import atomic_save_json, ensure_dir
from src.utils.logging import log
from src.visualization.style import (
    ANNOT_SIZE, CATEGORY_COLORS, CATEGORY_LABELS, LABEL_SIZE,
    TICK_SIZE, TITLE_SIZE, apply_style, label_panel, save_fig,
)

ALL_CATS = list(CATEGORY_MAP.keys())


def _lbl(cat: str) -> str:
    return CATEGORY_LABELS.get(cat, cat)


def _load_json(path: Path) -> dict | None:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _load_subgroup_dirs(run_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    path = run_dir / "analysis" / "subgroup_directions.npz"
    if not path.exists():
        return {}
    data = np.load(path, allow_pickle=True)
    result: dict[str, dict[str, np.ndarray]] = {}
    for key in data.files:
        if not key.startswith("subgroup_"):
            continue
        rest = key[len("subgroup_"):]
        for cat in ALL_CATS:
            if rest.startswith(f"{cat}_"):
                sg = rest[len(f"{cat}_"):]
                result.setdefault(cat, {})[sg] = data[key]
                break
    return result


# ===== Figures ==============================================================

def plot_fig63(ablation_data: dict, fig_dir: str) -> None:
    """Fig 63: Cross-subgroup ablation matrices per category."""
    apply_style()
    cats_data = ablation_data.get("categories", {})
    cats = [c for c in ALL_CATS if c in cats_data and "conditions" in cats_data[c]]
    if not cats:
        log("  No ablation data for fig_63")
        return

    n = len(cats)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4.5))
    if n == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    for idx, cat in enumerate(cats):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        conds = cats_data[cat]["conditions"]
        baseline_sg = conds.get("baseline", {}).get("subgroup_bias", {})
        sgs = sorted(baseline_sg.keys())
        ns = len(sgs)
        if ns < 2:
            ax.set_visible(False)
            continue

        matrix = np.zeros((ns, ns), dtype=np.float32)
        for i, ablated in enumerate(sgs):
            cond_key = f"ablate_{ablated}"
            if cond_key not in conds:
                continue
            abl_sg = conds[cond_key].get("subgroup_bias", {})
            for j, measured in enumerate(sgs):
                bl = baseline_sg.get(measured, 0.0) or 0.0
                ab = abl_sg.get(measured, 0.0) or 0.0
                matrix[i, j] = ab - bl

        vmax = max(abs(matrix.min()), abs(matrix.max()), 0.01)
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
        ax.set_xticks(range(ns))
        ax.set_yticks(range(ns))
        ax.set_xticklabels(sgs, rotation=45, ha="right", fontsize=ANNOT_SIZE - 1)
        ax.set_yticklabels(sgs, fontsize=ANNOT_SIZE - 1)
        for i in range(ns):
            for j in range(ns):
                val = matrix[i, j]
                clr = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                        fontsize=ANNOT_SIZE - 2, color=clr)
        ax.set_xlabel("Measured subgroup", fontsize=LABEL_SIZE - 2)
        ax.set_ylabel("Ablated subgroup", fontsize=LABEL_SIZE - 2)
        ax.set_title(_lbl(cat), fontsize=TITLE_SIZE - 1)

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    fig.suptitle("Cross-subgroup ablation effect matrices", fontsize=TITLE_SIZE + 1, y=1.01)
    save_fig(fig, f"{fig_dir}/fig_63_cross_subgroup_ablation_matrices.png")


def plot_fig64(
    ablation_data: dict,
    cat_sgs: dict[str, dict[str, np.ndarray]],
    mid_layer: int,
    fig_dir: str,
) -> None:
    """Fig 64: THE KEY SCATTER — cosine predicts cross-subgroup ablation effect."""
    apply_style()
    cosines_all: list[float] = []
    effects_all: list[float] = []
    cat_labels_all: list[str] = []

    cats_data = ablation_data.get("categories", {})
    for cat in ALL_CATS:
        if cat not in cats_data or cat not in cat_sgs:
            continue
        conds = cats_data[cat].get("conditions", {})
        baseline_sg = conds.get("baseline", {}).get("subgroup_bias", {})
        sg_dirs = cat_sgs[cat]
        sgs = sorted(set(baseline_sg.keys()) & set(sg_dirs.keys()))
        if len(sgs) < 2:
            continue

        sim, sim_names = cosine_similarity_matrix(sg_dirs, mid_layer)

        for i, ablated in enumerate(sgs):
            cond_key = f"ablate_{ablated}"
            if cond_key not in conds:
                continue
            abl_sg = conds[cond_key].get("subgroup_bias", {})
            for j, measured in enumerate(sgs):
                if ablated == measured:
                    continue
                bl = baseline_sg.get(measured, 0.0) or 0.0
                ab = abl_sg.get(measured, 0.0) or 0.0
                effect = ab - bl

                # Get cosine
                if ablated in sim_names and measured in sim_names:
                    ci = sim_names.index(ablated)
                    cj = sim_names.index(measured)
                    cos_val = float(sim[ci, cj])
                else:
                    continue

                cosines_all.append(cos_val)
                effects_all.append(effect)
                cat_labels_all.append(cat)

    if len(cosines_all) < 3:
        log("  <3 data points for fig_64, skipping")
        return

    cos_arr = np.array(cosines_all)
    eff_arr = np.array(effects_all)

    fig, ax = plt.subplots(figsize=(8, 7))
    for cat in ALL_CATS:
        mask = [c == cat for c in cat_labels_all]
        if not any(mask):
            continue
        cx = cos_arr[mask]
        cy = eff_arr[mask]
        ax.scatter(cx, cy, c=CATEGORY_COLORS.get(cat, "#999999"),
                   s=40, alpha=0.7, edgecolors="black", linewidths=0.3,
                   label=_lbl(cat))

    # Regression
    slope, intercept, r_val, p_val, _ = stats.linregress(cos_arr, eff_arr)
    x_line = np.linspace(cos_arr.min(), cos_arr.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, "k--", linewidth=1.5, alpha=0.7)
    r2 = r_val ** 2
    ax.text(0.05, 0.95, f"r²={r2:.3f}\np={p_val:.2e}\nn={len(cos_arr)}",
            transform=ax.transAxes, fontsize=TICK_SIZE, va="top",
            bbox=dict(facecolor="white", edgecolor="gray", alpha=0.8))

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Pairwise cosine (ablated ↔ measured subgroup)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Bias change for measured subgroup", fontsize=LABEL_SIZE)
    ax.set_title("Within-category cosine predicts cross-subgroup ablation effect",
                 fontsize=TITLE_SIZE)
    ax.legend(fontsize=TICK_SIZE - 1, bbox_to_anchor=(1.02, 1), loc="upper left")
    save_fig(fig, f"{fig_dir}/fig_64_backfire_magnitude_vs_cosine.png")


def plot_fig65(ablation_data: dict, fig_dir: str) -> None:
    """Fig 65: Family-level vs subgroup-level ablation comparison."""
    apply_style()
    cats_data = ablation_data.get("categories", {})
    cats_with_fam = []
    for cat in ALL_CATS:
        if cat not in cats_data:
            continue
        conds = cats_data[cat].get("conditions", {})
        if any(k.startswith("ablate_family_") for k in conds):
            cats_with_fam.append(cat)

    if not cats_with_fam:
        log("  No family ablation data, skipping fig_65")
        return

    n = len(cats_with_fam)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, cat in zip(axes, cats_with_fam):
        conds = cats_data[cat]["conditions"]
        baseline_sg = conds.get("baseline", {}).get("subgroup_bias", {})
        sgs = sorted(baseline_sg.keys())

        # Collect family conditions
        fam_conds = {k: v for k, v in conds.items() if k.startswith("ablate_family_")}

        x = np.arange(len(sgs))
        w = 0.25

        # Baseline
        bl_vals = [baseline_sg.get(sg, 0.0) or 0.0 for sg in sgs]
        ax.bar(x - w, bl_vals, w, label="Baseline", color="#999999",
               edgecolor="black", linewidth=0.4)

        # Best subgroup ablation per subgroup
        sg_best = []
        for sg in sgs:
            best = bl_vals[sgs.index(sg)]
            for other_sg in sgs:
                cond_key = f"ablate_{other_sg}"
                if cond_key in conds:
                    val = conds[cond_key].get("subgroup_bias", {}).get(sg, 0.0) or 0.0
                    if abs(val) < abs(best):
                        best = val
            sg_best.append(best)
        ax.bar(x, sg_best, w, label="Best subgroup abl.", color="#E69F00",
               edgecolor="black", linewidth=0.4)

        # Family ablation
        fam_vals = []
        for sg in sgs:
            best_fam = bl_vals[sgs.index(sg)]
            for fk, fv in fam_conds.items():
                val = fv.get("subgroup_bias", {}).get(sg, 0.0) or 0.0
                if abs(val) < abs(best_fam):
                    best_fam = val
            fam_vals.append(best_fam)
        ax.bar(x + w, fam_vals, w, label="Family ablation", color="#0072B2",
               edgecolor="black", linewidth=0.4)

        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(sgs, rotation=45, ha="right", fontsize=TICK_SIZE - 1)
        ax.set_title(_lbl(cat), fontsize=TITLE_SIZE)
        ax.legend(fontsize=TICK_SIZE - 1)

    axes[0].set_ylabel("Ambig bias score", fontsize=LABEL_SIZE)
    fig.suptitle("Family vs subgroup ablation", fontsize=TITLE_SIZE + 1, y=1.02)
    save_fig(fig, f"{fig_dir}/fig_65_family_vs_subgroup_ablation.png")


def plot_fig66(probe_data: dict, fig_dir: str) -> None:
    """Fig 66: Per-subgroup Probe S2 heatmaps for SO and Race."""
    apply_style()
    focus = ["so", "race"]
    available = [c for c in focus if c in probe_data and "probe_s2" in probe_data[c]]
    if not available:
        log("  No S2 probe data for SO/Race, skipping fig_66")
        return

    total_sgs = sum(len(probe_data[c]["probe_s2"]) for c in available)
    ncols = min(total_sgs, 6)
    fig, axes = plt.subplots(1, ncols, figsize=(ncols * 3.5, 5), sharey=True)
    if ncols == 1:
        axes = [axes]

    col = 0
    for cat in available:
        s2 = probe_data[cat]["probe_s2"]
        for sg_name, sg_info in sorted(s2.items()):
            if col >= ncols:
                break
            accs = np.array(sg_info["head_accuracies"])
            ax = axes[col]
            # Show as 1D bar plot (heads)
            ax.bar(range(len(accs)), accs, color=CATEGORY_COLORS.get(cat, "#999999"),
                   edgecolor="none", width=1.0)
            ax.axhline(0.5, color="gray", linewidth=0.5, linestyle="--")
            ax.set_xlabel("Head", fontsize=LABEL_SIZE - 2)
            ax.set_title(f"{_lbl(cat)[:4]}: {sg_name}\n(n={sg_info['n_items']})",
                         fontsize=TICK_SIZE)
            ax.set_ylim(0.35, 1.0)
            col += 1

    axes[0].set_ylabel("Probe S2 accuracy", fontsize=LABEL_SIZE)
    fig.suptitle("Per-subgroup stereotyping probe accuracy", fontsize=TITLE_SIZE + 1, y=1.02)
    save_fig(fig, f"{fig_dir}/fig_66_subgroup_probe_heatmaps.png")


def plot_fig69(
    ablation_data: dict,
    cat_sgs: dict[str, dict[str, np.ndarray]],
    mid_layer: int,
    fig_dir: str,
) -> None:
    """Fig 69: Race/Ethnicity deep dive."""
    apply_style()
    if "race" not in cat_sgs or len(cat_sgs["race"]) < 2:
        log("  No race subgroup data, skipping fig_69")
        return

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # Panel A: Cosine matrix with dendrogram
    sg_dirs = cat_sgs["race"]
    sim, names = cosine_similarity_matrix(sg_dirs, mid_layer)
    ns = len(names)
    ax = axes[0]
    im = ax.imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(ns))
    ax.set_yticks(range(ns))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=ANNOT_SIZE - 1)
    ax.set_yticklabels(names, fontsize=ANNOT_SIZE - 1)
    if ns <= 8:
        for i in range(ns):
            for j in range(ns):
                val = sim[i, j]
                clr = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=ANNOT_SIZE - 2, color=clr)
    ax.set_title("Race subgroup cosines", fontsize=TITLE_SIZE - 1)
    label_panel(ax, "A")

    # Panel B: PCA
    ax = axes[1]
    pca_r = run_pca(sg_dirs, mid_layer, n_components=min(3, ns))
    loadings = pca_r["loadings"]
    pca_names = pca_r["names"]
    for i, nm in enumerate(pca_names):
        ax.scatter(loadings[i, 0], loadings[i, 1], s=80, edgecolors="black", linewidths=0.7)
        ax.annotate(nm, (loadings[i, 0], loadings[i, 1]),
                    textcoords="offset points", xytext=(5, 5), fontsize=ANNOT_SIZE)
    ax.axhline(0, color="gray", linewidth=0.3)
    ax.axvline(0, color="gray", linewidth=0.3)
    ax.set_xlabel("PC1", fontsize=LABEL_SIZE - 1)
    ax.set_ylabel("PC2", fontsize=LABEL_SIZE - 1)
    ax.set_title("Race subgroups in PCA", fontsize=TITLE_SIZE - 1)
    label_panel(ax, "B")

    # Panel C: Ablation matrix
    ax = axes[2]
    cats_data = ablation_data.get("categories", {}).get("race", {})
    conds = cats_data.get("conditions", {})
    baseline_sg = conds.get("baseline", {}).get("subgroup_bias", {})
    sgs_abl = sorted(baseline_sg.keys())
    ns_a = len(sgs_abl)
    if ns_a >= 2:
        matrix = np.zeros((ns_a, ns_a), dtype=np.float32)
        for i, ablated in enumerate(sgs_abl):
            ckey = f"ablate_{ablated}"
            if ckey in conds:
                for j, measured in enumerate(sgs_abl):
                    bl = baseline_sg.get(measured, 0.0) or 0.0
                    ab = conds[ckey].get("subgroup_bias", {}).get(measured, 0.0) or 0.0
                    matrix[i, j] = ab - bl
        vmax = max(abs(matrix.min()), abs(matrix.max()), 0.01)
        im2 = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
        ax.set_xticks(range(ns_a))
        ax.set_yticks(range(ns_a))
        ax.set_xticklabels(sgs_abl, rotation=45, ha="right", fontsize=ANNOT_SIZE - 1)
        ax.set_yticklabels(sgs_abl, fontsize=ANNOT_SIZE - 1)
        for i in range(ns_a):
            for j in range(ns_a):
                val = matrix[i, j]
                clr = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                        fontsize=ANNOT_SIZE - 2, color=clr)
        fig.colorbar(im2, ax=ax, shrink=0.7)
    ax.set_title("Ablation effects", fontsize=TITLE_SIZE - 1)
    label_panel(ax, "C")

    fig.suptitle("Race/Ethnicity deep dive", fontsize=TITLE_SIZE + 1, y=1.02)
    save_fig(fig, f"{fig_dir}/fig_69_race_deep_dive.png")


def plot_fig70(
    frag_info: dict,
    ablation_data: dict,
    cat_sgs: dict[str, dict[str, np.ndarray]],
    mid_layer: int,
    fig_dir: str,
) -> None:
    """Fig 70: Universal finding summary (4 panels)."""
    apply_style()
    fig = plt.figure(figsize=(17, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.3)

    # Panel A: Fragmentation bars
    ax_a = fig.add_subplot(gs[0, 0])
    cats = sorted(frag_info.keys())
    display = [_lbl(c) for c in cats]
    x = np.arange(len(cats))
    w = 0.35
    within = [frag_info[c].get("within_family_cos", 0.0) for c in cats]
    between = [frag_info[c].get("between_family_cos", 0.0) for c in cats]
    ax_a.bar(x - w / 2, within, w, color="#0072B2", edgecolor="black", linewidth=0.4,
             label="Within-family")
    ax_a.bar(x + w / 2, between, w, color="#D55E00", edgecolor="black", linewidth=0.4,
             label="Between-family")
    ax_a.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(display, rotation=45, ha="right", fontsize=TICK_SIZE - 1)
    ax_a.set_ylabel("Mean cosine", fontsize=LABEL_SIZE - 1)
    ax_a.set_title("Fragmentation universality", fontsize=TITLE_SIZE - 1)
    ax_a.legend(fontsize=TICK_SIZE - 1)
    label_panel(ax_a, "A")

    # Panel B: Universal scatter (cosine vs ablation effect)
    ax_b = fig.add_subplot(gs[0, 1])
    cos_all, eff_all, cat_all = [], [], []
    cats_data = ablation_data.get("categories", {})
    for cat in ALL_CATS:
        if cat not in cats_data or cat not in cat_sgs:
            continue
        conds = cats_data[cat].get("conditions", {})
        baseline_sg = conds.get("baseline", {}).get("subgroup_bias", {})
        sg_dirs = cat_sgs[cat]
        sgs = sorted(set(baseline_sg.keys()) & set(sg_dirs.keys()))
        if len(sgs) < 2:
            continue
        sim, sim_names = cosine_similarity_matrix(sg_dirs, mid_layer)
        for ablated in sgs:
            ckey = f"ablate_{ablated}"
            if ckey not in conds:
                continue
            abl_sg = conds[ckey].get("subgroup_bias", {})
            for measured in sgs:
                if ablated == measured:
                    continue
                bl = baseline_sg.get(measured, 0.0) or 0.0
                ab = abl_sg.get(measured, 0.0) or 0.0
                if ablated in sim_names and measured in sim_names:
                    ci = sim_names.index(ablated)
                    cj = sim_names.index(measured)
                    cos_all.append(float(sim[ci, cj]))
                    eff_all.append(ab - bl)
                    cat_all.append(cat)

    if cos_all:
        cos_a = np.array(cos_all)
        eff_a = np.array(eff_all)
        for cat in set(cat_all):
            mask = [c == cat for c in cat_all]
            ax_b.scatter(cos_a[mask], eff_a[mask], c=CATEGORY_COLORS.get(cat, "#999999"),
                         s=25, alpha=0.6, edgecolors="none", label=_lbl(cat))
        if len(cos_a) > 2:
            sl, it, rv, pv, _ = stats.linregress(cos_a, eff_a)
            xl = np.linspace(cos_a.min(), cos_a.max(), 50)
            ax_b.plot(xl, sl * xl + it, "k--", linewidth=1.5, alpha=0.6)
            ax_b.text(0.05, 0.95, f"r²={rv**2:.3f}\np={pv:.1e}",
                      transform=ax_b.transAxes, fontsize=TICK_SIZE - 1, va="top")
    ax_b.axhline(0, color="gray", linewidth=0.3, linestyle="--")
    ax_b.set_xlabel("Subgroup cosine", fontsize=LABEL_SIZE - 1)
    ax_b.set_ylabel("Bias change", fontsize=LABEL_SIZE - 1)
    ax_b.set_title("Cosine predicts ablation effect", fontsize=TITLE_SIZE - 1)
    ax_b.legend(fontsize=TICK_SIZE - 2, bbox_to_anchor=(1.01, 1), loc="upper left")
    label_panel(ax_b, "B")

    # Panel C: Example ablation matrices (SO, Race, Religion)
    ax_c = fig.add_subplot(gs[1, 0])
    examples = [c for c in ["so", "race", "religion"] if c in cats_data]
    if examples:
        # Just show SO if available
        cat = examples[0]
        conds = cats_data[cat].get("conditions", {})
        baseline_sg = conds.get("baseline", {}).get("subgroup_bias", {})
        sgs = sorted(baseline_sg.keys())
        ns = len(sgs)
        if ns >= 2:
            mat = np.zeros((ns, ns), dtype=np.float32)
            for i, abl in enumerate(sgs):
                ck = f"ablate_{abl}"
                if ck in conds:
                    for j, meas in enumerate(sgs):
                        bl = baseline_sg.get(meas, 0.0) or 0.0
                        ab = conds[ck].get("subgroup_bias", {}).get(meas, 0.0) or 0.0
                        mat[i, j] = ab - bl
            vmax = max(abs(mat.min()), abs(mat.max()), 0.01)
            im = ax_c.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
            ax_c.set_xticks(range(ns))
            ax_c.set_yticks(range(ns))
            ax_c.set_xticklabels(sgs, rotation=45, ha="right", fontsize=ANNOT_SIZE - 1)
            ax_c.set_yticklabels(sgs, fontsize=ANNOT_SIZE - 1)
            for i in range(ns):
                for j in range(ns):
                    val = mat[i, j]
                    clr = "white" if abs(val) > vmax * 0.6 else "black"
                    ax_c.text(j, i, f"{val:+.2f}", ha="center", va="center",
                              fontsize=ANNOT_SIZE - 2, color=clr)
            fig.colorbar(im, ax=ax_c, shrink=0.7)
    ax_c.set_title(f"Example: {_lbl(examples[0]) if examples else '?'}", fontsize=TITLE_SIZE - 1)
    label_panel(ax_c, "C")

    # Panel D: Family ablation avoids backfire
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.text(0.5, 0.5, "Family-level ablation\navoids cross-family backfire\n(see fig_65)",
              ha="center", va="center", fontsize=LABEL_SIZE, transform=ax_d.transAxes,
              style="italic", color="gray")
    ax_d.set_title("Practical implication", fontsize=TITLE_SIZE - 1)
    label_panel(ax_d, "D")

    fig.suptitle("Universal subgroup fragmentation and ablation backfire",
                 fontsize=TITLE_SIZE + 2, y=0.98)
    save_fig(fig, f"{fig_dir}/fig_70_universal_finding_summary.png", tight=False)


# ===== Main =================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze all subgroup-level results.")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--chat_run_dir", type=str, default=None)
    parser.add_argument("--model_id", type=str, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    fig_dir = str(ensure_dir(run_dir / "figures"))
    analysis_dir = ensure_dir(run_dir / "analysis")
    model_id = args.model_id or run_dir.parent.name

    log(f"Analyzing subgroup results for {model_id}")

    # Load all data
    cat_sgs = _load_subgroup_dirs(run_dir)
    n_layers = next(iter(next(iter(cat_sgs.values())).values())).shape[0] if cat_sgs else 40
    mid_layer = n_layers // 2

    frag_data = _load_json(analysis_dir / "subgroup_fragmentation.json")
    frag_info = frag_data.get("categories", {}) if frag_data else {}

    ablation_data = _load_json(analysis_dir / "subgroup_ablation_results.json") or {}
    probe_data = _load_json(analysis_dir / "subgroup_probes.json") or {}

    chat_probe_data = None
    if args.chat_run_dir:
        chat_probe_data = _load_json(
            Path(args.chat_run_dir) / "analysis" / "subgroup_probes.json"
        )

    # Figures
    log("\n--- Generating figures ---")

    if ablation_data:
        plot_fig63(ablation_data, fig_dir)
        log("  Saved fig_63")

        plot_fig64(ablation_data, cat_sgs, mid_layer, fig_dir)
        log("  Saved fig_64")

        plot_fig65(ablation_data, fig_dir)
        log("  Saved fig_65")

    if probe_data:
        plot_fig66(probe_data, fig_dir)
        log("  Saved fig_66")

    # fig_67 + fig_68: base vs chat probes
    if probe_data and chat_probe_data:
        log("  fig_67/68: base vs chat comparison available")
        # Plot fig_67: scatter per subgroup
        apply_style()
        fig67, ax67 = plt.subplots(figsize=(7, 6))
        for cat in ALL_CATS:
            if cat not in probe_data or cat not in chat_probe_data:
                continue
            base_s2 = probe_data[cat].get("probe_s2", {})
            chat_s2 = chat_probe_data[cat].get("probe_s2", {})
            for sg in base_s2:
                if sg not in chat_s2:
                    continue
                ba = np.array(base_s2[sg]["head_accuracies"])
                ca = np.array(chat_s2[sg]["head_accuracies"])
                ax67.scatter(ba, ca, s=10, alpha=0.4,
                             c=CATEGORY_COLORS.get(cat, "#999999"), label=f"{_lbl(cat)[:3]}:{sg}")
        ax67.plot([0.4, 1], [0.4, 1], "k--", linewidth=0.8, alpha=0.5)
        ax67.set_xlabel("Base Probe S2 accuracy", fontsize=LABEL_SIZE)
        ax67.set_ylabel("Chat Probe S2 accuracy", fontsize=LABEL_SIZE)
        ax67.set_title("Base vs chat: per-subgroup stereotyping probes", fontsize=TITLE_SIZE)
        save_fig(fig67, f"{fig_dir}/fig_67_base_vs_chat_subgroup_probes.png")
        log("  Saved fig_67")

    if ablation_data and cat_sgs:
        plot_fig69(ablation_data, cat_sgs, mid_layer, fig_dir)
        log("  Saved fig_69")

    if frag_info and ablation_data:
        plot_fig70(frag_info, ablation_data, cat_sgs, mid_layer, fig_dir)
        log("  Saved fig_70")

    # Save analysis summary
    analysis = {
        "model_id": model_id,
        "n_categories_with_subgroups": len(cat_sgs),
        "categories_analyzed": sorted(cat_sgs.keys()),
    }
    atomic_save_json(analysis, analysis_dir / "subgroup_analysis.json")
    log(f"\nAnalysis -> {analysis_dir / 'subgroup_analysis.json'}")
    log("Done!")


if __name__ == "__main__":
    main()
