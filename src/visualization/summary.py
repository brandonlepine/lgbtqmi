"""Multi-panel publication-ready summary figures."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from typing import Optional

from src.visualization.style import (
    ANNOT_SIZE, CATEGORY_COLORS, CATEGORY_LABELS, CATEGORY_MARKERS,
    CLUSTER_COLORS, DPI, LABEL_SIZE, TICK_SIZE, TITLE_SIZE,
    apply_style, label_panel, save_fig,
)


def plot_representational_hierarchy_summary(
    cosine_matrix: np.ndarray,
    cosine_names: list[str],
    linkage_Z: np.ndarray,
    linkage_names: list[str],
    pca_loadings: np.ndarray,
    pca_names: list[str],
    pca_var_ratios: np.ndarray,
    variance_decomposition: dict[str, dict[str, float]],
    path: str,
    layer: int = 0,
) -> None:
    """Fig 20: 4-panel representational hierarchy summary.

    A: Cross-category cosine matrix
    B: Dendrogram
    C: PCA loading scatter
    D: Shared vs specific variance
    """
    apply_style()
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # Panel A: Cosine heatmap
    ax_a = fig.add_subplot(gs[0, 0])
    n = len(cosine_names)
    display = [CATEGORY_LABELS.get(nm, nm) for nm in cosine_names]
    im = ax_a.imshow(cosine_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax_a.set_xticks(range(n))
    ax_a.set_yticks(range(n))
    ax_a.set_xticklabels(display, rotation=45, ha="right", fontsize=TICK_SIZE - 1)
    ax_a.set_yticklabels(display, fontsize=TICK_SIZE - 1)
    if n <= 10:
        for i in range(n):
            for j in range(n):
                val = cosine_matrix[i, j]
                color = "white" if abs(val) > 0.6 else "black"
                ax_a.text(j, i, f"{val:.2f}", ha="center", va="center",
                          fontsize=ANNOT_SIZE - 1, color=color)
    fig.colorbar(im, ax=ax_a, shrink=0.7)
    ax_a.set_title(f"Direction cosines (Layer {layer})", fontsize=TITLE_SIZE - 1)
    label_panel(ax_a, "A")

    # Panel B: Dendrogram
    ax_b = fig.add_subplot(gs[0, 1])
    display_link = [CATEGORY_LABELS.get(nm, nm) for nm in linkage_names]
    dendrogram(linkage_Z, labels=display_link, ax=ax_b, leaf_rotation=45,
               leaf_font_size=TICK_SIZE - 1, color_threshold=0.5)
    ax_b.set_ylabel("Distance (1−|cos|)", fontsize=LABEL_SIZE - 1)
    ax_b.set_title("Hierarchical clustering", fontsize=TITLE_SIZE - 1)
    label_panel(ax_b, "B")

    # Panel C: PCA scatter
    ax_c = fig.add_subplot(gs[1, 0])
    for i, name in enumerate(pca_names):
        color = CLUSTER_COLORS.get(name, "#999999")
        marker = CATEGORY_MARKERS.get(name, "o")
        lbl = CATEGORY_LABELS.get(name, name)
        ax_c.scatter(pca_loadings[i, 0], pca_loadings[i, 1],
                     c=color, marker=marker, s=100, edgecolors="black", linewidths=0.7,
                     zorder=5, label=lbl)
        ax_c.annotate(lbl, (pca_loadings[i, 0], pca_loadings[i, 1]),
                      textcoords="offset points", xytext=(6, 4),
                      fontsize=ANNOT_SIZE - 1)
    xlabel = f"PC1 ({pca_var_ratios[0]:.1%})" if len(pca_var_ratios) > 0 else "PC1"
    ylabel = f"PC2 ({pca_var_ratios[1]:.1%})" if len(pca_var_ratios) > 1 else "PC2"
    ax_c.set_xlabel(xlabel, fontsize=LABEL_SIZE - 1)
    ax_c.set_ylabel(ylabel, fontsize=LABEL_SIZE - 1)
    ax_c.axhline(0, color="gray", linewidth=0.3)
    ax_c.axvline(0, color="gray", linewidth=0.3)
    ax_c.set_title("PCA of category directions", fontsize=TITLE_SIZE - 1)
    label_panel(ax_c, "C")

    # Panel D: Variance decomposition
    ax_d = fig.add_subplot(gs[1, 1])
    cats = sorted(variance_decomposition.keys())
    display_d = [CATEGORY_LABELS.get(c, c) for c in cats]
    x = np.arange(len(cats))
    shared = [variance_decomposition[c]["shared"] for c in cats]
    meso = [variance_decomposition[c]["meso"] for c in cats]
    specific = [variance_decomposition[c]["specific"] for c in cats]
    w = 0.6
    ax_d.bar(x, shared, w, label="Shared", color="#0072B2")
    ax_d.bar(x, meso, w, bottom=shared, label="Meso", color="#E69F00")
    bottoms = [s + m for s, m in zip(shared, meso)]
    ax_d.bar(x, specific, w, bottom=bottoms, label="Specific", color="#999999")
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(display_d, rotation=45, ha="right", fontsize=TICK_SIZE - 1)
    ax_d.set_ylabel("Fraction", fontsize=LABEL_SIZE - 1)
    ax_d.set_ylim(0, 1.05)
    ax_d.legend(fontsize=TICK_SIZE - 1)
    ax_d.set_title("Variance decomposition", fontsize=TITLE_SIZE - 1)
    label_panel(ax_d, "D")

    fig.suptitle("Representational hierarchy of identity directions",
                 fontsize=TITLE_SIZE + 2, y=0.98)
    save_fig(fig, path, tight=False)


def plot_rlhf_mechanism_summary(
    base_stereo_matrix: np.ndarray,
    chat_stereo_matrix: np.ndarray,
    ablation_results: dict[str, dict[str, float]],
    path: str,
) -> None:
    """Fig 21: 4-panel RLHF mechanism summary.

    A: Base Probe B heatmap
    B: Chat Probe B heatmap
    C: Difference heatmap
    D: Ablation bar chart
    """
    apply_style()
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.25)

    diff = base_stereo_matrix - chat_stereo_matrix
    vmax_diff = max(abs(diff.min()), abs(diff.max()), 0.05)

    # Panel A: Base
    ax_a = fig.add_subplot(gs[0, 0])
    im_a = ax_a.imshow(base_stereo_matrix, cmap="viridis", vmin=0.4, vmax=1.0,
                       aspect="auto", origin="upper")
    ax_a.set_xlabel("Head", fontsize=LABEL_SIZE - 1)
    ax_a.set_ylabel("Layer", fontsize=LABEL_SIZE - 1)
    ax_a.set_title("Base: stereotyping probe", fontsize=TITLE_SIZE - 1)
    fig.colorbar(im_a, ax=ax_a, shrink=0.7)
    label_panel(ax_a, "A")

    # Panel B: Chat
    ax_b = fig.add_subplot(gs[0, 1])
    im_b = ax_b.imshow(chat_stereo_matrix, cmap="viridis", vmin=0.4, vmax=1.0,
                       aspect="auto", origin="upper")
    ax_b.set_xlabel("Head", fontsize=LABEL_SIZE - 1)
    ax_b.set_ylabel("Layer", fontsize=LABEL_SIZE - 1)
    ax_b.set_title("Chat: stereotyping probe", fontsize=TITLE_SIZE - 1)
    fig.colorbar(im_b, ax=ax_b, shrink=0.7)
    label_panel(ax_b, "B")

    # Panel C: Difference
    ax_c = fig.add_subplot(gs[1, 0])
    im_c = ax_c.imshow(diff, cmap="RdBu_r", vmin=-vmax_diff, vmax=vmax_diff,
                       aspect="auto", origin="upper")
    ax_c.set_xlabel("Head", fontsize=LABEL_SIZE - 1)
    ax_c.set_ylabel("Layer", fontsize=LABEL_SIZE - 1)
    ax_c.set_title("Difference (base − chat)", fontsize=TITLE_SIZE - 1)
    fig.colorbar(im_c, ax=ax_c, shrink=0.7)
    label_panel(ax_c, "C")

    # Panel D: Ablation effects
    ax_d = fig.add_subplot(gs[1, 1])
    cats = sorted(ablation_results.keys())
    display = [CATEGORY_LABELS.get(c, c) for c in cats]
    x = np.arange(len(cats))
    w = 0.25

    base_bias = [ablation_results[c].get("base_baseline", 0) for c in cats]
    ablated_bias = [ablation_results[c].get("base_ablated", 0) for c in cats]
    chat_bias = [ablation_results[c].get("chat_baseline", 0) for c in cats]

    ax_d.bar(x - w, base_bias, w, label="Base baseline", color="#D55E00")
    ax_d.bar(x, ablated_bias, w, label="Base + head ablation", color="#E69F00")
    ax_d.bar(x + w, chat_bias, w, label="Chat baseline", color="#0072B2")

    ax_d.set_xticks(x)
    ax_d.set_xticklabels(display, rotation=45, ha="right", fontsize=TICK_SIZE - 1)
    ax_d.set_ylabel("Ambig bias score\n(+ stereo, − counter)", fontsize=LABEL_SIZE - 1)
    ax_d.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax_d.legend(fontsize=TICK_SIZE - 1)
    ax_d.set_title("RLHF-target ablation effect", fontsize=TITLE_SIZE - 1)
    label_panel(ax_d, "D")

    fig.suptitle("RLHF mechanism: localization and replication",
                 fontsize=TITLE_SIZE + 2, y=0.98)
    save_fig(fig, path, tight=False)


def plot_generalization_summary(
    transfer_matrix: np.ndarray,
    transfer_names: list[str],
    cross_benchmark_results: dict[str, dict[str, float]],
    cosine_values: np.ndarray,
    transfer_values: np.ndarray,
    pair_names: list[str],
    path: str,
) -> None:
    """Fig 22: 3-panel generalization summary.

    A: Cross-category transfer matrix
    B: Cross-benchmark bar chart
    C: Transfer vs cosine scatter
    """
    apply_style()
    fig = plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(1, 3, wspace=0.3)

    # Panel A: Transfer matrix
    ax_a = fig.add_subplot(gs[0, 0])
    n = len(transfer_names)
    display = [CATEGORY_LABELS.get(nm, nm) for nm in transfer_names]
    im = ax_a.imshow(transfer_matrix, cmap="viridis", vmin=0.4, vmax=1.0, aspect="equal")
    ax_a.set_xticks(range(n))
    ax_a.set_yticks(range(n))
    ax_a.set_xticklabels(display, rotation=45, ha="right", fontsize=TICK_SIZE - 2)
    ax_a.set_yticklabels(display, fontsize=TICK_SIZE - 2)
    ax_a.set_xlabel("Target", fontsize=LABEL_SIZE - 1)
    ax_a.set_ylabel("Source", fontsize=LABEL_SIZE - 1)
    if n <= 10:
        for i in range(n):
            for j in range(n):
                val = transfer_matrix[i, j]
                color = "white" if val < 0.65 else "black"
                ax_a.text(j, i, f"{val:.2f}", ha="center", va="center",
                          fontsize=ANNOT_SIZE - 1, color=color)
    fig.colorbar(im, ax=ax_a, shrink=0.7)
    ax_a.set_title("Probe transfer", fontsize=TITLE_SIZE - 1)
    label_panel(ax_a, "A")

    # Panel B: Cross-benchmark
    ax_b = fig.add_subplot(gs[0, 1])
    bbq_cats = sorted(cross_benchmark_results.keys())
    display_b = [CATEGORY_LABELS.get(c, c) for c in bbq_cats]
    accs = [cross_benchmark_results[c].get("accuracy", 0.5) for c in bbq_cats]
    colors = [CATEGORY_COLORS.get(c, "#999999") for c in bbq_cats]
    bars = ax_b.bar(range(len(bbq_cats)), accs, color=colors, edgecolor="black", linewidth=0.5)
    ax_b.axhline(0.5, color="gray", linewidth=1, linestyle="--", label="Chance")
    ax_b.set_xticks(range(len(bbq_cats)))
    ax_b.set_xticklabels(display_b, rotation=45, ha="right", fontsize=TICK_SIZE - 2)
    ax_b.set_ylabel("CrowS-Pairs Probe B accuracy", fontsize=LABEL_SIZE - 1)
    # Show below-chance values too (e.g., sign-flipped readouts)
    ax_b.set_ylim(0.0, 1.0)
    for rect, val in zip(bars, accs):
        ax_b.text(
            rect.get_x() + rect.get_width() / 2.0,
            min(max(val, 0.0) + 0.02, 0.98),
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=ANNOT_SIZE - 2,
        )
    ax_b.legend(fontsize=TICK_SIZE - 1)
    ax_b.set_title("Cross-benchmark transfer", fontsize=TITLE_SIZE - 1)
    label_panel(ax_b, "B")

    # Panel C: Transfer vs cosine
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.scatter(cosine_values, transfer_values, c="#0072B2", s=30, alpha=0.7,
                 edgecolors="black", linewidths=0.5)
    if len(cosine_values) > 2:
        coeffs = np.polyfit(cosine_values, transfer_values, 1)
        x_line = np.linspace(cosine_values.min(), cosine_values.max(), 100)
        ax_c.plot(x_line, np.polyval(coeffs, x_line), "r--", linewidth=1.5, alpha=0.7)
    ax_c.set_xlabel("Direction cosine", fontsize=LABEL_SIZE - 1)
    ax_c.set_ylabel("Transfer accuracy", fontsize=LABEL_SIZE - 1)
    ax_c.set_title("Cosine predicts transfer", fontsize=TITLE_SIZE - 1)
    label_panel(ax_c, "C")

    fig.suptitle("Generalization of identity representations",
                 fontsize=TITLE_SIZE + 2, y=1.02)
    save_fig(fig, path)


def plot_cross_model_stability(
    cosine_matrices: dict[str, tuple[np.ndarray, list[str]]],
    bias_scores: dict[str, dict[str, float]],
    path: str,
    title: str = "Cross-model geometry stability",
) -> None:
    """Fig 24: cosine matrices across models side by side."""
    apply_style()
    models = sorted(cosine_matrices.keys())
    n_models = len(models)

    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, model_id in zip(axes, models):
        matrix, names = cosine_matrices[model_id]
        n = len(names)
        display = [CATEGORY_LABELS.get(nm, nm) for nm in names]

        im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(display, rotation=45, ha="right", fontsize=TICK_SIZE - 2)
        ax.set_yticklabels(display, fontsize=TICK_SIZE - 2)
        ax.set_title(model_id, fontsize=TITLE_SIZE - 1)

        if n <= 10:
            for i in range(n):
                for j in range(n):
                    val = matrix[i, j]
                    color = "white" if abs(val) > 0.6 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=ANNOT_SIZE - 1, color=color)

    fig.colorbar(im, ax=axes, shrink=0.6, label="Cosine similarity")
    fig.suptitle(title, fontsize=TITLE_SIZE + 2, y=1.02)
    save_fig(fig, path)


def plot_ablation_grouped_bars(
    results: dict[str, dict[str, float]],
    path: str,
    conditions: list[str] = None,
    condition_labels: list[str] = None,
    condition_colors: list[str] = None,
    title: str = "Ablation effects on bias scores",
    ylabel: str = "Ambiguous bias score\n(+ stereo, − counter)",
) -> None:
    """Grouped bar chart for ablation comparison across categories."""
    apply_style()
    if conditions is None:
        conditions = ["baseline", "ablate_shared", "ablate_specific", "ablate_both"]
    if condition_labels is None:
        condition_labels = ["Baseline", "Shared ablated", "Specific ablated", "Both ablated"]
    if condition_colors is None:
        condition_colors = ["#999999", "#0072B2", "#E69F00", "#D55E00"]

    cats = sorted(results.keys())
    display = [CATEGORY_LABELS.get(c, c) for c in cats]
    n_cats = len(cats)
    n_conds = len(conditions)

    fig, ax = plt.subplots(figsize=(max(10, n_cats * 1.5), 5))
    x = np.arange(n_cats)
    total_w = 0.8
    w = total_w / n_conds

    for i, (cond, lbl, col) in enumerate(zip(conditions, condition_labels, condition_colors)):
        values = [results[c].get(cond, 0.0) for c in cats]
        offset = (i - n_conds / 2 + 0.5) * w
        bars = ax.bar(x + offset, values, w, label=lbl, color=col,
                      edgecolor="black", linewidth=0.5)
        # Error bars if available
        err_key = f"{cond}_se"
        errors = [results[c].get(err_key, 0.0) for c in cats]
        if any(e > 0 for e in errors):
            ax.errorbar(x + offset, values, yerr=errors, fmt="none",
                        ecolor="black", capsize=2, linewidth=0.8)

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(display, rotation=45, ha="right")
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.legend(fontsize=TICK_SIZE, loc="upper right")
    save_fig(fig, path)
