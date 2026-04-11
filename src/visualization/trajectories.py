"""Layer-wise trajectory plots: cosine similarity across layers."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from typing import Optional

from src.visualization.style import (
    CATEGORY_COLORS, CATEGORY_LABELS, LABEL_SIZE, TICK_SIZE, TITLE_SIZE,
    WONG_PALETTE, apply_style, label_panel, save_fig,
)


def plot_cosine_trajectories(
    trajectories: dict[str, np.ndarray],
    path: str,
    title: str = "Cross-category cosine similarity across layers",
    highlight_pairs: Optional[list[str]] = None,
    ylabel: str = "Cosine similarity",
) -> None:
    """Plot layer-wise cosine trajectories for category pairs.

    Args:
        trajectories: dict[pair_name -> (n_layers,) cosine values]
        path: output file path
        highlight_pairs: pair names to draw with thicker lines
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    highlight_set = set(highlight_pairs or [])
    colors = WONG_PALETTE
    for i, (name, cosines) in enumerate(sorted(trajectories.items())):
        color = colors[i % len(colors)]
        lw = 2.5 if name in highlight_set else 1.0
        alpha = 1.0 if name in highlight_set else 0.5
        ax.plot(cosines, label=name, color=color, linewidth=lw, alpha=alpha)

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Layer", fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=TICK_SIZE - 1)
    save_fig(fig, path)


def plot_cosine_trajectories_dual(
    raw_trajectories: dict[str, np.ndarray],
    residual_trajectories: dict[str, np.ndarray],
    path: str,
    suptitle: str = "Cross-category cosine trajectories",
) -> None:
    """Two-panel trajectory plot: raw directions (left) and after projection (right)."""
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

    colors = WONG_PALETTE
    for ax, traj, subtitle in [
        (ax1, raw_trajectories, "Raw directions"),
        (ax2, residual_trajectories, "After shared component removal"),
    ]:
        for i, (name, cosines) in enumerate(sorted(traj.items())):
            ax.plot(cosines, label=name, color=colors[i % len(colors)], linewidth=1.5)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_xlabel("Layer", fontsize=LABEL_SIZE)
        ax.set_title(subtitle, fontsize=TITLE_SIZE)

    ax1.set_ylabel("Cosine similarity", fontsize=LABEL_SIZE)
    ax2.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=TICK_SIZE - 1)

    label_panel(ax1, "A")
    label_panel(ax2, "B")
    fig.suptitle(suptitle, fontsize=TITLE_SIZE + 1, y=1.02)
    save_fig(fig, path)


def plot_pca_variance(
    variance_ratios_by_layer: dict[str, np.ndarray],
    path: str,
    title: str = "PCA variance explained by component",
) -> None:
    """Bar chart of PCA variance explained at representative layers.

    Args:
        variance_ratios_by_layer: dict[layer_label -> variance_ratio array]
    """
    apply_style()
    layers = sorted(variance_ratios_by_layer.keys())
    n_panels = len(layers)

    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4), sharey=True)
    if n_panels == 1:
        axes = [axes]

    colors = WONG_PALETTE
    for ax, layer_label in zip(axes, layers):
        ratios = variance_ratios_by_layer[layer_label]
        n_comp = len(ratios)
        bars = ax.bar(range(n_comp), ratios, color=colors[:n_comp], edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, ratios):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.1%}", ha="center", va="bottom", fontsize=TICK_SIZE)
        ax.set_xlabel("Principal component", fontsize=LABEL_SIZE)
        ax.set_title(f"Layer {layer_label}", fontsize=TITLE_SIZE)
        ax.set_xticks(range(n_comp))
        ax.set_xticklabels([f"PC{i+1}" for i in range(n_comp)])
        ax.set_ylim(0, 1.0)

    axes[0].set_ylabel("Variance explained", fontsize=LABEL_SIZE)
    fig.suptitle(title, fontsize=TITLE_SIZE + 1, y=1.02)
    save_fig(fig, path)


def plot_variance_decomposition(
    decomposition: dict[str, dict[str, float]],
    path: str,
    title: str = "Shared vs category-specific variance",
) -> None:
    """Stacked bar chart: shared / meso / specific variance per category."""
    apply_style()
    cats = sorted(decomposition.keys())
    display = [CATEGORY_LABELS.get(c, c) for c in cats]
    n = len(cats)

    shared = [decomposition[c]["shared"] for c in cats]
    meso = [decomposition[c]["meso"] for c in cats]
    specific = [decomposition[c]["specific"] for c in cats]

    fig, ax = plt.subplots(figsize=(max(8, n * 1.2), 5))
    x = np.arange(n)
    w = 0.6

    ax.bar(x, shared, w, label="Shared (PC1)", color="#0072B2")
    ax.bar(x, meso, w, bottom=shared, label="Meso (PC2-3)", color="#E69F00")
    bottoms = [s + m for s, m in zip(shared, meso)]
    ax.bar(x, specific, w, bottom=bottoms, label="Category-specific", color="#999999")

    ax.set_xticks(x)
    ax.set_xticklabels(display, rotation=45, ha="right")
    ax.set_ylabel("Fraction of variance", fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=TICK_SIZE)
    save_fig(fig, path)


def plot_dendrogram(
    linkage_matrices: dict[str, tuple[np.ndarray, list[str]]],
    path: str,
    title: str = "Hierarchical clustering of category directions",
) -> None:
    """Dendrogram at multiple layers as subplots.

    Args:
        linkage_matrices: dict[layer_label -> (linkage_matrix, names)]
    """
    apply_style()
    layers = sorted(linkage_matrices.keys())
    n_panels = len(layers)

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    for ax, layer_label in zip(axes, layers):
        Z, names = linkage_matrices[layer_label]
        display = [CATEGORY_LABELS.get(n, n) for n in names]
        dendrogram(Z, labels=display, ax=ax, leaf_rotation=45,
                   leaf_font_size=TICK_SIZE, color_threshold=0.5)
        ax.set_title(f"Layer {layer_label}", fontsize=TITLE_SIZE)
        ax.set_ylabel("Distance (1 − |cosine|)", fontsize=LABEL_SIZE)

    fig.suptitle(title, fontsize=TITLE_SIZE + 1, y=1.02)
    save_fig(fig, path)
