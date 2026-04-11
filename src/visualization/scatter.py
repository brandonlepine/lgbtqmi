"""Scatter plot visualizations: PCA loadings, probe comparisons, transfer analysis."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

from src.visualization.style import (
    ANNOT_SIZE, CATEGORY_COLORS, CATEGORY_LABELS, CATEGORY_MARKERS,
    CLUSTER_COLORS, LABEL_SIZE, TICK_SIZE, TITLE_SIZE, WONG_PALETTE,
    apply_style, label_panel, save_fig,
)


def plot_pca_loadings(
    loadings: np.ndarray,
    names: list[str],
    path: str,
    title: str = "Category directions in PCA space",
    variance_ratios: Optional[np.ndarray] = None,
) -> None:
    """Scatter plot of category loadings on PC1 vs PC2 (and PC1 vs PC3).

    Points colored by hypothesised cluster.
    """
    apply_style()
    n_pcs = loadings.shape[1]
    n_panels = 1 if n_pcs < 3 else 2

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    pc_pairs = [(0, 1)]
    if n_pcs >= 3:
        pc_pairs.append((0, 2))

    for ax, (pc_x, pc_y) in zip(axes, pc_pairs):
        for i, name in enumerate(names):
            color = CLUSTER_COLORS.get(name, "#999999")
            marker = CATEGORY_MARKERS.get(name, "o")
            label = CATEGORY_LABELS.get(name, name)
            ax.scatter(loadings[i, pc_x], loadings[i, pc_y],
                       c=color, marker=marker, s=120, edgecolors="black",
                       linewidths=0.8, zorder=5, label=label)
            ax.annotate(label, (loadings[i, pc_x], loadings[i, pc_y]),
                        textcoords="offset points", xytext=(8, 5),
                        fontsize=ANNOT_SIZE, ha="left")

        xlabel = f"PC{pc_x + 1}"
        ylabel = f"PC{pc_y + 1}"
        if variance_ratios is not None:
            xlabel += f" ({variance_ratios[pc_x]:.1%})"
            ylabel += f" ({variance_ratios[pc_y]:.1%})"
        ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
        ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
        ax.axhline(0, color="gray", linewidth=0.3)
        ax.axvline(0, color="gray", linewidth=0.3)

    if n_panels == 2:
        label_panel(axes[0], "A")
        label_panel(axes[1], "B")

    fig.suptitle(title, fontsize=TITLE_SIZE, y=1.02)
    save_fig(fig, path)


def plot_identity_vs_stereotyping(
    identity_accs: np.ndarray,
    stereo_accs: np.ndarray,
    path: str,
    title: str = "Identity encoding vs stereotyping",
    n_layers: Optional[int] = None,
) -> None:
    """Scatter: x = identity probe accuracy, y = stereotyping probe accuracy.

    Each point is one attention head. Color by layer depth.
    """
    apply_style()
    n_l, n_h = identity_accs.shape

    fig, ax = plt.subplots(figsize=(7, 6))

    # Flatten and create layer-based colors
    x = identity_accs.flatten()
    y = stereo_accs.flatten()
    layers = np.repeat(np.arange(n_l), n_h)

    scatter = ax.scatter(x, y, c=layers, cmap="viridis", s=15, alpha=0.6,
                         edgecolors="none")
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label("Layer", fontsize=LABEL_SIZE)

    # Diagonal reference
    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lims, lims, "k--", alpha=0.3, linewidth=0.8)

    # Annotate top-right heads (both high identity and stereotyping)
    combined = x + y
    top_idx = np.argsort(combined)[-5:]
    for idx in top_idx:
        l, h = divmod(idx, n_h)
        ax.annotate(f"L{l}H{h}", (x[idx], y[idx]),
                    textcoords="offset points", xytext=(4, 4),
                    fontsize=ANNOT_SIZE - 1, alpha=0.8)

    ax.set_xlabel("Identity probe accuracy", fontsize=LABEL_SIZE)
    ax.set_ylabel("Stereotyping probe accuracy", fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE)
    save_fig(fig, path)


def plot_identity_vs_stereotyping_dual(
    base_identity: np.ndarray,
    base_stereo: np.ndarray,
    chat_identity: np.ndarray,
    chat_stereo: np.ndarray,
    path: str,
    suptitle: str = "Identity vs stereotyping: base vs chat",
) -> None:
    """Two-panel scatter: base model (left) and chat model (right)."""
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

    for ax, id_acc, st_acc, label in [
        (ax1, base_identity, base_stereo, "Base model"),
        (ax2, chat_identity, chat_stereo, "Chat model"),
    ]:
        n_l, n_h = id_acc.shape
        x = id_acc.flatten()
        y = st_acc.flatten()
        layers = np.repeat(np.arange(n_l), n_h)

        scatter = ax.scatter(x, y, c=layers, cmap="viridis", s=12, alpha=0.5,
                             edgecolors="none")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=0.8)
        ax.set_xlabel("Identity probe accuracy", fontsize=LABEL_SIZE)
        ax.set_title(label, fontsize=TITLE_SIZE)

    ax1.set_ylabel("Stereotyping probe accuracy", fontsize=LABEL_SIZE)
    fig.colorbar(scatter, ax=[ax1, ax2], shrink=0.8, label="Layer")

    label_panel(ax1, "A")
    label_panel(ax2, "B")
    fig.suptitle(suptitle, fontsize=TITLE_SIZE + 1, y=1.02)
    save_fig(fig, path)


def plot_transfer_vs_cosine(
    cosine_values: np.ndarray,
    transfer_values: np.ndarray,
    pair_names: list[str],
    path: str,
    title: str = "Direction cosine vs probe transfer",
) -> None:
    """Scatter: x = direction cosine, y = probe transfer accuracy."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(cosine_values, transfer_values, c="#0072B2", s=40, alpha=0.7,
               edgecolors="black", linewidths=0.5)

    # Fit regression line
    if len(cosine_values) > 2:
        coeffs = np.polyfit(cosine_values, transfer_values, 1)
        x_line = np.linspace(cosine_values.min(), cosine_values.max(), 100)
        ax.plot(x_line, np.polyval(coeffs, x_line), "r--", linewidth=1.5, alpha=0.7,
                label=f"y = {coeffs[0]:.2f}x + {coeffs[1]:.2f}")
        ax.legend(fontsize=TICK_SIZE)

    # Annotate outliers
    residuals = transfer_values - np.polyval(coeffs, cosine_values) if len(cosine_values) > 2 else transfer_values
    outlier_idx = np.argsort(np.abs(residuals))[-3:]
    for idx in outlier_idx:
        ax.annotate(pair_names[idx], (cosine_values[idx], transfer_values[idx]),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=ANNOT_SIZE, ha="left")

    ax.set_xlabel("Direction cosine similarity", fontsize=LABEL_SIZE)
    ax.set_ylabel("Probe transfer accuracy", fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE)
    save_fig(fig, path)


def plot_confusion_matrix(
    conf_matrix: np.ndarray,
    class_names: list[str],
    path: str,
    title: str = "Sub-group probe confusion matrix",
) -> None:
    """Plot a confusion matrix heatmap with annotations."""
    apply_style()
    n = len(class_names)
    # Normalize rows
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1)
    norm_matrix = conf_matrix / row_sums

    fig, ax = plt.subplots(figsize=(max(5, n * 1.2), max(4, n)))
    im = ax.imshow(norm_matrix, cmap="Blues", vmin=0, vmax=1, aspect="equal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=TICK_SIZE)
    ax.set_yticklabels(class_names, fontsize=TICK_SIZE)
    ax.set_xlabel("Predicted", fontsize=LABEL_SIZE)
    ax.set_ylabel("True", fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE)

    for i in range(n):
        for j in range(n):
            val = norm_matrix[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=ANNOT_SIZE, color=color)

    fig.colorbar(im, ax=ax, shrink=0.8, label="Proportion")
    save_fig(fig, path)
