"""Heatmap visualizations: cosine matrices, probe accuracy maps, bias changes."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram

from src.visualization.style import (
    ANNOT_SIZE, CATEGORY_LABELS, DPI, LABEL_SIZE, TICK_SIZE, TITLE_SIZE,
    apply_style, label_panel, save_fig,
)


def plot_cosine_heatmap(
    sim_matrix: np.ndarray,
    names: list[str],
    layer: int,
    path: str,
    title: Optional[str] = None,
    vmin: float = -1.0,
    vmax: float = 1.0,
    order: Optional[list[str]] = None,
) -> None:
    """Plot a symmetric cosine similarity heatmap with annotations.

    Args:
        sim_matrix: (n, n) cosine similarity values
        names: labels for rows/columns
        layer: layer index (for title)
        path: output file path
        order: optional reordering of names (e.g., from clustering)
    """
    from typing import Optional as Opt  # avoid forward ref
    apply_style()

    if order is not None:
        idx = [names.index(n) for n in order]
        sim_matrix = sim_matrix[np.ix_(idx, idx)]
        names = order

    n = len(names)
    display_names = [CATEGORY_LABELS.get(n, n) for n in names]

    fig, ax = plt.subplots(figsize=(max(6, n * 0.9), max(5, n * 0.8)))
    im = ax.imshow(sim_matrix, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="equal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(display_names, rotation=45, ha="right", fontsize=TICK_SIZE)
    ax.set_yticklabels(display_names, fontsize=TICK_SIZE)

    # Annotate cells
    if n <= 10:
        for i in range(n):
            for j in range(n):
                val = sim_matrix[i, j]
                color = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=ANNOT_SIZE, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Cosine similarity", fontsize=LABEL_SIZE)

    if title is None:
        title = f"Cross-category direction cosines (Layer {layer})"
    ax.set_title(title, fontsize=TITLE_SIZE)

    save_fig(fig, path)


def plot_probe_heatmap(
    accuracy_matrix: np.ndarray,
    path: str,
    title: str = "Probe accuracy",
    vmin: float = 0.5,
    vmax: float = 1.0,
    cmap: str = "viridis",
    top_n_annotate: int = 10,
) -> None:
    """Plot a layer x head probe accuracy heatmap.

    Args:
        accuracy_matrix: (n_layers, n_heads)
        path: output path
        top_n_annotate: annotate the N highest-accuracy heads
    """
    apply_style()
    n_layers, n_heads = accuracy_matrix.shape

    fig, ax = plt.subplots(figsize=(max(8, n_heads * 0.3), max(6, n_layers * 0.2)))
    im = ax.imshow(accuracy_matrix, cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="auto", origin="upper")

    ax.set_xlabel("Head index", fontsize=LABEL_SIZE)
    ax.set_ylabel("Layer", fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE)

    # Annotate top heads
    if top_n_annotate > 0:
        flat = accuracy_matrix.flatten()
        top_indices = np.argsort(flat)[-top_n_annotate:]
        for idx in top_indices:
            layer = idx // n_heads
            head = idx % n_heads
            val = accuracy_matrix[layer, head]
            ax.text(head, layer, f"{val:.2f}", ha="center", va="center",
                    fontsize=ANNOT_SIZE - 1, color="white", fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Accuracy", fontsize=LABEL_SIZE)
    save_fig(fig, path)


def plot_probe_difference_heatmap(
    base_matrix: np.ndarray,
    chat_matrix: np.ndarray,
    path: str,
    title: str = "Stereotyping probe: base − chat",
) -> None:
    """Plot heatmap of probe accuracy difference (base - chat).

    Red = base encodes more (RLHF suppressed), Blue = chat encodes more.
    """
    apply_style()
    diff = base_matrix - chat_matrix
    n_layers, n_heads = diff.shape
    vmax = max(abs(diff.min()), abs(diff.max()), 0.1)

    fig, ax = plt.subplots(figsize=(max(8, n_heads * 0.3), max(6, n_layers * 0.2)))
    im = ax.imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   aspect="auto", origin="upper")

    ax.set_xlabel("Head index", fontsize=LABEL_SIZE)
    ax.set_ylabel("Layer", fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Accuracy difference (base − chat)", fontsize=LABEL_SIZE)
    save_fig(fig, path)


def plot_dual_heatmaps(
    matrix_left: np.ndarray,
    matrix_right: np.ndarray,
    names: list[str],
    path: str,
    title_left: str = "Original",
    title_right: str = "After removal",
    suptitle: str = "",
    vmin: float = -1.0,
    vmax: float = 1.0,
) -> None:
    """Plot two heatmaps side by side (e.g., before/after shared removal)."""
    apply_style()
    n = len(names)
    display = [CATEGORY_LABELS.get(nm, nm) for nm in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(12, n * 1.6), max(5, n * 0.8)))

    for ax, matrix, title in [(ax1, matrix_left, title_left), (ax2, matrix_right, title_right)]:
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(display, rotation=45, ha="right", fontsize=TICK_SIZE)
        ax.set_yticklabels(display, fontsize=TICK_SIZE)
        ax.set_title(title, fontsize=TITLE_SIZE)

        if n <= 10:
            for i in range(n):
                for j in range(n):
                    val = matrix[i, j]
                    color = "white" if abs(val) > 0.6 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=ANNOT_SIZE, color=color)

    fig.colorbar(im, ax=[ax1, ax2], shrink=0.8, label="Cosine similarity")
    if suptitle:
        fig.suptitle(suptitle, fontsize=TITLE_SIZE + 1, y=1.02)

    label_panel(ax1, "A")
    label_panel(ax2, "B")
    save_fig(fig, path)


def plot_fragmentation_grid(
    subgroup_cosines: dict[str, tuple[np.ndarray, list[str]]],
    path: str,
    title: str = "Within-category sub-group fragmentation",
) -> None:
    """Plot a grid of small heatmaps, one per category, showing sub-group cosines.

    Args:
        subgroup_cosines: dict[category -> (sim_matrix, subgroup_names)]
    """
    apply_style()
    cats = sorted(subgroup_cosines.keys())
    n_cats = len(cats)
    ncols = min(4, n_cats)
    nrows = (n_cats + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    for idx, cat in enumerate(cats):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        sim_matrix, names = subgroup_cosines[cat]
        n = len(names)

        im = ax.imshow(sim_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=ANNOT_SIZE)
        ax.set_yticklabels(names, fontsize=ANNOT_SIZE)
        ax.set_title(CATEGORY_LABELS.get(cat, cat), fontsize=TICK_SIZE + 1)

        if n <= 8:
            for i in range(n):
                for j in range(n):
                    val = sim_matrix[i, j]
                    color = "white" if abs(val) > 0.6 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=ANNOT_SIZE - 1, color=color)

    # Hide unused axes
    for idx in range(n_cats, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    fig.suptitle(title, fontsize=TITLE_SIZE, y=1.01)
    save_fig(fig, path)


def plot_transfer_matrix(
    transfer_matrix: np.ndarray,
    names: list[str],
    path: str,
    title: str = "Cross-category probe transfer",
) -> None:
    """Plot transfer accuracy matrix (source x target)."""
    apply_style()
    n = len(names)
    display = [CATEGORY_LABELS.get(nm, nm) for nm in names]

    fig, ax = plt.subplots(figsize=(max(7, n * 0.9), max(6, n * 0.8)))
    im = ax.imshow(transfer_matrix, cmap="viridis", vmin=0.4, vmax=1.0, aspect="equal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(display, rotation=45, ha="right", fontsize=TICK_SIZE)
    ax.set_yticklabels(display, fontsize=TICK_SIZE)
    ax.set_xlabel("Target category", fontsize=LABEL_SIZE)
    ax.set_ylabel("Source category (trained on)", fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE)

    if n <= 10:
        for i in range(n):
            for j in range(n):
                val = transfer_matrix[i, j]
                color = "white" if val < 0.7 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=ANNOT_SIZE, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Accuracy", fontsize=LABEL_SIZE)
    save_fig(fig, path)


# Need Optional import at module level for type hints in function signatures
from typing import Optional
