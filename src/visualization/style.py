"""Color palettes, formatting constants, and matplotlib configuration."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Wong colorblind-safe palette
# ---------------------------------------------------------------------------
ORANGE = "#E69F00"
BLUE = "#0072B2"
GREEN = "#009E73"
PURPLE = "#CC79A7"
RED_ORANGE = "#D55E00"
LIGHT_BLUE = "#56B4E9"
YELLOW = "#F0E442"
GRAY = "#999999"

WONG_PALETTE = [BLUE, ORANGE, GREEN, PURPLE, RED_ORANGE, LIGHT_BLUE, YELLOW, GRAY]

# ---------------------------------------------------------------------------
# Category-specific colours (consistent across all figures)
# ---------------------------------------------------------------------------
CATEGORY_COLORS: dict[str, str] = {
    "so": BLUE,
    "gi": PURPLE,
    "race": ORANGE,
    "religion": GREEN,
    "disability": RED_ORANGE,
    "physical_appearance": LIGHT_BLUE,
    "age": YELLOW,
}

CATEGORY_LABELS: dict[str, str] = {
    "so": "Sexual Orientation",
    "gi": "Gender Identity",
    "race": "Race/Ethnicity",
    "religion": "Religion",
    "disability": "Disability",
    "physical_appearance": "Physical Appearance",
    "age": "Age",
}

# ---------------------------------------------------------------------------
# Sub-group colours within Sexual Orientation
# ---------------------------------------------------------------------------
SO_COLORS: dict[str, str] = {
    "gay": BLUE,
    "lesbian": PURPLE,
    "bisexual": ORANGE,
    "pansexual": GREEN,
}

# ---------------------------------------------------------------------------
# Markers (second visual channel alongside colour)
# ---------------------------------------------------------------------------
CATEGORY_MARKERS: dict[str, str] = {
    "so": "o",
    "gi": "s",
    "race": "^",
    "religion": "D",
    "disability": "v",
    "physical_appearance": "<",
    "age": ">",
}

# ---------------------------------------------------------------------------
# Hypothesised cluster assignments for PCA scatter colouring
# ---------------------------------------------------------------------------
CLUSTER_COLORS: dict[str, str] = {
    "so": BLUE,
    "gi": BLUE,
    "race": ORANGE,
    "religion": ORANGE,
    "disability": GREEN,
    "physical_appearance": GREEN,
    "age": GRAY,
}

# ---------------------------------------------------------------------------
# Font sizes
# ---------------------------------------------------------------------------
TITLE_SIZE = 13
LABEL_SIZE = 12
TICK_SIZE = 10
ANNOT_SIZE = 9
PANEL_LABEL_SIZE = 14

# ---------------------------------------------------------------------------
# Figure defaults
# ---------------------------------------------------------------------------
DPI = 150
GRID_ALPHA = 0.25


def apply_style() -> None:
    """Apply project-wide matplotlib style settings."""
    plt.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "font.size": TICK_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": TICK_SIZE,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": GRID_ALPHA,
        "grid.color": "#cccccc",
    })


def label_panel(ax: plt.Axes, label: str, x: float = -0.08, y: float = 1.06) -> None:
    """Add a panel label (A, B, C, ...) in the upper-left corner."""
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        va="top", ha="left",
    )


def save_fig(fig: plt.Figure, path: str, tight: bool = True) -> None:
    """Save figure with tight layout and project DPI."""
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
