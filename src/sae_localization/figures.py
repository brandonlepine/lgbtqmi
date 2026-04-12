"""Figure generation for SAE analysis (Figures 10–19).

All figures use Wong colorblind-safe palette and save as PNG + PDF.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover
    pd = None  # type: ignore

from src.visualization.style import (
    BLUE,
    CATEGORY_COLORS,
    CATEGORY_LABELS,
    DPI,
    GRAY,
    GREEN,
    ORANGE,
    PURPLE,
    RED_ORANGE,
    WONG_PALETTE,
    apply_style,
    save_fig,
)
from src.utils.logging import log

# Wong-palette derived bias colours
VERMILLION = RED_ORANGE  # "#D55E00" — pro-bias / stereotyped
ANTI_BIAS_BLUE = BLUE    # "#0072B2" — anti-bias / non-stereotyped

apply_style()


def _require_pandas() -> None:
    global pd  # noqa: PLW0603
    if pd is not None:
        return
    try:
        import pandas as _pd  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pandas is required for SAE figure generation (DataFrame inputs). "
            "Install with: pip install pandas"
        ) from exc
    pd = _pd  # type: ignore


def _dual_save(fig: plt.Figure, path: str | Path, tight: bool = True) -> None:
    """Save figure as both PNG and PDF."""
    path = Path(path)
    save_fig(fig, str(path), tight=tight)
    pdf_path = path.with_suffix(".pdf")
    fig_copy_needed = False  # fig is already closed by save_fig
    # Re-open not possible — save PDF first
    # Instead, modify save_fig workflow: save both before close.
    pass


def _save_both(fig: plt.Figure, path: str | Path, tight: bool = True) -> None:
    """Save as PNG and PDF, then close."""
    path = Path(path)
    if tight:
        fig.tight_layout()
    fig.savefig(str(path), dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(
        str(path.with_suffix(".pdf")),
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 10: Volcano plots
# ---------------------------------------------------------------------------


def fig_volcano(
    df: pd.DataFrame,
    category: str,
    target_layer: int,
    output_dir: Path,
) -> None:
    """Volcano plot: Cohen's d vs -log10(FDR p-value).

    One plot per category + pooled.
    """
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    x = df["cohens_d"].values
    y = -np.log10(np.clip(df["p_value_fdr"].values, 1e-300, 1.0))
    sig = df["is_significant"].values
    direction = df["direction"].values

    # Non-significant: gray
    ns = ~sig
    ax.scatter(x[ns], y[ns], c=GRAY, s=8, alpha=0.3, label="Not significant", rasterized=True)

    # Pro-bias: vermillion
    pro = sig & (direction == "pro_bias")
    ax.scatter(x[pro], y[pro], c=VERMILLION, s=20, alpha=0.7, label="Pro-bias")

    # Anti-bias: blue
    anti = sig & (direction == "anti_bias")
    ax.scatter(x[anti], y[anti], c=ANTI_BIAS_BLUE, s=20, alpha=0.7, label="Anti-bias")

    # Threshold lines
    fdr_line = -np.log10(0.05)
    ax.axhline(fdr_line, ls="--", c=GRAY, lw=0.8, label="FDR = 0.05")
    ax.axvline(0.3, ls="--", c=GRAY, lw=0.8)
    ax.axvline(-0.3, ls="--", c=GRAY, lw=0.8)

    # Annotate top 5 by effect size
    if sig.any():
        top5 = df.loc[sig].nlargest(5, "cohens_d", keep="first")
        for _, row in top5.iterrows():
            ax.annotate(
                f"L{target_layer}_F{int(row['feature_idx'])}",
                (row["cohens_d"], -np.log10(max(row["p_value_fdr"], 1e-300))),
                fontsize=7,
                alpha=0.8,
                ha="left",
            )

    cat_label = CATEGORY_LABELS.get(category, category)
    ax.set_xlabel("Cohen's d (effect size)")
    ax.set_ylabel("$-\\log_{10}$(FDR-corrected p-value)")
    ax.set_title(f"Differential SAE feature activation \u2014 {cat_label}")
    ax.legend(fontsize=8, loc="upper left")

    fname = f"fig_volcano_{category}_layer_{target_layer}"
    _save_both(fig, output_dir / f"{fname}.png")
    log(f"    Saved {fname}")


# ---------------------------------------------------------------------------
# Figure 11: Feature overlap heatmap
# ---------------------------------------------------------------------------


def fig_feature_overlap_heatmap(
    overlap: dict[str, Any],
    target_layer: int,
    output_dir: Path,
) -> None:
    """Heatmap of Jaccard similarity between significant feature sets."""
    jaccard = overlap.get("cross_category_jaccard")
    if not jaccard:
        log("    Skipping overlap heatmap: no cross-category data")
        return

    cats = sorted(jaccard.keys())
    n = len(cats)
    if n < 2:
        return

    mat = np.zeros((n, n))
    for i, c1 in enumerate(cats):
        for j, c2 in enumerate(cats):
            mat[i, j] = jaccard[c1].get(c2, 0.0)

    labels = [CATEGORY_LABELS.get(c, c) for c in cats]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(mat, cmap="YlOrRd", vmin=0)

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=9)

    for i in range(n):
        for j in range(n):
            val = mat[i, j]
            text = f"{int(val)}" if i == j else f"{val:.2f}"
            color = "white" if val > 0.5 * mat.max() else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)

    fig.colorbar(im, ax=ax, label="Jaccard similarity", shrink=0.8)
    ax.set_title(f"Cross-category feature overlap (layer {target_layer})")

    fname = f"fig_feature_overlap_heatmap_layer_{target_layer}"
    _save_both(fig, output_dir / f"{fname}.png")
    log(f"    Saved {fname}")


# ---------------------------------------------------------------------------
# Figure 12: Feature breadth histogram
# ---------------------------------------------------------------------------


def fig_feature_breadth(
    overlap: dict[str, Any],
    target_layer: int,
    output_dir: Path,
) -> None:
    """Histogram of how many categories each significant feature appears in."""
    breadth = overlap.get("feature_breadth", {})
    dist = breadth.get("distribution", {})
    if not dist:
        log("    Skipping breadth histogram: no data")
        return

    x_vals = sorted(int(k) for k in dist.keys())
    counts = [dist.get(str(v), 0) for v in x_vals]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(x_vals, counts, color=BLUE, edgecolor="white", width=0.7)

    narrow = breadth.get("narrow_1_cat", 0)
    broad = breadth.get("broad_5plus_cats", 0)
    ax.annotate(
        f"{narrow} features in 1 category only",
        xy=(1, counts[0] if x_vals[0] == 1 else 0),
        xytext=(2.5, max(counts) * 0.85),
        arrowprops=dict(arrowstyle="->", color=GRAY),
        fontsize=8,
    )
    if broad > 0:
        ax.text(
            6, max(counts) * 0.7,
            f"{broad} features in 5+ categories",
            fontsize=8, ha="center",
        )

    ax.set_xlabel("Number of categories")
    ax.set_ylabel("Number of significant features")
    ax.set_title(f"Feature breadth distribution (layer {target_layer})")
    ax.set_xticks(x_vals)

    fname = f"fig_feature_breadth_layer_{target_layer}"
    _save_both(fig, output_dir / f"{fname}.png")
    log(f"    Saved {fname}")


# ---------------------------------------------------------------------------
# Figure 13: Subgroup specificity (UpSet-style)
# ---------------------------------------------------------------------------


def fig_subgroup_specificity(
    overlap: dict[str, Any],
    category: str,
    target_layer: int,
    output_dir: Path,
) -> None:
    """UpSet-style bar chart showing overlap of significant features across subcategories."""
    specificity = overlap.get("subgroup_specificity", {})
    if category not in specificity:
        return

    cat_data = specificity[category]
    counts = cat_data.get("feature_counts", {})
    jaccard = cat_data.get("jaccard", {})
    subs = sorted(counts.keys())

    if len(subs) < 2:
        return

    # Simple horizontal bar showing unique + shared counts
    fig, ax = plt.subplots(figsize=(8, max(4, len(subs) * 0.8 + 2)))

    y_pos = np.arange(len(subs))
    vals = [counts.get(s, 0) for s in subs]
    colors = [WONG_PALETTE[i % len(WONG_PALETTE)] for i in range(len(subs))]

    ax.barh(y_pos, vals, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(subs, fontsize=9)
    ax.set_xlabel("Number of significant features")
    cat_label = CATEGORY_LABELS.get(category, category)
    ax.set_title(f"Subgroup-specific bias features \u2014 {cat_label} (layer {target_layer})")

    for i, v in enumerate(vals):
        ax.text(v + 0.3, i, str(v), va="center", fontsize=8)

    fname = f"fig_subgroup_specificity_{category}_layer_{target_layer}"
    _save_both(fig, output_dir / f"{fname}.png")
    log(f"    Saved {fname}")


# ---------------------------------------------------------------------------
# Figure 14: Top feature profiles (small multiples)
# ---------------------------------------------------------------------------


def fig_top_feature_profiles(
    reports: list[dict[str, Any]],
    target_layer: int,
    output_dir: Path,
    max_features: int = 20,
) -> None:
    """Grid of bar charts showing per-category mean activation per feature."""
    reports = reports[:max_features]
    if not reports:
        return

    n = len(reports)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    for idx, report in enumerate(reports):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        cat_means = report.get("per_category_means", {})
        cats = sorted(cat_means.keys())
        if not cats:
            ax.set_visible(False)
            continue

        x = np.arange(len(cats))
        width = 0.35
        stereo_vals = [cat_means[c].get("stereotyped", 0) for c in cats]
        non_stereo_vals = [cat_means[c].get("non_stereotyped", 0) for c in cats]

        ax.bar(x - width / 2, stereo_vals, width, color=VERMILLION, alpha=0.8, label="Stereo")
        ax.bar(x + width / 2, non_stereo_vals, width, color=ANTI_BIAS_BLUE, alpha=0.8, label="Non-stereo")

        ax.set_xticks(x)
        ax.set_xticklabels(cats, rotation=45, ha="right", fontsize=6)
        ax.set_title(report.get("feature_label", f"F{report['feature_idx']}"), fontsize=9)
        if idx == 0:
            ax.legend(fontsize=6)

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle(f"Top feature profiles (layer {target_layer})", fontsize=12)
    fname = f"fig_top_feature_profiles_layer_{target_layer}"
    _save_both(fig, output_dir / f"{fname}.png")
    log(f"    Saved {fname}")


# ---------------------------------------------------------------------------
# Figure 15: Activation distributions (top 12)
# ---------------------------------------------------------------------------


def fig_activation_distributions(
    reports: list[dict[str, Any]],
    target_layer: int,
    output_dir: Path,
    max_features: int = 12,
) -> None:
    """Grid of overlapping histograms for top features by effect size."""
    reports = sorted(reports, key=lambda r: abs(r.get("cohens_d", 0)), reverse=True)
    reports = reports[:max_features]
    if not reports:
        return

    n = len(reports)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    for idx, report in enumerate(reports):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        dist = report.get("activation_dist", {})
        s_vals = np.array(dist.get("stereotyped_values", []))
        ns_vals = np.array(dist.get("non_stereotyped_values", []))

        if len(s_vals) == 0 and len(ns_vals) == 0:
            ax.set_visible(False)
            continue

        all_vals = np.concatenate([s_vals, ns_vals])
        if all_vals.max() - all_vals.min() < 1e-10:
            ax.set_visible(False)
            continue

        bins = np.linspace(all_vals.min(), all_vals.max(), 30)
        if len(s_vals) > 0:
            ax.hist(s_vals, bins=bins, color=VERMILLION, alpha=0.6, label="Stereotyped", density=True)
        if len(ns_vals) > 0:
            ax.hist(ns_vals, bins=bins, color=ANTI_BIAS_BLUE, alpha=0.6, label="Non-stereotyped", density=True)

        d_val = report.get("cohens_d", 0)
        label = report.get("feature_label", f"F{report['feature_idx']}")
        ax.set_title(f"{label}  d={d_val:.2f}", fontsize=8)
        if idx == 0:
            ax.legend(fontsize=6)

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle(f"Activation distributions \u2014 top features (layer {target_layer})", fontsize=12)
    fname = f"fig_activation_distributions_top_layer_{target_layer}"
    _save_both(fig, output_dir / f"{fname}.png")
    log(f"    Saved {fname}")


# ---------------------------------------------------------------------------
# Figure 16: Hybrid validation scatter
# ---------------------------------------------------------------------------


def fig_hybrid_validation(
    hybrid_summary: dict[str, Any],
    target_layer: int,
    output_dir: Path,
) -> None:
    """Scatter: |cos with DIM direction| vs |Cohen's d|."""
    directions = hybrid_summary.get("directions_analysed", [])
    if not directions:
        return

    for proj in directions:
        label = proj.get("direction_label", "unknown")
        scatter_path = (
            output_dir
            / "hybrid_projection"
            / f"scatter_{label}_layer_{target_layer}.json"
        )
        if not scatter_path.exists():
            continue

        with open(scatter_path) as f:
            scatter_data = json.load(f)
        if not scatter_data:
            continue

        abs_cos = np.array([d["abs_cosine"] for d in scatter_data])
        abs_d = np.array([d["abs_cohens_d"] for d in scatter_data])
        in_sig = np.array([d["in_sig_differential"] for d in scatter_data])
        in_aligned = np.array([d["in_top_aligned"] for d in scatter_data])

        fig, ax = plt.subplots(figsize=(7, 6))

        # Four categories
        both = in_sig & in_aligned
        dim_only = in_aligned & (~in_sig)
        diff_only = in_sig & (~in_aligned)
        neither = (~in_sig) & (~in_aligned)

        ax.scatter(abs_cos[neither], abs_d[neither], c=GRAY, s=6, alpha=0.2, label="Neither", rasterized=True)
        ax.scatter(abs_cos[diff_only], abs_d[diff_only], c=BLUE, s=15, alpha=0.6, label="Differential only")
        ax.scatter(abs_cos[dim_only], abs_d[dim_only], c=ORANGE, s=15, alpha=0.6, label="DIM-aligned only")
        ax.scatter(abs_cos[both], abs_d[both], c=GREEN, s=25, alpha=0.8, label="Both")

        rho = proj.get("spearman_r", None)
        p_rho = proj.get("spearman_p", None)
        if rho is not None:
            ax.text(
                0.95, 0.05,
                f"Spearman r = {rho:.3f}\np = {p_rho:.1e}",
                transform=ax.transAxes,
                ha="right", va="bottom",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
            )

        ax.set_xlabel("|Cosine similarity with DIM direction|")
        ax.set_ylabel("|Cohen's d| (differential analysis)")
        ax.set_title(
            f"DIM \u00d7 Differential convergence \u2014 {label} (layer {target_layer})"
        )
        ax.legend(fontsize=8, loc="upper left")

        fname = f"fig_hybrid_validation_{label}_layer_{target_layer}"
        _save_both(fig, output_dir / f"{fname}.png")
        log(f"    Saved {fname}")


# ---------------------------------------------------------------------------
# Figure 17: Logit attribution (top 10)
# ---------------------------------------------------------------------------


def fig_logit_attribution(
    reports: list[dict[str, Any]],
    target_layer: int,
    output_dir: Path,
    max_features: int = 10,
) -> None:
    """Horizontal bar charts of promoted/suppressed tokens per feature."""
    # Filter to reports with logit data
    with_logits = [
        r for r in reports
        if r.get("top_promoted_tokens") or r.get("top_suppressed_tokens")
    ]
    with_logits = with_logits[:max_features]
    if not with_logits:
        log("    Skipping logit attribution figure: no logit data")
        return

    n = len(with_logits)
    fig, axes = plt.subplots(n, 1, figsize=(8, 2.5 * n))
    if n == 1:
        axes = [axes]

    for idx, report in enumerate(with_logits):
        ax = axes[idx]
        label = report.get("feature_label", f"F{report['feature_idx']}")

        promoted = report.get("top_promoted_tokens", [])[:5]
        suppressed = report.get("top_suppressed_tokens", [])[:5]

        tokens = [p["token"] for p in promoted] + [s["token"] for s in suppressed]
        values = [p["logit_change"] for p in promoted] + [s["logit_change"] for s in suppressed]
        colors = [VERMILLION] * len(promoted) + [ANTI_BIAS_BLUE] * len(suppressed)

        y_pos = np.arange(len(tokens))
        ax.barh(y_pos, values, color=colors, height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(tokens, fontsize=8)
        ax.set_title(label, fontsize=9)
        ax.axvline(0, color="black", lw=0.5)

    fig.suptitle(
        f"Logit attribution \u2014 top features (layer {target_layer})",
        fontsize=12,
    )
    fname = f"fig_logit_attribution_top_layer_{target_layer}"
    _save_both(fig, output_dir / f"{fname}.png")
    log(f"    Saved {fname}")


# ---------------------------------------------------------------------------
# Figure 18: Co-activation network
# ---------------------------------------------------------------------------


def fig_co_activation_network(
    reports: list[dict[str, Any]],
    feature_results: dict[str, pd.DataFrame],
    target_layer: int,
    output_dir: Path,
    d_threshold: float = 0.5,
    corr_threshold: float = 0.3,
) -> None:
    """Force-directed network graph of co-activation among significant features."""
    try:
        import networkx as nx
    except ImportError:
        log("    Skipping co-activation network: networkx not installed")
        return

    # Build node set: features with |d| > threshold
    node_features: dict[int, dict[str, Any]] = {}
    for r in reports:
        fid = r["feature_idx"]
        if abs(r.get("cohens_d", 0)) > d_threshold:
            node_features[fid] = r

    if len(node_features) < 3:
        log("    Skipping co-activation network: <3 high-effect features")
        return

    G = nx.Graph()
    for fid, r in node_features.items():
        primary_cat = ""
        cats = r.get("categories_significant_in", [])
        if cats:
            primary_cat = cats[0]
        G.add_node(
            fid,
            cohens_d=abs(r.get("cohens_d", 0)),
            category=primary_cat,
        )

    # Add edges from co-activation data
    for r in reports:
        fid = r["feature_idx"]
        if fid not in node_features:
            continue
        for coact in r.get("co_activated_features", []):
            if isinstance(coact, dict):
                other = coact.get("feature_idx", coact)
                corr = coact.get("correlation", 0)
            else:
                other = coact
                corr = 0.5  # default if only IDs stored

            if other in node_features and abs(corr) > corr_threshold:
                G.add_edge(fid, other, weight=abs(corr))

    if G.number_of_edges() == 0:
        log("    Skipping co-activation network: no edges above threshold")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)

    # Node colours by category
    node_colors = []
    for n in G.nodes():
        cat = G.nodes[n].get("category", "")
        node_colors.append(CATEGORY_COLORS.get(cat, GRAY))

    node_sizes = [
        G.nodes[n].get("cohens_d", 0.3) * 200 for n in G.nodes()
    ]

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, width=1)
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.8,
    )
    labels = {n: f"F{n}" for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=6)

    ax.set_title(
        f"Co-activation network \u2014 bias features (layer {target_layer})"
    )
    ax.axis("off")

    fname = f"fig_co_activation_network_layer_{target_layer}"
    _save_both(fig, output_dir / f"{fname}.png")
    log(f"    Saved {fname}")


# ---------------------------------------------------------------------------
# Figure 19: Category feature count summary
# ---------------------------------------------------------------------------


def fig_category_feature_count(
    per_cat_df: pd.DataFrame,
    target_layer: int,
    output_dir: Path,
) -> None:
    """Stacked bar chart: pro-bias vs anti-bias feature counts per category."""
    if per_cat_df.empty:
        return

    sig = per_cat_df[per_cat_df["is_significant"]]
    if sig.empty:
        return

    cats = sorted(sig["category"].unique())

    pro_counts = []
    anti_counts = []
    for c in cats:
        cat_sig = sig[sig["category"] == c]
        pro_counts.append((cat_sig["direction"] == "pro_bias").sum())
        anti_counts.append((cat_sig["direction"] == "anti_bias").sum())

    x = np.arange(len(cats))
    labels = [CATEGORY_LABELS.get(c, c) for c in cats]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, pro_counts, color=VERMILLION, label="Pro-bias", width=0.6)
    ax.bar(x, anti_counts, bottom=pro_counts, color=ANTI_BIAS_BLUE, label="Anti-bias", width=0.6)

    for i in range(len(cats)):
        total = pro_counts[i] + anti_counts[i]
        ax.text(i, total + 0.5, str(total), ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Number of significant features")
    ax.set_title(f"Bias feature counts per category (layer {target_layer})")
    ax.legend()

    fname = f"fig_category_feature_count_summary_layer_{target_layer}"
    _save_both(fig, output_dir / f"{fname}.png")
    log(f"    Saved {fname}")


# ---------------------------------------------------------------------------
# Generate all figures
# ---------------------------------------------------------------------------


def generate_all_figures(
    feature_results: dict[str, pd.DataFrame],
    overlap: dict[str, Any],
    reports: list[dict[str, Any]],
    hybrid_summary: Optional[dict[str, Any]],
    target_layer: int,
    output_dir: Path,
) -> None:
    """Generate all SAE analysis figures (10–19)."""
    from src.utils.io import ensure_dir
    fig_dir = ensure_dir(output_dir / "figures")

    log("  Generating figures ...")

    # Fig 10: Volcano plots
    pooled_df = feature_results.get("pooled", pd.DataFrame())
    if not pooled_df.empty:
        fig_volcano(pooled_df, "pooled", target_layer, fig_dir)

    per_cat_df = feature_results.get("per_category", pd.DataFrame())
    if not per_cat_df.empty:
        for cat in per_cat_df["category"].unique():
            cat_df = per_cat_df[per_cat_df["category"] == cat]
            fig_volcano(cat_df, cat, target_layer, fig_dir)

    # Fig 11: Overlap heatmap
    fig_feature_overlap_heatmap(overlap, target_layer, fig_dir)

    # Fig 12: Breadth histogram
    fig_feature_breadth(overlap, target_layer, fig_dir)

    # Fig 13: Subgroup specificity
    for cat in overlap.get("subgroup_specificity", {}).keys():
        fig_subgroup_specificity(overlap, cat, target_layer, fig_dir)

    # Fig 14: Top feature profiles
    fig_top_feature_profiles(reports, target_layer, fig_dir)

    # Fig 15: Activation distributions
    fig_activation_distributions(reports, target_layer, fig_dir)

    # Fig 16: Hybrid validation
    if hybrid_summary:
        fig_hybrid_validation(hybrid_summary, target_layer, fig_dir)

    # Fig 17: Logit attribution
    fig_logit_attribution(reports, target_layer, fig_dir)

    # Fig 18: Co-activation network
    fig_co_activation_network(
        reports, feature_results, target_layer, fig_dir
    )

    # Fig 19: Category feature count
    if not per_cat_df.empty:
        fig_category_feature_count(per_cat_df, target_layer, fig_dir)

    log("  Figures complete.")
