"""SAE localization: analyse cosine convergence curves and generate figures.

Can be run standalone:
    python -m src.sae_localization.analyze --run_dir results/sae_localization/llama2-13b/2026-04-12/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils.io import atomic_save_json, ensure_dir
from src.utils.logging import log

# ── Wong palette + markers ──────────────────────────────────────────────────
ORANGE = "#E69F00"
BLUE = "#0072B2"
GREEN = "#009E73"
PURPLE = "#CC79A7"
VERMILLION = "#D55E00"
SKY = "#56B4E9"
YELLOW = "#F0E442"
GRAY = "#999999"

SEGMENT_STYLE: dict[str, dict[str, Any]] = {
    "stereotyped":            {"color": VERMILLION, "marker": "o", "label": "Stereotyped"},
    "non_stereotyped":        {"color": BLUE,       "marker": "s", "label": "Non-stereotyped"},
    "unknown_selected":       {"color": GREEN,      "marker": "^", "label": "Unknown selected"},
    "correct_aligned":        {"color": VERMILLION, "marker": "o", "label": "Correct aligned"},
    "correct_conflicting":    {"color": BLUE,       "marker": "s", "label": "Correct conflicting"},
    "incorrect_aligned":      {"color": ORANGE,     "marker": "D", "label": "Incorrect aligned"},
    "incorrect_conflicting":  {"color": PURPLE,     "marker": "v", "label": "Incorrect conflicting"},
}

CATEGORY_COLORS = {
    "so": BLUE, "gi": PURPLE, "race": ORANGE, "religion": GREEN,
    "disability": VERMILLION, "physical_appearance": SKY, "age": YELLOW,
}
CATEGORY_LABELS = {
    "so": "Sexual Orientation", "gi": "Gender Identity",
    "race": "Race/Ethnicity", "religion": "Religion",
    "disability": "Disability", "physical_appearance": "Physical Appearance",
    "age": "Age",
}
CATEGORY_MARKERS = {"so": "o", "gi": "s", "race": "^", "religion": "D",
                     "disability": "v", "physical_appearance": "<", "age": ">"}

DPI = 300
MARKEVERY = 4  # plot a marker every N layers for readability


def _save(fig: plt.Figure, path: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── Data loading ─────────────────────────────────────────────────────────────

def _parse_meta(raw: Any) -> dict:
    if isinstance(raw, np.ndarray):
        raw = raw.item() if raw.shape == () else raw.tolist()
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="replace")
    return json.loads(str(raw))


def load_category(act_dir: Path) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Load all items for one category.

    Returns hidden_states (N, n_layers, hidden_dim) float32,
            raw_norms (N, n_layers) float32,
            metadatas list[dict].
    """
    npz_files = sorted(act_dir.glob("item_*.npz"))
    hs_list, norm_list, meta_list = [], [], []
    for f in npz_files:
        d = np.load(f, allow_pickle=True)
        hs_list.append(d["hidden_states"].astype(np.float32))
        norm_list.append(d["hidden_states_raw_norms"])
        meta_list.append(_parse_meta(d["metadata_json"]))
    if not hs_list:
        return np.empty((0, 0, 0)), np.empty((0, 0)), []
    return np.stack(hs_list), np.stack(norm_list), meta_list


def _cosine_with_final(hs: np.ndarray) -> np.ndarray:
    """Compute cosine similarity of each layer with the final layer for each item.

    hs: (N, n_layers, dim) — already unit-normalised per layer.
    Returns (N, n_layers) array.
    """
    final = hs[:, -1:, :]  # (N, 1, dim)
    # dot product of each layer with final (both ~unit norm)
    cos = np.sum(hs * final, axis=2)  # (N, n_layers)
    return cos


def _segment_items(
    metas: list[dict],
) -> dict[str, np.ndarray]:
    """Partition item indices into segments. Returns dict[segment_name -> bool mask]."""
    n = len(metas)
    masks: dict[str, np.ndarray] = {
        "stereotyped": np.zeros(n, dtype=bool),
        "non_stereotyped": np.zeros(n, dtype=bool),
        "unknown_selected": np.zeros(n, dtype=bool),
        "correct_aligned": np.zeros(n, dtype=bool),
        "correct_conflicting": np.zeros(n, dtype=bool),
        "incorrect_aligned": np.zeros(n, dtype=bool),
        "incorrect_conflicting": np.zeros(n, dtype=bool),
    }
    for i, m in enumerate(metas):
        role = m.get("model_answer_role", "unknown")
        is_correct = m.get("is_correct", False)
        cond = m.get("context_condition", "")
        is_stereo = m.get("is_stereotyped_response", False)

        if role == "stereotyped_target":
            masks["stereotyped"][i] = True
        elif role == "non_stereotyped":
            masks["non_stereotyped"][i] = True
        else:
            masks["unknown_selected"][i] = True

        # Disambig breakdown
        if cond == "disambig":
            if is_correct and is_stereo:
                masks["correct_aligned"][i] = True
            elif is_correct and not is_stereo:
                masks["correct_conflicting"][i] = True
            elif not is_correct and is_stereo:
                masks["incorrect_aligned"][i] = True
            elif not is_correct:
                masks["incorrect_conflicting"][i] = True

    return masks


def _mean_sem(arr: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    mean = arr.mean(axis=axis)
    sem = arr.std(axis=axis) / max(np.sqrt(arr.shape[axis]), 1)
    return mean, sem


def _centered_diff(y: np.ndarray) -> np.ndarray:
    """Centered finite-difference derivative."""
    n = len(y)
    d = np.zeros(n, dtype=np.float64)
    d[0] = y[1] - y[0]
    d[-1] = y[-1] - y[-2]
    for i in range(1, n - 1):
        d[i] = (y[i + 1] - y[i - 1]) / 2.0
    return d.astype(np.float32)


# ── Per-category analysis ───────────────────────────────────────────────────

def analyze_category(
    act_dir: Path,
    fig_dir: Path,
    cat_short: str,
) -> dict | None:
    """Run full analysis for one category. Returns summary dict or None."""
    hs, norms, metas = load_category(act_dir)
    if hs.shape[0] == 0:
        log(f"  No items for {cat_short}")
        return None
    n_items, n_layers, hidden_dim = hs.shape
    log(f"  Loaded {n_items} items, {n_layers} layers, dim={hidden_dim}")

    cos = _cosine_with_final(hs)  # (N, n_layers)
    masks = _segment_items(metas)
    cat_label = CATEGORY_LABELS.get(cat_short, cat_short)

    # Per-segment curves
    curves: dict[str, dict] = {}
    for seg_name, mask in masks.items():
        n_seg = int(mask.sum())
        if n_seg < 2:
            continue
        seg_cos = cos[mask]
        mean_c, sem_c = _mean_sem(seg_cos)
        deriv = _centered_diff(mean_c)
        curves[seg_name] = {
            "mean": mean_c, "sem": sem_c, "deriv": deriv, "n": n_seg,
            "peak_deriv_layer": int(np.argmax(deriv)),
        }

    # Divergence layer: earliest layer where |stereo - non_stereo| > 2 * pooled_SE
    divergence_layer = n_layers - 1
    if "stereotyped" in curves and "non_stereotyped" in curves:
        s_mean = curves["stereotyped"]["mean"]
        ns_mean = curves["non_stereotyped"]["mean"]
        s_sem = curves["stereotyped"]["sem"]
        ns_sem = curves["non_stereotyped"]["sem"]
        pooled_se = np.sqrt(s_sem ** 2 + ns_sem ** 2)
        for l in range(n_layers):
            if abs(s_mean[l] - ns_mean[l]) > 2 * pooled_se[l] and pooled_se[l] > 0:
                divergence_layer = l
                break

    # Norm curves (for diagnostic fig)
    norm_curves: dict[str, dict] = {}
    for seg_name, mask in masks.items():
        if mask.sum() < 2:
            continue
        seg_norms = norms[mask]
        nm, nsem = _mean_sem(seg_norms)
        norm_curves[seg_name] = {"mean": nm, "sem": nsem}

    # ── Figure 1: Cosine convergence ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    for seg in ["stereotyped", "non_stereotyped", "unknown_selected"]:
        if seg not in curves:
            continue
        st = SEGMENT_STYLE[seg]
        c = curves[seg]
        ax.plot(c["mean"], color=st["color"], marker=st["marker"],
                markevery=MARKEVERY, markersize=5, linewidth=1.8, label=f"{st['label']} (n={c['n']})")
        ax.fill_between(range(n_layers), c["mean"] - c["sem"], c["mean"] + c["sem"],
                        color=st["color"], alpha=0.15)
    ax.axvline(divergence_layer, color="gray", linestyle="--", linewidth=1,
               label=f"Divergence @ layer {divergence_layer}")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Cosine similarity with final layer", fontsize=12)
    ax.set_title(f"Cosine convergence — {cat_label}", fontsize=13)
    ax.legend(fontsize=9)
    _save(fig, str(fig_dir / f"fig_cosine_convergence_{cat_short}.png"))

    # ── Figure 2: Cosine derivative ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    for seg in ["stereotyped", "non_stereotyped", "unknown_selected"]:
        if seg not in curves:
            continue
        st = SEGMENT_STYLE[seg]
        c = curves[seg]
        ax.plot(c["deriv"], color=st["color"], marker=st["marker"],
                markevery=MARKEVERY, markersize=5, linewidth=1.5, label=st["label"])
        ax.scatter([c["peak_deriv_layer"]], [c["deriv"][c["peak_deriv_layer"]]],
                   color=st["color"], s=80, edgecolors="black", zorder=5)
        ax.annotate(f"L{c['peak_deriv_layer']}", (c["peak_deriv_layer"], c["deriv"][c["peak_deriv_layer"]]),
                    textcoords="offset points", xytext=(5, 8), fontsize=8)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("d(cosine)/d(layer)", fontsize=12)
    ax.set_title(f"Rate of representational change — {cat_label}", fontsize=13)
    ax.legend(fontsize=9)
    _save(fig, str(fig_dir / f"fig_cosine_derivative_{cat_short}.png"))

    # ── Figure 5: Norm trajectory ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    for seg in ["stereotyped", "non_stereotyped", "unknown_selected"]:
        if seg not in norm_curves:
            continue
        st = SEGMENT_STYLE[seg]
        nc = norm_curves[seg]
        ax.plot(nc["mean"], color=st["color"], marker=st["marker"],
                markevery=MARKEVERY, markersize=5, linewidth=1.5, label=st["label"])
        ax.fill_between(range(n_layers), nc["mean"] - nc["sem"], nc["mean"] + nc["sem"],
                        color=st["color"], alpha=0.15)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("L2 norm of hidden state", fontsize=12)
    ax.set_title(f"Raw activation norm trajectory — {cat_label}", fontsize=13)
    ax.legend(fontsize=9)
    _save(fig, str(fig_dir / f"fig_norm_trajectory_{cat_short}.png"))

    # ── Figure 6: Disambig breakdown ────────────────────────────────────
    disambig_segs = ["correct_aligned", "correct_conflicting",
                     "incorrect_aligned", "incorrect_conflicting"]
    has_disambig = any(seg in curves for seg in disambig_segs)
    if has_disambig:
        fig, ax = plt.subplots(figsize=(10, 5))
        for seg in disambig_segs:
            if seg not in curves:
                continue
            st = SEGMENT_STYLE[seg]
            c = curves[seg]
            ax.plot(c["mean"], color=st["color"], marker=st["marker"],
                    markevery=MARKEVERY, markersize=5, linewidth=1.5,
                    label=f"{st['label']} (n={c['n']})")
            ax.fill_between(range(n_layers), c["mean"] - c["sem"],
                            c["mean"] + c["sem"], color=st["color"], alpha=0.15)
        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("Cosine with final layer", fontsize=12)
        ax.set_title(f"Disambig breakdown — {cat_label}", fontsize=13)
        ax.legend(fontsize=9)
        _save(fig, str(fig_dir / f"fig_disambig_breakdown_{cat_short}.png"))

    # ── Figure 7: Layer × layer cosine heatmap ──────────────────────────
    # Mean hidden state across all items at each layer, then layer-layer cosine
    mean_hs = hs.mean(axis=0)  # (n_layers, dim)
    hs_norms = np.linalg.norm(mean_hs, axis=1, keepdims=True)
    hs_norms = np.maximum(hs_norms, 1e-8)
    mean_hs_normed = mean_hs / hs_norms
    layer_layer_cos = mean_hs_normed @ mean_hs_normed.T

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(layer_layer_cos, cmap="viridis", aspect="equal", vmin=0, vmax=1)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)
    ax.set_title(f"Layer × layer cosine similarity — {cat_label}", fontsize=13)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Mean cosine similarity")
    _save(fig, str(fig_dir / f"fig_cosine_heatmap_{cat_short}.png"))

    # ── Figure 8: Subgroup convergence ──────────────────────────────────
    # Group stereotyped items by primary subgroup
    sg_items: dict[str, list[int]] = {}
    for i, m in enumerate(metas):
        if m.get("is_stereotyped_response"):
            groups = m.get("stereotyped_groups", [])
            sg = groups[0].lower() if groups else "unknown"
            sg_items.setdefault(sg, []).append(i)

    sg_with_enough = {sg: idxs for sg, idxs in sg_items.items() if len(idxs) >= 15}
    if len(sg_with_enough) >= 3:
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = [BLUE, VERMILLION, GREEN, ORANGE, PURPLE, SKY, YELLOW, GRAY]
        markers = ["o", "s", "^", "D", "v", "<", ">", "p"]
        for ci, (sg, idxs) in enumerate(sorted(sg_with_enough.items())):
            sg_cos = cos[idxs]
            m_c, _ = _mean_sem(sg_cos)
            ax.plot(m_c, color=colors[ci % len(colors)],
                    marker=markers[ci % len(markers)], markevery=MARKEVERY,
                    markersize=5, linewidth=1.5, label=f"{sg} (n={len(idxs)})")
        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("Cosine with final layer", fontsize=12)
        ax.set_title(f"Subgroup convergence (stereotyped) — {cat_label}", fontsize=13)
        ax.legend(fontsize=9, bbox_to_anchor=(1.02, 1), loc="upper left")
        _save(fig, str(fig_dir / f"fig_subgroup_convergence_{cat_short}.png"))
    elif len(sg_with_enough) >= 1:
        log(f"  Only {len(sg_with_enough)} subgroup(s) with ≥15 items; skipping subgroup figure")

    # Save aggregated cosine curves
    agg_dir = ensure_dir(fig_dir.parent / "cosine_curves")
    agg_arrays: dict[str, np.ndarray] = {}
    for seg, c in curves.items():
        agg_arrays[f"{seg}_mean"] = c["mean"]
        agg_arrays[f"{seg}_sem"] = c["sem"]
        agg_arrays[f"{seg}_deriv"] = c["deriv"]
    np.savez(agg_dir / f"{cat_short}.npz", **agg_arrays)

    # Build summary
    peak_stereo = curves.get("stereotyped", {}).get("peak_deriv_layer", n_layers // 2)
    peak_nstereo = curves.get("non_stereotyped", {}).get("peak_deriv_layer", n_layers // 2)
    rec_lo = max(0, divergence_layer)
    rec_hi = min(n_layers - 1, max(peak_stereo, peak_nstereo) + 2)

    return {
        "divergence_layer": divergence_layer,
        "peak_derivative_layer_stereotyped": peak_stereo,
        "peak_derivative_layer_non_stereotyped": peak_nstereo,
        "recommended_range": [rec_lo, rec_hi],
        "n_stereotyped": int(masks["stereotyped"].sum()),
        "n_non_stereotyped": int(masks["non_stereotyped"].sum()),
        "n_unknown": int(masks["unknown_selected"].sum()),
        "n_items": n_items,
    }


# ── Cross-category figures ──────────────────────────────────────────────────

def generate_cross_category_figures(
    per_cat: dict[str, dict],
    run_dir: Path,
) -> None:
    """Generate Figures 3, 4, and 9."""
    fig_dir = ensure_dir(run_dir / "figures")
    cosine_dir = run_dir / "cosine_curves"
    cats = sorted(per_cat.keys())

    # ── Figure 3: Divergence summary ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(10, len(cats) * 1.5), 5))
    x = np.arange(len(cats))
    w = 0.35
    div_layers = [per_cat[c]["divergence_layer"] for c in cats]
    peak_layers = [per_cat[c]["peak_derivative_layer_stereotyped"] for c in cats]
    display = [CATEGORY_LABELS.get(c, c) for c in cats]

    b1 = ax.bar(x - w / 2, div_layers, w, label="Divergence layer",
                color=BLUE, edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x + w / 2, peak_layers, w, label="Peak derivative (stereo)",
                color=VERMILLION, edgecolor="black", linewidth=0.5)
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3, f"{int(h)}",
                    ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(display, rotation=45, ha="right")
    ax.set_ylabel("Layer", fontsize=12)
    ax.set_title("Layer of bias incorporation across categories", fontsize=13)
    ax.legend(fontsize=10)
    _save(fig, str(fig_dir / "fig_divergence_summary.png"))

    # ── Figure 4: Overlay all categories ────────────────────────────────
    fig, (ax_s, ax_ns) = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    for ci, cat in enumerate(cats):
        cat_file = cosine_dir / f"{cat}.npz"
        if not cat_file.exists():
            continue
        d = np.load(cat_file)
        color = CATEGORY_COLORS.get(cat, GRAY)
        marker = CATEGORY_MARKERS.get(cat, "o")
        label = CATEGORY_LABELS.get(cat, cat)
        if "stereotyped_mean" in d:
            ax_s.plot(d["stereotyped_mean"], color=color, marker=marker,
                      markevery=MARKEVERY, markersize=4, linewidth=1.2, label=label)
        if "non_stereotyped_mean" in d:
            ax_ns.plot(d["non_stereotyped_mean"], color=color, marker=marker,
                       markevery=MARKEVERY, markersize=4, linewidth=1.2, label=label)

    ax_s.set_title("Stereotyped responses", fontsize=12)
    ax_ns.set_title("Non-stereotyped responses", fontsize=12)
    for ax in [ax_s, ax_ns]:
        ax.set_xlabel("Layer", fontsize=12)
    ax_s.set_ylabel("Cosine with final layer", fontsize=12)
    ax_ns.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("Cosine convergence across categories", fontsize=13, y=1.02)
    _save(fig, str(fig_dir / "fig_cosine_convergence_overlay.png"))

    # ── Figure 9: Layer recommendation ──────────────────────────────────
    n_layers = max(per_cat[c].get("recommended_range", [0, 1])[1] for c in cats) + 5
    # If we don't know n_layers exactly, estimate from data
    for cat_file in cosine_dir.glob("*.npz"):
        d = np.load(cat_file)
        for k in d.files:
            if k.endswith("_mean"):
                n_layers = max(n_layers, len(d[k]))
                break

    # Mean absolute divergence per layer
    divergences = np.zeros(n_layers, dtype=np.float32)
    derivs = np.zeros(n_layers, dtype=np.float32)
    n_contrib = 0
    for cat in cats:
        cat_file = cosine_dir / f"{cat}.npz"
        if not cat_file.exists():
            continue
        d = np.load(cat_file)
        if "stereotyped_mean" in d and "non_stereotyped_mean" in d:
            s = d["stereotyped_mean"]
            ns = d["non_stereotyped_mean"]
            ln = min(len(s), len(ns), n_layers)
            divergences[:ln] += np.abs(s[:ln] - ns[:ln])
            n_contrib += 1
        if "stereotyped_deriv" in d:
            sd = d["stereotyped_deriv"]
            ln = min(len(sd), n_layers)
            derivs[:ln] += np.abs(sd[:ln])

    if n_contrib > 0:
        divergences /= n_contrib
        derivs /= n_contrib

    # Recommended range
    all_recs = [per_cat[c]["recommended_range"] for c in cats]
    overall_lo = min(r[0] for r in all_recs)
    overall_hi = max(r[1] for r in all_recs)

    fig, ax = plt.subplots(figsize=(12, 5))
    layers_x = np.arange(n_layers)
    ax.plot(layers_x, divergences, color=VERMILLION, linewidth=2, label="Mean |stereo − non-stereo|")
    ax.plot(layers_x, derivs, color=BLUE, linewidth=2, label="Mean |derivative| (stereo)")
    ax.axvspan(overall_lo, overall_hi, alpha=0.15, color=GREEN,
               label=f"Recommended: layers {overall_lo}–{overall_hi}")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Magnitude", fontsize=12)
    ax.set_title("SAE injection layer recommendation", fontsize=13)
    ax.legend(fontsize=10)
    _save(fig, str(fig_dir / "fig_layer_recommendation.png"))

    return overall_lo, overall_hi


def run_analysis(run_dir: Path, categories: list[str] | None = None) -> dict:
    """Run full analysis on a completed extraction run."""
    run_dir = Path(run_dir)
    fig_dir = ensure_dir(run_dir / "figures")
    act_base = run_dir / "activations"

    if categories is None:
        categories = sorted([d.name for d in act_base.iterdir() if d.is_dir()])

    per_cat: dict[str, dict] = {}
    for cat in categories:
        act_dir = act_base / cat
        if not act_dir.exists():
            continue
        log(f"\nAnalyzing: {cat}")
        result = analyze_category(act_dir, fig_dir, cat)
        if result is not None:
            per_cat[cat] = result

    if not per_cat:
        log("No categories with data to analyze")
        return {}

    # Cross-category figures
    log("\nGenerating cross-category figures...")
    overall_lo, overall_hi = generate_cross_category_figures(per_cat, run_dir)

    # Build recommendation
    recommendation = {
        "per_category": per_cat,
        "overall_recommended_range": [int(overall_lo), int(overall_hi)],
        "notes": "Range selected as union of per-category recommended ranges, "
                 "bounded by earliest divergence and latest peak derivative",
    }
    rec_path = run_dir / "layer_recommendation.json"
    atomic_save_json(recommendation, rec_path)
    log(f"\nLayer recommendation -> {rec_path}")
    return recommendation


# ── CLI entry point ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse SAE localization results.")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--categories", type=str, default=None,
                        help="Comma-separated category short names (default: all found)")
    args = parser.parse_args()
    cats = args.categories.split(",") if args.categories else None
    run_analysis(Path(args.run_dir), cats)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    main()
