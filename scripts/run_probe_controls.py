#!/usr/bin/env python3
"""Probe selectivity controls for linear probing results.

Three controls that test whether probes learn identity-specific features
or exploit surface cues:

  A. Permutation baseline — shuffle labels, measure selectivity gap
  B. Structural control — probe for context condition and answer position
  C. Cross-category generalization — train on one category, test on another

Does NOT require model — uses saved hidden states from Stage 1 .npz files.

Usage
-----
python scripts/run_probe_controls.py \\
    --localization_dir results/sae_localization/llama-3.1-8b/2026-04-12/ \\
    --categories so,race,gi,disability,religion \\
    --n_permutations 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import atomic_save_json, ensure_dir
from src.utils.logging import log

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.visualization.style import (
    BLUE, CATEGORY_COLORS, CATEGORY_LABELS, DPI, GRAY, GREEN, ORANGE,
    WONG_PALETTE, apply_style,
)
apply_style()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe selectivity controls")
    p.add_argument("--localization_dir", required=True)
    p.add_argument("--categories", default="so,race,gi,disability,religion")
    p.add_argument("--n_permutations", type=int, default=10)
    p.add_argument("--layer_stride", type=int, default=2,
                   help="Probe every Nth layer (default 2)")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--skip_figures", action="store_true")
    return p.parse_args()


def _save_both(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(str(path), dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(str(path.with_suffix(".pdf")), bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_category_data(
    loc_dir: Path, cat: str,
) -> dict[str, Any]:
    """Load hidden states and metadata for a category.

    Returns dict with:
      hidden_states: ndarray (n_items, n_layers, hidden_dim) float32
      raw_norms: ndarray (n_items, n_layers) float32
      meta: list of dicts per item
    """
    act_dir = loc_dir / "activations" / cat
    if not act_dir.is_dir():
        return {}

    all_hs = []
    all_norms = []
    all_meta = []

    for npz_path in sorted(act_dir.glob("item_*.npz")):
        try:
            data = np.load(npz_path, allow_pickle=True)
            hs = data["hidden_states"]  # (n_layers, hidden_dim)
            raw_norms = data["hidden_states_raw_norms"]  # (n_layers,)
            meta_raw = data["metadata_json"]
            meta = json.loads(
                meta_raw.item() if meta_raw.shape == () else str(meta_raw)
            )

            all_hs.append(hs)
            all_norms.append(raw_norms)
            all_meta.append(meta)
        except Exception:
            continue

    if not all_hs:
        return {}

    hs_arr = np.stack(all_hs)  # (n_items, n_layers, hidden_dim)
    norms_arr = np.stack(all_norms)  # (n_items, n_layers)

    return {
        "hidden_states": hs_arr,
        "raw_norms": norms_arr,
        "meta": all_meta,
        "n_items": len(all_meta),
        "n_layers": hs_arr.shape[1],
    }


def _get_raw_hs(data: dict[str, Any], layer: int) -> np.ndarray:
    """Get de-normalized hidden states at a layer: (n_items, hidden_dim).

    Replaces NaN/inf with 0 to prevent downstream failures in PCA/probes.
    """
    hs = data["hidden_states"][:, layer, :].astype(np.float32)  # (n_items, hidden_dim)
    norms = data["raw_norms"][:, layer].astype(np.float32)  # (n_items,)
    result = hs * norms[:, None]
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# Probe training
# ---------------------------------------------------------------------------


def train_probe(X: np.ndarray, y: np.ndarray, n_folds: int = 5) -> float:
    """Train a logistic regression probe with PCA + stratified CV.

    Returns mean cross-validation accuracy.
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    n_classes = len(le.classes_)
    if n_classes < 2:
        return 1.0  # trivial

    # PCA reduction
    n_components = min(50, X.shape[0] - 1, X.shape[1])
    if n_components < 1:
        return 0.0
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # Adjust folds if too few samples per class
    min_class_count = min(np.bincount(y_enc))
    actual_folds = min(n_folds, min_class_count)
    if actual_folds < 2:
        return 0.0  # can't cross-validate

    skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
    accs = []
    for train_idx, test_idx in skf.split(X_pca, y_enc):
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        clf.fit(X_pca[train_idx], y_enc[train_idx])
        accs.append(clf.score(X_pca[test_idx], y_enc[test_idx]))

    return float(np.mean(accs))


def train_probe_transfer(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
) -> float:
    """Train on (X_train, y_train), test on (X_test, y_test). PCA fit on train."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    le.fit(np.concatenate([y_train, y_test]))
    y_tr = le.transform(y_train)
    y_te = le.transform(y_test)

    if len(set(y_tr)) < 2 or len(set(y_te)) < 2:
        return 0.5

    n_components = min(50, X_train.shape[0] - 1, X_train.shape[1])
    if n_components < 1:
        return 0.5
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    clf.fit(X_train_pca, y_tr)
    return float(clf.score(X_test_pca, y_te))


# ---------------------------------------------------------------------------
# Control A: Permutation baseline
# ---------------------------------------------------------------------------


def run_permutation_control(
    data: dict[str, Any],
    cat: str,
    layers: list[int],
    n_permutations: int,
) -> dict[str, Any]:
    """Permutation control: real vs shuffled subgroup labels."""
    meta = data["meta"]

    # Subgroup labels
    y_sub = np.array([
        m.get("stereotyped_groups", [""])[0]
        if isinstance(m.get("stereotyped_groups"), list) and m.get("stereotyped_groups")
        else str(m.get("model_answer_role", ""))
        for m in meta
    ])

    # Filter to items with valid labels
    valid_mask = np.array([len(str(y)) > 0 and y != "" for y in y_sub])
    if valid_mask.sum() < 10:
        return {"category": cat, "layers": [], "real_acc": [], "perm_mean": [], "perm_std": []}

    y_filtered = y_sub[valid_mask]
    rng = np.random.default_rng(42)

    results: dict[str, Any] = {
        "category": cat,
        "layers": [],
        "real_acc": [],
        "perm_mean": [],
        "perm_std": [],
        "selectivity": [],
    }

    for layer in layers:
        X = _get_raw_hs(data, layer)[valid_mask]
        real_acc = train_probe(X, y_filtered)

        perm_accs = []
        for _ in range(n_permutations):
            y_perm = rng.permutation(y_filtered)
            perm_accs.append(train_probe(X, y_perm))

        perm_mean = float(np.mean(perm_accs))
        perm_std = float(np.std(perm_accs))
        selectivity = real_acc - perm_mean

        results["layers"].append(layer)
        results["real_acc"].append(round(real_acc, 4))
        results["perm_mean"].append(round(perm_mean, 4))
        results["perm_std"].append(round(perm_std, 4))
        results["selectivity"].append(round(selectivity, 4))

        log(f"    Layer {layer:2d}: real={real_acc:.3f} perm={perm_mean:.3f}+/-{perm_std:.3f} "
            f"selectivity={selectivity:.3f}")

    return results


# ---------------------------------------------------------------------------
# Control B: Structural control tasks
# ---------------------------------------------------------------------------


def run_structural_control(
    data: dict[str, Any],
    cat: str,
    layers: list[int],
) -> dict[str, Any]:
    """Structural probes: context condition and answer position."""
    meta = data["meta"]

    # Load processed stimuli to get structural labels
    proc_dir = PROJECT_ROOT / "data" / "processed"
    files = sorted(proc_dir.glob(f"stimuli_{cat}_*.json"))
    stimuli_by_idx: dict[int, dict] = {}
    if files:
        with open(files[-1]) as f:
            for it in json.load(f):
                idx = it.get("item_idx", -1)
                if idx >= 0:
                    stimuli_by_idx[idx] = it

    # Context condition labels
    y_context = []
    for m in meta:
        idx = m.get("item_idx", -1)
        stim = stimuli_by_idx.get(idx, {})
        cond = stim.get("context_condition", m.get("context_condition", ""))
        y_context.append(str(cond))
    y_context = np.array(y_context)

    # Answer position labels (which letter is stereotyped)
    y_position = []
    for m in meta:
        idx = m.get("item_idx", -1)
        stim = stimuli_by_idx.get(idx, {})
        answer_roles = stim.get("answer_roles", {})
        stereo_letter = next(
            (l for l, r in answer_roles.items() if r == "stereotyped_target"), ""
        )
        y_position.append(stereo_letter)
    y_position = np.array(y_position)

    results: dict[str, Any] = {
        "category": cat,
        "layers": [],
        "context_acc": [],
        "position_acc": [],
    }

    # Filter for valid labels
    ctx_valid = np.array([len(y) > 0 for y in y_context])
    pos_valid = np.array([len(y) > 0 for y in y_position])

    for layer in layers:
        X_all = _get_raw_hs(data, layer)

        ctx_acc = 0.0
        if ctx_valid.sum() >= 10:
            ctx_acc = train_probe(X_all[ctx_valid], y_context[ctx_valid])

        pos_acc = 0.0
        if pos_valid.sum() >= 10:
            pos_acc = train_probe(X_all[pos_valid], y_position[pos_valid])

        results["layers"].append(layer)
        results["context_acc"].append(round(ctx_acc, 4))
        results["position_acc"].append(round(pos_acc, 4))

        log(f"    Layer {layer:2d}: context={ctx_acc:.3f} position={pos_acc:.3f}")

    return results


# ---------------------------------------------------------------------------
# Control C: Cross-category generalization
# ---------------------------------------------------------------------------


def run_cross_category_generalization(
    all_data: dict[str, dict[str, Any]],
    categories: list[str],
) -> dict[str, Any]:
    """Cross-category binary probe: train stereotyped/not on one cat, test on another."""
    # Find best layer per category (peak from permutation results, or use layer 14)
    best_layer: dict[str, int] = {}
    for cat in categories:
        data = all_data.get(cat, {})
        n_layers = data.get("n_layers", 32)
        best_layer[cat] = min(14, n_layers - 1)  # safe default

    matrix: dict[str, dict[str, float]] = {}

    for train_cat in categories:
        train_data = all_data.get(train_cat, {})
        if not train_data:
            continue
        layer = best_layer[train_cat]
        X_train = _get_raw_hs(train_data, layer)
        y_train = np.array([
            int(m.get("is_stereotyped_response", False))
            for m in train_data["meta"]
        ])

        if len(set(y_train)) < 2:
            continue

        matrix[train_cat] = {}
        for test_cat in categories:
            test_data = all_data.get(test_cat, {})
            if not test_data:
                continue

            if train_cat == test_cat:
                # Within-category: use CV
                acc = train_probe(X_train, y_train)
            else:
                # Cross-category: train/test split
                X_test = _get_raw_hs(test_data, layer)
                y_test = np.array([
                    int(m.get("is_stereotyped_response", False))
                    for m in test_data["meta"]
                ])
                if len(set(y_test)) < 2:
                    acc = 0.5
                else:
                    acc = train_probe_transfer(X_train, y_train, X_test, y_test)

            matrix[train_cat][test_cat] = round(acc, 4)
            log(f"    {train_cat} -> {test_cat}: {acc:.3f}")

    return matrix


def run_within_category_generalization(
    all_data: dict[str, dict[str, Any]],
    categories: list[str],
) -> dict[str, dict[str, dict[str, float]]]:
    """Within-category: train on subgroup A's items, test on subgroup B."""
    results: dict[str, dict[str, dict[str, float]]] = {}

    proc_dir = PROJECT_ROOT / "data" / "processed"

    for cat in categories:
        data = all_data.get(cat, {})
        if not data:
            continue

        meta = data["meta"]
        n_layers = data.get("n_layers", 32)
        layer = min(14, n_layers - 1)

        # Get subgroup labels from stimuli
        files = sorted(proc_dir.glob(f"stimuli_{cat}_*.json"))
        stimuli_by_idx: dict[int, dict] = {}
        if files:
            with open(files[-1]) as f:
                for it in json.load(f):
                    idx = it.get("item_idx", -1)
                    if idx >= 0:
                        stimuli_by_idx[idx] = it

        # Assign primary subgroup to each item
        subgroups: list[str] = []
        for m in meta:
            idx = m.get("item_idx", -1)
            stim = stimuli_by_idx.get(idx, {})
            groups = stim.get("stereotyped_groups", [])
            subgroups.append(groups[0] if groups else "")
        subgroups_arr = np.array(subgroups)

        unique_subs = sorted(set(s for s in subgroups if s))
        if len(unique_subs) < 2:
            log(f"    {cat}: only {len(unique_subs)} unique subgroup(s), skipping within-cat generalization")
            continue

        X = _get_raw_hs(data, layer)
        y = np.array([int(m.get("is_stereotyped_response", False)) for m in meta])

        cat_matrix: dict[str, dict[str, float]] = {}
        for train_sub in unique_subs:
            cat_matrix[train_sub] = {}
            train_mask = subgroups_arr == train_sub
            X_train = X[train_mask]
            y_train = y[train_mask]

            if len(set(y_train)) < 2 or train_mask.sum() < 10:
                continue

            for test_sub in unique_subs:
                if train_sub == test_sub:
                    acc = train_probe(X_train, y_train)
                else:
                    test_mask = subgroups_arr == test_sub
                    X_test = X[test_mask]
                    y_test = y[test_mask]
                    if len(set(y_test)) < 2 or test_mask.sum() < 5:
                        acc = 0.5
                    else:
                        acc = train_probe_transfer(X_train, y_train, X_test, y_test)

                cat_matrix[train_sub][test_sub] = round(acc, 4)

            log(f"    {cat}/{train_sub}: " + ", ".join(
                f"{t}={v:.2f}" for t, v in cat_matrix[train_sub].items()))

        results[cat] = cat_matrix

    return results


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_probe_selectivity(
    perm_results: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    """Probe accuracy vs layer with permutation baseline."""
    cats = sorted(perm_results.keys())
    if not cats:
        return

    n = len(cats)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for idx, cat in enumerate(cats):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        data = perm_results[cat]

        layers = data.get("layers", [])
        real = data.get("real_acc", [])
        perm_mean = data.get("perm_mean", [])
        perm_std = data.get("perm_std", [])

        if not layers:
            ax.set_visible(False)
            continue

        layers_arr = np.array(layers)
        real_arr = np.array(real)
        perm_arr = np.array(perm_mean)
        perm_std_arr = np.array(perm_std)

        ax.plot(layers_arr, real_arr, "o-", color=BLUE, label="Real labels", markersize=4)
        ax.plot(layers_arr, perm_arr, "--", color=GRAY, label="Permuted", linewidth=1)
        ax.fill_between(layers_arr, perm_arr - perm_std_arr, perm_arr + perm_std_arr,
                        color=GRAY, alpha=0.2)
        ax.fill_between(layers_arr, perm_arr, real_arr,
                        where=real_arr > perm_arr, color="#D55E00", alpha=0.15)

        # Annotate peak selectivity
        sel = data.get("selectivity", [])
        if sel:
            peak_idx = int(np.argmax(sel))
            peak_layer = layers[peak_idx]
            peak_val = sel[peak_idx]
            ax.annotate(f"Peak: L{peak_layer}\nsel={peak_val:.2f}",
                        xy=(peak_layer, real[peak_idx]),
                        fontsize=7, ha="center",
                        xytext=(0, 10), textcoords="offset points")

        ax.set_xlabel("Layer", fontsize=8)
        ax.set_ylabel("Accuracy", fontsize=8)
        cat_label = CATEGORY_LABELS.get(cat, cat)
        ax.set_title(cat_label, fontsize=9)
        if idx == 0:
            ax.legend(fontsize=7)

        ax.text(0.02, 0.95, chr(65 + idx), transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="top")

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle("Probe selectivity: real vs permuted labels", fontsize=11)
    _save_both(fig, output_dir / "fig_probe_selectivity.png")
    log("    Saved fig_probe_selectivity")


def fig_structural_comparison(
    perm_results: dict[str, dict[str, Any]],
    struct_results: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    """Identity probe vs structural control probes."""
    cats = sorted(set(perm_results.keys()) & set(struct_results.keys()))
    if not cats:
        return

    n = len(cats)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for idx, cat in enumerate(cats):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        perm = perm_results[cat]
        struct = struct_results[cat]

        layers = perm.get("layers", [])
        real = perm.get("real_acc", [])
        ctx = struct.get("context_acc", [])
        pos = struct.get("position_acc", [])

        if not layers:
            ax.set_visible(False)
            continue

        ax.plot(layers, real, "o-", color=BLUE, label="Identity subgroup", markersize=4)
        ax.plot(struct.get("layers", layers), ctx, "s--", color=ORANGE,
                label="Context condition", markersize=3)
        ax.plot(struct.get("layers", layers), pos, "^:", color=GREEN,
                label="Answer position", markersize=3)

        ax.set_xlabel("Layer", fontsize=8)
        ax.set_ylabel("Accuracy", fontsize=8)
        cat_label = CATEGORY_LABELS.get(cat, cat)
        ax.set_title(cat_label, fontsize=9)
        if idx == 0:
            ax.legend(fontsize=7)

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle("Identity vs structural control probes", fontsize=11)
    _save_both(fig, output_dir / "fig_probe_structural_comparison.png")
    log("    Saved fig_probe_structural_comparison")


def fig_generalization_matrix(
    gen_matrix: dict[str, dict[str, float]],
    output_dir: Path,
) -> None:
    """Cross-category generalization heatmap."""
    cats = sorted(gen_matrix.keys())
    if len(cats) < 2:
        return

    mat = np.zeros((len(cats), len(cats)))
    for i, tc in enumerate(cats):
        for j, te in enumerate(cats):
            mat[i, j] = gen_matrix.get(tc, {}).get(te, 0.5)

    fig, ax = plt.subplots(figsize=(max(5, len(cats) * 1.2), max(4, len(cats) * 1.0)))
    im = ax.imshow(mat, cmap="Blues", vmin=0.4, vmax=1.0, aspect="auto")

    labels = [CATEGORY_LABELS.get(c, c) for c in cats]
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Test category", fontsize=9)
    ax.set_ylabel("Train category", fontsize=9)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            color = "white" if mat[i, j] > 0.75 else "black"
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

    fig.colorbar(im, ax=ax, label="Accuracy", shrink=0.8)
    ax.set_title("Cross-category binary probe generalization", fontsize=10)

    _save_both(fig, output_dir / "fig_probe_generalization_matrix.png")
    log("    Saved fig_probe_generalization_matrix")


def fig_within_category_gen(
    within_gen: dict[str, dict[str, dict[str, float]]],
    output_dir: Path,
) -> None:
    """Within-category cross-subgroup generalization heatmaps."""
    cats = [c for c in sorted(within_gen.keys()) if len(within_gen[c]) >= 2]
    if not cats:
        return

    n = len(cats)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)

    for idx, cat in enumerate(cats):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        data = within_gen[cat]
        subs = sorted(data.keys())
        n_subs = len(subs)

        mat = np.full((n_subs, n_subs), 0.5)
        for i, train_sub in enumerate(subs):
            for j, test_sub in enumerate(subs):
                mat[i, j] = data.get(train_sub, {}).get(test_sub, 0.5)

        im = ax.imshow(mat, cmap="Blues", vmin=0.4, vmax=1.0, aspect="auto")
        ax.set_xticks(range(n_subs))
        ax.set_xticklabels(subs, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(n_subs))
        ax.set_yticklabels(subs, fontsize=7)

        for i in range(n_subs):
            for j in range(n_subs):
                color = "white" if mat[i, j] > 0.75 else "black"
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                        fontsize=6, color=color)

        cat_label = CATEGORY_LABELS.get(cat, cat)
        ax.set_title(f"Within-category -- {cat_label}", fontsize=9)
        ax.set_xlabel("Test subgroup", fontsize=8)
        ax.set_ylabel("Train subgroup", fontsize=8)

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle("Within-category cross-subgroup generalization", fontsize=11)
    _save_both(fig, output_dir / "fig_within_category_generalization.png")
    log("    Saved fig_within_category_generalization")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    t0 = time.time()
    args = parse_args()
    loc_dir = Path(args.localization_dir)

    categories = [c.strip() for c in args.categories.split(",")]

    output_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "results" / "probe_controls" / "llama-3.1-8b"
    )
    ensure_dir(output_dir)
    fig_dir = ensure_dir(output_dir / "figures")
    log(f"Output: {output_dir}")

    # Load all category data
    all_data: dict[str, dict[str, Any]] = {}
    for cat in categories:
        log(f"Loading {cat} ...")
        data = load_category_data(loc_dir, cat)
        if data:
            all_data[cat] = data
            log(f"  {data['n_items']} items, {data['n_layers']} layers")
        else:
            log(f"  WARNING: no data for {cat}")

    if not all_data:
        log("ERROR: no data loaded")
        sys.exit(1)

    # Determine layers to probe
    n_layers = max(d["n_layers"] for d in all_data.values())
    layers = list(range(0, n_layers, args.layer_stride))
    log(f"Probing {len(layers)} layers (stride={args.layer_stride})")

    # Control A: Permutation baseline
    log("\n--- Control A: Permutation baseline ---")
    perm_results: dict[str, dict[str, Any]] = {}
    for cat in categories:
        if cat not in all_data:
            continue
        log(f"\n  {cat}:")
        perm_results[cat] = run_permutation_control(
            all_data[cat], cat, layers, args.n_permutations,
        )
    atomic_save_json(perm_results, output_dir / "permutation_results.json")

    # Control B: Structural controls
    log("\n--- Control B: Structural control tasks ---")
    struct_results: dict[str, dict[str, Any]] = {}
    for cat in categories:
        if cat not in all_data:
            continue
        log(f"\n  {cat}:")
        struct_results[cat] = run_structural_control(all_data[cat], cat, layers)
    atomic_save_json(struct_results, output_dir / "structural_control_results.json")

    # Control C: Cross-category generalization
    log("\n--- Control C: Cross-category generalization ---")
    gen_matrix = run_cross_category_generalization(all_data, categories)
    atomic_save_json(gen_matrix, output_dir / "generalization_matrix.json")

    # Within-category cross-subgroup generalization
    log("\n--- Within-category cross-subgroup generalization ---")
    within_gen = run_within_category_generalization(all_data, categories)
    atomic_save_json(within_gen, output_dir / "within_category_generalization.json")

    # Figures
    if not args.skip_figures:
        log("\nGenerating figures ...")
        fig_probe_selectivity(perm_results, fig_dir)
        fig_structural_comparison(perm_results, struct_results, fig_dir)
        if gen_matrix:
            fig_generalization_matrix(gen_matrix, fig_dir)
        if within_gen:
            fig_within_category_gen(within_gen, fig_dir)

    total = time.time() - t0
    log(f"\nComplete in {total:.1f}s")
    log(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
