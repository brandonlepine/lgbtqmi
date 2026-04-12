#!/usr/bin/env python3
"""Train linear probes at each attention head to predict identity and stereotyping.

Generates figures 09-13: probe heatmaps, scatter plots, difference maps, confusion matrix.

Usage:
    # Single model
    python scripts/train_head_probes.py \
        --run_dir results/runs/llama2-13b/2026-04-10 \
        --n_heads 40 --head_dim 128

    # Base vs chat comparison
    python scripts/train_head_probes.py \
        --base_run_dir results/runs/llama2-13b/2026-04-10 \
        --chat_run_dir results/runs/llama2-13b-chat/2026-04-10 \
        --n_heads 40 --head_dim 128
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.metrics import confusion_matrix

from src.analysis.directions import load_activations_indexed
from src.analysis.probes import (
    build_identity_labels,
    build_stereotyping_labels,
    build_subgroup_labels,
    collect_head_features,
    run_head_probes,
    train_probe_cv,
)
from src.data.bbq_loader import CATEGORY_MAP, parse_categories
from src.utils.io import atomic_save_json, ensure_dir
from src.utils.logging import log
from src.visualization.heatmaps import (
    plot_probe_difference_heatmap,
    plot_probe_heatmap,
)
from src.visualization.scatter import (
    plot_confusion_matrix,
    plot_identity_vs_stereotyping,
    plot_identity_vs_stereotyping_dual,
)


def load_multi_category(
    run_dir: Path,
    categories: list[str],
    max_items: int | None = None,
) -> tuple[list[np.ndarray], list[dict], list[dict]]:
    """Load activations and stimuli across multiple categories.

    Returns combined hidden_finals, metadatas, and stimuli_items with
    a 'category' field set on each stimuli item.
    """
    all_finals: list[np.ndarray] = []
    all_metas: list[dict] = []
    all_stimuli: list[dict] = []

    for cat in categories:
        act_dir = run_dir / "activations" / cat
        if not act_dir.exists():
            log(f"  WARNING: {act_dir} not found, skipping {cat}")
            continue

        # Find stimuli
        stimuli_files = sorted((run_dir / "stimuli").glob(f"stimuli_{cat}_*.json"))
        if not stimuli_files:
            stimuli_files = sorted(Path("data/processed").glob(f"stimuli_{cat}_*.json"))
        if not stimuli_files:
            log(f"  WARNING: No stimuli for {cat}, skipping")
            continue

        with open(stimuli_files[-1]) as f:
            stimuli = json.load(f)

        finals_by_idx, _ids_by_idx, metas_by_idx = load_activations_indexed(
            act_dir,
            max_items=max_items,
            final_key="attn_pre_o_proj_final",
        )

        n_loaded = 0
        for item in stimuli[: (max_items or len(stimuli))]:
            idx = int(item["item_idx"])
            if idx in finals_by_idx and idx in metas_by_idx:
                item["_category_short"] = cat
                all_finals.append(finals_by_idx[idx])
                all_metas.append(metas_by_idx[idx])
                all_stimuli.append(item)
                n_loaded += 1

        log(f"  Loaded {n_loaded} aligned items from {cat}")

    return all_finals, all_metas, all_stimuli


def run_probes_for_model(
    run_dir: Path,
    categories: list[str],
    n_heads: int,
    head_dim: int,
    max_items: int | None = None,
    probe_type: str = "ridge",
) -> dict:
    """Run all three probes for one model.

    Returns dict with identity_matrix, stereo_matrix, subgroup results, etc.
    """
    finals, metas, stimuli = load_multi_category(run_dir, categories, max_items)
    n_items = len(finals)
    if n_items == 0:
        log("ERROR: No items loaded")
        return {}

    n_layers = finals[0].shape[0]
    log(f"  Total items: {n_items}, layers: {n_layers}, heads: {n_heads}")

    results: dict = {
        "n_items": n_items,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "head_dim": head_dim,
    }

    # Probe A: Identity category
    log("\n  === Probe A: Identity category classification ===")
    y_identity, le_identity = build_identity_labels(stimuli, "_category_short")
    log(f"  Classes: {list(le_identity.classes_)}")
    identity_matrix = run_head_probes(
        finals, y_identity, None, n_layers, n_heads, head_dim,
        probe_type=probe_type,
    )
    results["identity_matrix"] = identity_matrix
    results["identity_classes"] = list(le_identity.classes_)
    best_layer, best_head = np.unravel_index(identity_matrix.argmax(), identity_matrix.shape)
    log(f"  Best head: L{best_layer}H{best_head} = {identity_matrix.max():.3f}")

    # Probe B: Stereotyping prediction
    log("\n  === Probe B: Stereotyping prediction ===")
    mask_stereo, y_stereo = build_stereotyping_labels(stimuli, metas)
    n_stereo = mask_stereo.sum()
    log(f"  Stereotyping items: {n_stereo} ({y_stereo.mean():.2%} stereotyped)")

    if n_stereo >= 20:
        stereo_matrix = run_head_probes(
            finals, y_stereo, mask_stereo, n_layers, n_heads, head_dim,
            probe_type=probe_type,
        )
        results["stereo_matrix"] = stereo_matrix
        best_layer, best_head = np.unravel_index(stereo_matrix.argmax(), stereo_matrix.shape)
        log(f"  Best head: L{best_layer}H{best_head} = {stereo_matrix.max():.3f}")
    else:
        log(f"  WARNING: Too few stereotyping items ({n_stereo}), skipping Probe B")
        stereo_matrix = np.full((n_layers, n_heads), 0.5, dtype=np.float32)
        results["stereo_matrix"] = stereo_matrix

    # Probe C: Within-category subgroup classification (per-category)
    log("\n  === Probe C: Within-category subgroup classification (all categories) ===")
    # This is intentionally a *layer* probe (not head probe) to stay fast and avoid
    # an O(n_categories * n_layers * n_heads) sweep.
    MIN_SUBGROUP_ITEMS = 15
    subgroup_layer_probes: dict[str, dict] = {}

    # Imports are local to keep startup lean.
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder

    def _safe_pca_k(n_samples: int, n_features: int, desired: int, n_splits: int) -> int:
        # PCA runs inside CV folds; ensure k <= min(n_train, n_features) for all folds.
        test_max = int(np.ceil(n_samples / max(n_splits, 2)))
        n_train_min = max(n_samples - test_max, 1)
        return int(max(1, min(desired, n_train_min, n_features)))

    cats_in_data = sorted({s.get("_category_short", "") for s in stimuli if s.get("_category_short")})
    for cat in cats_in_data:
        cat_stimuli = [s for s in stimuli if s.get("_category_short") == cat]
        cat_indices = [i for i, s in enumerate(stimuli) if s.get("_category_short") == cat]
        if len(cat_stimuli) < 20:
            log(f"  {cat}: not enough items ({len(cat_stimuli)}), skipping")
            continue

        cat_finals = [finals[i] for i in cat_indices]

        raw = []
        for item in cat_stimuli:
            groups = item.get("stereotyped_groups", [])
            raw.append(groups[0].lower() if groups else "")

        counts = Counter([r for r in raw if r])
        eligible = sorted([g for g, n in counts.items() if n >= MIN_SUBGROUP_ITEMS])
        if len(eligible) < 2:
            log(
                f"  {cat}: only {len(eligible)} subgroup(s) with ≥{MIN_SUBGROUP_ITEMS} items; skipping"
            )
            continue

        finals_sg: list[np.ndarray] = []
        labels_sg: list[str] = []
        for item, f, r in zip(cat_stimuli, cat_finals, raw):
            if r and r in eligible:
                finals_sg.append(f)
                labels_sg.append(r)

        le = LabelEncoder()
        y = le.fit_transform(labels_sg)

        binc = np.bincount(y, minlength=len(le.classes_))
        min_per_class = int(binc.min()) if len(binc) else 0
        n_splits = int(min(5, max(2, min_per_class))) if min_per_class > 0 else 2
        if n_splits < 2:
            log(f"  {cat}: insufficient per-class counts for CV; skipping")
            continue
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Find best layer (PCA inside folds to avoid leakage)
        best_acc = 0.0
        best_layer = 0
        for layer in range(n_layers):
            X_raw = np.stack([f[layer] for f in finals_sg], axis=0)
            desired = 50
            k = _safe_pca_k(X_raw.shape[0], X_raw.shape[1], desired, n_splits)
            pipe = Pipeline([("pca", PCA(n_components=k)), ("clf", RidgeClassifier(alpha=1.0))])
            acc = float(np.mean(cross_val_score(pipe, X_raw, y, cv=skf)))
            if acc > best_acc:
                best_acc = acc
                best_layer = layer

        # Confusion at best layer
        X_best = np.stack([f[best_layer] for f in finals_sg], axis=0)
        desired = 50
        k = _safe_pca_k(X_best.shape[0], X_best.shape[1], desired, n_splits)
        pipe = Pipeline([("pca", PCA(n_components=k)), ("clf", RidgeClassifier(alpha=1.0))])
        y_pred = cross_val_predict(pipe, X_best, y, cv=skf)
        conf = confusion_matrix(y, y_pred, labels=list(range(len(le.classes_))))

        log(f"  {cat}: subgroups={list(le.classes_)}")
        log(f"  {cat}: best layer={best_layer} (acc={best_acc:.3f}), n={len(finals_sg)}")

        subgroup_layer_probes[cat] = {
            "n_items": int(len(finals_sg)),
            "min_items_per_subgroup": int(MIN_SUBGROUP_ITEMS),
            "classes": list(le.classes_),
            "best_layer": int(best_layer),
            "best_accuracy": float(best_acc),
            "confusion": conf,
        }

        # Back-compat keys for SO (previous behavior)
        if cat == "so":
            results["subgroup_best_layer"] = int(best_layer)
            results["subgroup_best_accuracy"] = float(best_acc)
            results["subgroup_confusion"] = conf
            results["subgroup_classes_list"] = list(le.classes_)

    results["subgroup_layer_probes"] = subgroup_layer_probes

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train head-level probes.")
    parser.add_argument("--run_dir", type=str, default=None, help="Single model run dir")
    parser.add_argument("--base_run_dir", type=str, default=None, help="Base model run dir")
    parser.add_argument("--chat_run_dir", type=str, default=None, help="Chat model run dir")
    parser.add_argument("--categories", type=str, default="all")
    parser.add_argument("--n_heads", type=int, required=True, help="Number of attention heads")
    parser.add_argument("--head_dim", type=int, required=True, help="Dimension per head")
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--probe_type", type=str, default="ridge", choices=["ridge", "logistic"])
    parser.add_argument("--model_id", type=str, default=None)
    args = parser.parse_args()

    categories = parse_categories(args.categories)

    # Determine mode: single model or base vs chat
    if args.base_run_dir and args.chat_run_dir:
        mode = "comparison"
        base_dir = Path(args.base_run_dir)
        chat_dir = Path(args.chat_run_dir)
        fig_dir = ensure_dir(base_dir / "figures")
        analysis_dir = ensure_dir(base_dir / "analysis")
    elif args.run_dir:
        mode = "single"
        base_dir = Path(args.run_dir)
        chat_dir = None
        fig_dir = ensure_dir(base_dir / "figures")
        analysis_dir = ensure_dir(base_dir / "analysis")
    else:
        parser.error("Provide --run_dir or both --base_run_dir and --chat_run_dir")
        return

    model_id = args.model_id or base_dir.parent.name

    log(f"Mode: {mode}")
    log(f"Categories: {categories}")

    # Run probes for base model
    log(f"\n{'='*60}")
    log(f"BASE MODEL: {base_dir}")
    log(f"{'='*60}")
    base_results = run_probes_for_model(
        base_dir, categories, args.n_heads, args.head_dim,
        max_items=args.max_items, probe_type=args.probe_type,
    )

    if not base_results:
        log("ERROR: No results from base model")
        return

    # Fig 09: Identity probe heatmap
    log("\n--- Fig 09: Identity probe heatmap ---")
    plot_probe_heatmap(
        base_results["identity_matrix"],
        path=str(fig_dir / "fig_09_identity_probe_heatmap.png"),
        title=f"Identity encoding ({model_id} base)",
    )
    log(f"  Saved fig_09")

    # Fig 10: Stereotyping probe heatmap
    log("\n--- Fig 10: Stereotyping probe heatmap ---")
    plot_probe_heatmap(
        base_results["stereo_matrix"],
        path=str(fig_dir / "fig_10_stereotyping_probe_heatmap.png"),
        title=f"Stereotyping encoding ({model_id} base)",
    )
    log(f"  Saved fig_10")

    chat_results = None
    if mode == "comparison":
        log(f"\n{'='*60}")
        log(f"CHAT MODEL: {chat_dir}")
        log(f"{'='*60}")
        chat_results = run_probes_for_model(
            chat_dir, categories, args.n_heads, args.head_dim,
            max_items=args.max_items, probe_type=args.probe_type,
        )

    if chat_results:
        # Fig 11: Identity vs stereotyping (dual)
        log("\n--- Fig 11: Identity vs stereotyping scatter ---")
        plot_identity_vs_stereotyping_dual(
            base_results["identity_matrix"], base_results["stereo_matrix"],
            chat_results["identity_matrix"], chat_results["stereo_matrix"],
            path=str(fig_dir / "fig_11_identity_vs_stereotyping_scatter.png"),
            suptitle=f"Identity vs stereotyping ({model_id})",
        )
        log(f"  Saved fig_11")

        # Fig 12: Probe difference heatmap
        log("\n--- Fig 12: Probe accuracy difference ---")
        plot_probe_difference_heatmap(
            base_results["stereo_matrix"], chat_results["stereo_matrix"],
            path=str(fig_dir / "fig_12_probe_accuracy_difference_heatmap.png"),
            title=f"RLHF localization: stereotyping probe (base − chat, {model_id})",
        )
        log(f"  Saved fig_12")
    else:
        # Single-model scatter
        log("\n--- Fig 11: Identity vs stereotyping scatter ---")
        plot_identity_vs_stereotyping(
            base_results["identity_matrix"], base_results["stereo_matrix"],
            path=str(fig_dir / "fig_11_identity_vs_stereotyping_scatter.png"),
            title=f"Identity vs stereotyping ({model_id})",
        )
        log(f"  Saved fig_11")

    # Fig 13: Sub-group confusion matrix (per-category)
    subgroup_probes = base_results.get("subgroup_layer_probes", {}) or {}
    if subgroup_probes:
        log("\n--- Fig 13: Sub-group probe confusion (per-category) ---")
        for cat, info in sorted(subgroup_probes.items()):
            conf = info["confusion"]
            classes = info["classes"]
            best_layer = info["best_layer"]
            fname = f"fig_13_subgroup_probe_confusion_{cat}.png"
            plot_confusion_matrix(
                conf,
                classes,
                path=str(fig_dir / fname),
                title=f"{cat} subgroup confusion (Layer {best_layer}, {model_id})",
            )
            log(f"  Saved {fname}")

        # Keep legacy filename for SO if present (downstream scripts may expect it).
        if "so" in subgroup_probes:
            info = subgroup_probes["so"]
            plot_confusion_matrix(
                info["confusion"],
                info["classes"],
                path=str(fig_dir / "fig_13_subgroup_probe_confusion.png"),
                title=f"SO sub-group confusion (Layer {info['best_layer']}, {model_id})",
            )
            log("  Saved fig_13_subgroup_probe_confusion.png")

    # Save probe results
    save_results = {
        "model_id": model_id,
        "mode": mode,
        "n_items": base_results["n_items"],
        "n_layers": base_results["n_layers"],
        "n_heads": base_results["n_heads"],
        "head_dim": base_results["head_dim"],
        "base_identity_max": float(base_results["identity_matrix"].max()),
        "base_stereo_max": float(base_results["stereo_matrix"].max()),
    }
    if subgroup_probes:
        save_results["subgroup_layer_probes"] = {
            cat: {
                "n_items": int(info["n_items"]),
                "classes": list(info["classes"]),
                "best_layer": int(info["best_layer"]),
                "best_accuracy": float(info["best_accuracy"]),
                "min_items_per_subgroup": int(info["min_items_per_subgroup"]),
            }
            for cat, info in sorted(subgroup_probes.items())
        }
    if "subgroup_best_accuracy" in base_results:
        # Legacy SO-only summary fields
        save_results["subgroup_best_layer"] = base_results["subgroup_best_layer"]
        save_results["subgroup_best_accuracy"] = base_results["subgroup_best_accuracy"]
    if chat_results:
        save_results["chat_identity_max"] = float(chat_results["identity_matrix"].max())
        save_results["chat_stereo_max"] = float(chat_results["stereo_matrix"].max())

    atomic_save_json(save_results, analysis_dir / "probe_results.json")

    # Save matrices as npz for downstream use
    save_arrays = {
        "base_identity": base_results["identity_matrix"],
        "base_stereo": base_results["stereo_matrix"],
    }
    if chat_results:
        save_arrays["chat_identity"] = chat_results["identity_matrix"]
        save_arrays["chat_stereo"] = chat_results["stereo_matrix"]

    np.savez(analysis_dir / "probe_matrices.npz", **save_arrays)
    log(f"\nResults saved to {analysis_dir}")
    log("Done!")


if __name__ == "__main__":
    main()
