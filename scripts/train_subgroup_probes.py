#!/usr/bin/env python3
"""Train per-subgroup linear probes at each attention head.

Probes S1 (subgroup identity), S2 (per-subgroup stereotyping), S3 (family).

Usage:
    python scripts/train_subgroup_probes.py \
        --run_dir results/runs/llama2-13b-hf/2026-04-11/ \
        --n_heads 40 --head_dim 128

    # Chat model (run separately, compare later)
    python scripts/train_subgroup_probes.py \
        --run_dir results/runs/llama2-13b-chat-hf/2026-04-11/ \
        --n_heads 40 --head_dim 128
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.analysis.directions import load_activations_indexed
from src.analysis.probes import collect_head_features, train_probe_cv, build_subgroup_labels
from src.data.bbq_loader import CATEGORY_MAP, parse_categories
from src.utils.io import atomic_save_json, ensure_dir
from src.utils.logging import log

ALL_CATS = list(CATEGORY_MAP.keys())
MIN_ITEMS_PROBE = 20


def _load_cat_data(
    run_dir: Path, cat: str, max_items: int | None,
) -> tuple[list[np.ndarray], list[dict], list[dict]] | None:
    act_dir = run_dir / "activations" / cat
    if not act_dir.exists():
        return None
    stim_files = sorted((run_dir / "stimuli").glob(f"stimuli_{cat}_*.json"))
    if not stim_files:
        stim_files = sorted(Path("data/processed").glob(f"stimuli_{cat}_*.json"))
    if not stim_files:
        return None
    with open(stim_files[-1]) as f:
        stimuli = json.load(f)
    finals_by, _ids, metas_by = load_activations_indexed(
        act_dir, max_items=max_items, final_key="attn_pre_o_proj_final",
    )
    aligned_f, aligned_m, aligned_s = [], [], []
    for item in stimuli[: (max_items or len(stimuli))]:
        idx = int(item["item_idx"])
        if idx in finals_by and idx in metas_by:
            aligned_f.append(finals_by[idx])
            aligned_m.append(metas_by[idx])
            aligned_s.append(item)
    return (aligned_f, aligned_m, aligned_s) if aligned_f else None


def _build_per_subgroup_stereo_masks(
    stimuli: list[dict], metas: list[dict],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Build per-subgroup stereotyping masks and labels.

    Returns dict[subgroup -> (mask, y)] where mask is bool over items and y is binary.
    """
    sg_data: dict[str, tuple[list[int], list[int]]] = {}
    for i, (item, meta) in enumerate(zip(stimuli, metas)):
        if item.get("context_condition") != "ambig":
            continue
        groups = item.get("stereotyped_groups", [])
        if not groups:
            continue
        sg = groups[0].lower()

        logits = {}
        for letter in ["A", "B", "C"]:
            key = f"logit_{letter}"
            if key in meta:
                logits[letter] = meta[key]
        pred_letter = meta.get("predicted_letter", "")
        if not pred_letter and logits:
            pred_letter = max(logits, key=logits.get)
        pred_role = item.get("answer_roles", {}).get(str(pred_letter).upper(), "unknown")
        if pred_role == "unknown":
            continue

        sg_data.setdefault(sg, ([], []))
        sg_data[sg][0].append(i)
        sg_data[sg][1].append(1 if pred_role == "stereotyped_target" else 0)

    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sg, (indices, labels) in sg_data.items():
        if len(indices) < MIN_ITEMS_PROBE:
            continue
        mask = np.zeros(len(stimuli), dtype=bool)
        mask[indices] = True
        result[sg] = (mask, np.array(labels, dtype=np.int64))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train per-subgroup probes.")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--categories", type=str, default="all")
    parser.add_argument("--n_heads", type=int, required=True)
    parser.add_argument("--head_dim", type=int, required=True)
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--model_id", type=str, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    analysis_dir = ensure_dir(run_dir / "analysis")
    model_id = args.model_id or run_dir.parent.name
    categories = parse_categories(args.categories)

    log(f"Training subgroup probes for {model_id}")
    log(f"Heads: {args.n_heads}, head_dim: {args.head_dim}")

    results: dict[str, dict] = {"model_id": model_id}

    for cat in categories:
        log(f"\n{'='*60}")
        log(f"Category: {cat}")
        log(f"{'='*60}")

        data = _load_cat_data(run_dir, cat, args.max_items)
        if data is None:
            log(f"  No data, skipping")
            continue
        finals, metas, stimuli = data
        n_items = len(finals)
        n_layers = finals[0].shape[0]
        log(f"  {n_items} items, {n_layers} layers")

        cat_results: dict[str, dict] = {}

        # Probe S1: Subgroup identity classification
        log("  --- Probe S1: Subgroup identity ---")
        groups_present = [item["stereotyped_groups"][0].lower() for item in stimuli
                          if item.get("stereotyped_groups")]
        unique_groups = sorted(set(groups_present))
        if len(unique_groups) >= 2:
            mask_s1, y_s1, le_s1 = build_subgroup_labels(stimuli)
            # Run at 3 representative layers + best search
            best_acc = 0.0
            best_layer = 0
            for layer in [n_layers // 5, n_layers // 2, int(n_layers * 0.8)]:
                for head in range(min(args.n_heads, 5)):  # Quick scan
                    s1_finals = [f for f, m in zip(finals, mask_s1) if m]
                    X = collect_head_features(s1_finals, layer, head, args.head_dim)
                    r = train_probe_cv(X, y_s1)
                    if r["mean_accuracy"] > best_acc:
                        best_acc = r["mean_accuracy"]
                        best_layer = layer

            log(f"  Best S1 layer: {best_layer} (acc={best_acc:.3f})")
            log(f"  Classes: {list(le_s1.classes_)}")

            # Full head scan at best layer
            s1_accs = np.zeros(args.n_heads, dtype=np.float32)
            for head in range(args.n_heads):
                s1_finals = [f for f, m in zip(finals, mask_s1) if m]
                X = collect_head_features(s1_finals, best_layer, head, args.head_dim)
                r = train_probe_cv(X, y_s1)
                s1_accs[head] = r["mean_accuracy"]

            cat_results["probe_s1"] = {
                "best_layer": best_layer,
                "best_accuracy": best_acc,
                "classes": list(le_s1.classes_),
                "head_accuracies": s1_accs.tolist(),
            }
        else:
            log(f"  <2 subgroups, skipping S1")

        # Probe S2: Per-subgroup stereotyping
        log("  --- Probe S2: Per-subgroup stereotyping ---")
        sg_masks = _build_per_subgroup_stereo_masks(stimuli, metas)
        s2_results: dict[str, dict] = {}

        for sg_name, (mask, y_sg) in sorted(sg_masks.items()):
            n_sg = int(mask.sum())
            sg_finals = [finals[i] for i, m in enumerate(mask) if m]

            best_acc = 0.0
            best_layer = n_layers // 2
            for layer in [n_layers // 5, n_layers // 2, int(n_layers * 0.8)]:
                for head in range(min(args.n_heads, 5)):
                    X = collect_head_features(sg_finals, layer, head, args.head_dim)
                    r = train_probe_cv(X, y_sg)
                    if r["mean_accuracy"] > best_acc:
                        best_acc = r["mean_accuracy"]
                        best_layer = layer

            # Full head scan at best layer
            head_accs = np.zeros(args.n_heads, dtype=np.float32)
            for head in range(args.n_heads):
                X = collect_head_features(sg_finals, best_layer, head, args.head_dim)
                r = train_probe_cv(X, y_sg)
                head_accs[head] = r["mean_accuracy"]

            s2_results[sg_name] = {
                "n_items": n_sg,
                "stereo_rate": float(y_sg.mean()),
                "best_layer": best_layer,
                "best_accuracy": best_acc,
                "head_accuracies": head_accs.tolist(),
            }
            log(f"    {sg_name}: n={n_sg}, best_acc={best_acc:.3f} @ L{best_layer}")

        cat_results["probe_s2"] = s2_results

        # Probe S3: Family classification (if fragmentation data exists)
        frag_path = analysis_dir / "subgroup_fragmentation.json"
        if frag_path.exists():
            with open(frag_path) as f:
                frag_data = json.load(f)
            cat_frag = frag_data.get("categories", {}).get(cat, {})
            families = cat_frag.get("families", {})
            if len(families) >= 2:
                log("  --- Probe S3: Family classification ---")
                sg_to_fam = {}
                for fid, members in families.items():
                    for m in members:
                        sg_to_fam[m] = int(fid)

                fam_labels = []
                fam_mask = np.zeros(n_items, dtype=bool)
                for i, item in enumerate(stimuli):
                    groups = item.get("stereotyped_groups", [])
                    if groups:
                        sg = groups[0].lower()
                        if sg in sg_to_fam:
                            fam_labels.append(sg_to_fam[sg])
                            fam_mask[i] = True

                if len(set(fam_labels)) >= 2 and sum(fam_mask) >= MIN_ITEMS_PROBE:
                    y_fam = np.array(fam_labels, dtype=np.int64)
                    fam_finals = [finals[i] for i, m in enumerate(fam_mask) if m]

                    best_acc = 0.0
                    best_layer = n_layers // 2
                    for layer in range(0, n_layers, max(n_layers // 10, 1)):
                        for head in range(min(args.n_heads, 5)):
                            X = collect_head_features(fam_finals, layer, head, args.head_dim)
                            r = train_probe_cv(X, y_fam)
                            if r["mean_accuracy"] > best_acc:
                                best_acc = r["mean_accuracy"]
                                best_layer = layer

                    head_accs = np.zeros(args.n_heads, dtype=np.float32)
                    for head in range(args.n_heads):
                        X = collect_head_features(fam_finals, best_layer, head, args.head_dim)
                        r = train_probe_cv(X, y_fam)
                        head_accs[head] = r["mean_accuracy"]

                    cat_results["probe_s3"] = {
                        "best_layer": best_layer,
                        "best_accuracy": best_acc,
                        "families": families,
                        "head_accuracies": head_accs.tolist(),
                    }
                    log(f"  Family probe: best_acc={best_acc:.3f} @ L{best_layer}")

        results[cat] = cat_results

    # Save
    out_path = analysis_dir / "subgroup_probes.json"
    atomic_save_json(results, out_path)
    log(f"\nResults -> {out_path}")
    log("Done!")


if __name__ == "__main__":
    main()
