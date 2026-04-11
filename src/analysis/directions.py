"""Direction computation utilities: within-item deltas and category-level directions."""

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.utils.logging import ProgressLogger, log


def _parse_metadata_json_field(meta_raw: Any) -> dict:
    """Parse metadata_json saved in .npz (may be numpy scalar/0-d array/bytes)."""
    if isinstance(meta_raw, np.ndarray):
        if meta_raw.shape == ():
            meta_raw = meta_raw.item()
        elif meta_raw.size == 1:
            meta_raw = meta_raw.reshape(()).item()
        else:
            meta_raw = meta_raw.tolist()
    if isinstance(meta_raw, (bytes, bytearray)):
        meta_raw = meta_raw.decode("utf-8", errors="replace")
    return json.loads(str(meta_raw))


def load_activations(
    activations_dir: str | Path,
    max_items: Optional[int] = None,
    *,
    final_key: str = "hidden_final",
    identity_key: str = "hidden_identity",
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict]]:
    """Load all item .npz files from an activations directory.

    Returns:
        hidden_finals: list of arrays, each (n_layers, hidden_dim)
        hidden_identities: list of arrays, each (n_layers, n_pos, hidden_dim)
        metadatas: list of parsed metadata dicts
    """
    act_dir = Path(activations_dir)
    npz_files = sorted(act_dir.glob("item_*.npz"))
    if max_items is not None:
        npz_files = npz_files[:max_items]

    hidden_finals: list[np.ndarray] = []
    hidden_identities: list[np.ndarray] = []
    metadatas: list[dict] = []

    for f in npz_files:
        data = np.load(f, allow_pickle=True)
        if final_key not in data.files:
            # Backward-compat: fall back to hidden_final
            final_arr = data["hidden_final"]
        else:
            final_arr = data[final_key]
        if identity_key not in data.files:
            ident_arr = data["hidden_identity"]
        else:
            ident_arr = data[identity_key]

        hidden_finals.append(final_arr)
        hidden_identities.append(ident_arr)
        meta = _parse_metadata_json_field(data["metadata_json"])
        metadatas.append(meta)

    return hidden_finals, hidden_identities, metadatas


def load_activations_indexed(
    activations_dir: str | Path,
    max_items: Optional[int] = None,
    *,
    final_key: str = "hidden_final",
    identity_key: str = "hidden_identity",
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, dict]]:
    """Load activations keyed by item_idx for robust alignment."""
    finals, identities, metas = load_activations(
        activations_dir,
        max_items=max_items,
        final_key=final_key,
        identity_key=identity_key,
    )
    finals_by_idx: dict[int, np.ndarray] = {}
    id_by_idx: dict[int, np.ndarray] = {}
    meta_by_idx: dict[int, dict] = {}
    for f, i, m in zip(finals, identities, metas):
        idx = int(m.get("item_idx", -1))
        if idx < 0:
            continue
        finals_by_idx[idx] = f
        id_by_idx[idx] = i
        meta_by_idx[idx] = m
    return finals_by_idx, id_by_idx, meta_by_idx


def compute_item_delta(
    hidden_identity: np.ndarray,
    metadata: dict,
    stimuli_item: dict,
) -> Optional[np.ndarray]:
    """Compute within-item identity delta from identity-position hidden states.

    For items with identity token positions tagged to stereotyped vs contrast groups,
    computes h_stereotyped - h_contrast, normalized by mean activation norm.

    Args:
        hidden_identity: (n_layers, n_positions, hidden_dim)
        metadata: extraction metadata with identity_positions dict
        stimuli_item: processed stimuli item with identity_role_tags, answer_roles

    Returns:
        delta: (n_layers, hidden_dim) or None if positions couldn't be separated
    """
    n_layers, n_pos, hidden_dim = hidden_identity.shape
    if n_pos == 0:
        return None

    # Preferred: answer-span based split (robust across categories)
    all_positions = metadata.get("all_identity_token_positions", [])
    stereo_positions = metadata.get("stereotyped_token_positions", [])
    non_positions = metadata.get("non_stereotyped_token_positions", [])
    if not all_positions or not stereo_positions or not non_positions:
        return None

    pos_to_idx = {int(p): i for i, p in enumerate(all_positions)}
    stereo_pos_indices = [pos_to_idx[p] for p in stereo_positions if p in pos_to_idx]
    contrast_pos_indices = [pos_to_idx[p] for p in non_positions if p in pos_to_idx]

    if not stereo_pos_indices or not contrast_pos_indices:
        return None

    # Mean hidden state per group, per layer
    h_stereo = hidden_identity[:, stereo_pos_indices, :].mean(axis=1)  # (n_layers, dim)
    h_contrast = hidden_identity[:, contrast_pos_indices, :].mean(axis=1)

    # Normalize by mean activation norm per layer
    norm_s = np.linalg.norm(h_stereo, axis=1, keepdims=True)  # (n_layers, 1)
    norm_c = np.linalg.norm(h_contrast, axis=1, keepdims=True)
    mean_norm = (norm_s + norm_c) / 2.0
    mean_norm = np.maximum(mean_norm, 1e-8)  # avoid division by zero

    delta = (h_stereo - h_contrast) / mean_norm
    return delta.astype(np.float32)


def compute_category_direction(
    deltas: list[np.ndarray],
) -> np.ndarray:
    """Compute category-level direction from per-item deltas.

    Takes the mean delta across items, then unit-normalizes per layer.

    Args:
        deltas: list of (n_layers, hidden_dim) arrays

    Returns:
        direction: (n_layers, hidden_dim), unit-normalized per layer
    """
    stacked = np.stack(deltas, axis=0)  # (n_items, n_layers, dim)
    mean_delta = stacked.mean(axis=0)  # (n_layers, dim)
    norms = np.linalg.norm(mean_delta, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return (mean_delta / norms).astype(np.float32)


def compute_subgroup_directions(
    deltas: list[np.ndarray],
    metadatas: list[dict],
    stimuli_items: list[dict],
) -> dict[str, np.ndarray]:
    """Compute per-subgroup directions within a category.

    Groups items by their primary stereotyped group and computes a
    direction for each sub-group.

    Returns:
        dict mapping sub-group name -> (n_layers, hidden_dim) direction
    """
    group_deltas: dict[str, list[np.ndarray]] = {}

    for delta, meta, item in zip(deltas, metadatas, stimuli_items):
        if delta is None:
            continue
        groups = item.get("stereotyped_groups", [])
        if not groups:
            continue
        # Use the first listed stereotyped group as the primary group
        primary = groups[0].lower()
        if primary not in group_deltas:
            group_deltas[primary] = []
        group_deltas[primary].append(delta)

    directions: dict[str, np.ndarray] = {}
    for group, group_d in group_deltas.items():
        if len(group_d) >= 5:  # need minimum samples
            directions[group] = compute_category_direction(group_d)
            log(f"    Sub-group '{group}': {len(group_d)} items")
        else:
            log(f"    Sub-group '{group}': {len(group_d)} items (too few, skipped)")

    return directions


def compute_gender_decomposition(
    subgroup_dirs: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Compute gender decomposition for Sexual Orientation directions.

    gender_dir = (gay_dir - lesbian_dir) / 2
    orientation_dir = (gay_dir + lesbian_dir) / 2

    Also computes family directions:
    GL = mean(gay, lesbian)
    BP = mean(bisexual, pansexual)  [if available]

    Returns dict with computed directions, unit-normalized per layer.
    """
    results: dict[str, np.ndarray] = {}

    if "gay" in subgroup_dirs and "lesbian" in subgroup_dirs:
        gay = subgroup_dirs["gay"]
        lesbian = subgroup_dirs["lesbian"]

        gender_raw = (gay - lesbian) / 2.0
        orientation_raw = (gay + lesbian) / 2.0

        # Unit-normalize per layer
        for name, raw in [("gender", gender_raw), ("orientation", orientation_raw)]:
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            results[name] = (raw / norms).astype(np.float32)

        # GL family direction
        results["gl_family"] = compute_category_direction([gay, lesbian])
        log("    Computed gender decomposition: gender, orientation, GL family")

    if "bisexual" in subgroup_dirs and "pansexual" in subgroup_dirs:
        bp_raw = (subgroup_dirs["bisexual"] + subgroup_dirs["pansexual"]) / 2.0
        norms = np.linalg.norm(bp_raw, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        results["bp_family"] = (bp_raw / norms).astype(np.float32)
        log("    Computed BP family direction")

    return results
