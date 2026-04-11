"""Head ablation: zero out specific attention heads.

Methodologically correct implementation zeros head slices on the attention
output *before* the output projection (pre-o_proj), not on the layer residual.
"""

from typing import Any

import numpy as np
import torch

from src.utils.logging import log


def make_head_ablation_pre_hook(
    head_indices: list[int],
    head_dim: int,
) -> Any:
    """Create a forward_pre_hook that zeros specific attention head slices.

    Args:
        head_indices: which heads to ablate (0-indexed)
        head_dim: dimension per head

    Returns:
        hook_fn suitable for register_forward_pre_hook on o_proj
    """
    def pre_hook_fn(module: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
        if not args or not isinstance(args[0], torch.Tensor):
            raise ValueError("Unexpected o_proj pre-hook args")
        x = args[0]
        if x.dim() != 3:
            raise ValueError(f"Expected (batch, seq, dim) tensor, got {tuple(x.shape)}")
        x = x.clone()
        for head_idx in head_indices:
            start = head_idx * head_dim
            end = start + head_dim
            x[:, :, start:end] = 0.0
        return (x,) + args[1:]

    return pre_hook_fn


def apply_head_ablation(
    get_o_proj_fn: Any,
    targets: dict[int, list[int]],
    head_dim: int,
) -> list:
    """Register head ablation hooks on specified layers (pre-o_proj).

    Args:
        get_o_proj_fn: callable(layer_idx) -> o_proj module
        targets: dict[layer_idx -> list of head indices to ablate]
        head_dim: dimension per head

    Returns:
        List of hook handles
    """
    hooks = []
    total_heads = 0
    for layer_idx, head_indices in targets.items():
        o_proj = get_o_proj_fn(layer_idx)
        hook = o_proj.register_forward_pre_hook(
            make_head_ablation_pre_hook(head_indices, head_dim)
        )
        hooks.append(hook)
        total_heads += len(head_indices)

    log(f"  Registered head ablation: {total_heads} heads across {len(targets)} layers")
    return hooks


def identify_rlhf_targets(
    base_probe_matrix: np.ndarray,
    chat_probe_matrix: np.ndarray,
    top_k: int = 20,
    min_base_acc: float = 0.6,
) -> dict[int, list[int]]:
    """Identify attention heads where RLHF suppressed stereotyping signal.

    Finds heads where base model has high stereotyping probe accuracy
    but chat model has low accuracy (the "RLHF-suppressed" heads).

    Args:
        base_probe_matrix: (n_layers, n_heads) base model Probe B accuracies
        chat_probe_matrix: (n_layers, n_heads) chat model Probe B accuracies
        top_k: number of top heads to return
        min_base_acc: minimum base accuracy to consider

    Returns:
        dict[layer_idx -> list of head indices]
    """
    diff = base_probe_matrix - chat_probe_matrix
    n_layers, n_heads = diff.shape

    # Filter: base must have meaningful accuracy
    mask = base_probe_matrix >= min_base_acc
    masked_diff = np.where(mask, diff, -np.inf)

    # Get top-k heads by difference
    flat_indices = np.argsort(masked_diff.flatten())[-top_k:]

    targets: dict[int, list[int]] = {}
    for flat_idx in flat_indices:
        layer = flat_idx // n_heads
        head = flat_idx % n_heads
        if masked_diff[layer, head] > 0:
            if layer not in targets:
                targets[layer] = []
            targets[layer].append(head)

    # Sort heads within each layer
    for layer in targets:
        targets[layer].sort()

    log(f"  Identified {sum(len(v) for v in targets.values())} RLHF-target heads "
        f"across {len(targets)} layers")
    return targets


def remove_hooks(hooks: list) -> None:
    """Remove all registered hooks."""
    for h in hooks:
        h.remove()
