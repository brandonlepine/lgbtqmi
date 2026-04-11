"""Direction ablation: project out identity directions from hidden states."""

from typing import Any, Optional

import numpy as np
import torch

from src.utils.logging import log


def make_direction_ablation_hook(
    direction: np.ndarray,
    device: str,
) -> Any:
    """Create a forward hook that projects out a direction from the hidden state.

    Args:
        direction: (hidden_dim,) unit-normalized direction to ablate
        device: torch device string

    Returns:
        hook_fn suitable for register_forward_hook
    """
    d = torch.tensor(direction, dtype=torch.float32).to(device)
    d = d / d.norm().clamp(min=1e-8)

    def hook_fn(module: Any, args: Any, output: Any) -> Any:
        if isinstance(output, tuple):
            hidden = output[0]
            # Project out the direction: h = h - (h . d) * d
            proj = (hidden @ d).unsqueeze(-1) * d.unsqueeze(0).unsqueeze(0)
            modified = hidden - proj
            return (modified,) + output[1:]
        else:
            proj = (output @ d).unsqueeze(-1) * d.unsqueeze(0).unsqueeze(0)
            return output - proj

    return hook_fn


def apply_direction_ablation(
    model: Any,
    get_layer_fn: Any,
    direction: np.ndarray,
    layer_indices: list[int],
    device: str,
) -> list:
    """Register direction ablation hooks on specified layers.

    Args:
        model: the model (for reference)
        get_layer_fn: callable(layer_idx) -> layer module
        direction: (hidden_dim,) direction to ablate (will be normalized)
        layer_indices: which layers to apply ablation on
        device: torch device

    Returns:
        List of hook handles (call .remove() to undo)
    """
    # Normalize direction
    d = direction / max(np.linalg.norm(direction), 1e-8)

    hooks = []
    for layer_idx in layer_indices:
        layer = get_layer_fn(layer_idx)
        hook = layer.register_forward_hook(make_direction_ablation_hook(d, device))
        hooks.append(hook)

    log(f"  Registered direction ablation on {len(layer_indices)} layers")
    return hooks


def apply_multi_direction_ablation(
    model: Any,
    get_layer_fn: Any,
    directions_per_layer: dict[int, list[np.ndarray]],
    device: str,
) -> list:
    """Register ablation hooks that project out multiple directions per layer.

    Args:
        directions_per_layer: dict[layer_idx -> list of (hidden_dim,) directions]

    Returns:
        List of hook handles
    """
    hooks = []
    for layer_idx, directions in directions_per_layer.items():
        # Stack and orthogonalize via Gram-Schmidt
        basis = _gram_schmidt(directions)
        layer = get_layer_fn(layer_idx)

        d_tensors = [
            torch.tensor(d, dtype=torch.float32).to(device)
            for d in basis
        ]

        def make_multi_hook(d_list):
            def hook_fn(module, args, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                    for d in d_list:
                        proj = (hidden @ d).unsqueeze(-1) * d.unsqueeze(0).unsqueeze(0)
                        hidden = hidden - proj
                    return (hidden,) + output[1:]
                else:
                    h = output
                    for d in d_list:
                        proj = (h @ d).unsqueeze(-1) * d.unsqueeze(0).unsqueeze(0)
                        h = h - proj
                    return h
            return hook_fn

        hook = layer.register_forward_hook(make_multi_hook(d_tensors))
        hooks.append(hook)

    log(f"  Registered multi-direction ablation on {len(directions_per_layer)} layers")
    return hooks


def _gram_schmidt(vectors: list[np.ndarray]) -> list[np.ndarray]:
    """Orthogonalize a list of vectors via Gram-Schmidt."""
    basis: list[np.ndarray] = []
    for v in vectors:
        w = v.copy().astype(np.float64)
        for b in basis:
            w -= np.dot(w, b) * b
        norm = np.linalg.norm(w)
        if norm > 1e-8:
            basis.append((w / norm).astype(np.float32))
    return basis


def remove_hooks(hooks: list) -> None:
    """Remove all registered hooks."""
    for h in hooks:
        h.remove()
