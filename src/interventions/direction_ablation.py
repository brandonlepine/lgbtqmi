"""Direction ablation: project out identity directions from hidden states.

Robustness note:
Different transformer families can return different forward-hook output tuple
structures for decoder layers. When `hidden_dim` is provided, we locate the
(batch, seq, hidden_dim) hidden-state tensor within the output rather than
assuming it is `output[0]`.
"""

from typing import Any, Optional

import numpy as np
import torch

from src.utils.logging import log


def _locate_hidden_in_output(
    output: Any,
    *,
    hidden_dim: int | None,
) -> tuple[torch.Tensor, int | None]:
    """Return (hidden, idx) where idx is the tuple index, or None if output is a Tensor."""
    if isinstance(output, torch.Tensor):
        return output, None

    if isinstance(output, tuple):
        if hidden_dim is None:
            if not output or not isinstance(output[0], torch.Tensor):
                raise ValueError("Expected output[0] tensor for tuple output")
            return output[0], 0

        for i, x in enumerate(output):
            if isinstance(x, torch.Tensor) and x.dim() == 3 and x.shape[-1] == hidden_dim:
                return x, i
        raise ValueError(
            "Could not locate hidden state tensor in output tuple "
            f"(hidden_dim={hidden_dim})."
        )

    raise ValueError(f"Unexpected hook output type: {type(output)}")


def _replace_tuple_index(output: tuple[Any, ...], idx: int, value: Any) -> tuple[Any, ...]:
    out = list(output)
    out[idx] = value
    return tuple(out)


def make_direction_ablation_hook(
    direction: np.ndarray,
    device: str,
    *,
    hidden_dim: int | None = None,
    alpha: float = 1.0,
) -> Any:
    """Create a forward hook that projects out a direction from the hidden state.

    Args:
        direction: (hidden_dim,) unit-normalized direction to ablate
        device: torch device string
        hidden_dim: if provided, used to locate the hidden-state tensor in tuple outputs
        alpha: scaling for projection removal. alpha=1.0 is standard ablation; alpha<0 amplifies.

    Returns:
        hook_fn suitable for register_forward_hook
    """
    d_base = torch.tensor(direction, dtype=torch.float32).to(device)
    d_base = d_base / d_base.norm().clamp(min=1e-8)
    d_cache: dict[torch.dtype, torch.Tensor] = {}

    def hook_fn(module: Any, args: Any, output: Any) -> Any:
        hidden, idx = _locate_hidden_in_output(output, hidden_dim=hidden_dim)
        if not isinstance(hidden, torch.Tensor):
            return output

        # Match the direction dtype to the hidden state dtype (e.g., fp16 on MPS/CUDA)
        dt = hidden.dtype
        if dt not in d_cache:
            d_cache[dt] = d_base.to(dtype=dt)
        d = d_cache[dt]

        # Project out the direction: h_new = h - alpha * (h . d) * d
        proj = (hidden @ d).unsqueeze(-1) * d.unsqueeze(0).unsqueeze(0)
        modified = hidden - float(alpha) * proj

        if idx is None:
            return modified
        return _replace_tuple_index(output, idx, modified)

    return hook_fn


def apply_direction_ablation(
    model: Any,
    get_layer_fn: Any,
    direction: np.ndarray,
    layer_indices: list[int],
    device: str,
    *,
    hidden_dim: int | None = None,
    alpha: float = 1.0,
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
        hook = layer.register_forward_hook(
            make_direction_ablation_hook(d, device, hidden_dim=hidden_dim, alpha=alpha)
        )
        hooks.append(hook)

    log(f"  Registered direction ablation on {len(layer_indices)} layers")
    return hooks


def apply_multi_direction_ablation(
    model: Any,
    get_layer_fn: Any,
    directions_per_layer: dict[int, list[np.ndarray]],
    device: str,
    *,
    hidden_dim: int | None = None,
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

        d_bases = []
        for d_np in basis:
            t = torch.tensor(d_np, dtype=torch.float32).to(device)
            t = t / t.norm().clamp(min=1e-8)
            d_bases.append(t)

        def make_multi_hook(d_list):
            d_cache_multi: dict[torch.dtype, list[torch.Tensor]] = {}
            def hook_fn(module, args, output):
                hidden, idx = _locate_hidden_in_output(output, hidden_dim=hidden_dim)
                if not isinstance(hidden, torch.Tensor):
                    return output
                dt = hidden.dtype
                if dt not in d_cache_multi:
                    d_cache_multi[dt] = [dd.to(dtype=dt) for dd in d_list]
                d_use = d_cache_multi[dt]

                h = hidden
                for d in d_use:
                    proj = (h @ d).unsqueeze(-1) * d.unsqueeze(0).unsqueeze(0)
                    h = h - proj

                if idx is None:
                    return h
                return _replace_tuple_index(output, idx, h)
            return hook_fn

        hook = layer.register_forward_hook(make_multi_hook(d_bases))
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
