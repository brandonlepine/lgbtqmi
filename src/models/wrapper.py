"""Architecture-agnostic model wrapper.

This wrapper exists to avoid hardcoding internal module paths and to provide
validated, portable hook registration across decoder-only transformer families.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch

from src.utils.logging import log


def _resolve_attr_path(obj: Any, attr_path: str) -> Any | None:
    cur = obj
    for attr in attr_path.split("."):
        cur = getattr(cur, attr, None)
        if cur is None:
            return None
    return cur


def _pick_first_attr(obj: Any, names: list[str]) -> Any | None:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _select_dtype_for_device(device: str) -> torch.dtype:
    # float16 on CPU is often unsupported/slow and can error; keep CPU float32.
    if device == "cpu":
        return torch.float32
    return torch.float16


def _extract_hidden_candidate(output: Any, hidden_dim: int) -> torch.Tensor:
    """Find (batch, seq, hidden_dim) hidden state tensor in a module output."""
    if isinstance(output, torch.Tensor):
        candidate = output
        if candidate.dim() == 3 and candidate.shape[-1] == hidden_dim:
            return candidate
        raise ValueError(f"Unexpected tensor output shape: {tuple(candidate.shape)}")

    if isinstance(output, tuple):
        for x in output:
            if isinstance(x, torch.Tensor) and x.dim() == 3 and x.shape[-1] == hidden_dim:
                return x
        # Common case: output[0] exists but isn't hidden states; surface a useful error
        if output and isinstance(output[0], torch.Tensor):
            raise ValueError(
                f"Could not locate hidden state in output tuple; "
                f"output[0] shape={tuple(output[0].shape)} hidden_dim={hidden_dim}"
            )
        raise ValueError("Could not locate hidden state in output tuple")

    raise ValueError(f"Unexpected output type: {type(output)}")


@dataclass
class _HookCounter:
    name: str
    count: int = 0


class ModelWrapper:
    """Thin wrapper around a HF causal LM + tokenizer."""

    def __init__(self, model: Any, tokenizer: Any, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        cfg = getattr(model, "config", None)
        if cfg is None:
            raise ValueError("Model has no config; cannot infer dimensions")

        self.n_layers: int = int(cfg.num_hidden_layers)
        self.hidden_dim: int = int(cfg.hidden_size)
        self.n_heads: int | None = int(getattr(cfg, "num_attention_heads", 0)) or None
        self.head_dim: int | None = (
            (self.hidden_dim // self.n_heads) if self.n_heads else None
        )

        layers = self._resolve_layers()
        if layers is None:
            raise RuntimeError(
                f"Cannot find decoder layers for {type(model).__name__}. "
                f"Add support in src/models/wrapper.py."
            )
        self._layers = layers

        self._hook_counters: dict[int, _HookCounter] = {}

    @classmethod
    def from_pretrained(cls, model_path: str, device: str = "cpu") -> "ModelWrapper":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = _select_dtype_for_device(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype).to(device)
        model.eval()

        wrapper = cls(model=model, tokenizer=tokenizer, device=device)
        log(
            f"ModelWrapper loaded {type(model).__name__}: "
            f"layers={wrapper.n_layers} hidden_dim={wrapper.hidden_dim} "
            f"heads={wrapper.n_heads} head_dim={wrapper.head_dim}"
        )
        return wrapper

    def _resolve_layers(self) -> Any | None:
        # Most decoder-only models fall into one of these.
        candidates = [
            "model.layers",  # Llama/Mistral/Qwen/Gemma-style (via AutoModelForCausalLM)
            "transformer.h",  # GPT-2 style
            "gpt_neox.layers",  # GPT-NeoX style
            "model.decoder.layers",  # some decoder wrappers
        ]
        for path in candidates:
            layers = _resolve_attr_path(self.model, path)
            if layers is not None:
                log(f"ModelWrapper: decoder layers at model.{path}")
                return layers
        return None

    def get_layer(self, layer_idx: int) -> Any:
        return self._layers[layer_idx]

    def get_attn_module(self, layer_idx: int) -> Any:
        layer = self.get_layer(layer_idx)
        attn = _pick_first_attr(layer, ["self_attn", "attn", "attention"])
        if attn is None:
            raise RuntimeError(
                f"Cannot find attention module on layer {layer_idx} ({type(layer).__name__})"
            )
        return attn

    def get_o_proj(self, layer_idx: int) -> Any:
        attn = self.get_attn_module(layer_idx)
        o_proj = _pick_first_attr(attn, ["o_proj", "out_proj", "c_proj"])
        if o_proj is None:
            raise RuntimeError(
                f"Cannot find attention output projection on layer {layer_idx} "
                f"(attn={type(attn).__name__})"
            )
        return o_proj

    def _new_counter(self, name: str) -> _HookCounter:
        counter = _HookCounter(name=name, count=0)
        self._hook_counters[id(counter)] = counter
        return counter

    def register_residual_hook(
        self,
        layer_idx: int,
        hook_fn: Callable[[Any, Any, Any], Any],
        *,
        name: str = "residual_hook",
    ) -> Any:
        """Register a forward hook on the decoder layer output (residual stream)."""
        layer = self.get_layer(layer_idx)
        counter = self._new_counter(f"{name}[layer={layer_idx}]")

        def wrapped(module: Any, args: Any, output: Any) -> Any:
            counter.count += 1
            # Validate we can locate the hidden state (architecture safety)
            _ = _extract_hidden_candidate(output, self.hidden_dim)
            out2 = hook_fn(module, args, output)
            return output if out2 is None else out2

        handle = layer.register_forward_hook(wrapped)
        return (handle, counter)

    def register_o_proj_pre_hook(
        self,
        layer_idx: int,
        pre_hook_fn: Callable[[Any, tuple[Any, ...]], tuple[Any, ...] | None],
        *,
        name: str = "o_proj_pre_hook",
    ) -> Any:
        """Register a forward_pre_hook on the attention o_proj module."""
        o_proj = self.get_o_proj(layer_idx)
        counter = self._new_counter(f"{name}[layer={layer_idx}]")

        def wrapped(module: Any, args: tuple[Any, ...]) -> tuple[Any, ...] | None:
            counter.count += 1
            return pre_hook_fn(module, args)

        handle = o_proj.register_forward_pre_hook(wrapped)
        return (handle, counter)

    def register_head_ablation_hook(
        self,
        layer_idx: int,
        heads: list[int],
        *,
        name: str = "head_ablation",
    ) -> Any:
        """Zero specific attention heads by slicing the pre-o_proj activations."""
        if self.head_dim is None or self.n_heads is None:
            raise RuntimeError("head_dim/n_heads unknown; cannot register head ablation")

        head_dim = self.head_dim

        def pre_hook(_module: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
            if not args or not isinstance(args[0], torch.Tensor):
                raise ValueError("Unexpected o_proj pre-hook args")
            x = args[0]
            if x.dim() != 3 or x.shape[-1] != self.hidden_dim:
                raise ValueError(f"Unexpected o_proj input shape: {tuple(x.shape)}")
            x = x.clone()
            for h in heads:
                start = h * head_dim
                end = start + head_dim
                x[:, :, start:end] = 0.0
            return (x,) + args[1:]

        return self.register_o_proj_pre_hook(layer_idx, pre_hook, name=name)

    def validate_hooks(self, prompt: str = "test") -> dict[str, int]:
        """Run a single forward pass and report hook invocation counts."""
        before = {c.name: c.count for c in self._hook_counters.values()}
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            _ = self.model(**inputs)
        after = {c.name: c.count for c in self._hook_counters.values()}
        deltas = {name: after[name] - before.get(name, 0) for name in after}
        return deltas

