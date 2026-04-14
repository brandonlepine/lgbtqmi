"""SAE steering engine: apply feature-direction interventions during forward passes.

Constructs steering vectors from SAE decoder columns and modifies the residual
stream at a target layer via hooks.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional

import numpy as np
import torch

from src.utils.logging import log

try:
    import pandas as pd
except ImportError:
    pd = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_pandas():
    global pd
    if pd is not None:
        return
    try:
        import pandas as _pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required for SAE steering (parquet I/O + analysis). "
            "Install with: pip install pandas pyarrow"
        ) from exc
    pd = _pd


def _locate_hidden_in_output(output: Any, hidden_dim: int) -> tuple[torch.Tensor, int | None]:
    """Locate the (batch, seq, hidden_dim) residual tensor in a layer hook output.

    Returns (hidden, tuple_index). tuple_index is None when output is a Tensor.
    """
    if isinstance(output, torch.Tensor):
        if output.dim() != 3 or output.shape[-1] != hidden_dim:
            raise ValueError(f"Unexpected hidden tensor shape: {tuple(output.shape)} (expected (*,*,{hidden_dim}))")
        return output, None
    if isinstance(output, tuple):
        for i, x in enumerate(output):
            if isinstance(x, torch.Tensor) and x.dim() == 3 and x.shape[-1] == hidden_dim:
                return x, i
        raise ValueError("Could not locate hidden state tensor in hook output tuple")
    raise ValueError(f"Unexpected hook output type: {type(output)}")


def _replace_tuple_index(t: tuple[Any, ...], idx: int, value: Any) -> tuple[Any, ...]:
    lst = list(t)
    lst[idx] = value
    return tuple(lst)


DAMPEN_ALPHAS = [-50, -40, -30, -20, -15, -10, -5, -2, -1]
AMPLIFY_ALPHAS = [1, 2, 5, 10, 15, 20, 30, 40, 50]


# ---------------------------------------------------------------------------
# Feature selection helpers
# ---------------------------------------------------------------------------


def load_significant_features(
    analysis_dir: "Path", layer: int, granularity: str = "per_category",
) -> "pd.DataFrame":
    """Load significant features from Stage-2 parquet files."""
    _require_pandas()
    from pathlib import Path
    path = Path(analysis_dir) / "features" / f"{granularity}_layer_{layer}.parquet"
    if not path.exists():
        log(f"  WARNING: feature file not found: {path}")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    return df[df["is_significant"] == True].copy()  # noqa: E712


def get_feature_set(
    df: "pd.DataFrame",
    category: str | None = None,
    subcategory: str | None = None,
    direction: str = "pro_bias",
) -> list[int]:
    """Get feature indices for a given category/subcategory/direction."""
    mask = df["direction"] == direction
    if category is not None:
        mask = mask & (df["category"] == category)
    if subcategory is not None:
        mask = mask & (df["subcategory"] == subcategory)
    return df.loc[mask, "feature_idx"].tolist()


# ---------------------------------------------------------------------------
# SAESteerer
# ---------------------------------------------------------------------------


class SAESteerer:
    """Core engine for SAE feature steering experiments.

    Parameters
    ----------
    wrapper : ModelWrapper
        Loaded model wrapper with hook registration.
    sae : SAEWrapper
        Loaded SAE for the target layer.
    layer : int
        Which transformer layer to intervene at.
    """

    def __init__(self, wrapper: Any, sae: Any, layer: int) -> None:
        self.wrapper = wrapper
        self.sae = sae
        self.layer = layer
        self._active_hooks: list[Any] = []  # handles to remove
        self._validated: bool = False

    # ------------------------------------------------------------------
    # Steering vector construction
    # ------------------------------------------------------------------

    def get_steering_vector(
        self,
        feature_indices: list[int],
        alphas: list[float],
    ) -> torch.Tensor:
        """Build a composite steering vector.

        steering_vec = sum(alpha_j * d_j)  for (j, alpha_j) in zip(indices, alphas)

        Returns shape ``(hidden_dim,)`` on the model device.
        """
        if len(feature_indices) != len(alphas):
            raise ValueError("feature_indices and alphas must have the same length")

        vec = torch.zeros(self.sae.hidden_dim, dtype=torch.float32)
        for fidx, alpha in zip(feature_indices, alphas):
            direction = torch.from_numpy(
                self.sae.get_feature_direction(fidx)
            )
            vec += alpha * direction

        return vec.to(self.wrapper.model.dtype).to(self.wrapper.device)

    def get_composite_steering(
        self,
        feature_indices: list[int],
        alpha: float,
        scale_by_sqrt_n: bool = True,
    ) -> torch.Tensor:
        """Build composite steering: same alpha for all features, optionally scaled by 1/sqrt(N).

        Parameters
        ----------
        feature_indices : list[int]
        alpha : float
            Base coefficient (negative to dampen, positive to amplify).
        scale_by_sqrt_n : bool
            If True, multiply alpha by 1/sqrt(N) to normalise perturbation magnitude.

        .. note:: Magnitude convention — this function applies alpha/sqrt(n) per
           feature then sums, so total perturbation magnitude scales as
           ~alpha * sqrt(k) for k features.  This differs from
           ``build_subgroup_steering_vector()`` in ``subgroup_steering.py``,
           which takes the mean of unit directions then multiplies by alpha
           (magnitude ~alpha/sqrt(k)).  Same alpha value produces different
           perturbation magnitudes by factor of k.  Do NOT compare alpha values
           across these two conventions.
        """
        n = len(feature_indices)
        if n == 0:
            return torch.zeros(
                self.sae.hidden_dim,
                dtype=self.wrapper.model.dtype,
                device=self.wrapper.device,
            )
        effective_alpha = alpha / math.sqrt(n) if scale_by_sqrt_n else alpha
        alphas = [effective_alpha] * n
        return self.get_steering_vector(feature_indices, alphas)

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def _make_steering_hook(
        self,
        steering_vec: torch.Tensor,
        *,
        scope: str = "last",
    ) -> Callable:
        """Return a hook function that adds steering_vec to the residual stream.

        scope:
          - "last": only last token position (good for BBQ letter selection)
          - "all": all token positions (required for sequence log-prob scoring)
        """
        if scope not in ("last", "all"):
            raise ValueError(f"Unknown steering hook scope: {scope!r}")
        def hook_fn(module: Any, args: Any, output: Any) -> Any:
            hidden, tup_idx = _locate_hidden_in_output(output, self.wrapper.hidden_dim)

            seq_len = hidden.shape[1]
            # Modify last token position (clone to avoid mutating shared buffers)
            hidden2 = hidden.clone()
            vec = steering_vec.to(hidden.dtype)
            if scope == "all":
                hidden2 = hidden2 + vec.view(1, 1, -1)
            else:
                hidden2[:, seq_len - 1, :] = hidden2[:, seq_len - 1, :] + vec

            if tup_idx is None:
                return hidden2
            return _replace_tuple_index(output, tup_idx, hidden2)

        return hook_fn

    def _install_hook(self, steering_vec: torch.Tensor, *, scope: str = "last") -> None:
        """Register the steering hook on self.layer."""
        hook_fn = self._make_steering_hook(steering_vec, scope=scope)
        handle, counter = self.wrapper.register_residual_hook(
            self.layer, hook_fn, name="sae_steering"
        )
        self._active_hooks.append(handle)
        if not self._validated:
            deltas = self.wrapper.validate_hooks(prompt="test")
            if deltas.get(counter.name, 0) <= 0:
                raise RuntimeError(
                    f"Steering hook did not fire during validation (counter={counter.name}). "
                    "This usually indicates an architecture mismatch in hook output structure."
                )
            self._validated = True

    def clear_hooks(self) -> None:
        """Remove all active steering hooks."""
        for h in self._active_hooks:
            h.remove()
        self._active_hooks.clear()

    # ------------------------------------------------------------------
    # Forward pass with steering
    # ------------------------------------------------------------------

    def steer_and_evaluate(
        self,
        prompt: str,
        steering_vec: torch.Tensor,
        *,
        letters: tuple[str, ...] = ("A", "B", "C"),
    ) -> dict[str, Any]:
        """Run forward pass with steering and extract the model's answer.

        Returns dict with model_answer, answer_logits, top5_tokens,
        degenerated, degenerated_soft.
        """
        from src.utils.answers import best_choice_from_logits

        self.clear_hooks()
        self._install_hook(steering_vec, scope="last")

        tokenizer = self.wrapper.tokenizer
        inputs = tokenizer(prompt, return_tensors="pt").to(self.wrapper.device)
        seq_len = inputs["input_ids"].shape[1]
        last_pos = seq_len - 1

        with torch.no_grad():
            outputs = self.wrapper.model(**inputs)

        self.clear_hooks()

        logits = outputs.logits[0, last_pos, :]
        best_letter, letter_logits = best_choice_from_logits(logits, tokenizer, letters=letters)

        # Top-5 tokens
        top5_vals, top5_idx = torch.topk(logits, 5)
        top5_tokens = [
            (tokenizer.decode([int(t)]), float(v))
            for t, v in zip(top5_idx, top5_vals)
        ]
        top_logit = float(top5_vals[0])

        # Degeneration detection
        answer_logits_vals = [v for v in letter_logits.values()]
        max_answer_logit = max(answer_logits_vals) if answer_logits_vals else -float("inf")
        hard_degen = all(v < -10.0 for v in answer_logits_vals)
        soft_degen = (top_logit - max_answer_logit) > 5.0

        return {
            "model_answer": best_letter or "",
            "answer_logits": letter_logits,
            "top5_tokens": top5_tokens,
            "degenerated": hard_degen,
            "degenerated_soft": soft_degen,
        }

    def evaluate_baseline(self, prompt: str) -> dict[str, Any]:
        """Run forward pass WITHOUT steering to get baseline answer."""
        from src.utils.answers import best_choice_from_logits

        self.clear_hooks()

        tokenizer = self.wrapper.tokenizer
        inputs = tokenizer(prompt, return_tensors="pt").to(self.wrapper.device)
        seq_len = inputs["input_ids"].shape[1]
        last_pos = seq_len - 1

        with torch.no_grad():
            outputs = self.wrapper.model(**inputs)

        logits = outputs.logits[0, last_pos, :]
        best_letter, letter_logits = best_choice_from_logits(logits, tokenizer)

        return {
            "model_answer": best_letter or "",
            "answer_logits": letter_logits,
        }

    def evaluate_baseline_mcq(
        self,
        prompt: str,
        *,
        letters: tuple[str, ...] = ("A", "B", "C"),
    ) -> dict[str, Any]:
        """Baseline answer extraction for arbitrary MCQ letter sets (e.g., A/B/C/D)."""
        from src.utils.answers import best_choice_from_logits

        self.clear_hooks()
        tokenizer = self.wrapper.tokenizer
        inputs = tokenizer(prompt, return_tensors="pt").to(self.wrapper.device)
        seq_len = inputs["input_ids"].shape[1]
        last_pos = seq_len - 1
        with torch.no_grad():
            outputs = self.wrapper.model(**inputs)
        logits = outputs.logits[0, last_pos, :]
        best_letter, letter_logits = best_choice_from_logits(logits, tokenizer, letters=letters)
        return {"model_answer": best_letter or "", "answer_logits": letter_logits}

    # ------------------------------------------------------------------
    # Alpha sweep
    # ------------------------------------------------------------------

    def run_alpha_sweep(
        self,
        items: list[dict[str, Any]],
        feature_indices: list[int],
        alpha_values: list[float],
        prompt_formatter: Callable[[dict[str, Any]], str],
        scale_by_sqrt_n: bool = True,
    ) -> "pd.DataFrame":
        """Run all items at each alpha value and record results.

        Returns DataFrame with columns: item_idx, alpha, original_answer,
        original_role, steered_answer, steered_role, flipped, degenerated,
        degenerated_soft, category, stereotyped_groups.
        """
        _require_pandas()

        records: list[dict[str, Any]] = []
        n_items = len(items)
        n_alphas = len(alpha_values)
        log(f"    Alpha sweep: {n_items} items x {n_alphas} alphas "
            f"x {len(feature_indices)} features")

        for i, item in enumerate(items):
            prompt = prompt_formatter(item)
            baseline = self.evaluate_baseline(prompt)
            orig_answer = baseline["model_answer"]
            orig_role = item.get("answer_roles", {}).get(orig_answer, "unknown")

            for alpha in alpha_values:
                vec = self.get_composite_steering(
                    feature_indices, alpha, scale_by_sqrt_n=scale_by_sqrt_n,
                )
                result = self.steer_and_evaluate(prompt, vec)
                steered_answer = result["model_answer"]
                steered_role = item.get("answer_roles", {}).get(steered_answer, "unknown")

                records.append({
                    "item_idx": item.get("item_idx", i),
                    "alpha": alpha,
                    "original_answer": orig_answer,
                    "original_role": orig_role,
                    "steered_answer": steered_answer,
                    "steered_role": steered_role,
                    "flipped": orig_answer != steered_answer,
                    "degenerated": result["degenerated"],
                    "degenerated_soft": result["degenerated_soft"],
                    "category": item.get("category", ""),
                    "stereotyped_groups": item.get("stereotyped_groups", []),
                    "original_logits": baseline["answer_logits"],
                    "steered_logits": result["answer_logits"],
                })

            if (i + 1) % 25 == 0 or i == 0:
                log(f"      [{i + 1}/{n_items}] done")

            # Memory management
            if self.wrapper.device == "mps" and (i + 1) % 50 == 0:
                torch.mps.empty_cache()
            elif str(self.wrapper.device).startswith("cuda") and (i + 1) % 100 == 0:
                torch.cuda.empty_cache()

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Individual feature testing
    # ------------------------------------------------------------------

    def test_individual_features(
        self,
        items: list[dict[str, Any]],
        feature_indices: list[int],
        alpha: float,
        prompt_formatter: Callable[[dict[str, Any]], str],
        max_features: int = 30,
    ) -> "pd.DataFrame":
        """Test each feature individually to identify causally active features.

        Returns DataFrame with columns: feature_idx, n_items, correction_rate,
        degeneration_rate, n_flipped.
        """
        _require_pandas()

        features_to_test = feature_indices[:max_features]
        n_items = len(items)
        log(f"    Individual feature test: {len(features_to_test)} features x {n_items} items")

        records: list[dict[str, Any]] = []

        for fi, fidx in enumerate(features_to_test):
            n_flipped = 0
            n_degen = 0

            for item in items:
                prompt = prompt_formatter(item)
                vec = self.get_steering_vector([fidx], [alpha])
                result = self.steer_and_evaluate(prompt, vec)

                if result["model_answer"] != item.get("_baseline_answer", ""):
                    n_flipped += 1
                if result["degenerated"]:
                    n_degen += 1

            records.append({
                "feature_idx": fidx,
                "alpha": alpha,
                "n_items": n_items,
                "n_flipped": n_flipped,
                "correction_rate": n_flipped / max(n_items, 1),
                "degeneration_rate": n_degen / max(n_items, 1),
            })

            if (fi + 1) % 5 == 0 or fi == 0:
                log(f"      Feature [{fi + 1}/{len(features_to_test)}] "
                    f"L{self.layer}_F{fidx}: flipped {n_flipped}/{n_items}")

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # CrowS-Pairs log-probability scoring
    # ------------------------------------------------------------------

    def compute_log_prob(self, text: str) -> float:
        """Compute mean per-token log-probability for a text string (autoregressive)."""
        tokenizer = self.wrapper.tokenizer
        inputs = tokenizer(text, return_tensors="pt").to(self.wrapper.device)
        input_ids = inputs["input_ids"]
        seq_len = input_ids.shape[1]
        if seq_len < 2:
            return 0.0

        with torch.no_grad():
            outputs = self.wrapper.model(**inputs)

        logits = outputs.logits[0, :-1, :]  # (seq_len-1, vocab)
        targets = input_ids[0, 1:]  # (seq_len-1,)

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        return float(token_log_probs.mean())

    def compute_log_prob_steered(
        self, text: str, steering_vec: torch.Tensor,
    ) -> float:
        """Compute mean per-token log-prob under steering."""
        self.clear_hooks()
        # For sequence scoring, we must steer *all* positions. If we only steer the
        # last token, compute_log_prob() excludes that position and the score won't change.
        self._install_hook(steering_vec, scope="all")

        result = self.compute_log_prob(text)
        self.clear_hooks()
        return result
