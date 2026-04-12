#!/usr/bin/env bash
set -euo pipefail

# Local smoketest runner for the full pipeline on a small subset across categories.
# Safe to re-run: uses a unique RUN_DATE by default to avoid overwriting/colliding.

on_err() {
  local status="$1"
  local lineno="$2"
  echo
  echo "ERROR: smoketest failed (exit=${status}) at line ${lineno}"
  echo "Run dir (if created): ${RUN_DIR:-<unset>}"
}
trap 'on_err $? $LINENO' ERR

# ----- Config (override via env vars) -----
MODEL_ID="${MODEL_ID:-llama2-7b}"
MODEL_PATH="${MODEL_PATH:-/Users/brandonlepine/Repositories/Research_Repositories/lgbtqmi/src/models/llama2-7b}"
DEVICE="${DEVICE:-mps}"
SEED="${SEED:-42}"

# Optional: override head geometry. If unset, we infer from model config.
N_HEADS="${N_HEADS:-}"
HEAD_DIM="${HEAD_DIM:-}"

# Optional per-category cap. If unset/empty, runs all items.
MAX_ITEMS="${MAX_ITEMS:-}"

RUN_DATE="${RUN_DATE:-$(date +%F)-smoke$(date +%H%M%S)}"
RUN_DIR="results/runs/${MODEL_ID}/${RUN_DATE}"

# Optional: run meso-level analysis/ablations (can be compute-heavy).
RUN_MESO="${RUN_MESO:-0}"
MESO_MAX_ITEMS="${MESO_MAX_ITEMS:-${MAX_ITEMS}}"
MESO_ALPHA="${MESO_ALPHA:-14.0}"

# Optional: run pairwise shared-direction interventions (additive; can be compute-heavy).
# Default: if you enabled RUN_MESO, also run pairwise unless explicitly disabled.
RUN_PAIRWISE="${RUN_PAIRWISE:-${RUN_MESO}}"
PAIRWISE_THRESHOLD="${PAIRWISE_THRESHOLD:-0.4}"
PAIRWISE_MAX_PAIRS="${PAIRWISE_MAX_PAIRS:-}"
PAIRWISE_ALPHA="${PAIRWISE_ALPHA:-14.0}"

# Optional: run subgroup-level pipeline (directions + fragmentation + probes + ablation + summary).
RUN_SUBGROUP="${RUN_SUBGROUP:-0}"
SUBGROUP_MAX_ITEMS="${SUBGROUP_MAX_ITEMS:-${MAX_ITEMS}}"
SUBGROUP_ALPHA="${SUBGROUP_ALPHA:-14.0}"

# Optional: include CrowS-Pairs in the same run if the CSV exists.
CROWS_PAIRS_PATH="${CROWS_PAIRS_PATH:-data/raw/crows_pairs.csv}"
CROWS_MAX_ITEMS="${CROWS_MAX_ITEMS:-${MAX_ITEMS}}"

MAX_ITEMS_ARGS=()
if [[ -n "${MAX_ITEMS}" ]]; then
  MAX_ITEMS_ARGS=(--max_items "${MAX_ITEMS}")
fi

MESO_MAX_ITEMS_ARGS=()
if [[ -n "${MESO_MAX_ITEMS}" ]]; then
  MESO_MAX_ITEMS_ARGS=(--max_items "${MESO_MAX_ITEMS}")
fi

CROWS_ARGS=()
if [[ -f "${CROWS_PAIRS_PATH}" ]]; then
  CROWS_ARGS=(--crows_pairs_path "${CROWS_PAIRS_PATH}")
  if [[ -n "${CROWS_MAX_ITEMS}" ]]; then
    CROWS_ARGS+=(--crows_max_items "${CROWS_MAX_ITEMS}")
  fi
fi

echo "============================================================"
echo "LOCAL SMOKETEST"
echo "============================================================"
echo "MODEL_ID:   ${MODEL_ID}"
echo "MODEL_PATH: ${MODEL_PATH}"
echo "DEVICE:     ${DEVICE}"
echo "SEED:       ${SEED}"
echo "MAX_ITEMS:  ${MAX_ITEMS:-<none>} (per category)"
echo "RUN_DATE:   ${RUN_DATE}"
echo "RUN_DIR:    ${RUN_DIR}"
echo "RUN_MESO:   ${RUN_MESO}"
echo "MESO_MAX_ITEMS: ${MESO_MAX_ITEMS:-<none>}"
echo "MESO_ALPHA: ${MESO_ALPHA}"
echo "RUN_PAIRWISE: ${RUN_PAIRWISE}"
echo "PAIRWISE_THRESHOLD: ${PAIRWISE_THRESHOLD}"
echo "PAIRWISE_MAX_PAIRS: ${PAIRWISE_MAX_PAIRS:-<none>}"
echo "PAIRWISE_ALPHA: ${PAIRWISE_ALPHA}"
echo "RUN_SUBGROUP: ${RUN_SUBGROUP}"
echo "SUBGROUP_MAX_ITEMS: ${SUBGROUP_MAX_ITEMS:-<none>}"
echo "SUBGROUP_ALPHA: ${SUBGROUP_ALPHA}"
echo "CROWS_PAIRS_PATH: ${CROWS_PAIRS_PATH}"
echo "CROWS_MAX_ITEMS:  ${CROWS_MAX_ITEMS:-<none>}"
echo

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "ERROR: MODEL_PATH does not exist: ${MODEL_PATH}"
  exit 2
fi

# Infer head geometry once (used by probe scripts). This loads config only (no weights).
if [[ -z "${N_HEADS}" || -z "${HEAD_DIM}" ]]; then
  echo "Inferring N_HEADS/HEAD_DIM from model config..."
  read -r N_HEADS HEAD_DIM < <(MODEL_PATH="${MODEL_PATH}" python - <<'PY'
from __future__ import annotations
import os
from pathlib import Path

mp = os.environ.get("MODEL_PATH", "")
if not mp:
    raise SystemExit("ERROR: MODEL_PATH env var not set for head-geometry inference")
model_path = Path(mp)
try:
    from transformers import AutoConfig
except Exception as e:
    raise SystemExit(f"ERROR: transformers not available to infer model config: {e}")

cfg = AutoConfig.from_pretrained(str(model_path))

def _get_int(*names: str) -> int | None:
    for n in names:
        v = getattr(cfg, n, None)
        if isinstance(v, int) and v > 0:
            return int(v)
    return None

n_heads = _get_int("num_attention_heads", "n_head", "n_heads")
hidden = _get_int("hidden_size", "n_embd", "d_model")
if n_heads is None or hidden is None:
    raise SystemExit(
        f"ERROR: could not infer n_heads/hidden_size from config class {type(cfg).__name__} "
        f"(num_attention_heads={getattr(cfg,'num_attention_heads',None)}, hidden_size={getattr(cfg,'hidden_size',None)})"
    )
if hidden % n_heads != 0:
    raise SystemExit(f"ERROR: hidden_size {hidden} not divisible by n_heads {n_heads}")
head_dim = hidden // n_heads
print(f"{n_heads} {head_dim}")
PY
)
  if [[ -z "${N_HEADS}" || -z "${HEAD_DIM}" ]]; then
    echo "ERROR: Failed to infer N_HEADS/HEAD_DIM from model config."
    exit 3
  fi
  echo "  N_HEADS=${N_HEADS}  HEAD_DIM=${HEAD_DIM}"
fi

# Optional: install deps if requested
if [[ "${INSTALL_DEPS:-0}" == "1" ]]; then
  echo "Installing dependencies from requirements.txt..."
  python -m pip install -r requirements.txt
  echo
fi


echo "----- 1) Prepare stimuli + extract activations (all categories) -----"
python scripts/run_extraction_pipeline.py \
  --model_path "${MODEL_PATH}" \
  --model_id "${MODEL_ID}" \
  --device "${DEVICE}" \
  --categories all \
  "${MAX_ITEMS_ARGS[@]}" \
  --run_date "${RUN_DATE}" \
  --seed "${SEED}" \
  "${CROWS_ARGS[@]}"

echo
echo "----- 2) Compute directions (uses item_idx alignment) -----"
python scripts/compute_directions.py \
  --run_dir "${RUN_DIR}" \
  --categories all \
  "${MAX_ITEMS_ARGS[@]}"

echo
echo "----- 3) Cross-category geometry figs/results -----"
python scripts/analyze_cross_category.py \
  --run_dir "${RUN_DIR}"

echo
echo "----- 4) Head probes (uses attn_pre_o_proj_final) -----"
python scripts/train_head_probes.py \
  --run_dir "${RUN_DIR}" \
  --categories all \
  --n_heads "${N_HEADS}" \
  --head_dim "${HEAD_DIM}" \
  "${MAX_ITEMS_ARGS[@]}"

echo
echo "----- 5) Probe generalization figs/results -----"
python scripts/analyze_probe_generalization.py \
  --run_dir "${RUN_DIR}" \
  --categories all \
  --n_heads "${N_HEADS}" \
  --head_dim "${HEAD_DIM}" \
  "${MAX_ITEMS_ARGS[@]}"

echo
echo "----- 6) Causal ablations (shared/specific + RLHF head ablation if probes exist) -----"
python scripts/causal_ablation_hierarchy.py \
  --run_dir "${RUN_DIR}" \
  --model_path "${MODEL_PATH}" \
  --device "${DEVICE}" \
  --categories all \
  "${MAX_ITEMS_ARGS[@]}" \
  --model_id "${MODEL_ID}"

echo
echo "----- 7) Summary figure bundle -----"
python scripts/generate_summary_figures.py \
  --run_dir "${RUN_DIR}"

if [[ "${RUN_MESO}" == "1" ]]; then
  echo
  echo "----- 8) Meso directions + meso cluster ablations (optional) -----"
  python scripts/compute_meso_directions.py \
    --run_dir "${RUN_DIR}"

  python scripts/ablate_meso_clusters.py \
    --run_dir "${RUN_DIR}" \
    --model_path "${MODEL_PATH}" \
    --model_id "${MODEL_ID}" \
    --device "${DEVICE}" \
    --alpha "${MESO_ALPHA}" \
    "${MESO_MAX_ITEMS_ARGS[@]}" \
    --categories all

  python scripts/analyze_meso_ablation.py \
    --run_dir "${RUN_DIR}"
fi

if [[ "${RUN_PAIRWISE}" == "1" ]]; then
  echo
  echo "----- 9) Pairwise shared-direction extraction + ablation (optional) -----"
  python scripts/extract_pairwise_shared.py \
    --run_dir "${RUN_DIR}" \
    --threshold "${PAIRWISE_THRESHOLD}"

  PAIRWISE_MAX_PAIRS_ARGS=()
  if [[ -n "${PAIRWISE_MAX_PAIRS}" ]]; then
    PAIRWISE_MAX_PAIRS_ARGS=(--max_pairs "${PAIRWISE_MAX_PAIRS}")
  fi

  python scripts/ablate_pairwise_shared.py \
    --run_dir "${RUN_DIR}" \
    --model_path "${MODEL_PATH}" \
    --model_id "${MODEL_ID}" \
    --device "${DEVICE}" \
    --alpha "${PAIRWISE_ALPHA}" \
    "${MAX_ITEMS_ARGS[@]}" \
    "${PAIRWISE_MAX_PAIRS_ARGS[@]}" \
    --categories all

  python scripts/analyze_pairwise_ablation.py \
    --run_dir "${RUN_DIR}"
fi

if [[ "${RUN_SUBGROUP}" == "1" ]]; then
  echo
  echo "----- 10) Subgroup directions + fragmentation + probes + cross-subgroup ablations (optional) -----"

  SUBGROUP_MAX_ITEMS_ARGS=()
  if [[ -n "${SUBGROUP_MAX_ITEMS}" ]]; then
    SUBGROUP_MAX_ITEMS_ARGS=(--max_items "${SUBGROUP_MAX_ITEMS}")
  fi

  python scripts/compute_subgroup_directions.py \
    --run_dir "${RUN_DIR}" \
    --categories all \
    "${SUBGROUP_MAX_ITEMS_ARGS[@]}"

  python scripts/analyze_subgroup_fragmentation.py \
    --run_dir "${RUN_DIR}"

  # NOTE: you must pass the model's n_heads/head_dim for probe slicing
  python scripts/train_subgroup_probes.py \
    --run_dir "${RUN_DIR}" \
    --categories all \
    --n_heads "${N_HEADS}" \
    --head_dim "${HEAD_DIM}" \
    "${SUBGROUP_MAX_ITEMS_ARGS[@]}"

  python scripts/ablate_cross_subgroup.py \
    --run_dir "${RUN_DIR}" \
    --model_path "${MODEL_PATH}" \
    --model_id "${MODEL_ID}" \
    --device "${DEVICE}" \
    --alpha "${SUBGROUP_ALPHA}" \
    "${SUBGROUP_MAX_ITEMS_ARGS[@]}" \
    --categories all

  python scripts/analyze_subgroup_results.py \
    --run_dir "${RUN_DIR}"
fi

echo
echo "Smoke run complete. Outputs are under: ${RUN_DIR}"