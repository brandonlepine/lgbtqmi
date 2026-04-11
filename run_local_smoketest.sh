#!/usr/bin/env bash
set -euo pipefail

# Local smoketest runner for the full pipeline on a small subset across categories.
# Safe to re-run: uses a unique RUN_DATE by default to avoid overwriting/colliding.

on_err() {
  echo
  echo "ERROR: smoketest failed (exit=$?) at line $1"
  echo "Run dir (if created): ${RUN_DIR:-<unset>}"
}
trap 'on_err $LINENO' ERR

# ----- Config (override via env vars) -----
MODEL_ID="${MODEL_ID:-llama2-7b}"
MODEL_PATH="${MODEL_PATH:-/Users/brandonlepine/Repositories/Research_Repositories/lgbtqmi/src/models/llama2-7b}"
DEVICE="${DEVICE:-mps}"
SEED="${SEED:-42}"

# Small subset per category for a fast-but-meaningful correctness check.
# 20 can be too small to hit multiple subgroups; 40 is a better default.
MAX_ITEMS="${MAX_ITEMS:-200}"

RUN_DATE="${RUN_DATE:-$(date +%F)-smoke$(date +%H%M%S)}"
RUN_DIR="results/runs/${MODEL_ID}/${RUN_DATE}"

# Optional: run meso-level analysis/ablations (can be compute-heavy).
RUN_MESO="${RUN_MESO:-0}"
MESO_MAX_ITEMS="${MESO_MAX_ITEMS:-${MAX_ITEMS}}"
MESO_ALPHA="${MESO_ALPHA:-14.0}"

# Optional: include CrowS-Pairs in the same run if the CSV exists.
CROWS_PAIRS_PATH="${CROWS_PAIRS_PATH:-data/raw/crows_pairs.csv}"
CROWS_MAX_ITEMS="${CROWS_MAX_ITEMS:-${MAX_ITEMS}}"

echo "============================================================"
echo "LOCAL SMOKETEST"
echo "============================================================"
echo "MODEL_ID:   ${MODEL_ID}"
echo "MODEL_PATH: ${MODEL_PATH}"
echo "DEVICE:     ${DEVICE}"
echo "SEED:       ${SEED}"
echo "MAX_ITEMS:  ${MAX_ITEMS} (per category)"
echo "RUN_DATE:   ${RUN_DATE}"
echo "RUN_DIR:    ${RUN_DIR}"
echo "RUN_MESO:   ${RUN_MESO}"
echo "MESO_MAX_ITEMS: ${MESO_MAX_ITEMS}"
echo "MESO_ALPHA: ${MESO_ALPHA}"
echo "CROWS_PAIRS_PATH: ${CROWS_PAIRS_PATH}"
echo "CROWS_MAX_ITEMS:  ${CROWS_MAX_ITEMS}"
echo

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "ERROR: MODEL_PATH does not exist: ${MODEL_PATH}"
  exit 2
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
  --max_items "${MAX_ITEMS}" \
  --run_date "${RUN_DATE}" \
  --seed "${SEED}" \
  $( [[ -f "${CROWS_PAIRS_PATH}" ]] && printf '%s' "--crows_pairs_path ${CROWS_PAIRS_PATH} --crows_max_items ${CROWS_MAX_ITEMS}" )

echo
echo "----- 2) Compute directions (uses item_idx alignment) -----"
python scripts/compute_directions.py \
  --run_dir "${RUN_DIR}" \
  --categories all \
  --max_items "${MAX_ITEMS}"

echo
echo "----- 3) Cross-category geometry figs/results -----"
python scripts/analyze_cross_category.py \
  --run_dir "${RUN_DIR}"

echo
echo "----- 4) Head probes (uses attn_pre_o_proj_final) -----"
# Llama2-7b: n_heads=32, head_dim=128
python scripts/train_head_probes.py \
  --run_dir "${RUN_DIR}" \
  --categories all \
  --n_heads 32 \
  --head_dim 128 \
  --max_items "${MAX_ITEMS}"

echo
echo "----- 5) Probe generalization figs/results -----"
python scripts/analyze_probe_generalization.py \
  --run_dir "${RUN_DIR}" \
  --categories all \
  --n_heads 32 \
  --head_dim 128 \
  --max_items "${MAX_ITEMS}"

echo
echo "----- 6) Causal ablations (shared/specific + RLHF head ablation if probes exist) -----"
python scripts/causal_ablation_hierarchy.py \
  --run_dir "${RUN_DIR}" \
  --model_path "${MODEL_PATH}" \
  --device "${DEVICE}" \
  --categories all \
  --max_items "${MAX_ITEMS}" \
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
    --max_items "${MESO_MAX_ITEMS}" \
    --categories all

  python scripts/analyze_meso_ablation.py \
    --run_dir "${RUN_DIR}"
fi

echo
echo "Smoke run complete. Outputs are under: ${RUN_DIR}"