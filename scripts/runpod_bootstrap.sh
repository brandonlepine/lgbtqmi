#!/usr/bin/env bash
set -euo pipefail

# RunPod bootstrap for this repo:
# - creates/uses a venv
# - installs a CUDA-enabled PyTorch
# - installs repo Python deps
# - clones BBQ into datasets/bbq if missing
# - (optionally) runs a small smoketest
#
# Expected usage on RunPod (from repo root):
#   bash scripts/runpod_bootstrap.sh
#
# Optional env vars:
#   VENV_DIR=.venv
#   TORCH_CUDA_INDEX_URL=https://download.pytorch.org/whl/cu124
#   RUN_SMOKETEST=1
#   MODEL_PATH=/workspace/lgbtqmi/models/llama2-7b
#   MODEL_ID=llama2-7b
#   MAX_ITEMS=40

VENV_DIR="${VENV_DIR:-.venv}"
TORCH_CUDA_INDEX_URL="${TORCH_CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu124}"

echo "=== RunPod bootstrap ==="
echo "VENV_DIR: ${VENV_DIR}"
echo "TORCH_CUDA_INDEX_URL: ${TORCH_CUDA_INDEX_URL}"
echo

if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install -U pip

echo
echo "Installing CUDA PyTorch..."
python -m pip install --index-url "${TORCH_CUDA_INDEX_URL}" torch

echo
echo "Installing repo dependencies..."
python -m pip install -r requirements.txt

echo
echo "Ensuring BBQ dataset is present..."
if [[ ! -d "datasets/bbq/.git" ]]; then
  mkdir -p datasets
  git clone --depth 1 https://github.com/nyu-mll/BBQ.git datasets/bbq
else
  echo "BBQ already cloned at datasets/bbq"
fi

echo
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device", torch.cuda.get_device_name(0))
PY

if [[ "${RUN_SMOKETEST:-0}" == "1" ]]; then
  MODEL_PATH="${MODEL_PATH:-/workspace/lgbtqmi/models/llama2-7b}"
  MODEL_ID="${MODEL_ID:-llama2-7b}"
  MAX_ITEMS="${MAX_ITEMS:-40}"
  RUN_DATE="${RUN_DATE:-$(date +%F)-runpod_smoke$(date +%H%M%S)}"

  echo
  echo "Running smoketest..."
  echo "MODEL_PATH=${MODEL_PATH}"
  echo "MODEL_ID=${MODEL_ID}"
  echo "MAX_ITEMS=${MAX_ITEMS}"
  echo "RUN_DATE=${RUN_DATE}"
  echo

  python scripts/run_extraction_pipeline.py \
    --model_path "${MODEL_PATH}" \
    --model_id "${MODEL_ID}" \
    --device cuda \
    --categories all \
    --max_items "${MAX_ITEMS}" \
    --run_date "${RUN_DATE}" \
    --seed 42 \
    --crows_pairs_path data/raw/crows_pairs.csv \
    --crows_max_items "${MAX_ITEMS}"

  python scripts/compute_directions.py \
    --run_dir "results/runs/${MODEL_ID}/${RUN_DATE}" \
    --categories all \
    --max_items "${MAX_ITEMS}"
fi

echo
echo "Bootstrap complete."

