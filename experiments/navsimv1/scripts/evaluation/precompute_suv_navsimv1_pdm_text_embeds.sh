#!/usr/bin/env bash
# Build the text-embedding cache for NAVSIM v1 PDM evaluation.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

# ======================== User configuration ========================
# Edit the paths below before running this script.

# Local checkpoint root
# Expected Wan layout:
#   ${CKPT_LOCAL_DIR}/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth
#   ${CKPT_LOCAL_DIR}/Wan2.2-TI2V-5B/google/umt5-xxl/
CKPT_LOCAL_DIR="/path/to/checkpoints"

# Text-embedding output directory
TEXT_EMBEDDING_CACHE_DIR="/path/to/suv_text_embeds_cache_test/navsim_v1"

# NAVSIM v1 data
OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/path/to/navsim}"
NAVSIM_LOG_PATH="${NAVSIM_LOG_PATH:-${OPENSCENE_DATA_ROOT}/navsim_logs/test}"
SENSOR_BLOBS_PATH="${SENSOR_BLOBS_PATH:-${OPENSCENE_DATA_ROOT}/sensor_blobs/test}"

# Text encoder settings: embeddings are sharded across four GPUs by default
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
BATCH_SIZE="${BATCH_SIZE:-16}"
OVERWRITE="${OVERWRITE:-false}"
# ====================================================================

export CKPT_LOCAL_DIR
export CUDA_VISIBLE_DEVICES
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFSYNTH_SKIP_DOWNLOAD=true

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#GPU_IDS[@]}"

torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
  "${REPO_ROOT}/experiments/navsimv1/precompute_text_embeds.py" \
  --train-test-split navtest \
  --navsim-log-path "${NAVSIM_LOG_PATH}" \
  --sensor-blobs-path "${SENSOR_BLOBS_PATH}" \
  --text-embedding-cache-dir "${TEXT_EMBEDDING_CACHE_DIR}" \
  --model-id Wan2.2-TI2V-5B \
  --tokenizer-model-id Wan2.2-TI2V-5B \
  --num-history-frames 4 \
  --num-future-frames 8 \
  --batch-size "${BATCH_SIZE}" \
  --overwrite "${OVERWRITE}" \
  "$@"
