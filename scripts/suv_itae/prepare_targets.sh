#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

ENV_NAME="${ENV_NAME:-py312torch210cu126}"
TOKENIZER_DIR="${TOKENIZER_DIR:-output/navsim_trainval_v4_scratch_4gpu}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/suv_itae/action_targets}"
TRAIN_FEATURE_CACHE="${TRAIN_FEATURE_CACHE:-/inspire/qb-ilm2/project/spatiotemporal-intelligence-research/ky26298/itae_nvsim_cache/vggt_omega_cache/navsim_trainval_front_4s_train_rich}"
VAL_FEATURE_CACHE="${VAL_FEATURE_CACHE:-/inspire/qb-ilm2/project/spatiotemporal-intelligence-research/ky26298/itae_nvsim_cache/vggt_omega_cache/navsim_trainval_front_4s_val_rich}"
GPU="${GPU:-0}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${OUTPUT_ROOT}"

run_split() {
  local split="$1"
  local manifest="$2"
  local feature_cache="$3"
  local output="$4"
  if [[ -s "${output}" && -s "${output%.safetensors}.json" ]]; then
    echo "[${split}] target cache already complete: ${output}"
    return
  fi
  CUDA_VISIBLE_DEVICES="${GPU}" conda run --no-capture-output -n "${ENV_NAME}" \
    python tools/features/cache_tokenizer_action_targets.py \
      --tokenizer-config "${TOKENIZER_DIR}/resolved_config.json" \
      --manifest "${manifest}" \
      --feature-cache "${feature_cache}" \
      --checkpoint "${TOKENIZER_DIR}/best.pt" \
      --output "${output}" \
      --batch-size "${BATCH_SIZE:-256}" \
      --num-workers "${NUM_WORKERS:-8}"
}

run_split train \
  data/manifests/navsim_trainval_train_4s.jsonl \
  "${TRAIN_FEATURE_CACHE}" \
  "${OUTPUT_ROOT}/train.safetensors"
run_split validation \
  data/manifests/navsim_trainval_val_4s.jsonl \
  "${VAL_FEATURE_CACHE}" \
  "${OUTPUT_ROOT}/validation.safetensors"
