#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

ENV_NAME="${ENV_NAME:-suv-navsim1}"
GPUS="${GPUS:-0,1}"
CONFIG="${CONFIG:-configs/suv_itae/navsim_trainval_action_tokens.yaml}"
OUTPUT="${OUTPUT:-output/suv_itae/navsim_trainval}"
NPROC="$(awk -F, '{print NF}' <<< "${GPUS}")"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"

export CKPT_LOCAL_DIR="${CKPT_LOCAL_DIR:-/inspire/hdd/project/spatiotemporal-intelligence-research/ky26298/Projects/pure_checkpoints/SUV_ckpt}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}/third_party/SUV/src:${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "${NNODES}" == "1" ]]; then
  RENDEZVOUS_ARGS=(--standalone)
else
  : "${MASTER_ADDR:?Set MASTER_ADDR for multi-node training}"
  MASTER_PORT="${MASTER_PORT:-29500}"
  RENDEZVOUS_ARGS=(
    --nnodes="${NNODES}"
    --node_rank="${NODE_RANK}"
    --master_addr="${MASTER_ADDR}"
    --master_port="${MASTER_PORT}"
  )
fi

CUDA_VISIBLE_DEVICES="${GPUS}" conda run --no-capture-output -n "${ENV_NAME}" \
  torchrun "${RENDEZVOUS_ARGS[@]}" --nproc_per_node="${NPROC}" \
    -m tools.suv_itae.train \
    --config "${CONFIG}" \
    --output "${OUTPUT}" \
    "$@"
