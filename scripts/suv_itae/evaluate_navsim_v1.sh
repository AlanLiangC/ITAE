#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

ENV_NAME="${ENV_NAME:-suv-navsim1}"
GPUS="${GPUS:-0,1}"
ADAPTER="${ADAPTER:-output/suv_itae/navsim_trainval/best.pt}"
TOKENIZER_DIR="${TOKENIZER_DIR:-output/navsim_trainval_v4_scratch_4gpu}"
OUTPUT_DIR="${OUTPUT_DIR:-output/suv_itae/navsim_trainval/navsim_v1_evaluation}"

conda run --no-capture-output -n "${ENV_NAME}" \
  python -m tools.suv.evaluate_navsim_v1 evaluate \
    --gpus "${GPUS}" \
    --itae-adapter "${ADAPTER}" \
    --action-tokenizer-config "${TOKENIZER_DIR}/resolved_config.json" \
    --action-tokenizer-checkpoint "${TOKENIZER_DIR}/best.pt" \
    --output-dir "${OUTPUT_DIR}" \
    "$@"

