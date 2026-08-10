#!/usr/bin/env bash
# Run the official NAVSIM v1 PDM evaluation on the selected GPUs.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

# ======================== User configuration ========================
# Edit the paths below before running this script.

# Local checkpoint root
# Expected layout:
#   ${CKPT_LOCAL_DIR}/suv_navsim.pt
#   ${CKPT_LOCAL_DIR}/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
CKPT_LOCAL_DIR="/path/to/checkpoints"

# Precomputed text embeddings
TEXT_EMBEDDING_CACHE_DIR="/path/to/suv_text_embeds_cache_test/navsim_v1"

# NAVSIM v1 data, maps, and metric cache
OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/path/to/navsim}"
NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/path/to/navsim/maps}"
NAVSIM_LOG_PATH="${NAVSIM_LOG_PATH:-${OPENSCENE_DATA_ROOT}/navsim_logs/test}"
SENSOR_BLOBS_PATH="${SENSOR_BLOBS_PATH:-${OPENSCENE_DATA_ROOT}/sensor_blobs/test}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-/path/to/navsim_v1/metric_cache}"

# Evaluation output root
NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/path/to/evaluation_outputs}"

# GPU IDs: four-GPU sharding by default; use "0" for single-GPU evaluation
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
# ====================================================================

OUTPUT_DIR="${NAVSIM_EXP_ROOT}/suv_navsimv1_navtest"

export OPENSCENE_DATA_ROOT NUPLAN_MAPS_ROOT NAVSIM_EXP_ROOT
export CKPT_LOCAL_DIR CUDA_VISIBLE_DEVICES
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.2}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFSYNTH_SKIP_DOWNLOAD=true

python -u "${REPO_ROOT}/experiments/navsimv1/run_pdm_score_multigpu.py" \
  --cuda-visible-devices "${CUDA_VISIBLE_DEVICES}" \
  --output-dir "${OUTPUT_DIR}" \
  --overrides \
    train_test_split=navtest \
    train_test_split.scene_filter.num_history_frames=4 \
    train_test_split.scene_filter.num_future_frames=8 \
    metric_cache_path="${METRIC_CACHE_PATH}" \
    navsim_log_path="${NAVSIM_LOG_PATH}" \
    sensor_blobs_path="${SENSOR_BLOBS_PATH}" \
    output_dir="${OUTPUT_DIR}" \
    experiment_name=suv_navsimv1_navtest \
    worker=single_machine_thread_pool \
    worker.max_workers=1 \
    worker.use_process_pool=false \
    agent._target_=experiments.navsimv1.pdm_agent.SUVNavsimV1Agent \
    agent.checkpoint_path="${CKPT_LOCAL_DIR}/suv_navsim.pt" \
    ++agent.model_config_path="${REPO_ROOT}/experiments/navsimv1/config/model/suv_navsim.yaml" \
    ++agent.text_embedding_cache_dir="${TEXT_EMBEDDING_CACHE_DIR}" \
    ++agent.visual_conditioning=history_4 \
    ++agent.slot_inference=true \
    ++agent.num_inference_steps=10 \
    "$@"
