#!/usr/bin/env bash
set -euo pipefail

# Default: use both GPUs for train, then both GPUs for validation. For two
# nodes, run `all-part 0` and `all-part 1` against the same shared CACHE_ROOT.
action=${1:-local}
partition_index=${2:-${PARTITION_INDEX:-}}
num_partitions=${NUM_PARTITIONS:-2}
cuda_device=${CUDA_DEVICE:-0}
batch_size=${BATCH_SIZE:-4}
num_workers=${NUM_WORKERS:-2}
image_cache_size=${IMAGE_CACHE_SIZE:-64}
cache_root=${CACHE_ROOT:-/inspire/qb-ilm2/project/spatiotemporal-intelligence-research/ky26298/itae_nvsim_cache/vggt_omega_cache}

config=configs/tokenizer/navsim_trainval_v4_scratch_4s.yaml
train_manifest=data/manifests/navsim_trainval_train_4s.jsonl
val_manifest=data/manifests/navsim_trainval_val_4s.jsonl
train_name=navsim_trainval_front_4s_train_rich
val_name=navsim_trainval_front_4s_val_rich
parts_root=${cache_root}/${train_name}.parts
val_parts_root=${cache_root}/${val_name}.parts
train_output=${cache_root}/${train_name}
val_output=${cache_root}/${val_name}

mkdir -p output "$cache_root" "$parts_root" "$val_parts_root"

ensure_config_cache_link() {
  local name=$1
  local target=$2
  local link=data/vggt_omega_cache/${name}
  mkdir -p data/vggt_omega_cache
  if [[ -L "$link" ]]; then
    if [[ "$(readlink -f "$link")" != "$(readlink -f "$target")" ]]; then
      echo "cache link points elsewhere: $link" >&2
      exit 1
    fi
  elif [[ -e "$link" ]]; then
    echo "cache config path already exists and is not a symlink: $link" >&2
    exit 1
  else
    ln -s "$target" "$link"
  fi
}

part_path() {
  printf '%s/part-%03d-of-%03d' "$parts_root" "$1" "$num_partitions"
}

val_part_path() {
  printf '%s/part-%03d-of-%03d' "$val_parts_root" "$1" "$num_partitions"
}

run_train_partition() {
  local index=$1
  local device=$2
  local destination
  destination=$(part_path "$index")
  CUDA_VISIBLE_DEVICES=$device python -m tools.features.cache_vggt_omega_features \
    --config "$config" \
    --manifest "$train_manifest" \
    --output "$destination" \
    --num-partitions "$num_partitions" --partition-index "$index" \
    --batch-size "$batch_size" --shard-size 256 \
    --num-workers "$num_workers" --image-cache-size "$image_cache_size"
}

merge_train() {
  local parts=()
  local index
  for ((index = 0; index < num_partitions; index++)); do
    parts+=("$(part_path "$index")")
  done
  python -m tools.features.merge_vggt_omega_feature_caches \
    --parts "${parts[@]}" \
    --manifest "$train_manifest" \
    --output "$train_output" \
    --mode hardlink
  ensure_config_cache_link "$train_name" "$train_output"
}

run_validation_partition() {
  local index=$1
  local device=$2
  local destination
  destination=$(val_part_path "$index")
  CUDA_VISIBLE_DEVICES=$device python -m tools.features.cache_vggt_omega_features \
    --config "$config" \
    --manifest "$val_manifest" \
    --output "$destination" \
    --num-partitions "$num_partitions" --partition-index "$index" \
    --batch-size "$batch_size" --shard-size 256 \
    --num-workers "$num_workers" --image-cache-size "$image_cache_size"
}

merge_validation() {
  local parts=()
  local index
  for ((index = 0; index < num_partitions; index++)); do
    parts+=("$(val_part_path "$index")")
  done
  python -m tools.features.merge_vggt_omega_feature_caches \
    --parts "${parts[@]}" \
    --manifest "$val_manifest" \
    --output "$val_output" \
    --mode hardlink
  ensure_config_cache_link "$val_name" "$val_output"
}

case "$action" in
  train-part)
    if [[ -z "$partition_index" ]]; then
      echo "usage: bash scripts/temp.sh train-part PARTITION_INDEX" >&2
      exit 2
    fi
    run_train_partition "$partition_index" "$cuda_device" \
      > "output/navsim_trainval_cache_train_part_${partition_index}.log" 2>&1
    ;;
  merge-train)
    merge_train
    ;;
  val-part)
    if [[ -z "$partition_index" ]]; then
      echo "usage: bash scripts/temp.sh val-part PARTITION_INDEX" >&2
      exit 2
    fi
    run_validation_partition "$partition_index" "$cuda_device" \
      > "output/navsim_trainval_cache_val_part_${partition_index}.log" 2>&1
    ;;
  all-part)
    if [[ -z "$partition_index" ]]; then
      echo "usage: bash scripts/temp.sh all-part PARTITION_INDEX" >&2
      exit 2
    fi
    run_train_partition "$partition_index" "$cuda_device" \
      > "output/navsim_trainval_cache_train_part_${partition_index}.log" 2>&1
    run_validation_partition "$partition_index" "$cuda_device" \
      > "output/navsim_trainval_cache_val_part_${partition_index}.log" 2>&1
    ;;
  merge-val)
    merge_validation
    ;;
  merge-all)
    merge_train
    merge_validation
    ;;
  local)
    if [[ "$num_partitions" -ne 2 ]]; then
      echo "local mode expects NUM_PARTITIONS=2" >&2
      exit 2
    fi
    run_train_partition 0 0 > output/navsim_trainval_cache_train_part_0.log 2>&1 &
    pid0=$!
    run_train_partition 1 1 > output/navsim_trainval_cache_train_part_1.log 2>&1 &
    pid1=$!
    wait "$pid0"
    wait "$pid1"
    merge_train
    run_validation_partition 0 0 > output/navsim_trainval_cache_val_part_0.log 2>&1 &
    pid0=$!
    run_validation_partition 1 1 > output/navsim_trainval_cache_val_part_1.log 2>&1 &
    pid1=$!
    wait "$pid0"
    wait "$pid1"
    merge_validation
    ;;
  *)
    echo "usage: bash scripts/temp.sh [local|all-part INDEX|train-part INDEX|val-part INDEX|merge-all|merge-train|merge-val]" >&2
    exit 2
    ;;
esac
