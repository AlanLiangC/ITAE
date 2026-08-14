#!/usr/bin/env python3
"""Cache frozen VGGT-Omega CameraHead hidden features in strict shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from itertools import islice
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Subset

from vision_action_tokenizer.config import load_config, stable_hash
from vision_action_tokenizer.data.dataset import ActionWindowDataset, VGGTOmegaResize
from vision_action_tokenizer.models.vggt_omega import OmegaCameraFeatureExtractor, file_sha256


def _sample_token_order_sha256(windows, sample_count: int) -> str:
    """Hash a JSON string list without materializing all trainval tokens."""
    digest = hashlib.sha256()
    digest.update(b"[")
    for index, window in enumerate(windows):
        if index >= sample_count:
            break
        if index:
            digest.update(b", ")
        digest.update(
            json.dumps(
                window.sample_token,
                ensure_ascii=False,
                default=str,
            ).encode("utf-8")
        )
    digest.update(b"]")
    return digest.hexdigest()


def _partition_bounds(
    sample_count: int,
    num_partitions: int,
    partition_index: int,
    max_samples: int | None = None,
) -> tuple[int, int]:
    """Return one balanced, contiguous half-open manifest range."""
    if sample_count <= 0:
        raise ValueError("Cannot partition an empty manifest")
    if num_partitions <= 0 or not 0 <= partition_index < num_partitions:
        raise ValueError("Invalid cache partition selection")
    start = sample_count * partition_index // num_partitions
    end = sample_count * (partition_index + 1) // num_partitions
    if max_samples is not None:
        end = min(end, start + max_samples)
    if end <= start:
        raise ValueError("Selected cache partition is empty")
    return start, end


def _git_commit(repository: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _atomic_write_index(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def _validate_shards(
    output: Path, shards: list[dict[str, object]], expected_count: int
) -> None:
    cursor = 0
    for shard in shards:
        if int(shard["start"]) != cursor or int(shard["end"]) <= cursor:
            raise ValueError("VGGT-Omega cache shards are not contiguous")
        path = output / str(shard["file"])
        if not path.is_file() or file_sha256(path) != shard.get("sha256"):
            raise ValueError(f"VGGT-Omega cache shard is missing or corrupt: {path}")
        cursor = int(shard["end"])
    if cursor != expected_count:
        raise ValueError(
            f"VGGT-Omega cache shard range ends at {cursor}, expected {expected_count}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--image-cache-size", type=int, default=64)
    parser.add_argument("--num-partitions", type=int, default=1)
    parser.add_argument("--partition-index", type=int, default=0)
    args = parser.parse_args()
    if args.shard_size <= 0:
        raise ValueError("--shard-size must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.prefetch_factor <= 0:
        raise ValueError("--prefetch-factor must be positive")
    if args.image_cache_size < 0:
        raise ValueError("--image-cache-size must be non-negative")
    if args.num_partitions <= 0:
        raise ValueError("--num-partitions must be positive")
    if not 0 <= args.partition_index < args.num_partitions:
        raise ValueError("--partition-index must be in [0, num-partitions)")
    if not torch.cuda.is_available():
        raise RuntimeError("VGGT-Omega feature extraction requires CUDA")

    config = load_config(args.config)
    backbone = config["vision_backbone"]
    token_mode = str(backbone["cache_token_mode"])
    if token_mode not in {"camera_register_mean", "camera_register_tokens"}:
        raise ValueError("Unsupported vision_backbone.cache_token_mode")
    transform = VGGTOmegaResize(
        image_resolution=int(backbone["image_resolution"]),
        mode=str(backbone["resize_mode"]),
        patch_size=int(backbone["patch_size"]),
    )
    base = ActionWindowDataset(
        args.manifest,
        transform=transform,
        load_images=True,
        image_cache_size=args.image_cache_size,
    )
    manifest_sample_count = len(base)
    range_start, range_end = _partition_bounds(
        manifest_sample_count,
        args.num_partitions,
        args.partition_index,
        args.max_samples,
    )
    sample_count = range_end - range_start
    checkpoint = Path(backbone["checkpoint_path"])
    checkpoint_sha = file_sha256(checkpoint)
    configured_sha = backbone.get("checkpoint_sha256")
    if configured_sha is not None and checkpoint_sha != configured_sha:
        raise ValueError(f"Checkpoint SHA256 mismatch: {checkpoint_sha} != {configured_sha}")
    source_path = Path(backbone["source_path"])
    metadata: dict[str, object] = {
        "cache_type": "vggt_omega_camera_head_hidden_v1",
        "dataset_names": sorted({window.dataset_name for window in base.windows}),
        "manifest_schema_versions": sorted({window.schema_version for window in base.windows}),
        "sample_token_order_sha256": _sample_token_order_sha256(
            islice(base.windows, range_start, range_end), sample_count
        ),
        "manifest_sha256": file_sha256(args.manifest),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "third_party_commit": _git_commit(source_path),
        "image_resolution": int(backbone["image_resolution"]),
        "resize_mode": str(backbone["resize_mode"]),
        "patch_size": int(backbone["patch_size"]),
        "token_mode": token_mode,
        "camera_hidden_shape": [
            len(config["data"]["frame_offsets_s"]),
            int(backbone.get("feature_dim", 2048)),
        ],
        "register_hidden_mean_shape": [
            len(config["data"]["frame_offsets_s"]),
            int(backbone.get("feature_dim", 2048)),
        ],
        "pose_enc_shape": [len(config["data"]["frame_offsets_s"]), 9],
        "feature_dtype": "float16",
        "preprocessing_hash": stable_hash(
            {
                "image_resolution": int(backbone["image_resolution"]),
                "resize_mode": str(backbone["resize_mode"]),
                "patch_size": int(backbone["patch_size"]),
                "frame_offsets_s": config["data"]["frame_offsets_s"],
                "token_mode": str(backbone["cache_token_mode"]),
            }
        ),
        "expected_num_samples": sample_count,
    }
    if args.num_partitions > 1:
        metadata.update(
            {
                "manifest_num_samples": manifest_sample_count,
                "num_partitions": args.num_partitions,
                "partition_index": args.partition_index,
                "range_start": range_start,
                "range_end": range_end,
            }
        )
    if token_mode == "camera_register_tokens":
        metadata["register_hidden_shape"] = [
            len(config["data"]["frame_offsets_s"]),
            16,
            int(backbone.get("feature_dim", 2048)),
        ]

    args.output.mkdir(parents=True, exist_ok=True)
    index_path = args.output / "index.json"
    shards: list[dict[str, object]] = []
    count = 0
    previous_elapsed = 0.0
    if any(args.output.iterdir()):
        if not index_path.is_file():
            raise FileExistsError(
                f"Non-empty cache has no resumable index.json: {args.output}"
            )
        existing = json.loads(index_path.read_text(encoding="utf-8"))
        for key, expected in metadata.items():
            if key in {"expected_num_samples", "preprocessing_hash"} and key not in existing:
                continue
            if existing.get(key) != expected:
                raise ValueError(
                    f"Cannot resume cache: metadata mismatch for {key}: "
                    f"{existing.get(key)!r} != {expected!r}"
                )
        count = int(existing["num_samples"])
        if not 0 <= count <= sample_count:
            raise ValueError("Existing cache num_samples is outside the requested range")
        shards = list(existing.get("shards", []))
        _validate_shards(args.output, shards, count)
        previous_elapsed = float(existing.get("elapsed_seconds", 0.0))
        if existing.get("complete", count == sample_count):
            if count != sample_count:
                raise ValueError("Cache is marked complete but has the wrong sample count")
            print(f"Cache already complete and verified: {args.output}", flush=True)
            return
        if count == sample_count:
            existing["complete"] = True
            _atomic_write_index(index_path, existing)
            print(f"Finalized complete cache index: {args.output}", flush=True)
            return
        print(f"Resuming verified cache at sample {count}/{sample_count}", flush=True)

    dataset = Subset(base, range(range_start + count, range_end))
    loader_options: dict[str, object] = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        loader_options["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(
        dataset,
        **loader_options,
    )
    extractor = OmegaCameraFeatureExtractor(
        checkpoint_path=checkpoint,
        expected_sha256=None,
        freeze=True,
    ).cuda().eval()
    torch.cuda.reset_peak_memory_stats()
    pending: dict[str, list[torch.Tensor]] = {
        "camera_hidden": [],
        "register_hidden_mean": [],
        "pose_enc": [],
    }
    if token_mode == "camera_register_tokens":
        pending["register_hidden"] = []
    shard_index = len(shards)
    resume_count = count
    started = time.time()

    def write_progress(complete: bool) -> None:
        index = {
            **metadata,
            "num_samples": count,
            "complete": complete,
            "elapsed_seconds": previous_elapsed + time.time() - started,
            "shards": shards,
        }
        _atomic_write_index(index_path, index)

    def flush() -> None:
        nonlocal shard_index
        if not pending["camera_hidden"]:
            return
        tensors = {key: torch.cat(value, dim=0).contiguous() for key, value in pending.items()}
        start = count - tensors["camera_hidden"].shape[0]
        filename = f"features_{shard_index:05d}.safetensors"
        destination = args.output / filename
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        save_file(
            tensors,
            temporary,
            metadata={
                "cache_type": "vggt_omega_camera_head_hidden_v1",
                "checkpoint_sha256": checkpoint_sha,
                "token_mode": token_mode,
            },
        )
        os.replace(temporary, destination)
        shards.append(
            {
                "file": filename,
                "start": start,
                "end": count,
                "sha256": file_sha256(destination),
                "size_bytes": destination.stat().st_size,
            }
        )
        shard_index += 1
        for values in pending.values():
            values.clear()
        write_progress(complete=False)

    write_progress(complete=False)
    with torch.inference_mode():
        for batch in loader:
            output = extractor(batch["images"].cuda(non_blocking=True))
            actual_shapes = {
                "camera_hidden": tuple(output.camera_hidden.shape[1:]),
                "register_hidden_mean": tuple(output.register_hidden_mean.shape[1:]),
                "pose_enc": tuple(output.pose_enc.shape[1:]),
            }
            if token_mode == "camera_register_tokens":
                actual_shapes["register_hidden"] = tuple(output.register_hidden.shape[1:])
            for key, actual_shape in actual_shapes.items():
                expected_shape = tuple(metadata[f"{key}_shape"])
                if actual_shape != expected_shape:
                    raise ValueError(
                        f"Extractor output shape mismatch for {key}: "
                        f"{actual_shape} != {expected_shape}; check manifest frame count"
                    )
            pending["camera_hidden"].append(output.camera_hidden.half().cpu())
            pending["register_hidden_mean"].append(
                output.register_hidden_mean.half().cpu()
            )
            if token_mode == "camera_register_tokens":
                pending["register_hidden"].append(output.register_hidden.half().cpu())
            pending["pose_enc"].append(output.pose_enc.float().cpu())
            count += output.camera_hidden.shape[0]
            pending_count = sum(value.shape[0] for value in pending["camera_hidden"])
            if pending_count >= args.shard_size:
                flush()
            if count % 10 == 0 or count == sample_count:
                elapsed = time.time() - started
                rate = (count - resume_count) / max(elapsed, 1e-6)
                print(
                    f"cached={count}/{sample_count} "
                    f"global={range_start + count}/{manifest_sample_count} "
                    f"rate={rate:.2f} windows/s",
                    flush=True,
                )
        flush()
    write_progress(complete=True)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    print(
        f"peak_cuda_memory_gib={torch.cuda.max_memory_allocated() / 2**30:.2f}",
        flush=True,
    )
    print(json.dumps(index, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
