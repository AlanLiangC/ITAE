#!/usr/bin/env python3
"""Cache frozen VGGT-Omega CameraHead hidden features in strict shards."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Subset

from vision_action_tokenizer.config import load_config, stable_hash
from vision_action_tokenizer.data.dataset import NuScenesWindowDataset, VGGTOmegaResize
from vision_action_tokenizer.models.vggt_omega import OmegaCameraFeatureExtractor, file_sha256


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
    args = parser.parse_args()
    if args.shard_size <= 0:
        raise ValueError("--shard-size must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("VGGT-Omega feature extraction requires CUDA")

    config = load_config(args.config)
    backbone = config["vision_backbone"]
    transform = VGGTOmegaResize(
        image_resolution=int(backbone["image_resolution"]),
        mode=str(backbone["resize_mode"]),
        patch_size=int(backbone["patch_size"]),
    )
    base = NuScenesWindowDataset(args.manifest, transform=transform, load_images=True)
    sample_count = len(base) if args.max_samples is None else min(args.max_samples, len(base))
    checkpoint = Path(backbone["checkpoint_path"])
    checkpoint_sha = file_sha256(checkpoint)
    configured_sha = backbone.get("checkpoint_sha256")
    if configured_sha is not None and checkpoint_sha != configured_sha:
        raise ValueError(f"Checkpoint SHA256 mismatch: {checkpoint_sha} != {configured_sha}")
    source_path = Path(backbone["source_path"])
    metadata: dict[str, object] = {
        "cache_type": "vggt_omega_camera_head_hidden_v1",
        "manifest_sha256": file_sha256(args.manifest),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "third_party_commit": _git_commit(source_path),
        "image_resolution": int(backbone["image_resolution"]),
        "resize_mode": str(backbone["resize_mode"]),
        "patch_size": int(backbone["patch_size"]),
        "token_mode": str(backbone["cache_token_mode"]),
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

    dataset = Subset(base, range(count, sample_count))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    extractor = OmegaCameraFeatureExtractor(
        checkpoint_path=checkpoint,
        expected_sha256=None,
        freeze=True,
    ).cuda().eval()
    pending: dict[str, list[torch.Tensor]] = {
        "camera_hidden": [],
        "register_hidden_mean": [],
        "pose_enc": [],
    }
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
                "token_mode": str(backbone["cache_token_mode"]),
            },
        )
        os.replace(temporary, destination)
        shards.append(
            {
                "file": filename,
                "start": start,
                "end": count,
                "sha256": file_sha256(destination),
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
            pending["camera_hidden"].append(output.camera_hidden.half().cpu())
            pending["register_hidden_mean"].append(
                output.register_hidden_mean.half().cpu()
            )
            pending["pose_enc"].append(output.pose_enc.float().cpu())
            count += output.camera_hidden.shape[0]
            pending_count = sum(value.shape[0] for value in pending["camera_hidden"])
            if pending_count >= args.shard_size:
                flush()
            if count % 10 == 0 or count == sample_count:
                elapsed = time.time() - started
                rate = (count - resume_count) / max(elapsed, 1e-6)
                print(
                    f"cached={count}/{sample_count} rate={rate:.2f} windows/s",
                    flush=True,
                )
        flush()
    write_progress(complete=True)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    print(json.dumps(index, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
