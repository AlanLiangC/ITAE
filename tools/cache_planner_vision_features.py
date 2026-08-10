#!/usr/bin/env python3
"""Cache configured current-frame planner vision features in verified shards."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset, Subset

from vision_action_tokenizer.config import load_config, stable_hash
from vision_action_tokenizer.data.manifest import load_manifest
from vision_action_tokenizer.data.planner_dataset import file_sha256
from vision_action_tokenizer.models.vision_backbones import (
    build_planner_vision_backbone,
    build_planner_vision_transform,
)


class CurrentImageDataset(Dataset[dict[str, torch.Tensor | str]]):
    def __init__(self, manifest: Path, transform: object, limit: int | None) -> None:
        windows = load_manifest(manifest)
        self.windows = windows if limit is None else windows[:limit]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        window = self.windows[index]
        with Image.open(window.image_paths[0]) as image:
            tensor = self.transform(image)  # type: ignore[operator]
        return {"image": tensor, "sample_token": window.sample_token}


def _atomic_write(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _git_commit(path: str | Path | None) -> str | None:
    if path is None or not Path(path).is_dir():
        return None
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.shard_size <= 0:
        raise ValueError("batch-size and shard-size must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("max-samples must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Planner vision feature caching requires CUDA")

    config = load_config(args.config)
    vision_config = config["vision_condition"]
    if not bool(vision_config.get("freeze", True)):
        raise ValueError("Feature caching requires a frozen vision backbone")
    transform = build_planner_vision_transform(config)
    dataset = CurrentImageDataset(args.manifest, transform, args.max_samples)
    extractor = build_planner_vision_backbone(config).cuda().eval()
    backbone_metadata = extractor.preprocessing_metadata()
    feature_dtype = str(vision_config.get("cache_dtype", "float16"))
    if feature_dtype not in {"float16", "bfloat16", "float32"}:
        raise ValueError("vision_condition.cache_dtype is unsupported")
    output_dtype = getattr(torch, feature_dtype)
    metadata: dict[str, object] = {
        "cache_type": "planner_vision_condition_v1",
        "manifest_sha256": file_sha256(args.manifest),
        "backbone_type": str(vision_config["type"]),
        "model_name": str(vision_config.get("model_name", vision_config["type"])),
        "checkpoint_sha256": vision_config.get("checkpoint_sha256"),
        "source_commit": _git_commit(vision_config.get("source_path")),
        "preprocessing": backbone_metadata,
        "preprocessing_hash": stable_hash(backbone_metadata),
        "condition_shape": [extractor.output_token_count, extractor.feature_dim],
        "condition_grid": backbone_metadata.get("pool_grid"),
        "feature_dtype": feature_dtype,
        "expected_num_samples": len(dataset),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    index_path = args.output / "index.json"
    shards: list[dict[str, object]] = []
    count = 0
    previous_elapsed = 0.0
    if any(args.output.iterdir()):
        if not index_path.is_file():
            raise FileExistsError(
                f"Non-empty planner cache has no resumable index: {args.output}"
            )
        existing = json.loads(index_path.read_text(encoding="utf-8"))
        for key, expected in metadata.items():
            if existing.get(key) != expected:
                raise ValueError(
                    f"Cannot resume planner cache: {key} differs "
                    f"({existing.get(key)!r} != {expected!r})"
                )
        shards = list(existing.get("shards", []))
        count = int(existing.get("num_samples", 0))
        cursor = 0
        for shard in shards:
            if int(shard["start"]) != cursor or int(shard["end"]) <= cursor:
                raise ValueError("Existing planner cache shards are not contiguous")
            shard_path = args.output / str(shard["file"])
            if not shard_path.is_file() or file_sha256(shard_path) != shard["sha256"]:
                raise ValueError(f"Existing planner cache shard is corrupt: {shard_path}")
            cursor = int(shard["end"])
        if cursor != count or not 0 <= count <= len(dataset):
            raise ValueError("Existing planner cache index has an invalid sample range")
        if bool(existing.get("complete", False)):
            if count != len(dataset):
                raise ValueError("Complete planner cache has the wrong sample count")
            print(f"Planner vision cache already complete: {args.output}")
            return
        previous_elapsed = float(existing.get("elapsed_seconds", 0.0))
        print(f"Resuming planner vision cache at {count}/{len(dataset)}", flush=True)

    loader = DataLoader(
        Subset(dataset, range(count, len(dataset))),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    pending_tokens: list[torch.Tensor] = []
    pending_masks: list[torch.Tensor] = []
    shard_index = len(shards)
    resume_count = count
    started = time.time()

    def write_index(complete: bool) -> None:
        _atomic_write(
            index_path,
            {
                **metadata,
                "num_samples": count,
                "complete": complete,
                "elapsed_seconds": previous_elapsed + time.time() - started,
                "shards": shards,
            },
        )

    def flush() -> None:
        nonlocal shard_index
        if not pending_tokens:
            return
        tokens = torch.cat(pending_tokens).contiguous()
        masks = torch.cat(pending_masks).contiguous()
        start = count - len(tokens)
        filename = f"features_{shard_index:05d}.safetensors"
        destination = args.output / filename
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        save_file(
            {"condition_tokens": tokens, "condition_mask": masks},
            str(temporary),
            metadata={"cache_type": "planner_vision_condition_v1"},
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
        pending_tokens.clear()
        pending_masks.clear()
        shard_index += 1
        write_index(complete=False)

    write_index(complete=False)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.inference_mode():
        for batch in loader:
            with torch.autocast("cuda", dtype=amp_dtype):
                condition = extractor(batch["image"].cuda(non_blocking=True))
            pending_tokens.append(condition.tokens.to(dtype=output_dtype).cpu())
            pending_masks.append(condition.token_mask.cpu())
            count += condition.tokens.shape[0]
            if sum(len(value) for value in pending_tokens) >= args.shard_size:
                flush()
            if count % 100 == 0 or count == len(dataset):
                print(
                    f"cached={count}/{len(dataset)} "
                    f"rate={(count - resume_count) / max(time.time() - started, 1e-6):.2f} "
                    "images/s",
                    flush=True,
                )
        flush()
    write_index(complete=True)
    print(f"Planner vision cache complete: {args.output}")


if __name__ == "__main__":
    main()
