#!/usr/bin/env python3
"""Cache configurable single- or multi-frame planner vision features."""

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


class ConditionImageDataset(Dataset[dict[str, torch.Tensor | str]]):
    def __init__(
        self,
        manifest: Path,
        transform: object,
        limit: int | None,
        frame_indices: list[int],
        expected_offsets_s: list[float],
        max_frame_time_error_s: float,
        causal_tolerance_s: float,
        export_ego_motion: bool,
    ) -> None:
        windows = load_manifest(manifest)
        self.windows = windows if limit is None else windows[:limit]
        self.transform = transform
        self.frame_indices = frame_indices
        self.expected_offsets_s = expected_offsets_s
        self.max_frame_time_error_s = max_frame_time_error_s
        self.causal_tolerance_s = causal_tolerance_s
        self.export_ego_motion = export_ego_motion
        if not frame_indices or len(frame_indices) != len(expected_offsets_s):
            raise ValueError("frame_indices and frame_offsets_s must be non-empty and aligned")
        if len(set(frame_indices)) != len(frame_indices):
            raise ValueError("Planner condition frame_indices must be unique")
        if any(
            right <= left
            for left, right in zip(
                expected_offsets_s, expected_offsets_s[1:], strict=False
            )
        ):
            raise ValueError("Planner condition frame_offsets_s must be strictly increasing")
        if any(offset > 0 for offset in expected_offsets_s):
            raise ValueError("Planner condition frames cannot request future timestamps")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        window = self.windows[index]
        images = []
        actual_times = []
        for frame_index, expected_time in zip(
            self.frame_indices, self.expected_offsets_s, strict=True
        ):
            if not 0 <= frame_index < len(window.image_paths):
                raise IndexError(
                    f"Condition frame index {frame_index} is invalid for "
                    f"sample {window.sample_token}"
                )
            actual_time = float(window.frame_times_s[frame_index])
            if abs(actual_time - expected_time) > self.max_frame_time_error_s + 1e-9:
                raise ValueError(
                    f"Condition frame time mismatch for {window.sample_token}: "
                    f"actual={actual_time:.3f}s expected={expected_time:.3f}s"
                )
            if actual_time > self.causal_tolerance_s + 1e-9:
                raise ValueError(
                    f"Condition frame leaks future data for {window.sample_token}: "
                    f"t={actual_time:.3f}s"
                )
            with Image.open(window.image_paths[frame_index]) as image:
                images.append(self.transform(image))  # type: ignore[operator]
            actual_times.append(actual_time)
        sample: dict[str, torch.Tensor | str] = {
            "images": torch.stack(images),
            "frame_times": torch.tensor(actual_times, dtype=torch.float32),
            "sample_token": window.sample_token,
        }
        if self.export_ego_motion:
            if window.ego_motion_states is None or window.ego_motion_times_s is None:
                raise ValueError(
                    f"Manifest sample {window.sample_token} has no ego-motion condition"
                )
            states = torch.tensor(window.ego_motion_states, dtype=torch.float32)
            state_times = torch.tensor(window.ego_motion_times_s, dtype=torch.float32)
            if states.shape != (len(self.frame_indices), 6):
                raise ValueError("Ego-motion state must have shape [condition_frames,6]")
            if state_times.shape != (len(self.frame_indices),):
                raise ValueError("Ego-motion times must align with condition frames")
            if not torch.isfinite(states).all() or not torch.isfinite(state_times).all():
                raise ValueError("Ego-motion condition contains non-finite values")
            if torch.any(state_times > self.causal_tolerance_s + 1e-6):
                raise ValueError("Ego-motion condition leaks a future state")
            sample["ego_motion_states"] = states
            sample["ego_motion_times"] = state_times
        return sample


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
    explicit_frame_setting = "frame_indices" in vision_config
    frame_indices = list(map(int, vision_config.get("frame_indices", [0])))
    frame_offsets_s = list(
        map(float, vision_config.get("frame_offsets_s", [0.0]))
    )
    max_frame_time_error_s = float(
        vision_config.get("max_frame_time_error_s", 0.25)
    )
    causal_tolerance_s = float(vision_config.get("causal_tolerance_s", 0.25))
    current_frame_index = int(vision_config.get("current_frame_index", frame_indices[-1]))
    if current_frame_index not in frame_indices:
        raise ValueError("current_frame_index must be one of vision_condition.frame_indices")
    current_position = frame_indices.index(current_frame_index)
    if abs(frame_offsets_s[current_position]) > max_frame_time_error_s:
        raise ValueError("current_frame_index must select the condition frame at t=0")
    ego_config = config.get("ego_motion_condition", {})
    export_ego_motion = bool(ego_config.get("enabled", False))
    ego_state_dim = int(ego_config.get("state_dim", 6))
    ego_tokens = int(ego_config.get("num_tokens", len(frame_indices)))
    if export_ego_motion and (ego_state_dim != 6 or ego_tokens != len(frame_indices)):
        raise ValueError(
            "Ego-motion export requires state_dim=6 and one token per condition frame"
        )
    transform = build_planner_vision_transform(config)
    dataset = ConditionImageDataset(
        args.manifest,
        transform,
        args.max_samples,
        frame_indices,
        frame_offsets_s,
        max_frame_time_error_s,
        causal_tolerance_s,
        export_ego_motion,
    )
    extractor = build_planner_vision_backbone(config).cuda().eval()
    backbone_metadata = extractor.preprocessing_metadata()
    feature_dtype = str(vision_config.get("cache_dtype", "float16"))
    if feature_dtype not in {"float16", "bfloat16", "float32"}:
        raise ValueError("vision_condition.cache_dtype is unsupported")
    output_dtype = getattr(torch, feature_dtype)
    cache_type = (
        "planner_vision_condition_v3"
        if export_ego_motion
        else (
            "planner_vision_condition_v2"
            if explicit_frame_setting
            else "planner_vision_condition_v1"
        )
    )
    tokens_per_frame = extractor.output_token_count
    metadata: dict[str, object] = {
        "cache_type": cache_type,
        "manifest_sha256": file_sha256(args.manifest),
        "backbone_type": str(vision_config["type"]),
        "model_name": str(vision_config.get("model_name", vision_config["type"])),
        "checkpoint_sha256": vision_config.get("checkpoint_sha256"),
        "source_commit": _git_commit(vision_config.get("source_path")),
        "preprocessing": backbone_metadata,
        "preprocessing_hash": stable_hash(backbone_metadata),
        "condition_shape": [
            len(frame_indices) * tokens_per_frame,
            extractor.feature_dim,
        ],
        "condition_grid": backbone_metadata.get("pool_grid"),
        "feature_dtype": feature_dtype,
        "expected_num_samples": len(dataset),
    }
    if cache_type in {"planner_vision_condition_v2", "planner_vision_condition_v3"}:
        metadata.update(
            {
                "frame_indices": frame_indices,
                "current_frame_index": current_frame_index,
                "condition_frame_offsets_s": frame_offsets_s,
                "num_condition_frames": len(frame_indices),
                "tokens_per_frame": tokens_per_frame,
                "condition_time_shape": [len(frame_indices) * tokens_per_frame],
                "max_frame_time_error_s": max_frame_time_error_s,
                "causal_tolerance_s": causal_tolerance_s,
            }
        )
    if cache_type == "planner_vision_condition_v3":
        metadata.update(
            {
                "ego_motion_shape": [ego_tokens, ego_state_dim],
                "ego_motion_state_fields": list(ego_config["state_fields"]),
                "ego_motion_scales": list(map(float, ego_config["scales"])),
            }
        )
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
    pending_times: list[torch.Tensor] = []
    pending_ego_states: list[torch.Tensor] = []
    pending_ego_times: list[torch.Tensor] = []
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
        condition_times = torch.cat(pending_times).contiguous()
        start = count - len(tokens)
        filename = f"features_{shard_index:05d}.safetensors"
        destination = args.output / filename
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        tensors = {"condition_tokens": tokens, "condition_mask": masks}
        if cache_type in {"planner_vision_condition_v2", "planner_vision_condition_v3"}:
            tensors["condition_times"] = condition_times
        if cache_type == "planner_vision_condition_v3":
            tensors["ego_motion_states"] = torch.cat(pending_ego_states).contiguous()
            tensors["ego_motion_times"] = torch.cat(pending_ego_times).contiguous()
        save_file(
            tensors,
            str(temporary),
            metadata={"cache_type": cache_type},
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
        pending_times.clear()
        pending_ego_states.clear()
        pending_ego_times.clear()
        shard_index += 1
        write_index(complete=False)

    write_index(complete=False)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.inference_mode():
        for batch in loader:
            images = batch["images"]
            batch_size, frame_count = images.shape[:2]
            with torch.autocast("cuda", dtype=amp_dtype):
                condition = extractor(
                    images.flatten(0, 1).cuda(non_blocking=True)
                )
            token_count, feature_dim = condition.tokens.shape[1:]
            tokens = condition.tokens.reshape(
                batch_size, frame_count * token_count, feature_dim
            )
            masks = condition.token_mask.reshape(batch_size, frame_count * token_count)
            times = batch["frame_times"].unsqueeze(-1).expand(
                -1, -1, token_count
            ).reshape(batch_size, frame_count * token_count)
            pending_tokens.append(tokens.to(dtype=output_dtype).cpu())
            pending_masks.append(masks.cpu())
            pending_times.append(times.cpu())
            if cache_type == "planner_vision_condition_v3":
                pending_ego_states.append(batch["ego_motion_states"].cpu())
                pending_ego_times.append(batch["ego_motion_times"].cpu())
            count += batch_size
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
