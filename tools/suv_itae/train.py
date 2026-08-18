#!/usr/bin/env python3
"""Fine-tune SUV's action expert on frozen ITAE action-token targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from PIL import Image
from safetensors.torch import load_file
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset
from torch.utils.tensorboard import SummaryWriter

from experiments.navsimv1.data.prompts import STAGEA_RGB_PROMPT
from experiments.suv_itae.runtime import (
    ACTION_SHAPE,
    ADAPTER_FORMAT,
    instantiate_suv_itae,
    load_suv_base_for_itae,
    predict_action_velocity,
    prepare_video_kv_cache,
    sample_action_tokens,
    shifted_sigma,
)
from experiments.suv_itae.visualization import (
    encode_mp4,
    load_navsim_frame,
    render_trajectory_clip,
)
from vision_action_tokenizer.config import load_config, seed_everything, stable_hash
from vision_action_tokenizer.data.planner_dataset import (
    PlannerTargetNormalizer,
    file_sha256,
)
from vision_action_tokenizer.models.factory import (
    build_tokenizer,
    tokenizer_state_from_checkpoint,
)


class IndexedActionDataset(Dataset[dict[str, Any]]):
    """Random-access JSONL reader paired with a safetensors teacher cache."""

    def __init__(
        self,
        manifest: Path,
        action_targets: Path,
        index_path: Path,
        *,
        image_size: tuple[int, int],
        crop_top_bottom: int,
        expected_teacher_sha256: str,
        expected_tokenizer_config_hash: str,
    ) -> None:
        self.manifest = manifest.resolve()
        self.action_targets_path = action_targets.resolve()
        self.image_size = tuple(map(int, image_size))
        self.crop_top_bottom = int(crop_top_bottom)
        self.offsets = np.load(index_path, mmap_mode="r")
        metadata_path = action_targets.with_suffix(".json")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing action-target metadata: {metadata_path}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("cache_type") != "v4_action_targets_v1":
            raise ValueError(f"Unsupported action-target cache: {action_targets}")
        if self.metadata.get("tokenizer_checkpoint_sha256") != expected_teacher_sha256:
            raise ValueError(
                f"Action targets use another tokenizer checkpoint: {action_targets}"
            )
        if self.metadata.get("tokenizer_config_hash") != expected_tokenizer_config_hash:
            raise ValueError(
                f"Action targets use another tokenizer config: {action_targets}"
            )
        self.sample_tokens = self.metadata.get("sample_tokens")
        if not isinstance(self.sample_tokens, list):
            raise ValueError("Action-target metadata has no sample_tokens list")
        self.tensors = load_file(str(action_targets), device="cpu")
        if tuple(self.tensors["action_tokens"].shape[1:]) != ACTION_SHAPE:
            raise ValueError(
                f"Expected cached targets [N,{ACTION_SHAPE}], "
                f"got {tuple(self.tensors['action_tokens'].shape)}"
            )
        if len(self.offsets) != len(self.sample_tokens) or len(self.offsets) != len(
            self.tensors["action_tokens"]
        ):
            raise ValueError("Manifest index and action-target cache lengths differ")
        self._handle = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_handle"] = None
        return state

    def __len__(self) -> int:
        return len(self.offsets)

    def _record(self, index: int) -> dict[str, Any]:
        if self._handle is None:
            self._handle = self.manifest.open("rb")
        self._handle.seek(int(self.offsets[index]))
        record = json.loads(self._handle.readline())
        if record["sample_token"] != self.sample_tokens[index]:
            raise ValueError(
                f"Target/manifest order mismatch at {index}: "
                f"{record['sample_token']} != {self.sample_tokens[index]}"
            )
        return record

    def _image(self, path: str) -> torch.Tensor:
        with Image.open(path) as source:
            image = source.convert("RGB")
            array = np.asarray(image, dtype=np.uint8)
        crop = self.crop_top_bottom
        if crop > 0 and array.shape[0] > 2 * crop:
            array = array[crop:-crop]
        height, width = self.image_size
        resized = Image.fromarray(array).resize(
            (width, height), resample=Image.Resampling.BILINEAR
        )
        value = np.asarray(resized, dtype=np.float32).copy()
        return torch.from_numpy(value).permute(2, 0, 1).div_(127.5).sub_(1.0)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self._record(index)
        return {
            "image": self._image(record["image_paths"][0]),
            "target": self.tensors["action_tokens"][index].float(),
            "sample_token": record["sample_token"],
            "scene_token": record["scene_token"],
            "group_token": record["group_token"],
            "anchor_timestamp_us": int(record["anchor_timestamp_us"]),
            "image_path": record["image_paths"][0],
            "trajectory": torch.as_tensor(record["trajectory"], dtype=torch.float32),
            "future_times": torch.as_tensor(
                record["future_times_s"], dtype=torch.float32
            ),
        }


class ActionFlowModule(nn.Module):
    def __init__(self, suv) -> None:
        super().__init__()
        self.suv = suv

    def forward(self, noisy, timestep, context, context_mask, cache):
        return predict_action_velocity(
            self.suv, noisy, timestep, context, context_mask, cache
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--overfit-samples", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--max-validation-batches", type=int)
    parser.add_argument("--qualitative-every-steps", type=int)
    parser.add_argument(
        "--num-qualitative-clips",
        "--num-qualitative-videos",
        dest="num_qualitative_clips",
        type=int,
    )
    parser.add_argument("--qualitative-clip-frames", type=int)
    parser.add_argument(
        "--last-n-blocks",
        type=int,
        help="Debug/memory override; omit to use train.last_n_blocks from config",
    )
    parser.add_argument(
        "--profile-batch-sizes",
        help="Comma-separated per-GPU batch sizes; run real optimizer steps and exit",
    )
    return parser.parse_args()


def distributed_context() -> tuple[int, int, int, torch.device]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if world > 1 and not dist.is_initialized():
        if device.type == "cuda":
            torch.cuda.set_device(device)
            dist.init_process_group("nccl", device_id=device)
        else:
            dist.init_process_group("gloo")
    return rank, world, local_rank, device


def build_offsets(manifest: Path, index_path: Path, rank: int, world: int) -> None:
    if rank == 0 and not index_path.is_file():
        index_path.parent.mkdir(parents=True, exist_ok=True)
        offsets: list[int] = []
        with manifest.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    offsets.append(offset)
        temporary = index_path.with_suffix(index_path.suffix + ".tmp.npy")
        np.save(temporary, np.asarray(offsets, dtype=np.int64))
        os.replace(temporary, index_path)
        print(f"Indexed {len(offsets):,} records: {index_path}", flush=True)
    if world > 1:
        dist.barrier()


def load_context(cache_dir: Path) -> tuple[torch.Tensor, torch.Tensor, Path]:
    digest = hashlib.sha256(STAGEA_RGB_PROMPT.encode("utf-8")).hexdigest()
    candidates = sorted(cache_dir.glob(f"{digest}.t5_len512.*.pt"))
    if not candidates:
        raise FileNotFoundError(
            "Static RGB text context is absent. Run the guide's `precompute --prompt-mode "
            f"static` command first. Expected under {cache_dir}"
        )
    payload = torch.load(candidates[-1], map_location="cpu", weights_only=False)
    context = payload["context"].float().clone()
    mask = payload["mask"].bool().clone()
    context[~mask] = 0
    mask.fill_(True)
    return context, mask, candidates[-1]


def configure_trainable(model, last_n_blocks: int | None) -> list[nn.Parameter]:
    model.requires_grad_(False)
    action = model.action_expert
    action.head.requires_grad_(True)
    if last_n_blocks is not None and int(last_n_blocks) <= 0:
        raise ValueError("train.last_n_blocks must be positive or null")
    # Full fine-tuning updates every projection and block. In the last-N mode,
    # keep the pre-DiT projections frozen too: the frozen prefix then runs
    # without an autograd graph and actually reduces activation memory.
    if last_n_blocks is None:
        action.action_encoder.requires_grad_(True)
        action.text_embedding.requires_grad_(True)
        action.time_embedding.requires_grad_(True)
        action.time_projection.requires_grad_(True)
    first = (
        0
        if last_n_blocks is None
        else max(len(action.blocks) - int(last_n_blocks), 0)
    )
    for index, block in enumerate(action.blocks):
        block.requires_grad_(index >= first)
    parameters = [parameter for parameter in action.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("No trainable action-expert parameters selected")
    return parameters


def flow_loss(
    module: nn.Module,
    clean: torch.Tensor,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    cache: dict[str, Any],
    *,
    scheduler_shift: float,
    num_timesteps: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    batch = clean.shape[0]
    uniform = torch.rand((batch,), device=clean.device, dtype=torch.float32)
    sigma = shifted_sigma(uniform, scheduler_shift).to(clean.dtype)
    noise = torch.randn_like(clean)
    weight = sigma.view(batch, 1, 1)
    noisy = (1.0 - weight) * clean + weight * noise
    target_velocity = noise - clean
    timestep = sigma * float(num_timesteps)
    prediction = module(noisy, timestep, context, context_mask, cache)
    loss = F.mse_loss(prediction.float(), target_velocity.float())
    return loss, {
        "loss": float(loss.detach()),
        "sigma": float(sigma.float().mean()),
        "prediction_rms": float(prediction.float().square().mean().sqrt().detach()),
    }


def select_qualitative_clips(
    dataset: IndexedActionDataset,
    *,
    available_count: int,
    requested_count: int,
    clip_frames: int,
    max_gap_s: float = 0.65,
) -> list[dict[str, Any]]:
    """Choose stable, genuinely contiguous NAVSIM clips without loading images."""
    count = min(max(int(available_count), 0), len(dataset))
    requested = max(int(requested_count), 0)
    if requested == 0:
        return []
    if clip_frames <= 0:
        raise ValueError("qualitative clip_frames must be positive")
    if max_gap_s <= 0:
        raise ValueError("qualitative max_gap_s must be positive")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    with dataset.manifest.open("r", encoding="utf-8") as handle:
        index = 0
        for line in handle:
            if not line.strip():
                continue
            if index >= count:
                break
            record = json.loads(line)
            trajectory = np.asarray(record["trajectory"], dtype=np.float64)
            grouped[(str(record["group_token"]), str(record["scene_token"]))].append(
                {
                    "index": index,
                    "timestamp": int(record["anchor_timestamp_us"]),
                    "sample_token": str(record["sample_token"]),
                    "distance": float(np.linalg.norm(trajectory[-1, :2])),
                    "lateral": float(np.max(np.abs(trajectory[:, 1]))),
                    "yaw": float(np.max(np.abs(trajectory[:, 2]))),
                }
            )
            index += 1

    candidates: dict[str, list[dict[str, Any]]] = {
        "straight": [],
        "turn": [],
        "dynamic": [],
    }
    for (group_token, scene_token), entries in grouped.items():
        entries.sort(key=lambda item: item["timestamp"])
        runs: list[list[dict[str, Any]]] = []
        for entry in entries:
            if not runs:
                runs.append([entry])
                continue
            gap_s = (entry["timestamp"] - runs[-1][-1]["timestamp"]) / 1e6
            if 0.0 < gap_s <= max_gap_s:
                runs[-1].append(entry)
            else:
                runs.append([entry])
        for run in runs:
            for start in range(0, len(run) - clip_frames + 1):
                clip = run[start : start + clip_frames]
                distance = float(np.mean([item["distance"] for item in clip]))
                lateral = float(np.mean([item["lateral"] for item in clip]))
                yaw = float(np.mean([item["yaw"] for item in clip]))
                base = {
                    "indices": [int(item["index"]) for item in clip],
                    "sample_tokens": [str(item["sample_token"]) for item in clip],
                    "group_token": group_token,
                    "scene_token": scene_token,
                    "start_timestamp_us": int(clip[0]["timestamp"]),
                    "end_timestamp_us": int(clip[-1]["timestamp"]),
                }
                if distance > 4.0 and (lateral > 1.5 or yaw > 0.2):
                    candidates["turn"].append(
                        {**base, "score": lateral + 10.0 * yaw + 0.02 * distance}
                    )
                if distance > 8.0 and lateral < 2.0 and yaw < 0.2:
                    candidates["straight"].append(
                        {**base, "score": distance - 2.0 * lateral - 8.0 * yaw}
                    )
                if distance > 4.0:
                    candidates["dynamic"].append({**base, "score": distance})

    selected: list[dict[str, Any]] = []
    used_groups: set[str] = set()
    used_indices: set[int] = set()
    for label in ("turn", "straight", "dynamic"):
        if len(selected) >= requested:
            break
        ranked = sorted(candidates[label], key=lambda item: item["score"], reverse=True)
        choice = next(
            (
                item
                for item in ranked
                if item["group_token"] not in used_groups
                and not used_indices.intersection(item["indices"])
            ),
            None,
        )
        if choice is None:
            choice = next(
                (
                    item
                    for item in ranked
                    if not used_indices.intersection(item["indices"])
                ),
                None,
            )
        if choice is not None:
            selected.append({**choice, "label": label})
            used_indices.update(choice["indices"])
            used_groups.add(str(choice["group_token"]))

    fallback = sorted(
        candidates["dynamic"], key=lambda item: item["score"], reverse=True
    )
    for choice in fallback:
        if len(selected) >= requested:
            break
        if used_indices.intersection(choice["indices"]):
            continue
        selected.append({**choice, "label": f"clip_{len(selected) + 1}"})
        used_indices.update(choice["indices"])
    if not selected:
        raise ValueError(
            f"No contiguous {clip_frames}-frame clip exists in {dataset.manifest}; "
            "reduce tensorboard.qualitative.clip_frames"
        )
    return selected


def main() -> None:
    args = parse_args()
    rank, world, local_rank, device = distributed_context()
    writer: SummaryWriter | None = None
    try:
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        paths = config["paths"]
        train_cfg = config["train"]
        required_files = [
            Path(paths[key])
            for key in (
                "base_checkpoint",
                "model_config",
                "tokenizer_config",
                "tokenizer_checkpoint",
                "train_manifest",
                "val_manifest",
                "train_action_targets",
                "val_action_targets",
            )
        ]
        required_files.extend(
            path.with_suffix(".json")
            for path in (
                Path(paths["train_action_targets"]),
                Path(paths["val_action_targets"]),
            )
        )
        missing_files = [str(path) for path in required_files if not path.is_file()]
        if missing_files:
            raise FileNotFoundError(
                "SUV-ITAE training inputs are incomplete:\n- "
                + "\n- ".join(missing_files)
            )
        context_cpu, context_mask_cpu, context_path = load_context(
            Path(paths["text_cache"])
        )
        output = (args.output or Path(paths["output"])).resolve()
        output.mkdir(parents=True, exist_ok=True)
        seed = int(config.get("seed", 42))
        seed_everything(seed + rank)

        index_dir = output / "manifest_indices"
        train_manifest = Path(paths["train_manifest"])
        val_manifest = Path(paths["val_manifest"])
        train_index = index_dir / "train.offsets.npy"
        val_index = index_dir / "validation.offsets.npy"
        build_offsets(train_manifest, train_index, rank, world)
        build_offsets(val_manifest, val_index, rank, world)

        data_cfg = config.get("data", {})
        tokenizer_config = load_config(paths["tokenizer_config"])
        dataset_args = {
            "image_size": tuple(data_cfg.get("image_size", [384, 640])),
            "crop_top_bottom": int(data_cfg.get("crop_top_bottom", 28)),
            "expected_teacher_sha256": file_sha256(paths["tokenizer_checkpoint"]),
            "expected_tokenizer_config_hash": stable_hash(tokenizer_config),
        }
        train_base = IndexedActionDataset(
            train_manifest, Path(paths["train_action_targets"]), train_index, **dataset_args
        )
        val_base = IndexedActionDataset(
            val_manifest, Path(paths["val_action_targets"]), val_index, **dataset_args
        )
        if args.overfit_samples:
            count = min(int(args.overfit_samples), len(train_base))
            train_dataset: Dataset = Subset(train_base, range(count))
            val_dataset: Dataset = Subset(train_base, range(count))
            fit_targets = train_base.tensors["action_tokens"][:count]
        else:
            train_dataset = train_base
            val_dataset = val_base
            fit_targets = train_base.tensors["action_tokens"]
        # Qualitative clips always come from the real validation split, including
        # overfit/debug runs, so temporal continuity is never fabricated.
        qualitative_base = val_base
        qualitative_count = len(val_base)

        tensorboard_cfg = config.get("tensorboard", {})
        qualitative_cfg = tensorboard_cfg.get("qualitative", {})
        qualitative_clips = (
            select_qualitative_clips(
                qualitative_base,
                available_count=qualitative_count,
                requested_count=int(
                    args.num_qualitative_clips
                    if args.num_qualitative_clips is not None
                    else qualitative_cfg.get("num_clips", 3)
                ),
                clip_frames=int(
                    args.qualitative_clip_frames
                    if args.qualitative_clip_frames is not None
                    else qualitative_cfg.get("clip_frames", 16)
                ),
                max_gap_s=float(qualitative_cfg.get("max_gap_s", 0.65)),
            )
            if rank == 0 and bool(qualitative_cfg.get("enabled", True))
            else []
        )
        if rank == 0:
            for clip in qualitative_clips:
                duration = (
                    int(clip["end_timestamp_us"])
                    - int(clip["start_timestamp_us"])
                ) / 1e6
                print(
                    f"Selected qualitative {clip['label']} clip: "
                    f"frames={len(clip['indices'])} duration={duration:.2f}s "
                    f"{clip['sample_tokens'][0]} -> {clip['sample_tokens'][-1]}",
                    flush=True,
                )

        normalizer = PlannerTargetNormalizer(
            ACTION_SHAPE, epsilon=float(train_cfg.get("normalizer_epsilon", 1e-4))
        )
        normalizer.fit(fit_targets.float())
        normalizer.to(device)

        dtype = torch.bfloat16 if train_cfg.get("precision", "bf16") == "bf16" else torch.float16
        model = instantiate_suv_itae(
            paths["model_config"], device=device, model_dtype=dtype
        )
        # The base checkpoint is ~12 GB. Stagger loads within each node so DDP
        # startup does not multiply host-RAM peaks by the local GPU count.
        local_world = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
        for load_turn in range(local_world):
            if local_rank == load_turn:
                load_suv_base_for_itae(model, paths["base_checkpoint"])
            if world > 1:
                dist.barrier()
        last_n_blocks = (
            args.last_n_blocks
            if args.last_n_blocks is not None
            else train_cfg.get("last_n_blocks")
        )
        parameters = configure_trainable(model, last_n_blocks)
        model.video_expert.eval()
        model.vae.eval()
        model.action_expert.train()
        model.mot.train()
        wrapper = ActionFlowModule(model).to(device)
        train_model: nn.Module = wrapper
        if world > 1:
            train_model = DistributedDataParallel(
                wrapper,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )

        batch_size = int(
            args.batch_size
            if args.batch_size is not None
            else train_cfg.get("batch_size", 1)
        )
        workers = int(
            args.num_workers
            if args.num_workers is not None
            else data_cfg.get("num_workers", 4)
        )
        train_sampler = (
            DistributedSampler(train_dataset, world, rank, shuffle=True, seed=seed)
            if world > 1
            else None
        )
        val_sampler = (
            DistributedSampler(val_dataset, world, rank, shuffle=False, drop_last=False)
            if world > 1
            else None
        )
        loader_args = dict(
            batch_size=batch_size,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=workers > 0,
        )
        train_loader = DataLoader(
            train_dataset,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            drop_last=True,
            **loader_args,
        )
        val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            sampler=val_sampler,
            drop_last=False,
            **loader_args,
        )
        optimizer = torch.optim.AdamW(
            parameters,
            lr=float(train_cfg["learning_rate"]),
            betas=tuple(map(float, train_cfg.get("betas", [0.9, 0.95]))),
            weight_decay=float(train_cfg.get("weight_decay", 0.01)),
            fused=device.type == "cuda",
        )
        max_steps = int(args.max_steps or train_cfg["max_steps"])
        warmup = int(train_cfg.get("warmup_steps", 500))

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return (step + 1) / max(warmup, 1)
            progress = (step - warmup) / max(max_steps - warmup, 1)
            minimum = float(train_cfg.get("min_lr_ratio", 0.1))
            return minimum + 0.5 * (1 - minimum) * (
                1 + math.cos(math.pi * min(max(progress, 0), 1))
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        global_step = 0
        epoch = 0
        best_val = float("inf")
        resume = args.resume
        if resume is None and not args.no_resume and (output / "last.pt").is_file():
            resume = output / "last.pt"
        if resume is not None:
            checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
            if checkpoint.get("format") != ADAPTER_FORMAT:
                raise ValueError(f"Not a SUV-ITAE adapter: {resume}")
            model.action_expert.load_state_dict(checkpoint["action_expert"], strict=True)
            normalizer.load_state_dict(checkpoint["normalizer"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            global_step = int(checkpoint["global_step"])
            epoch = int(checkpoint.get("epoch", 0))
            best_val = float(checkpoint.get("best_val_loss", best_val))
            if rank == 0:
                print(f"Resumed {resume} at step {global_step}", flush=True)

        resolved = {
            **config,
            "runtime": {
                "world_size": world,
                "train_samples": len(train_dataset),
                "validation_samples": len(val_dataset),
                "trainable_parameters": sum(p.numel() for p in parameters),
                "text_context": str(context_path),
                "action_shape": list(ACTION_SHAPE),
                "last_n_blocks": last_n_blocks,
                "qualitative_clips": [
                    {
                        "label": clip["label"],
                        "indices": clip["indices"],
                        "sample_tokens": clip["sample_tokens"],
                        "group_token": clip["group_token"],
                        "scene_token": clip["scene_token"],
                    }
                    for clip in qualitative_clips
                ],
            },
        }
        if rank == 0:
            (output / "resolved_config.json").write_text(
                json.dumps(resolved, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(
                f"SUV-ITAE world={world} samples={len(train_dataset):,} "
                f"trainable={sum(p.numel() for p in parameters):,}",
                flush=True,
            )

        def adapter_payload(include_optimizer: bool) -> dict[str, Any]:
            value = {
                "format": ADAPTER_FORMAT,
                "action_shape": ACTION_SHAPE,
                "action_expert": model.action_expert.state_dict(),
                "normalizer": normalizer.state_dict(),
                "global_step": global_step,
                "epoch": epoch,
                "best_val_loss": best_val,
                "config": resolved,
            }
            if include_optimizer:
                value.update(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict())
            return value

        accumulation = int(
            args.gradient_accumulation_steps
            if args.gradient_accumulation_steps is not None
            else train_cfg.get("gradient_accumulation_steps", 1)
        )
        eval_every = int(train_cfg.get("eval_every_steps", 1000))
        save_every = int(train_cfg.get("save_every_steps", eval_every))
        log_every = int(train_cfg.get("log_every_steps", 10))
        qualitative_every = int(
            args.qualitative_every_steps
            if args.qualitative_every_steps is not None
            else qualitative_cfg.get("every_steps", eval_every)
        )
        if min(
            accumulation, eval_every, save_every, log_every, qualitative_every
        ) <= 0:
            raise ValueError("accumulation/eval/save/log intervals must all be positive")
        max_val_batches = int(
            args.max_validation_batches
            if args.max_validation_batches is not None
            else config.get("validation", {}).get("max_batches", 100)
        )
        scheduler_shift = float(config.get("flow", {}).get("shift", 5.0))
        num_timesteps = int(config.get("flow", {}).get("num_timesteps", 1000))
        grad_clip = float(train_cfg.get("grad_clip_norm", 1.0))
        context = context_cpu.to(device=device, dtype=dtype).unsqueeze(0)
        context_mask = context_mask_cpu.to(device=device).unsqueeze(0)
        optimizer.zero_grad(set_to_none=True)
        started = time.time()
        starting_step = global_step

        if args.profile_batch_sizes:
            profile_sizes = [
                int(item.strip())
                for item in args.profile_batch_sizes.split(",")
                if item.strip()
            ]
            if not profile_sizes or min(profile_sizes) <= 0:
                raise ValueError("--profile-batch-sizes must contain positive integers")
            if profile_sizes != sorted(set(profile_sizes)):
                raise ValueError("--profile-batch-sizes must be unique and ascending")
            profile_sampler = (
                DistributedSampler(
                    train_dataset,
                    world,
                    rank,
                    shuffle=False,
                    drop_last=True,
                )
                if world > 1
                else None
            )
            profile_loader = DataLoader(
                train_dataset,
                batch_size=max(profile_sizes),
                shuffle=False,
                sampler=profile_sampler,
                num_workers=workers,
                pin_memory=True,
                drop_last=True,
            )
            try:
                profile_batch = next(iter(profile_loader))
            except StopIteration as error:
                raise ValueError(
                    "Profiling dataset is smaller than the largest requested batch"
                ) from error

            def profile_step(per_gpu_batch: int, *, measure: bool) -> dict[str, float]:
                images = profile_batch["image"][:per_gpu_batch].to(
                    device, dtype=dtype, non_blocking=True
                )
                clean = normalizer.normalize(
                    profile_batch["target"][:per_gpu_batch].to(
                        device, dtype=torch.float32, non_blocking=True
                    )
                ).to(dtype)
                batch_context = context.expand(per_gpu_batch, -1, -1)
                batch_mask = context_mask.expand(per_gpu_batch, -1)
                optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(device)
                    torch.cuda.synchronize(device)
                if world > 1:
                    dist.barrier()
                step_started = time.perf_counter()
                cache = prepare_video_kv_cache(
                    model, images, batch_context, batch_mask
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=dtype,
                    enabled=device.type == "cuda",
                ):
                    loss, _ = flow_loss(
                        train_model,
                        clean,
                        batch_context,
                        batch_mask,
                        cache,
                        scheduler_shift=float(config.get("flow", {}).get("shift", 5.0)),
                        num_timesteps=int(
                            config.get("flow", {}).get("num_timesteps", 1000)
                        ),
                    )
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    parameters, float(train_cfg.get("grad_clip_norm", 1.0))
                )
                optimizer.step()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - step_started
                stats = torch.tensor(
                    [
                        elapsed,
                        (
                            torch.cuda.max_memory_allocated(device) / 2**30
                            if device.type == "cuda"
                            else 0.0
                        ),
                        (
                            torch.cuda.max_memory_reserved(device) / 2**30
                            if device.type == "cuda"
                            else 0.0
                        ),
                    ],
                    device=device,
                    dtype=torch.float64,
                )
                if world > 1:
                    dist.all_reduce(stats, op=dist.ReduceOp.MAX)
                del cache, images, clean, loss
                if measure:
                    return {
                        "seconds": float(stats[0]),
                        "allocated_gib": float(stats[1]),
                        "reserved_gib": float(stats[2]),
                        "samples_per_second": per_gpu_batch * world / float(stats[0]),
                    }
                return {}

            # Warm up kernels and allocate AdamW state before measuring.
            profile_step(profile_sizes[0], measure=False)
            if rank == 0:
                print(
                    "batch_profile columns: per_gpu global seconds samples/s "
                    "peak_allocated_GiB peak_reserved_GiB",
                    flush=True,
                )
            for profile_size in profile_sizes:
                result = profile_step(profile_size, measure=True)
                if rank == 0:
                    print(
                        f"batch_profile per_gpu={profile_size} "
                        f"global={profile_size * world} "
                        f"seconds={result['seconds']:.3f} "
                        f"samples/s={result['samples_per_second']:.3f} "
                        f"peak_allocated_GiB={result['allocated_gib']:.2f} "
                        f"peak_reserved_GiB={result['reserved_gib']:.2f}",
                        flush=True,
                    )
            return

        if rank == 0 and bool(tensorboard_cfg.get("enabled", True)):
            configured_log_dir = tensorboard_cfg.get("log_dir")
            if configured_log_dir is None:
                log_dir = output / "tensorboard"
            else:
                log_dir = Path(configured_log_dir)
                if not log_dir.is_absolute():
                    log_dir = output / log_dir
            writer_kwargs: dict[str, Any] = {
                "log_dir": str(log_dir),
                "flush_secs": int(tensorboard_cfg.get("flush_secs", 30)),
            }
            if resume is not None and global_step > 0:
                writer_kwargs["purge_step"] = global_step + 1
            writer = SummaryWriter(**writer_kwargs)
            writer.add_text(
                "run/config",
                "```json\n" + json.dumps(resolved, indent=2, ensure_ascii=False) + "\n```",
                global_step,
            )
            print(f"TensorBoard log_dir={log_dir}", flush=True)

        action_decoder: nn.Module | None = None

        @torch.no_grad()
        def qualitative_validation() -> list[dict[str, Any]]:
            nonlocal action_decoder
            if rank != 0 or not qualitative_clips:
                return []
            if action_decoder is None:
                action_decoder = build_tokenizer(tokenizer_config).to(device).eval()
                tokenizer_checkpoint = torch.load(
                    paths["tokenizer_checkpoint"],
                    map_location="cpu",
                    weights_only=False,
                )
                action_decoder.load_state_dict(
                    tokenizer_state_from_checkpoint(tokenizer_checkpoint), strict=True
                )
                action_decoder.requires_grad_(False)

            navsim_root = Path(
                qualitative_cfg.get(
                    "navsim_data_root",
                    "/inspire/hdd/global_public/public_datas/NAVSIM",
                )
            )
            navsim_split = str(qualitative_cfg.get("navsim_split", "trainval"))
            inference_steps = int(qualitative_cfg.get("inference_steps", 10))
            inference_batch_size = int(
                qualitative_cfg.get("inference_batch_size", 8)
            )
            fps = float(qualitative_cfg.get("fps", 2))
            frame_size = tuple(
                map(int, qualitative_cfg.get("frame_size", [432, 768]))
            )
            if (
                inference_steps <= 0
                or inference_batch_size <= 0
                or fps <= 0
                or len(frame_size) != 2
            ):
                raise ValueError(
                    "TensorBoard qualitative inference/batch/fps/frame_size are invalid"
                )
            step_dir = output / "qualitative" / f"step_{global_step:08d}"
            reports: list[dict[str, Any]] = []
            model.eval()
            configured_seed = int(qualitative_cfg.get("seed", 2026))
            for clip in qualitative_clips:
                label = str(clip["label"])
                items = [qualitative_base[int(index)] for index in clip["indices"]]
                predictions: list[torch.Tensor] = []
                for batch_start in range(0, len(items), inference_batch_size):
                    batch_items = items[
                        batch_start : batch_start + inference_batch_size
                    ]
                    images = torch.stack([item["image"] for item in batch_items]).to(
                        device=device, dtype=dtype, non_blocking=True
                    )
                    batch_context = context.expand(len(batch_items), -1, -1)
                    batch_mask = context_mask.expand(len(batch_items), -1)
                    seeds = [
                        (
                            configured_seed
                            + int.from_bytes(
                                hashlib.sha256(
                                    str(item["sample_token"]).encode("utf-8")
                                ).digest()[:8],
                                byteorder="big",
                            )
                        )
                        % (2**63 - 1)
                        for item in batch_items
                    ]
                    normalized_tokens = sample_action_tokens(
                        model,
                        images,
                        batch_context,
                        batch_mask,
                        num_inference_steps=inference_steps,
                        sigma_shift=scheduler_shift,
                        seeds=seeds,
                    )
                    action_tokens = normalizer.denormalize(
                        normalized_tokens.float()
                    )
                    future_times = torch.stack(
                        [item["future_times"] for item in batch_items]
                    ).to(device)
                    predictions.append(
                        action_decoder.decode(action_tokens, future_times).float().cpu()
                    )
                prediction = torch.cat(predictions, dim=0)
                if not torch.isfinite(prediction).all():
                    raise FloatingPointError(
                        f"Non-finite qualitative prediction in {label} clip"
                    )

                raw_frames = []
                for item in items:
                    sample_token = str(item["sample_token"])
                    raw_frame = load_navsim_frame(
                        navsim_root,
                        navsim_split,
                        str(item["group_token"]),
                        sample_token,
                    )
                    expected_image = Path(
                        str(raw_frame["cams"]["CAM_F0"]["data_path"])
                    ).name
                    if expected_image != Path(str(item["image_path"])).name:
                        raise ValueError(
                            f"Manifest/raw CAM_F0 mismatch for {sample_token}: "
                            f"{item['image_path']} != {expected_image}"
                        )
                    raw_frames.append(raw_frame)

                video, metrics = render_trajectory_clip(
                    [str(item["image_path"]) for item in items],
                    raw_frames,
                    torch.stack([item["trajectory"] for item in items]).numpy(),
                    prediction.numpy(),
                    torch.stack([item["future_times"] for item in items]).numpy(),
                    sample_tokens=[str(item["sample_token"]) for item in items],
                    anchor_timestamps_us=[
                        int(item["anchor_timestamp_us"]) for item in items
                    ],
                    clip_label=label,
                    global_step=global_step,
                    frame_size=frame_size,
                )
                start_token = str(items[0]["sample_token"])
                end_token = str(items[-1]["sample_token"])
                safe_start = re.sub(r"[^A-Za-z0-9_.-]+", "_", start_token)
                safe_end = re.sub(r"[^A-Za-z0-9_.-]+", "_", end_token)
                if bool(qualitative_cfg.get("save_mp4", True)):
                    encode_mp4(
                        video,
                        step_dir / f"{label}_{safe_start}_to_{safe_end}.mp4",
                        fps,
                    )
                if writer is not None:
                    tag = f"qualitative_clip/{label}"
                    writer.add_video(tag, video.unsqueeze(0), global_step, fps=fps)
                    for name in (
                        "mean_ade_m",
                        "mean_fde_m",
                        "max_fde_m",
                        "mean_yaw_mae_deg",
                    ):
                        writer.add_scalar(
                            f"qualitative_clip/{label}/{name}",
                            metrics[name],
                            global_step,
                        )
                reports.append(
                    {
                        "label": label,
                        "dataset_indices": clip["indices"],
                        "start_sample_token": start_token,
                        "end_sample_token": end_token,
                        "group_token": clip["group_token"],
                        "scene_token": clip["scene_token"],
                        **metrics,
                    }
                )
                print(
                    f"qualitative clip step={global_step} {label} "
                    f"frames={len(items)} ADE={metrics['mean_ade_m']:.3f}m "
                    f"FDE={metrics['mean_fde_m']:.3f}m",
                    flush=True,
                )
            if reports:
                mean_ade = float(
                    np.mean([item["mean_ade_m"] for item in reports])
                )
                mean_fde = float(
                    np.mean([item["mean_fde_m"] for item in reports])
                )
                if writer is not None:
                    writer.add_scalar(
                        "qualitative_clip/mean_ade_m", mean_ade, global_step
                    )
                    writer.add_scalar(
                        "qualitative_clip/mean_fde_m", mean_fde, global_step
                    )
                    writer.flush()
                step_dir.mkdir(parents=True, exist_ok=True)
                (step_dir / "metrics.json").write_text(
                    json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8"
                )
            model.video_expert.eval()
            model.vae.eval()
            model.action_expert.train()
            model.mot.train()
            return reports

        def validate() -> float:
            model.action_expert.eval()
            total = torch.zeros(2, device=device, dtype=torch.float64)
            with torch.no_grad():
                for batch_index, batch in enumerate(val_loader):
                    if batch_index >= max_val_batches:
                        break
                    images = batch["image"].to(device, dtype=dtype, non_blocking=True)
                    clean = normalizer.normalize(
                        batch["target"].to(device, dtype=torch.float32, non_blocking=True)
                    ).to(dtype)
                    batch_context = context.expand(len(images), -1, -1)
                    batch_mask = context_mask.expand(len(images), -1)
                    cache = prepare_video_kv_cache(
                        model, images, batch_context, batch_mask
                    )
                    loss, _ = flow_loss(
                        wrapper, clean, batch_context, batch_mask, cache,
                        scheduler_shift=scheduler_shift, num_timesteps=num_timesteps,
                    )
                    total[0] += float(loss) * len(images)
                    total[1] += len(images)
            if world > 1:
                dist.all_reduce(total)
            model.action_expert.train()
            return float(total[0] / total[1].clamp_min(1))

        micro_step = 0
        while global_step < max_steps:
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            for batch in train_loader:
                if global_step >= max_steps:
                    break
                images = batch["image"].to(device, dtype=dtype, non_blocking=True)
                clean = normalizer.normalize(
                    batch["target"].to(device, dtype=torch.float32, non_blocking=True)
                ).to(dtype)
                batch_context = context.expand(len(images), -1, -1)
                batch_mask = context_mask.expand(len(images), -1)
                cache = prepare_video_kv_cache(
                    model, images, batch_context, batch_mask
                )
                should_sync = (micro_step + 1) % accumulation == 0
                sync_context = (
                    train_model.no_sync()  # type: ignore[attr-defined]
                    if world > 1 and not should_sync
                    else torch.enable_grad()
                )
                with sync_context:
                    with torch.autocast(
                        device_type=device.type,
                        dtype=dtype,
                        enabled=device.type == "cuda",
                    ):
                        loss, metrics = flow_loss(
                            train_model, clean, batch_context, batch_mask, cache,
                            scheduler_shift=scheduler_shift, num_timesteps=num_timesteps,
                        )
                        (loss / accumulation).backward()
                micro_step += 1
                del cache
                if not should_sync:
                    continue
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                if global_step % log_every == 0:
                    log_values = torch.tensor(
                        [
                            metrics["loss"],
                            metrics["sigma"],
                            metrics["prediction_rms"],
                        ],
                        device=device,
                        dtype=torch.float64,
                    )
                    if world > 1:
                        dist.all_reduce(log_values)
                        log_values.div_(world)
                if rank == 0 and global_step % log_every == 0:
                    elapsed = max(time.time() - started, 1e-6)
                    completed = max(global_step - starting_step, 1)
                    samples_per_second = (
                        completed * batch_size * world * accumulation / elapsed
                    )
                    current_lr = scheduler.get_last_lr()[0]
                    logged_loss, logged_sigma, logged_prediction_rms = map(
                        float, log_values
                    )
                    print(
                        f"step={global_step}/{max_steps} loss={logged_loss:.6f} "
                        f"sigma={logged_sigma:.3f} lr={current_lr:.2e} "
                        f"grad={float(grad_norm):.3f} samples/s={samples_per_second:.2f}",
                        flush=True,
                    )
                    if writer is not None:
                        writer.add_scalar("train/flow_loss", logged_loss, global_step)
                        writer.add_scalar("train/sigma", logged_sigma, global_step)
                        writer.add_scalar(
                            "train/prediction_rms", logged_prediction_rms, global_step
                        )
                        writer.add_scalar("train/learning_rate", current_lr, global_step)
                        writer.add_scalar(
                            "train/gradient_norm", float(grad_norm), global_step
                        )
                        writer.add_scalar(
                            "train/samples_per_second", samples_per_second, global_step
                        )
                        writer.add_scalar("train/epoch", epoch, global_step)
                        if device.type == "cuda":
                            writer.add_scalar(
                                "system/gpu_memory_allocated_gib",
                                torch.cuda.memory_allocated(device) / 2**30,
                                global_step,
                            )
                            writer.add_scalar(
                                "system/gpu_memory_reserved_gib",
                                torch.cuda.memory_reserved(device) / 2**30,
                                global_step,
                            )
                if global_step % eval_every == 0 or global_step == max_steps:
                    value = validate()
                    if rank == 0:
                        print(f"validation step={global_step} flow_loss={value:.6f}", flush=True)
                        if writer is not None:
                            writer.add_scalar("validation/flow_loss", value, global_step)
                        if value < best_val:
                            best_val = value
                            torch.save(adapter_payload(False), output / "best.pt")
                        if bool(qualitative_cfg.get("enabled", True)) and (
                            global_step % qualitative_every == 0
                        ):
                            qualitative_validation()
                if rank == 0 and (
                    global_step % save_every == 0 or global_step == max_steps
                ):
                    torch.save(adapter_payload(True), output / "last.pt")
                if world > 1 and (
                    global_step % eval_every == 0
                    or global_step % save_every == 0
                    or global_step == max_steps
                ):
                    dist.barrier()
            epoch += 1
    finally:
        if writer is not None:
            writer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
