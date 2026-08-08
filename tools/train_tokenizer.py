#!/usr/bin/env python3
"""Train the VGGT-Omega visual action tokenizer from online or cached features."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset, WeightedRandomSampler

from vision_action_tokenizer.config import (
    load_config,
    resolve_resume_checkpoint,
    seed_everything,
)
from vision_action_tokenizer.data.dataset import (
    CachedVGGTOmegaFeatureDataset,
    NuScenesWindowDataset,
    VGGTOmegaResize,
)
from vision_action_tokenizer.data.manifest import WindowRecord, load_manifest, manifest_scene_tokens
from vision_action_tokenizer.distributed import cleanup_distributed, initialize_distributed
from vision_action_tokenizer.losses import LossConfig, TokenizerLoss
from vision_action_tokenizer.models.factory import build_training_model
from vision_action_tokenizer.trainer import TokenizerTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--val-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", type=Path, help="Explicit checkpoint override")
    resume_group.add_argument(
        "--no-resume", action="store_true", help="Ignore config/auto-resume and start fresh"
    )
    parser.add_argument("--overfit-samples", type=int)
    parser.add_argument("--epochs", type=int, help="Override train.epochs for smoke tests")
    parser.add_argument("--train-feature-cache", type=Path)
    parser.add_argument("--val-feature-cache", type=Path)
    return parser.parse_args()


def cache_for_split(config: dict, split: str):
    cache_config = config["data"].get("feature_cache")
    return cache_config.get(split) if isinstance(cache_config, dict) else cache_config


def make_dataset(manifest: Path, config: dict, split: str, overfit_samples: int | None = None):
    cache = cache_for_split(config, split)
    base = NuScenesWindowDataset(
        manifest,
        transform=VGGTOmegaResize(
            image_resolution=int(config["vision_backbone"]["image_resolution"]),
            mode=str(config["vision_backbone"]["resize_mode"]),
            patch_size=int(config["vision_backbone"]["patch_size"]),
        ),
        load_images=cache is None,
    )
    dataset = (
        base
        if cache is None
        else CachedVGGTOmegaFeatureDataset(
            base,
            cache,
            manifest_path=manifest,
            expected_metadata={
                "checkpoint_sha256": config["vision_backbone"].get(
                    "checkpoint_sha256"
                ),
                "image_resolution": int(config["vision_backbone"]["image_resolution"]),
                "resize_mode": str(config["vision_backbone"]["resize_mode"]),
                "token_mode": str(config["vision_backbone"]["cache_token_mode"]),
            },
        )
    )
    if overfit_samples is not None:
        dataset = Subset(dataset, range(min(overfit_samples, len(dataset))))
    return dataset


def motion_bucket(window: WindowRecord) -> str:
    """Use the same coarse motion groups reported by the evaluator."""
    endpoint = torch.as_tensor(window.trajectory[-1])
    distance = float(torch.linalg.vector_norm(endpoint[:2]))
    lateral = abs(float(endpoint[1]))
    yaw = abs(float(endpoint[2]))
    if distance < 2.0:
        return "stationary"
    if lateral > 2.0 or yaw > 0.15:
        return "turn"
    return "straight_slow" if distance < 20.0 else "straight_fast"


def speed_trend(
    window: WindowRecord,
    steps_per_interval: int,
    threshold_mps: float,
) -> str:
    """Classify acceleration from the first and last one-second mean speeds."""
    if motion_bucket(window) == "stationary":
        return "stationary"
    trajectory = torch.as_tensor(window.trajectory, dtype=torch.float32)
    times = torch.as_tensor(window.future_times_s, dtype=torch.float32)
    if len(trajectory) < 2 * steps_per_interval or len(times) != len(trajectory):
        raise ValueError("Trajectory is too short for configured speed-trend intervals")
    previous_xy = torch.cat([torch.zeros_like(trajectory[:1, :2]), trajectory[:-1, :2]])
    delta_t = torch.diff(torch.cat([torch.zeros_like(times[:1]), times]))
    if torch.any(delta_t <= 0):
        raise ValueError("Manifest future_times_s must be strictly increasing")
    speed = torch.linalg.vector_norm(trajectory[:, :2] - previous_xy, dim=-1) / delta_t
    speed_change = float(
        speed[-steps_per_interval:].mean() - speed[:steps_per_interval].mean()
    )
    if speed_change > threshold_mps:
        return "accelerating"
    if speed_change < -threshold_mps:
        return "decelerating"
    return "steady"


def make_motion_sampler(
    manifest: Path,
    config: dict,
    dataset_size: int,
    overfit_samples: int | None,
    world_size: int,
) -> WeightedRandomSampler | None:
    balancing = config["train"].get("motion_balancing", {})
    if not bool(balancing.get("enabled", False)):
        return None
    if world_size > 1:
        raise ValueError(
            "train.motion_balancing currently supports one process; disable it for DDP"
        )
    configured = balancing.get("bucket_weights", {})
    bucket_names = {"stationary", "straight_slow", "straight_fast", "turn"}
    if set(configured) != bucket_names:
        raise ValueError(
            "train.motion_balancing.bucket_weights must define exactly "
            f"{sorted(bucket_names)}"
        )
    bucket_weights = {name: float(value) for name, value in configured.items()}
    if any(value <= 0 for value in bucket_weights.values()):
        raise ValueError("Motion bucket weights must all be positive")
    configured_trends = balancing.get("speed_trend_weights")
    trend_names = {"stationary", "steady", "accelerating", "decelerating"}
    if configured_trends is None:
        trend_weights = {name: 1.0 for name in trend_names}
    else:
        if set(configured_trends) != trend_names:
            raise ValueError(
                "train.motion_balancing.speed_trend_weights must define exactly "
                f"{sorted(trend_names)}"
            )
        trend_weights = {
            name: float(value) for name, value in configured_trends.items()
        }
        if any(value <= 0 for value in trend_weights.values()):
            raise ValueError("Speed-trend weights must all be positive")
    trend_threshold = float(
        balancing.get("speed_trend_threshold_mps", 0.5)
    )
    if trend_threshold <= 0:
        raise ValueError("speed_trend_threshold_mps must be positive")

    windows = load_manifest(manifest)
    if overfit_samples is not None:
        windows = windows[:overfit_samples]
    if len(windows) != dataset_size:
        raise ValueError(
            f"Sampler manifest/dataset mismatch: {len(windows)} != {dataset_size}"
        )
    buckets = [motion_bucket(window) for window in windows]
    trends = [
        speed_trend(
            window,
            steps_per_interval=int(config["action_tokenizer"]["steps_per_token"]),
            threshold_mps=trend_threshold,
        )
        for window in windows
    ]
    sample_weights = torch.tensor(
        [
            bucket_weights[bucket] * trend_weights[trend]
            for bucket, trend in zip(buckets, trends, strict=True)
        ],
        dtype=torch.double,
    )
    generator = torch.Generator().manual_seed(int(config["seed"]))
    counts = Counter(buckets)
    trend_counts = Counter(trends)
    weighted_counts = {
        name: sum(
            float(weight)
            for bucket, weight in zip(buckets, sample_weights, strict=True)
            if bucket == name
        )
        for name in sorted(bucket_names)
    }
    normalization = sum(weighted_counts.values())
    expected = {
        name: weighted_counts[name] / normalization for name in sorted(bucket_names)
    }
    print(
        f"Motion-balanced sampling counts={dict(counts)} trends={dict(trend_counts)} "
        f"expected_fractions={expected}",
        flush=True,
    )
    return WeightedRandomSampler(
        sample_weights,
        num_samples=dataset_size,
        replacement=True,
        generator=generator,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        if args.epochs <= 0:
            raise ValueError("--epochs must be positive")
        config["train"]["epochs"] = args.epochs
    if (args.train_feature_cache is None) != (args.val_feature_cache is None):
        raise ValueError("Train and val feature-cache overrides must be provided together")
    if args.train_feature_cache is not None:
        config["data"]["feature_cache"] = {
            "train": str(args.train_feature_cache),
            "val": str(args.val_feature_cache),
        }
    resume_checkpoint = resolve_resume_checkpoint(
        config,
        args.output,
        cli_resume=args.resume,
        no_resume=args.no_resume,
    )
    context = initialize_distributed()
    if context.is_main:
        if resume_checkpoint is None:
            print(f"Starting fresh; no resume checkpoint selected in {args.output}", flush=True)
        else:
            print(f"Resuming training from {resume_checkpoint}", flush=True)
    seed_everything(int(config["seed"]) + context.rank)
    train_scenes = manifest_scene_tokens(args.train_manifest)
    val_scenes = manifest_scene_tokens(args.val_manifest)
    overlap = train_scenes & val_scenes
    if overlap:
        examples = sorted(overlap)[:5]
        raise ValueError(
            f"Train/val scene leakage: {len(overlap)} overlapping scenes; examples={examples}"
        )
    train_dataset = make_dataset(args.train_manifest, config, "train", args.overfit_samples)
    val_dataset = make_dataset(args.val_manifest, config, "val", args.overfit_samples)
    train_sampler = make_motion_sampler(
        args.train_manifest,
        config,
        len(train_dataset),
        args.overfit_samples,
        context.world_size,
    )
    if train_sampler is None and context.world_size > 1:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if context.world_size > 1 else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=context.device.type == "cuda",
        persistent_workers=int(config["data"]["num_workers"]) > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=False,
        sampler=val_sampler,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=context.device.type == "cuda",
        persistent_workers=int(config["data"]["num_workers"]) > 0,
    )
    train_cached = cache_for_split(config, "train") is not None
    val_cached = cache_for_split(config, "val") is not None
    if train_cached != val_cached:
        raise ValueError("Train and val must both use online VGGT-Omega or feature caches")
    cached = train_cached
    if not cached and int(config["train"]["batch_size"]) != 1:
        raise ValueError("Online VGGT-Omega training requires train.batch_size=1 on this GPU")
    model = build_training_model(config, cached=cached).to(context.device)
    if context.world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[context.local_rank] if context.device.type == "cuda" else None,
        )
    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    parameters = [parameter for _, parameter in named_parameters]
    total_params = sum(parameter.numel() for parameter in parameters)
    print(f"Total trainable parameters: {total_params:,}")
    learning_rate = float(config["train"]["learning_rate"])
    base_learning_rate_scale = float(
        config["train"].get("base_learning_rate_scale", 1.0)
    )
    if base_learning_rate_scale <= 0:
        raise ValueError("train.base_learning_rate_scale must be positive")
    residual_parameters = [
        parameter
        for name, parameter in named_parameters
        if "register_residual_" in name
    ]
    residual_parameter_ids = {id(parameter) for parameter in residual_parameters}
    base_parameters = [
        parameter for parameter in parameters if id(parameter) not in residual_parameter_ids
    ]
    if residual_parameters and base_learning_rate_scale != 1.0:
        optimizer_groups = [
            {
                "params": base_parameters,
                "lr": learning_rate * base_learning_rate_scale,
                "name": "base",
            },
            {
                "params": residual_parameters,
                "lr": learning_rate,
                "name": "register_residual",
            },
        ]
        print(
            "Optimizer parameter groups: "
            f"base={sum(p.numel() for p in base_parameters):,} "
            f"lr={learning_rate * base_learning_rate_scale:g}; "
            f"register_residual={sum(p.numel() for p in residual_parameters):,} "
            f"lr={learning_rate:g}",
            flush=True,
        )
    else:
        optimizer_groups = [{"params": parameters, "lr": learning_rate, "name": "all"}]
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        weight_decay=float(config["train"]["weight_decay"]),
    )
    total_steps = max(1, len(train_loader) * int(config["train"]["epochs"]))
    warmup = int(config["train"]["warmup_steps"])

    def schedule(step: int) -> float:
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    trainer = TokenizerTrainer(
        model,
        TokenizerLoss(
            LossConfig(
                **config["loss"],
                steps_per_token=int(config["action_tokenizer"]["steps_per_token"]),
            )
        ).to(context.device),
        optimizer,
        scheduler,
        context,
        args.output,
        precision=str(config["train"]["precision"]),
        grad_clip_norm=float(config["train"]["grad_clip_norm"]),
        config=config,
    )
    if resume_checkpoint is not None:
        start_epoch = trainer.load_checkpoint(resume_checkpoint)
    else:
        start_epoch = 0
        initial_checkpoint = config["train"].get("initial_checkpoint")
        if initial_checkpoint not in (None, ""):
            trainer.load_initial_weights(
                initial_checkpoint,
                allowed_missing_prefixes=("tokenizer.encoder.register_residual_",),
            )
            if context.is_main:
                trainer.save_checkpoint("initial.pt", epoch=-1)
                trainer.save_checkpoint("best.pt", epoch=-1)
    try:
        trainer.fit(
            train_loader,
            val_loader,
            epochs=int(config["train"]["epochs"]),
            log_every=int(config["train"]["log_every"]),
            start_epoch=start_epoch,
        )
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
