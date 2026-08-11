#!/usr/bin/env python3
"""Train the VGGT-Omega visual action tokenizer from online or cached features."""

from __future__ import annotations

import argparse
import hashlib
import math
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset

from vision_action_tokenizer.config import (
    load_config,
    resolve_resume_checkpoint,
    seed_everything,
)
from vision_action_tokenizer.data.dataset import (
    ActionWindowDataset,
    CachedVGGTOmegaFeatureDataset,
    MultiSourceActionDataset,
    VGGTOmegaResize,
)
from vision_action_tokenizer.data.manifest import (
    WindowRecord,
    manifest_group_tokens,
)
from vision_action_tokenizer.data.sampler import DeterministicDistributedWeightedSampler
from vision_action_tokenizer.data.trajectory import shift_se2_reference_point
from vision_action_tokenizer.distributed import cleanup_distributed, initialize_distributed
from vision_action_tokenizer.losses import LossConfig, TokenizerLoss
from vision_action_tokenizer.models.factory import build_training_model
from vision_action_tokenizer.trainer import (
    TokenizerTrainer,
    is_visual_residual_parameter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--val-manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", type=Path, help="Explicit checkpoint override")
    resume_group.add_argument(
        "--no-resume", action="store_true", help="Ignore config/auto-resume and start fresh"
    )
    parser.add_argument("--overfit-samples", type=int)
    parser.add_argument(
        "--overfit-on-train",
        action="store_true",
        help="Diagnostic only: evaluate the selected training subset as validation",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        help="Optional subset of config data.sources (for source-only ablations)",
    )
    parser.add_argument("--epochs", type=int, help="Override train.epochs for smoke tests")
    parser.add_argument("--train-feature-cache", type=Path)
    parser.add_argument("--val-feature-cache", type=Path)
    return parser.parse_args()


def cache_for_split(config: dict, split: str):
    cache_config = config["data"].get("feature_cache")
    return cache_config.get(split) if isinstance(cache_config, dict) else cache_config


def _source_specs(
    args: argparse.Namespace, config: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    configured = config["data"].get("sources")
    if configured:
        if args.train_manifest is not None or args.val_manifest is not None:
            raise ValueError("Manifest CLI overrides are only valid for single-source configs")
        if args.train_feature_cache is not None or args.val_feature_cache is not None:
            raise ValueError("Feature-cache CLI overrides are only valid for single-source configs")
        result = {}
        for name, values in configured.items():
            reference_offset = tuple(
                map(float, values.get("reference_point_offset_m", [0.0, 0.0]))
            )
            if len(reference_offset) != 2 or not all(math.isfinite(v) for v in reference_offset):
                raise ValueError(
                    f"{name}.reference_point_offset_m must contain two finite values"
                )
            result[str(name)] = {
                "train_manifest": Path(values["train_manifest"]),
                "val_manifest": Path(values["val_manifest"]),
                "train_cache": Path(values["train_feature_cache"])
                if values.get("train_feature_cache")
                else None,
                "val_cache": Path(values["val_feature_cache"])
                if values.get("val_feature_cache")
                else None,
                "reference_point_offset_m": reference_offset,
            }
        return result
    if args.train_manifest is None or args.val_manifest is None:
        raise ValueError(
            "Single-source configs require --train-manifest and --val-manifest"
        )
    if (args.train_feature_cache is None) != (args.val_feature_cache is None):
        raise ValueError("Train and val feature-cache overrides must be provided together")
    train_cache = args.train_feature_cache or cache_for_split(config, "train")
    val_cache = args.val_feature_cache or cache_for_split(config, "val")
    return {
        "nuscenes": {
            "train_manifest": args.train_manifest,
            "val_manifest": args.val_manifest,
            "train_cache": Path(train_cache) if train_cache else None,
            "val_cache": Path(val_cache) if val_cache else None,
            "reference_point_offset_m": (0.0, 0.0),
        }
    }


def make_dataset(
    manifest: Path,
    cache: Path | None,
    config: dict,
    overfit_samples: int | None = None,
    reference_point_offset_m: tuple[float, float] = (0.0, 0.0),
):
    base = ActionWindowDataset(
        manifest,
        transform=VGGTOmegaResize(
            image_resolution=int(config["vision_backbone"]["image_resolution"]),
            mode=str(config["vision_backbone"]["resize_mode"]),
            patch_size=int(config["vision_backbone"]["patch_size"]),
        ),
        load_images=cache is None,
        reference_point_offset_m=reference_point_offset_m,
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
    windows = [
        replace(
            window,
            trajectory=shift_se2_reference_point(
                window.trajectory, reference_point_offset_m
            ).tolist(),
        )
        for window in base.windows
    ]
    if overfit_samples is not None:
        count = min(overfit_samples, len(dataset))
        dataset = Subset(dataset, range(count))
        windows = windows[:count]
    return dataset, windows


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


def make_weighted_sampler(
    windows: list[WindowRecord],
    source_names: list[str],
    config: dict,
    dataset_size: int,
    world_size: int,
    rank: int,
) -> DeterministicDistributedWeightedSampler | None:
    if len(windows) != dataset_size or len(source_names) != dataset_size:
        raise ValueError("Sampler windows/source names do not match the dataset")
    sampling = config["data"].get("sampling", {})
    strategy = str(sampling.get("strategy", "proportional"))
    if strategy not in {"proportional", "balanced", "weighted"}:
        raise ValueError("data.sampling.strategy must be proportional, balanced or weighted")
    balancing = config["train"].get("motion_balancing", {})
    motion_enabled = bool(balancing.get("enabled", False))
    source_counts = Counter(source_names)
    configured = balancing.get("bucket_weights", {}) if motion_enabled else {
        "stationary": 1.0,
        "straight_slow": 1.0,
        "straight_fast": 1.0,
        "turn": 1.0,
    }
    bucket_names = {"stationary", "straight_slow", "straight_fast", "turn"}
    if set(configured) != bucket_names:
        raise ValueError(
            "train.motion_balancing.bucket_weights must define exactly "
            f"{sorted(bucket_names)}"
        )
    bucket_weights = {name: float(value) for name, value in configured.items()}
    if any(value <= 0 for value in bucket_weights.values()):
        raise ValueError("Motion bucket weights must all be positive")
    configured_trends = balancing.get("speed_trend_weights") if motion_enabled else None
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

    buckets = [motion_bucket(window) for window in windows]
    trends = [
        speed_trend(
            window,
            steps_per_interval=int(config["action_tokenizer"]["steps_per_token"]),
            threshold_mps=trend_threshold,
        )
        for window in windows
    ]
    desired_source_weights = sampling.get("source_weights", {})
    if strategy == "balanced":
        desired_source_weights = {name: 1.0 for name in source_counts}
    elif strategy == "proportional":
        desired_source_weights = {name: float(count) for name, count in source_counts.items()}
    if set(desired_source_weights) != set(source_counts):
        raise ValueError(
            "data.sampling.source_weights must define exactly the configured sources"
        )
    if any(float(value) <= 0 for value in desired_source_weights.values()):
        raise ValueError("Source sampling weights must be positive")
    motion_factors = [
        bucket_weights[bucket] * trend_weights[trend]
        for bucket, trend in zip(buckets, trends, strict=True)
    ]
    source_motion_totals = {
        name: sum(
            factor
            for source, factor in zip(source_names, motion_factors, strict=True)
            if source == name
        )
        for name in source_counts
    }
    sample_weights = torch.tensor(
        [
            float(desired_source_weights[source])
            * factor
            / source_motion_totals[source]
            for source, factor in zip(source_names, motion_factors, strict=True)
        ],
        dtype=torch.double,
    )
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
    weighted_source_totals = {
        name: sum(
            float(weight)
            for source, weight in zip(source_names, sample_weights, strict=True)
            if source == name
        )
        for name in source_counts
    }
    source_normalization = sum(weighted_source_totals.values())
    expected_sources = {
        name: value / source_normalization
        for name, value in sorted(weighted_source_totals.items())
    }
    weighted_trend_totals = {
        name: sum(
            float(weight)
            for trend, weight in zip(trends, sample_weights, strict=True)
            if trend == name
        )
        for name in sorted(trend_names)
    }
    expected_trends = {
        name: value / normalization for name, value in weighted_trend_totals.items()
    }
    config.setdefault("data_runtime", {})["sampling_expectation"] = {
        "source": expected_sources,
        "motion": expected,
        "speed_trend": expected_trends,
    }
    print(
        f"Weighted sampling sources={dict(source_counts)} motion={dict(counts)} "
        f"trends={dict(trend_counts)} expected_sources={expected_sources} "
        f"expected_motion_fractions={expected}",
        flush=True,
    )
    if strategy == "proportional" and not motion_enabled:
        return None
    requested_samples = sampling.get("samples_per_epoch")
    num_samples = dataset_size if requested_samples is None else int(requested_samples)
    return DeterministicDistributedWeightedSampler(
        sample_weights,
        num_samples=num_samples,
        seed=int(config["seed"]),
        num_replicas=world_size,
        rank=rank,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.sources:
        configured_sources = config["data"].get("sources")
        if not configured_sources:
            raise ValueError("--sources requires a multi-source config")
        requested = list(dict.fromkeys(args.sources))
        missing = set(requested) - set(configured_sources)
        if missing:
            raise ValueError(f"Unknown --sources: {sorted(missing)}")
        config["data"]["sources"] = {
            name: configured_sources[name] for name in requested
        }
        sampling_weights = config["data"].get("sampling", {}).get("source_weights")
        if sampling_weights is not None:
            config["data"]["sampling"]["source_weights"] = {
                name: sampling_weights[name] for name in requested
            }
    if args.epochs is not None:
        if args.epochs <= 0:
            raise ValueError("--epochs must be positive")
        config["train"]["epochs"] = args.epochs
    source_specs = _source_specs(args, config)
    if args.overfit_on_train:
        if args.overfit_samples is None:
            raise ValueError("--overfit-on-train requires --overfit-samples")
        for spec in source_specs.values():
            spec["val_manifest"] = spec["train_manifest"]
            spec["val_cache"] = spec["train_cache"]
    runtime_sources: dict[str, dict[str, Any]] = {}
    for source_name, spec in source_specs.items():
        train_manifest = spec["train_manifest"]
        val_manifest = spec["val_manifest"]
        assert isinstance(train_manifest, Path) and isinstance(val_manifest, Path)
        if not train_manifest.is_file() or not val_manifest.is_file():
            raise FileNotFoundError(
                f"Missing {source_name} manifest: {train_manifest} or {val_manifest}"
            )
        overlap = manifest_group_tokens(train_manifest) & manifest_group_tokens(val_manifest)
        if overlap and not args.overfit_on_train:
            raise ValueError(
                f"{source_name} train/val group leakage: {len(overlap)} groups; "
                f"examples={sorted(overlap)[:5]}"
            )
        runtime_sources[source_name] = {
            "train_manifest_sha256": hashlib.sha256(train_manifest.read_bytes()).hexdigest(),
            "val_manifest_sha256": hashlib.sha256(val_manifest.read_bytes()).hexdigest(),
            "train_cache": str(spec["train_cache"]) if spec["train_cache"] else None,
            "val_cache": str(spec["val_cache"]) if spec["val_cache"] else None,
            "reference_point_offset_m": list(spec["reference_point_offset_m"]),
        }
    config["data_runtime"] = {"sources": runtime_sources}
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
    train_sources = {}
    val_sources = {}
    train_windows: list[WindowRecord] = []
    train_source_names: list[str] = []
    cache_modes: set[bool] = set()
    for source_name, spec in source_specs.items():
        train_manifest = spec["train_manifest"]
        val_manifest = spec["val_manifest"]
        assert isinstance(train_manifest, Path) and isinstance(val_manifest, Path)
        train_cache = spec["train_cache"]
        val_cache = spec["val_cache"]
        assert train_cache is None or isinstance(train_cache, Path)
        assert val_cache is None or isinstance(val_cache, Path)
        if (train_cache is None) != (val_cache is None):
            raise ValueError(f"{source_name} train/val cache mode must match")
        cache_modes.add(train_cache is not None)
        train_dataset_part, source_windows = make_dataset(
            train_manifest,
            train_cache,
            config,
            args.overfit_samples,
            reference_point_offset_m=spec["reference_point_offset_m"],
        )
        val_dataset_part, _ = make_dataset(
            val_manifest,
            val_cache,
            config,
            args.overfit_samples,
            reference_point_offset_m=spec["reference_point_offset_m"],
        )
        train_sources[source_name] = train_dataset_part
        val_sources[source_name] = val_dataset_part
        train_windows.extend(source_windows)
        train_source_names.extend([source_name] * len(source_windows))
        runtime_sources[source_name]["num_train_windows"] = len(source_windows)
        runtime_sources[source_name]["num_val_windows"] = len(val_dataset_part)
    if len(cache_modes) != 1:
        raise ValueError("All sources must consistently use online features or caches")
    train_dataset = MultiSourceActionDataset(train_sources)
    val_dataset = MultiSourceActionDataset(val_sources)
    train_sampler = make_weighted_sampler(
        train_windows,
        train_source_names,
        config,
        len(train_dataset),
        context.world_size,
        context.rank,
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
    cached = cache_modes.pop()
    if not cached and int(config["train"]["batch_size"]) != 1:
        raise ValueError("Online VGGT-Omega training requires train.batch_size=1 on this GPU")
    model = build_training_model(config, cached=cached)
    if bool(config["train"].get("freeze_base", False)):
        residual_count = 0
        for name, parameter in model.named_parameters():
            residual = is_visual_residual_parameter(name)
            parameter.requires_grad_(residual)
            residual_count += parameter.numel() if residual else 0
        if residual_count == 0:
            raise ValueError("train.freeze_base requires visual residual parameters")
        print(f"Permanently froze motion path; residual parameters={residual_count:,}")
    model = model.to(context.device)
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
        if is_visual_residual_parameter(name)
    ]
    residual_parameter_ids = {id(parameter) for parameter in residual_parameters}
    base_parameters = [
        parameter for parameter in parameters if id(parameter) not in residual_parameter_ids
    ]
    if residual_parameters and not base_parameters:
        optimizer_groups = [
            {
                "params": residual_parameters,
                "lr": learning_rate,
                "name": "visual_residual",
            }
        ]
    elif residual_parameters and base_learning_rate_scale != 1.0:
        optimizer_groups = [
            {
                "params": base_parameters,
                "lr": learning_rate * base_learning_rate_scale,
                "name": "base",
            },
            {
                "params": residual_parameters,
                "lr": learning_rate,
                "name": "visual_residual",
            },
        ]
        print(
            "Optimizer parameter groups: "
            f"base={sum(p.numel() for p in base_parameters):,} "
            f"lr={learning_rate * base_learning_rate_scale:g}; "
            f"visual_residual={sum(p.numel() for p in residual_parameters):,} "
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
        if config["data"].get("sources") and initial_checkpoint not in (None, ""):
            raise ValueError("Joint scratch training forbids train.initial_checkpoint")
        if initial_checkpoint not in (None, ""):
            evaluate_initial = bool(
                config["train"].get("evaluate_initial_checkpoint", False)
            )
            trainer.load_initial_weights(
                initial_checkpoint,
                allowed_missing_prefixes=(
                    "tokenizer.encoder.register_residual_",
                    "tokenizer.visual_residual_",
                ),
                inherit_best_metric=not evaluate_initial,
            )
            if evaluate_initial:
                trainer.establish_initial_baseline(val_loader)
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
