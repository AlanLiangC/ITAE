#!/usr/bin/env python3
"""Train the visual action tokenizer with online or cached PE features."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset

from vision_action_tokenizer.config import load_config, seed_everything
from vision_action_tokenizer.data.dataset import CachedPEFeatureDataset, NuScenesWindowDataset
from vision_action_tokenizer.data.manifest import manifest_scene_tokens
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
    parser.add_argument("--resume", type=Path)
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
        image_size=int(config["data"]["image_size"]),
        load_images=cache is None,
    )
    dataset = (
        base
        if cache is None
        else CachedPEFeatureDataset(
            base,
            cache,
            manifest_path=manifest,
            expected_metadata={
                "model_name": config["pe"]["model_name"],
                "checkpoint_path": config["pe"].get("checkpoint_path"),
                "layer_idx": config["pe"].get("layer_idx"),
                "pool_size": config["pe"]["pool_size"],
            },
        )
    )
    if overfit_samples is not None:
        dataset = Subset(dataset, range(min(overfit_samples, len(dataset))))
    return dataset


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
    context = initialize_distributed()
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
    train_sampler = (
        DistributedSampler(train_dataset, shuffle=True) if context.world_size > 1 else None
    )
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
        raise ValueError("Train and val must both use online PE or both use feature caches")
    cached = train_cached
    model = build_training_model(config, cached=cached).to(context.device)
    if context.world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[context.local_rank] if context.device.type == "cuda" else None,
        )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config["train"]["learning_rate"]),
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
        TokenizerLoss(LossConfig(**config["loss"])).to(context.device),
        optimizer,
        scheduler,
        context,
        args.output,
        precision=str(config["train"]["precision"]),
        grad_clip_norm=float(config["train"]["grad_clip_norm"]),
        config=config,
    )
    start_epoch = trainer.load_checkpoint(args.resume) if args.resume else 0
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