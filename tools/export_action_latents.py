#!/usr/bin/env python3
"""Export deterministic VGGT-Omega interval action tokens for the action expert."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from vision_action_tokenizer.config import load_config
from vision_action_tokenizer.data.dataset import (
    CachedVGGTOmegaFeatureDataset,
    NuScenesWindowDataset,
    VGGTOmegaResize,
)
from vision_action_tokenizer.models.expert import LatentNormalizer
from vision_action_tokenizer.models.factory import build_tokenizer, tokenizer_state_from_checkpoint
from vision_action_tokenizer.models.vggt_omega import OmegaCameraFeatureExtractor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    config = load_config(args.config)
    backbone = config["vision_backbone"]
    cache_config = config["data"].get("feature_cache")
    configured_cache = cache_config.get("train") if isinstance(cache_config, dict) else cache_config
    cache = args.feature_cache or configured_cache
    transform = VGGTOmegaResize(
        int(backbone["image_resolution"]),
        str(backbone["resize_mode"]),
        int(backbone["patch_size"]),
    )
    base = NuScenesWindowDataset(
        args.manifest, transform=transform, load_images=cache is None
    )
    dataset = (
        base
        if cache is None
        else CachedVGGTOmegaFeatureDataset(
            base,
            cache,
            manifest_path=args.manifest,
            expected_metadata={
                "checkpoint_sha256": backbone.get("checkpoint_sha256"),
                "image_resolution": int(backbone["image_resolution"]),
                "resize_mode": str(backbone["resize_mode"]),
                "token_mode": str(backbone["cache_token_mode"]),
            },
        )
    )
    if cache is None and args.batch_size != 1:
        raise ValueError("Online VGGT-Omega export requires --batch-size 1")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = build_tokenizer(config).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tokenizer.load_state_dict(tokenizer_state_from_checkpoint(checkpoint), strict=True)
    extractor = (
        None
        if cache is not None
        else OmegaCameraFeatureExtractor(
            backbone["checkpoint_path"], backbone.get("checkpoint_sha256"), freeze=True
        ).to(device)
    )

    latents = []
    sample_tokens: list[str] = []
    with torch.inference_mode():
        for batch in loader:
            if extractor is None:
                camera = batch["camera_hidden"].to(device).float()
                registers = batch["register_hidden_mean"].to(device).float()
            else:
                features = extractor(batch["images"].to(device))
                camera = features.camera_hidden
                registers = features.register_hidden_mean
            output = tokenizer(
                camera,
                registers,
                batch["frame_times"].to(device),
                batch["future_times"].to(device),
            )
            latents.append(output.action_tokens.float().cpu())
            sample_tokens.extend(batch["sample_token"])
    all_latents = torch.cat(latents)
    normalizer = LatentNormalizer(tuple(all_latents.shape[1:]))
    normalizer.fit(all_latents)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "latents": all_latents.contiguous(),
            "normalizer_mean": normalizer.mean.contiguous(),
            "normalizer_std": normalizer.std.contiguous(),
        },
        args.output,
        metadata={"checkpoint": str(args.checkpoint), "target": "vggt_interval_action_tokens"},
    )
    args.output.with_suffix(".json").write_text(
        json.dumps({"sample_tokens": sample_tokens}, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
