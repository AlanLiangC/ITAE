#!/usr/bin/env python3
"""Export visual posterior means and train-split normalizer for an action expert."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from vision_action_tokenizer.config import load_config
from vision_action_tokenizer.data.dataset import CachedPEFeatureDataset, NuScenesWindowDataset
from vision_action_tokenizer.models.expert import LatentNormalizer
from vision_action_tokenizer.models.factory import build_tokenizer, tokenizer_state_from_checkpoint
from vision_action_tokenizer.models.pe import PEFeatureExtractor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_config = config["data"].get("feature_cache")
    configured_cache = cache_config.get("train") if isinstance(cache_config, dict) else cache_config
    cache = args.feature_cache or configured_cache
    base = NuScenesWindowDataset(
        args.manifest, int(config["data"]["image_size"]), load_images=cache is None
    )
    dataset = (
        base
        if cache is None
        else CachedPEFeatureDataset(
            base,
            cache,
            manifest_path=args.manifest,
            expected_metadata={
                "model_name": config["pe"]["model_name"],
                "checkpoint_path": config["pe"].get("checkpoint_path"),
                "layer_idx": config["pe"].get("layer_idx"),
                "pool_size": config["pe"]["pool_size"],
            },
        )
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    tokenizer = build_tokenizer(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tokenizer.load_state_dict(tokenizer_state_from_checkpoint(checkpoint), strict=True)
    tokenizer.eval()
    extractor = None
    if cache is None:
        pe = config["pe"]
        extractor = PEFeatureExtractor(
            model_name=pe["model_name"],
            checkpoint_path=pe.get("checkpoint_path"),
            layer_idx=pe.get("layer_idx"),
            pool_size=int(pe["pool_size"]),
            forward_batch_size=pe.get("forward_batch_size"),
            freeze=True,
        ).to(device)

    latents = []
    sample_tokens: list[str] = []
    with torch.inference_mode():
        for batch in loader:
            if extractor is None:
                features = batch["visual_features"].to(device).float()
            else:
                features = extractor(batch["images"].to(device))
            output = tokenizer(
                features,
                batch["trajectory"].to(device),
                batch["frame_times"].to(device),
                batch["future_times"].to(device),
                batch["trajectory_mask"].to(device),
                sample_posterior=False,
            )
            latents.append(output.mean_vis.float().cpu())
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
        metadata={"checkpoint": str(args.checkpoint), "target": "posterior_mean"},
    )
    args.output.with_suffix(".json").write_text(
        json.dumps({"sample_tokens": sample_tokens}, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
