#!/usr/bin/env python3
"""Evaluate visual- and trajectory-latent reconstructions on a manifest."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from vision_action_tokenizer.config import load_config
from vision_action_tokenizer.data.dataset import CachedPEFeatureDataset, NuScenesWindowDataset
from vision_action_tokenizer.metrics import trajectory_metrics
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
    base = NuScenesWindowDataset(
        args.manifest,
        int(config["data"]["image_size"]),
        load_images=args.feature_cache is None,
    )
    dataset = (
        base
        if args.feature_cache is None
        else CachedPEFeatureDataset(
            base,
            args.feature_cache,
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
    tokenizer = build_tokenizer(config).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tokenizer.load_state_dict(tokenizer_state_from_checkpoint(checkpoint), strict=True)
    extractor = None
    if args.feature_cache is None:
        pe = config["pe"]
        extractor = PEFeatureExtractor(
            model_name=pe["model_name"],
            checkpoint_path=pe.get("checkpoint_path"),
            layer_idx=pe.get("layer_idx"),
            pool_size=int(pe["pool_size"]),
            forward_batch_size=pe.get("forward_batch_size"),
            freeze=True,
        ).to(device)
    totals: dict[str, float] = defaultdict(float)
    batches = 0
    with torch.inference_mode():
        for batch in loader:
            features = (
                batch["visual_features"].to(device).float()
                if extractor is None
                else extractor(batch["images"].to(device))
            )
            trajectory = batch["trajectory"].to(device)
            times = batch["future_times"].to(device)
            mask = batch["trajectory_mask"].to(device)
            output = tokenizer(
                features,
                trajectory,
                batch["frame_times"].to(device),
                times,
                mask,
                sample_posterior=False,
            )
            for prefix, prediction in (
                ("visual", output.reconstruction_vis),
                ("trajectory", output.reconstruction_traj),
            ):
                metrics = trajectory_metrics(prediction, trajectory, times, mask)
                for key, value in metrics.items():
                    totals[f"{prefix}/{key.removeprefix('metric/')}"] += float(value)
            batches += 1
    report = {key: value / max(batches, 1) for key, value in sorted(totals.items())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
