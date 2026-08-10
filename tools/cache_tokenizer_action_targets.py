#!/usr/bin/env python3
"""Export frozen V4 action-token targets and their oracle reconstructions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Subset

from vision_action_tokenizer.config import load_config, stable_hash
from vision_action_tokenizer.data.dataset import (
    CachedVGGTOmegaFeatureDataset,
    NuScenesWindowDataset,
)
from vision_action_tokenizer.data.manifest import load_manifest
from vision_action_tokenizer.data.planner_dataset import file_sha256
from vision_action_tokenizer.models.factory import build_tokenizer, tokenizer_state_from_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        help="Manifest used to build feature-cache; target manifest may be an ordered subset",
    )
    parser.add_argument("--feature-cache", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError(f"Action target cache already exists: {args.output}")

    config = load_config(args.tokenizer_config)
    backbone = config["vision_backbone"]
    source_manifest = args.source_manifest or args.manifest
    base = NuScenesWindowDataset(source_manifest, load_images=False)
    source_dataset = CachedVGGTOmegaFeatureDataset(
        base,
        args.feature_cache,
        manifest_path=source_manifest,
        expected_metadata={
            "checkpoint_sha256": backbone.get("checkpoint_sha256"),
            "image_resolution": int(backbone["image_resolution"]),
            "resize_mode": str(backbone["resize_mode"]),
            "token_mode": str(backbone["cache_token_mode"]),
        },
    )
    target_windows = load_manifest(args.manifest)
    source_index = {
        window.sample_token: index for index, window in enumerate(base.windows)
    }
    if len(source_index) != len(base.windows):
        raise ValueError("Source manifest contains duplicate sample tokens")
    try:
        selected_indices = [source_index[window.sample_token] for window in target_windows]
    except KeyError as error:
        raise ValueError(
            f"Target manifest sample is absent from source manifest: {error}"
        ) from error
    dataset = Subset(source_dataset, selected_indices)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = build_tokenizer(config).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tokenizer.load_state_dict(tokenizer_state_from_checkpoint(checkpoint), strict=True)
    tokenizer.requires_grad_(False)

    action_tokens: list[torch.Tensor] = []
    oracle_trajectories: list[torch.Tensor] = []
    future_times: list[torch.Tensor] = []
    sample_tokens: list[str] = []
    ade_sum = 0.0
    fde_sum = 0.0
    sample_count = 0
    with torch.inference_mode():
        for batch in loader:
            future = batch["future_times"].to(device).float()
            output = tokenizer(
                batch["camera_hidden"].to(device).float(),
                batch["register_hidden_mean"].to(device).float(),
                batch["frame_times"].to(device).float(),
                future,
                register_hidden=batch["register_hidden"].to(device).float(),
                pose_enc=batch["pose_enc"].to(device).float(),
            )
            tokens = output.action_tokens.float()
            oracle = tokenizer.decode(tokens, future).float()
            target = batch["trajectory"].to(device).float()
            position_error = torch.linalg.vector_norm(
                oracle[..., :2] - target[..., :2], dim=-1
            )
            ade_sum += float(position_error.mean(dim=1).sum())
            fde_sum += float(position_error[:, -1].sum())
            sample_count += len(tokens)
            action_tokens.append(tokens.cpu())
            oracle_trajectories.append(oracle.cpu())
            future_times.append(future.cpu())
            sample_tokens.extend(batch["sample_token"])

    all_tokens = torch.cat(action_tokens).contiguous()
    all_oracle = torch.cat(oracle_trajectories).contiguous()
    all_future = torch.cat(future_times).contiguous()
    if tuple(all_tokens.shape[1:]) != (4, 192):
        raise ValueError(f"Expected V4 targets [N,4,192], got {tuple(all_tokens.shape)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "action_tokens": all_tokens,
            "oracle_trajectory": all_oracle,
            "future_times": all_future,
        },
        str(args.output),
        metadata={"cache_type": "v4_action_targets_v1"},
    )
    metadata = {
        "cache_type": "v4_action_targets_v1",
        "manifest": str(args.manifest),
        "manifest_sha256": file_sha256(args.manifest),
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": file_sha256(source_manifest),
        "rich_feature_cache": str(args.feature_cache),
        "rich_feature_cache_index_sha256": file_sha256(args.feature_cache / "index.json"),
        "tokenizer_config": str(args.tokenizer_config),
        "tokenizer_config_hash": stable_hash(config),
        "tokenizer_checkpoint": str(args.checkpoint),
        "tokenizer_checkpoint_sha256": file_sha256(args.checkpoint),
        "sample_tokens": sample_tokens,
        "target_shape": list(all_tokens.shape),
        "oracle_ade_m": ade_sum / sample_count,
        "oracle_fde_m": fde_sum / sample_count,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"Exported {sample_count} V4 targets | oracle "
        f"ADE={metadata['oracle_ade_m']:.6f} FDE={metadata['oracle_fde_m']:.6f}"
    )


if __name__ == "__main__":
    main()
