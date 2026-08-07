#!/usr/bin/env python3
"""Evaluate VGGT-Omega interval action reconstruction."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from vggt_omega.utils.pose_enc import encoding_to_camera

from vision_action_tokenizer.config import load_config
from vision_action_tokenizer.data.dataset import (
    CachedVGGTOmegaFeatureDataset,
    NuScenesWindowDataset,
    VGGTOmegaResize,
)
from vision_action_tokenizer.data.manifest import load_manifest
from vision_action_tokenizer.metrics import trajectory_metrics
from vision_action_tokenizer.models.factory import build_tokenizer, tokenizer_state_from_checkpoint
from vision_action_tokenizer.models.vggt_omega import OmegaCameraFeatureExtractor
from vision_action_tokenizer.visualization import render_vggt_evaluation_diagnostic


def _motion_bucket(trajectory: torch.Tensor) -> str:
    distance = float(torch.linalg.vector_norm(trajectory[-1, :2]))
    lateral = abs(float(trajectory[-1, 1]))
    yaw = abs(float(trajectory[-1, 2]))
    if distance < 2.0:
        return "stationary"
    if lateral > 2.0 or yaw > 0.15:
        return "turn"
    return "straight_slow" if distance < 20.0 else "straight_fast"


def _accumulate(destination: dict[str, float], metrics: dict[str, torch.Tensor]) -> None:
    for key, value in metrics.items():
        destination[key.removeprefix("metric/")] += float(value)


def _pose_baseline_trajectory(
    pose_enc: torch.Tensor,
    frame_times: torch.Tensor,
    future_times: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    extrinsics, _ = encoding_to_camera(pose_enc, (1, 1), build_intrinsics=False)
    rotation = extrinsics[..., :3, :3]
    translation = extrinsics[..., :3, 3]
    centers = -(rotation.transpose(-1, -2) @ translation.unsqueeze(-1)).squeeze(-1)
    centers = centers - centers[:, :1]
    keyframe_xy = torch.stack([centers[..., 2], -centers[..., 0]], dim=-1) * scale
    trajectories = []
    for batch_index in range(pose_enc.shape[0]):
        right = torch.searchsorted(frame_times[batch_index], future_times[batch_index])
        right = right.clamp(1, frame_times.shape[1] - 1)
        left = right - 1
        left_time = frame_times[batch_index, left]
        right_time = frame_times[batch_index, right]
        alpha = ((future_times[batch_index] - left_time) / (right_time - left_time)).clamp(0, 1)
        xy = keyframe_xy[batch_index, left] + alpha.unsqueeze(-1) * (
            keyframe_xy[batch_index, right] - keyframe_xy[batch_index, left]
        )
        previous_xy = torch.cat([xy[:1].new_zeros(1, 2), xy[:-1]], dim=0)
        delta = xy - previous_xy
        yaw = torch.atan2(delta[:, 1], delta[:, 0])
        yaw = torch.where(torch.linalg.vector_norm(delta, dim=-1) > 1e-3, yaw, 0.0)
        trajectories.append(torch.cat([xy, yaw.unsqueeze(-1)], dim=-1))
    return torch.stack(trajectories)


def _load_camera_window(paths_json: str, transform: VGGTOmegaResize) -> torch.Tensor:
    frames = []
    for path in json.loads(paths_json):
        with Image.open(path) as image:
            frames.append(transform(image))
    return torch.stack(frames)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument(
        "--train-manifest",
        type=Path,
        help="Optional train split used to report a leak-free train-mean baseline",
    )
    parser.add_argument(
        "--pose-calibration",
        type=Path,
        help="Train-fitted camera_motion_calibration.json for the VGGT pose baseline",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tensorboard-dir", type=Path)
    parser.add_argument("--visualize-items", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    backbone = config["vision_backbone"]
    tensorboard_config = config.get("tensorboard", {})
    visualize_items = (
        int(tensorboard_config.get("evaluation_visualization_items", 0))
        if args.visualize_items is None
        else args.visualize_items
    )
    if visualize_items < 0:
        raise ValueError("--visualize-items must be non-negative")
    include_images = bool(
        tensorboard_config.get("evaluation_visualization_include_images", True)
    )
    cache_config = config["data"].get("feature_cache")
    configured_cache = cache_config.get("val") if isinstance(cache_config, dict) else cache_config
    cache = args.feature_cache or configured_cache
    transform = VGGTOmegaResize(
        image_resolution=int(backbone["image_resolution"]),
        mode=str(backbone["resize_mode"]),
        patch_size=int(backbone["patch_size"]),
    )
    base = NuScenesWindowDataset(
        args.manifest,
        transform=transform,
        load_images=cache is None,
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
        raise ValueError("Online VGGT-Omega evaluation requires --batch-size 1")
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
    mean_trajectory = None
    if args.train_manifest is not None:
        train_windows = load_manifest(args.train_manifest)
        if not train_windows:
            raise ValueError("--train-manifest is empty")
        mean_trajectory = torch.tensor(
            [window.trajectory for window in train_windows], dtype=torch.float32
        ).mean(dim=0).to(device)
    pose_scale = None
    if args.pose_calibration is not None:
        pose_calibration = json.loads(args.pose_calibration.read_text(encoding="utf-8"))
        pose_scale = float(pose_calibration["global_scale"])

    writer = None
    if bool(tensorboard_config.get("enabled", False)) or args.tensorboard_dir is not None:
        log_dir = args.tensorboard_dir or args.output.parent / "tensorboard_eval"
        writer = SummaryWriter(
            log_dir=str(log_dir),
            flush_secs=int(tensorboard_config.get("flush_secs", 30)),
        )
    step = int(checkpoint.get("global_step", 0))
    model_totals: dict[str, float] = defaultdict(float)
    stationary_totals: dict[str, float] = defaultdict(float)
    mean_totals: dict[str, float] = defaultdict(float)
    pose_totals: dict[str, float] = defaultdict(float)
    bucket_totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    bucket_counts: dict[str, int] = defaultdict(int)
    sample_count = 0
    visualization_count = 0
    visualized_scenes: set[str] = set()
    distinct_scenes = bool(
        tensorboard_config.get("evaluation_visualization_distinct_scenes", True)
    )
    try:
        with torch.inference_mode():
            for batch in loader:
                if extractor is None:
                    camera_hidden = batch["camera_hidden"].to(device).float()
                    registers = batch["register_hidden_mean"].to(device).float()
                    pose_enc = batch["pose_enc"].to(device).float()
                else:
                    features = extractor(batch["images"].to(device))
                    camera_hidden = features.camera_hidden
                    registers = features.register_hidden_mean
                    pose_enc = features.pose_enc
                trajectory = batch["trajectory"].to(device)
                times = batch["future_times"].to(device)
                mask = batch["trajectory_mask"].to(device)
                output = tokenizer(
                    camera_hidden,
                    registers,
                    batch["frame_times"].to(device),
                    times,
                )
                pose_prediction = (
                    None
                    if pose_scale is None
                    else _pose_baseline_trajectory(
                        pose_enc,
                        batch["frame_times"].to(device),
                        times,
                        pose_scale,
                    )
                )
                steps_per_token = int(config["action_tokenizer"]["steps_per_token"])
                for item_index in range(trajectory.shape[0]):
                    selection = slice(item_index, item_index + 1)
                    metric_args = (
                        trajectory[selection],
                        times[selection],
                        mask[selection],
                    )
                    model_metrics = trajectory_metrics(
                        output.reconstruction[selection],
                        *metric_args,
                        steps_per_token=steps_per_token,
                    )
                    stationary_metrics = trajectory_metrics(
                        torch.zeros_like(trajectory[selection]),
                        *metric_args,
                        steps_per_token=steps_per_token,
                    )
                    _accumulate(model_totals, model_metrics)
                    _accumulate(stationary_totals, stationary_metrics)
                    bucket = _motion_bucket(trajectory[item_index])
                    _accumulate(bucket_totals[bucket], model_metrics)
                    bucket_counts[bucket] += 1
                    if mean_trajectory is not None:
                        if mean_trajectory.shape != trajectory[item_index].shape:
                            raise ValueError(
                                "Train-mean trajectory shape does not match evaluation data"
                            )
                        mean_prediction = mean_trajectory.unsqueeze(0)
                        mean_metrics = trajectory_metrics(
                            mean_prediction,
                            *metric_args,
                            steps_per_token=steps_per_token,
                        )
                        _accumulate(mean_totals, mean_metrics)
                    if pose_prediction is not None:
                        pose_metrics = trajectory_metrics(
                            pose_prediction[selection],
                            *metric_args,
                            steps_per_token=steps_per_token,
                        )
                        _accumulate(pose_totals, pose_metrics)
                    sample_count += 1

                if writer is not None and visualization_count < visualize_items:
                    sample_tokens = batch.get("sample_token", [""] * trajectory.shape[0])
                    scene_tokens = batch.get("scene_token", [""] * trajectory.shape[0])
                    for item_index in range(trajectory.shape[0]):
                        if visualization_count >= visualize_items:
                            break
                        scene_token = str(scene_tokens[item_index])
                        if distinct_scenes and scene_token in visualized_scenes:
                            continue
                        camera_images = None
                        if include_images:
                            if "images" in batch:
                                camera_images = batch["images"][item_index]
                            else:
                                camera_images = _load_camera_window(
                                    batch["image_paths_json"][item_index], transform
                                )
                        image = render_vggt_evaluation_diagnostic(
                            trajectory[item_index],
                            output.reconstruction[item_index],
                            output.predicted_increments[item_index],
                            times[item_index],
                            camera_images=camera_images,
                            frame_times=batch["frame_times"][item_index],
                            sample_token=str(sample_tokens[item_index]),
                            mask=mask[item_index],
                        )
                        writer.add_image(
                            f"evaluation/vggt_diagnostic_2x2/item_{visualization_count:03d}",
                            image,
                            step,
                        )
                        visualized_scenes.add(scene_token)
                        visualization_count += 1
    finally:
        if writer is not None:
            writer.flush()

    if sample_count == 0:
        raise ValueError("Evaluation dataset is empty")

    def averaged(values: dict[str, float], count: int) -> dict[str, float]:
        return {key: value / count for key, value in sorted(values.items())}

    report: dict[str, object] = {
        "num_samples": sample_count,
        "model": averaged(model_totals, sample_count),
        "baselines": {
            "stationary": averaged(stationary_totals, sample_count),
        },
        "buckets": {
            name: {
                "num_samples": bucket_counts[name],
                **averaged(values, bucket_counts[name]),
            }
            for name, values in sorted(bucket_totals.items())
        },
    }
    baselines = report["baselines"]
    assert isinstance(baselines, dict)
    if mean_trajectory is not None:
        baselines["train_mean"] = averaged(mean_totals, sample_count)
    if pose_scale is not None:
        baselines["vggt_pose_train_calibrated"] = averaged(pose_totals, sample_count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if writer is not None:
        model_report = report["model"]
        assert isinstance(model_report, dict)
        for key, value in model_report.items():
            writer.add_scalar(f"evaluation/{key}", value, step)
        writer.close()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
