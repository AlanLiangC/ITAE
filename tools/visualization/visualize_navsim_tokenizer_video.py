#!/usr/bin/env python3
"""Render NAVSIM tokenizer GT/reconstruction overlays as an MP4 video."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from pyquaternion import Quaternion
from torch.utils.data import DataLoader, Subset

from vision_action_tokenizer.config import load_config
from vision_action_tokenizer.data.dataset import (
    ActionWindowDataset,
    CachedVGGTOmegaFeatureDataset,
    configured_reference_point_offset,
)
from vision_action_tokenizer.data.manifest import WindowRecord, load_manifest
from vision_action_tokenizer.models.factory import (
    build_tokenizer,
    tokenizer_state_from_checkpoint,
)
from vision_action_tokenizer.visualization import trajectory_time_color

GT_COLOR = (202, 225, 235)
PANEL_BACKGROUND = (10, 14, 20, 210)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/tokenizer/nuscenes_navsim_v4_scratch_4s.yaml"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/navsim_mini_val_4s.jsonl"),
    )
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=Path(
            "/home/alan/AlanLiang/Dataset/vggt_omega_cache/"
            "navsim_mini_front_4s_val_rich"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("output/itae_v4_scratch_nuscenes_navsim/best.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "output/itae_v4_scratch_nuscenes_navsim/"
            "navsim_gt_vs_reconstruction.mp4"
        ),
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument("--sample-token", help="Start at this namespaced NAVSIM token")
    parser.add_argument("--scene-token", help="Restrict automatic selection to one scene")
    parser.add_argument("--group-token", help="Restrict automatic selection to one log")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--selection",
        choices=("turn", "first"),
        default="turn",
        help="Automatic clip selection when --sample-token is absent",
    )
    return parser.parse_args()


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def _motion_score(window: WindowRecord) -> float:
    endpoint = np.asarray(window.trajectory[-1], dtype=np.float64)
    distance = float(np.linalg.norm(endpoint[:2]))
    if distance < 2.0:
        return -10.0
    return 20.0 * min(abs(float(endpoint[2])), 1.5) + min(
        abs(float(endpoint[1])), 30.0
    ) + 0.02 * distance


def _contiguous_runs(
    indexed_windows: list[tuple[int, WindowRecord]], max_gap_s: float = 0.65
) -> list[list[tuple[int, WindowRecord]]]:
    runs: list[list[tuple[int, WindowRecord]]] = []
    for item in sorted(indexed_windows, key=lambda value: value[1].anchor_timestamp_us):
        if not runs:
            runs.append([item])
            continue
        previous = runs[-1][-1][1]
        gap_s = (item[1].anchor_timestamp_us - previous.anchor_timestamp_us) / 1e6
        if 0.0 < gap_s <= max_gap_s:
            runs[-1].append(item)
        else:
            runs.append([item])
    return runs


def _select_windows(
    windows: list[WindowRecord],
    *,
    num_frames: int,
    sample_token: str | None,
    scene_token: str | None,
    group_token: str | None,
    selection: str,
) -> list[tuple[int, WindowRecord]]:
    if num_frames <= 0:
        raise ValueError("--num-frames must be positive")
    normalized_sample = None
    if sample_token:
        normalized_sample = (
            sample_token if sample_token.startswith("navsim:") else f"navsim:{sample_token}"
        )
    normalized_scene = None
    if scene_token:
        normalized_scene = (
            scene_token if scene_token.startswith("navsim:") else f"navsim:{scene_token}"
        )
    normalized_group = None
    if group_token:
        normalized_group = (
            group_token if group_token.startswith("navsim:") else f"navsim:{group_token}"
        )

    grouped: dict[tuple[str, str], list[tuple[int, WindowRecord]]] = defaultdict(list)
    for index, window in enumerate(windows):
        if window.dataset_name != "navsim":
            continue
        if normalized_scene and window.scene_token != normalized_scene:
            continue
        current_group = window.group_token or window.scene_token
        if normalized_group and current_group != normalized_group:
            continue
        grouped[(current_group, window.scene_token)].append((index, window))
    runs = [
        run
        for group in grouped.values()
        for run in _contiguous_runs(group)
        if len(run) >= num_frames
    ]
    if not runs:
        raise ValueError("No NAVSIM scene has enough contiguous windows for this clip")

    if normalized_sample:
        for run in runs:
            positions = [
                position
                for position, (_, window) in enumerate(run)
                if window.sample_token == normalized_sample
            ]
            if positions:
                start = positions[0]
                if start + num_frames > len(run):
                    start = len(run) - num_frames
                return run[start : start + num_frames]
        raise ValueError(f"Sample token is absent from a valid contiguous clip: {sample_token}")

    candidates: list[tuple[float, list[tuple[int, WindowRecord]]]] = []
    for run in runs:
        for start in range(len(run) - num_frames + 1):
            clip = run[start : start + num_frames]
            score = (
                float(np.mean([_motion_score(window) for _, window in clip]))
                if selection == "turn"
                else -float(clip[0][1].anchor_timestamp_us)
            )
            candidates.append((score, clip))
    return max(candidates, key=lambda value: value[0])[1]


def _decode_selected(
    config: dict[str, Any],
    manifest: Path,
    cache: Path,
    checkpoint_path: Path,
    windows: list[WindowRecord],
    selected_indices: list[int],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    reference_offset = configured_reference_point_offset(config, windows)
    base = ActionWindowDataset(
        windows,
        load_images=False,
        reference_point_offset_m=reference_offset,
    )
    cached = CachedVGGTOmegaFeatureDataset(
        base,
        cache,
        manifest_path=manifest,
        expected_metadata={
            "checkpoint_sha256": config["vision_backbone"].get("checkpoint_sha256"),
            "image_resolution": int(config["vision_backbone"]["image_resolution"]),
            "resize_mode": str(config["vision_backbone"]["resize_mode"]),
            "token_mode": str(config["vision_backbone"]["cache_token_mode"]),
        },
    )
    loader = DataLoader(
        Subset(cached, selected_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = build_tokenizer(config).to(device).eval()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    tokenizer.load_state_dict(tokenizer_state_from_checkpoint(checkpoint), strict=True)
    predictions = []
    targets = []
    with torch.inference_mode():
        for batch in loader:
            future_times = batch["future_times"].to(device).float()
            output = tokenizer(
                batch["camera_hidden"].to(device).float(),
                batch["register_hidden_mean"].to(device).float(),
                batch["frame_times"].to(device).float(),
                future_times,
                register_hidden=batch.get("register_hidden", None).to(device).float()
                if "register_hidden" in batch
                else None,
                pose_enc=batch.get("pose_enc", None).to(device).float()
                if "pose_enc" in batch
                else None,
            )
            predictions.append(output.reconstruction.float().cpu())
            targets.append(batch["trajectory"].float())
    return torch.cat(targets).numpy(), torch.cat(predictions).numpy()


def _load_log_frames(data_root: Path, split: str, group_token: str) -> dict[str, dict]:
    log_name = group_token.removeprefix("navsim:")
    log_path = data_root / "navsim_logs" / split / f"{log_name}.pkl"
    if not log_path.is_file():
        raise FileNotFoundError(f"NAVSIM log is missing: {log_path}")
    with log_path.open("rb") as handle:
        frames = pickle.load(handle)  # noqa: S301 - trusted local NAVSIM dataset
    if not isinstance(frames, list):
        raise ValueError(f"Expected a list in NAVSIM log {log_path}")
    return {str(frame["token"]): frame for frame in frames}


def _project_ego_ground_to_camera(
    trajectory: np.ndarray,
    raw_frame: dict,
    reference_offset_m: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    camera = raw_frame["cams"]["CAM_F0"]
    sensor_to_lidar_rotation = np.asarray(
        camera["sensor2lidar_rotation"], dtype=np.float64
    )
    sensor_to_lidar_translation = np.asarray(
        camera["sensor2lidar_translation"], dtype=np.float64
    )
    intrinsic = np.asarray(camera["cam_intrinsic"], dtype=np.float64)
    lidar_to_ego = np.eye(4, dtype=np.float64)
    lidar_to_ego[:3, :3] = Quaternion(
        raw_frame["lidar2ego_rotation"]
    ).rotation_matrix
    lidar_to_ego[:3, 3] = np.asarray(
        raw_frame["lidar2ego_translation"], dtype=np.float64
    )

    points_ego = np.column_stack(
        [
            trajectory[:, 0] + reference_offset_m[0],
            trajectory[:, 1] + reference_offset_m[1],
            np.zeros(len(trajectory)),
            np.ones(len(trajectory)),
        ]
    )
    points_lidar = (np.linalg.inv(lidar_to_ego) @ points_ego.T).T[:, :3]
    lidar_to_camera_rotation = np.linalg.inv(sensor_to_lidar_rotation)
    lidar_to_camera_translation = (
        sensor_to_lidar_translation @ lidar_to_camera_rotation.T
    )
    points_camera = (
        lidar_to_camera_rotation @ points_lidar.T
    ).T - lidar_to_camera_translation
    homogeneous = (intrinsic @ points_camera.T).T
    depth = points_camera[:, 2]
    pixels = np.full((len(trajectory), 2), np.nan, dtype=np.float64)
    valid_depth = depth > 1e-3
    pixels[valid_depth] = homogeneous[valid_depth, :2] / depth[valid_depth, None]
    return pixels, depth


def _visible_mask(pixels: np.ndarray, depth: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    return (
        (depth > 0.1)
        & np.isfinite(pixels).all(axis=1)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )


def _draw_dashed_projected_path(
    draw: ImageDraw.ImageDraw,
    pixels: np.ndarray,
    visible: np.ndarray,
    *,
    fill: tuple[int, int, int, int],
    width: int,
) -> None:
    for index in range(1, len(pixels)):
        if index % 2 == 0 and visible[index - 1] and visible[index]:
            draw.line(
                (*pixels[index - 1].tolist(), *pixels[index].tolist()),
                fill=fill,
                width=width,
            )


def _draw_prediction_projected_path(
    draw: ImageDraw.ImageDraw,
    pixels: np.ndarray,
    visible: np.ndarray,
    times: np.ndarray,
) -> None:
    for index in range(1, len(pixels)):
        if not (visible[index - 1] and visible[index]):
            continue
        color = trajectory_time_color(index / max(len(pixels) - 1, 1))
        draw.line(
            (*pixels[index - 1].tolist(), *pixels[index].tolist()),
            fill=(*color, 250),
            width=7,
        )
    for index in range(4, len(pixels), 5):
        if not visible[index]:
            continue
        x, y = pixels[index]
        color = trajectory_time_color(index / max(len(pixels) - 1, 1))
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=(*color, 255))
        draw.text((x + 8, y - 12), f"{times[index]:.1f}s", fill=(*color, 255))


def _draw_bev(
    draw: ImageDraw.ImageDraw,
    gt: np.ndarray,
    prediction: np.ndarray,
    canvas_size: tuple[int, int],
) -> None:
    width, height = canvas_size
    panel_width, panel_height = 430, 470
    left, top = width - panel_width - 22, height - panel_height - 22
    right, bottom = left + panel_width, top + panel_height
    draw.rounded_rectangle(
        (left, top, right, bottom),
        radius=16,
        fill=PANEL_BACKGROUND,
        outline=(90, 100, 115, 230),
        width=2,
    )
    all_xy = np.vstack([np.zeros((1, 2)), gt[:, :2], prediction[:, :2]])
    x_min, y_min = all_xy.min(axis=0)
    x_max, y_max = all_xy.max(axis=0)
    x_span = max(float(x_max - x_min), 10.0)
    y_span = max(float(y_max - y_min), 10.0)
    x_center, y_center = 0.5 * (x_min + x_max), 0.5 * (y_min + y_max)
    x_min, x_max = x_center - 0.6 * x_span, x_center + 0.6 * x_span
    y_min, y_max = y_center - 0.6 * y_span, y_center + 0.6 * y_span
    plot_left, plot_right = left + 35, right - 25
    plot_top, plot_bottom = top + 55, bottom - 42

    def project(xy: np.ndarray) -> list[tuple[float, float]]:
        u = plot_left + (y_max - xy[:, 1]) / (y_max - y_min) * (
            plot_right - plot_left
        )
        v = plot_top + (x_max - xy[:, 0]) / (x_max - x_min) * (
            plot_bottom - plot_top
        )
        return list(zip(u.astype(float), v.astype(float), strict=True))

    origin = project(np.zeros((1, 2)))[0]
    draw.line((plot_left, origin[1], plot_right, origin[1]), fill=(85, 90, 100), width=2)
    draw.line((origin[0], plot_top, origin[0], plot_bottom), fill=(85, 90, 100), width=2)
    gt_points = [origin, *project(gt[:, :2])]
    for index in range(1, len(gt_points)):
        if index % 2:
            draw.line((*gt_points[index - 1], *gt_points[index]), fill=GT_COLOR, width=8)
    prediction_points = [origin, *project(prediction[:, :2])]
    for index in range(1, len(prediction_points)):
        color = trajectory_time_color(index / max(len(prediction_points) - 1, 1))
        draw.line(
            (*prediction_points[index - 1], *prediction_points[index]),
            fill=color,
            width=5,
        )
    draw.ellipse(
        (origin[0] - 6, origin[1] - 6, origin[0] + 6, origin[1] + 6),
        fill=(80, 180, 255),
    )
    draw.text((left + 16, top + 12), "BEV | x forward, y left", fill="white", font=_font(20))
    draw.text((origin[0] + 6, plot_top), "x+", fill=(185, 190, 200), font=_font(15))
    draw.text((plot_left, origin[1] - 20), "y+", fill=(185, 190, 200), font=_font(15))
    draw.text(
        (left + 16, bottom - 30),
        "dashed GT | colored decode",
        fill=(220, 225, 232),
        font=_font(16),
    )


def _render_frame(
    image: Image.Image,
    window: WindowRecord,
    raw_frame: dict,
    gt: np.ndarray,
    prediction: np.ndarray,
    reference_offset_m: tuple[float, float],
    clip_elapsed_s: float,
) -> tuple[Image.Image, dict[str, float]]:
    frame = image.convert("RGB")
    overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    gt_pixels, gt_depth = _project_ego_ground_to_camera(
        gt, raw_frame, reference_offset_m
    )
    prediction_pixels, prediction_depth = _project_ego_ground_to_camera(
        prediction, raw_frame, reference_offset_m
    )
    gt_visible = _visible_mask(gt_pixels, gt_depth, frame.size)
    prediction_visible = _visible_mask(prediction_pixels, prediction_depth, frame.size)
    _draw_dashed_projected_path(
        draw, gt_pixels, gt_visible, fill=(*GT_COLOR, 245), width=13
    )
    _draw_prediction_projected_path(
        draw,
        prediction_pixels,
        prediction_visible,
        np.asarray(window.future_times_s),
    )

    xy_error = np.linalg.norm(prediction[:, :2] - gt[:, :2], axis=-1)
    yaw_error = np.abs(
        (prediction[:, 2] - gt[:, 2] + np.pi) % (2.0 * np.pi) - np.pi
    )
    metrics = {
        "ade_m": float(xy_error.mean()),
        "fde_m": float(xy_error[-1]),
        "yaw_mae_rad": float(yaw_error.mean()),
        "gt_visible_points": int(gt_visible.sum()),
        "prediction_visible_points": int(prediction_visible.sum()),
    }
    width, height = frame.size
    draw.rounded_rectangle((16, 14, 1050, 124), radius=12, fill=(0, 0, 0, 185))
    draw.text(
        (31, 25),
        "NAVSIM CAM_F0 | V4 action-tokenizer reconstruction",
        fill="white",
        font=_font(25),
    )
    draw.text(
        (31, 60),
        (
            f"clip t={clip_elapsed_s:.1f}s | ADE={metrics['ade_m']:.3f}m | "
            f"FDE={metrics['fde_m']:.3f}m | yaw MAE={math.degrees(metrics['yaw_mae_rad']):.2f}deg"
        ),
        fill=(230, 233, 240),
        font=_font(20),
    )
    draw.text(
        (31, 91),
        f"sample {window.sample_token} | dashed GT | colored decode (green -> yellow -> red)",
        fill=(205, 212, 222),
        font=_font(17),
    )
    _draw_bev(draw, gt, prediction, (width, height))
    return Image.alpha_composite(frame.convert("RGBA"), overlay).convert("RGB"), metrics


def _encode_video(frames: list[Image.Image], output: Path, fps: float) -> None:
    if not frames:
        raise ValueError("No video frames were rendered")
    if fps <= 0:
        raise ValueError("--fps must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    width, height = frames[0].size
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None and process.stderr is not None
    for frame in frames:
        if frame.size != (width, height):
            raise ValueError("All rendered video frames must have the same size")
        process.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
    process.stdin.close()
    error = process.stderr.read().decode("utf-8", errors="replace")
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}: {error}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    windows = load_manifest(args.manifest)
    if not windows:
        raise ValueError("NAVSIM manifest is empty")
    selected = _select_windows(
        windows,
        num_frames=args.num_frames,
        sample_token=args.sample_token,
        scene_token=args.scene_token,
        group_token=args.group_token,
        selection=args.selection,
    )
    selected_indices = [index for index, _ in selected]
    selected_windows = [window for _, window in selected]
    group_tokens = {window.group_token for window in selected_windows}
    if len(group_tokens) != 1 or None in group_tokens:
        raise ValueError("Selected clip must belong to one NAVSIM log")
    group_token = next(iter(group_tokens))
    assert group_token is not None
    data_root = Path(config["navsim_export"]["data_root"])
    split = str(config["navsim_export"].get("split", "mini"))
    raw_frames = _load_log_frames(data_root, split, group_token)
    reference_offset = configured_reference_point_offset(config, windows)
    targets, predictions = _decode_selected(
        config,
        args.manifest,
        args.feature_cache,
        args.checkpoint,
        windows,
        selected_indices,
        args.batch_size,
    )

    rendered_frames = []
    frame_reports = []
    first_timestamp = selected_windows[0].anchor_timestamp_us
    for window, gt, prediction in zip(
        selected_windows, targets, predictions, strict=True
    ):
        raw_token = window.sample_token.removeprefix("navsim:")
        if raw_token not in raw_frames:
            raise KeyError(f"NAVSIM sample token is absent from its log: {raw_token}")
        raw_frame = raw_frames[raw_token]
        expected_image = str(raw_frame["cams"]["CAM_F0"]["data_path"])
        if Path(expected_image).name != Path(window.image_paths[0]).name:
            raise ValueError(
                f"Manifest/raw CAM_F0 mismatch for {window.sample_token}: "
                f"{window.image_paths[0]} != {expected_image}"
            )
        with Image.open(window.image_paths[0]) as image:
            rendered, metrics = _render_frame(
                image,
                window,
                raw_frame,
                gt,
                prediction,
                reference_offset,
                (window.anchor_timestamp_us - first_timestamp) / 1e6,
            )
        rendered_frames.append(rendered)
        frame_reports.append(
            {
                "sample_token": window.sample_token,
                "anchor_timestamp_us": window.anchor_timestamp_us,
                **metrics,
            }
        )
    _encode_video(rendered_frames, args.output, args.fps)

    report = {
        "output": str(args.output),
        "config": str(args.config),
        "manifest": str(args.manifest),
        "feature_cache": str(args.feature_cache),
        "checkpoint": str(args.checkpoint),
        "group_token": group_token,
        "scene_token": selected_windows[0].scene_token,
        "num_frames": len(rendered_frames),
        "fps": args.fps,
        "clip_dataset_duration_s": (
            selected_windows[-1].anchor_timestamp_us - first_timestamp
        )
        / 1e6,
        "mean_ade_m": float(np.mean([item["ade_m"] for item in frame_reports])),
        "mean_fde_m": float(np.mean([item["fde_m"] for item in frame_reports])),
        "max_fde_m": float(max(item["fde_m"] for item in frame_reports)),
        "reference_point_offset_m": list(reference_offset),
        "projection": "rear-axle ground plane via lidar2ego and CAM_F0 sensor2lidar",
        "frames": frame_reports,
    }
    report_path = args.report or args.output.with_suffix(".json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
