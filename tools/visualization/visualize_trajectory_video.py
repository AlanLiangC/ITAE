#!/usr/bin/env python3
"""Project parsed future ego trajectories onto a CAM_FRONT video.

The overlay uses the same planar trajectory as the training manifest and the
full nuScenes camera calibration chain:

    future ego origin -> anchor ego -> z=0 plane -> anchor CAM_FRONT -> pixels

It also independently transforms the future ego origins through global space
and checks that their anchor-frame x/y values agree with the parser.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from vision_action_tokenizer.config import load_config
from vision_action_tokenizer.data.geometry import (
    poses_to_local_trajectory,
)
from vision_action_tokenizer.data.lidar import LidarPoseRecord, load_lidar_pose_records
from vision_action_tokenizer.data.schema import (
    FrameRecord,
    InfoSchemaAdapter,
    load_info_pickle,
    load_nuscenes_scene_lookup,
    resolve_data_path,
)
from vision_action_tokenizer.data.temporal import TemporalIndex
from vision_action_tokenizer.visualization import trajectory_time_color


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--info", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, help="Optional override for config data_root")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--scene-token", type=str)
    parser.add_argument(
        "--start-offset-s",
        type=float,
        help="Clip start relative to the selected scene; defaults to an automatic turning clip.",
    )
    parser.add_argument("--duration-s", type=float, default=8.0)
    parser.add_argument("--fps", type=float, default=12.0)
    return parser.parse_args()


def _group_scenes(records: list[FrameRecord]) -> dict[str, list[FrameRecord]]:
    scenes: dict[str, list[FrameRecord]] = {}
    for record in records:
        scenes.setdefault(record.scene_token, []).append(record)
    for scene_records in scenes.values():
        scene_records.sort(key=lambda item: item.timestamp_us)
    return scenes


def _endpoint_score(anchor: FrameRecord, future: FrameRecord) -> float:
    endpoint = poses_to_local_trajectory(anchor.ego_to_global, [future.ego_to_global])[0]
    distance = float(np.linalg.norm(endpoint[:2]))
    if distance < 5.0:
        return -math.inf
    # Prefer a clearly curved, moving example without requiring map annotations.
    yaw = min(abs(float(endpoint[2])), 1.2)
    lateral = min(abs(float(endpoint[1])), 25.0)
    return 18.0 * yaw + lateral + 0.03 * distance


def _select_clip(
    scenes: dict[str, list[FrameRecord]],
    scene_token: str | None,
    start_offset_s: float | None,
    duration_s: float,
    horizon_s: float,
    max_error_us: int,
) -> tuple[str, list[FrameRecord], int, FrameRecord]:
    if scene_token is not None and scene_token not in scenes:
        raise ValueError(f"Unknown scene token: {scene_token}")
    candidate_scenes = {scene_token: scenes[scene_token]} if scene_token else scenes
    best: tuple[float, str, FrameRecord] | None = None
    horizon_us = round(horizon_s * 1_000_000)
    for token, scene_records in candidate_scenes.items():
        pose_controls = [item for item in scene_records if not item.pose_interpolated]
        if len(pose_controls) < 2:
            pose_controls = scene_records
        index = TemporalIndex([item.timestamp_us for item in pose_controls], pose_controls)
        for anchor in pose_controls:
            future = index.nearest(anchor.timestamp_us + horizon_us, max_error_us)
            if future is None:
                continue
            score = _endpoint_score(anchor, future.value)
            if best is None or score > best[0]:
                best = (score, token, anchor)
    if best is None:
        raise RuntimeError("No scene has a valid future trajectory for the requested horizon")

    _, selected_token, reference = best
    scene_records = scenes[selected_token]
    selected_controls = [item for item in scene_records if not item.pose_interpolated]
    if len(selected_controls) < 2:
        selected_controls = scene_records
    first_us = selected_controls[0].timestamp_us
    last_valid_us = selected_controls[-1].timestamp_us - horizon_us
    duration_us = round(duration_s * 1_000_000)
    latest_start_us = max(first_us, last_valid_us - duration_us)
    if start_offset_s is None:
        desired_start_us = reference.timestamp_us - 2_000_000
    else:
        desired_start_us = first_us + round(start_offset_s * 1_000_000)
    start_us = min(max(desired_start_us, first_us), latest_start_us)
    return selected_token, scene_records, start_us, reference


def _future_poses(
    anchor_timestamp_us: int,
    pose_index: TemporalIndex[LidarPoseRecord],
    future_times_s: np.ndarray,
    max_time_error_us: int,
) -> tuple[list[np.ndarray], np.ndarray, int]:
    matches = [
        pose_index.nearest(
            anchor_timestamp_us + round(float(time_s) * 1_000_000), max_time_error_us
        )
        for time_s in future_times_s
    ]
    if any(match is None for match in matches):
        raise RuntimeError(f"Incomplete future trajectory at {anchor_timestamp_us}")
    valid_matches = [match for match in matches if match is not None]
    timestamps_us = [match.value.timestamp_us for match in valid_matches]
    if any(
        right <= left
        for left, right in zip(timestamps_us, timestamps_us[1:], strict=False)
    ):
        raise RuntimeError(
            f"Repeated or out-of-order LiDAR trajectory pose at {anchor_timestamp_us}"
        )
    actual_times_s = np.asarray(
        [(timestamp_us - anchor_timestamp_us) / 1_000_000 for timestamp_us in timestamps_us]
    )
    return (
        [match.value.ego_to_global for match in valid_matches],
        actual_times_s,
        max(match.error_us for match in valid_matches),
    )


def _project_trajectory(
    anchor: FrameRecord, anchor_pose: np.ndarray, future_poses: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    if anchor.camera_to_ego is None or anchor.camera_intrinsic is None:
        raise RuntimeError("CAM_FRONT extrinsics/intrinsics are missing from the info pickle")
    global_points = np.stack([pose[:, 3] for pose in future_poses])
    local_trajectory = poses_to_local_trajectory(anchor_pose, future_poses)
    global_to_anchor = np.linalg.inv(anchor_pose)
    local_points_full = (global_to_anchor @ global_points.T).T
    parser_consistency_error_m = float(
        np.max(np.abs(local_points_full[:, :2] - local_trajectory[:, :2]))
    )
    planarization_z_abs_max_m = float(np.max(np.abs(local_points_full[:, 2])))
    local_points = np.column_stack(
        [local_trajectory[:, :2], np.zeros(len(local_trajectory)), np.ones(len(local_trajectory))]
    )
    camera_ego_to_global = anchor.camera_ego_to_global
    if camera_ego_to_global is None:
        camera_ego_to_global = anchor_pose
    camera_to_global = camera_ego_to_global @ anchor.camera_to_ego
    camera_from_anchor = np.linalg.inv(camera_to_global) @ anchor_pose
    camera_points = (camera_from_anchor @ local_points.T).T[:, :3]

    intrinsic = np.asarray(anchor.camera_intrinsic, dtype=np.float64)
    if intrinsic.shape == (4, 4):
        intrinsic = intrinsic[:3, :3]
    if intrinsic.shape != (3, 3):
        raise RuntimeError(f"Expected a 3x3 camera intrinsic, got {intrinsic.shape}")
    homogeneous_pixels = (intrinsic @ camera_points.T).T
    depth = camera_points[:, 2]
    pixels = np.full((len(camera_points), 2), np.nan, dtype=np.float64)
    in_front = depth > 1e-3
    pixels[in_front] = homogeneous_pixels[in_front, :2] / depth[in_front, None]
    return (
        pixels,
        depth,
        local_trajectory,
        parser_consistency_error_m,
        planarization_z_abs_max_m,
    )


def _time_color(fraction: float) -> tuple[int, int, int]:
    return trajectory_time_color(fraction)


def _draw_bev(draw: ImageDraw.ImageDraw, trajectory: np.ndarray, width: int, height: int) -> None:
    box_w, box_h = 280, 300
    left, top = width - box_w - 20, height - box_h - 20
    draw.rounded_rectangle((left, top, left + box_w, top + box_h), radius=12, fill=(0, 0, 0, 175))
    origin_x, origin_y = left + box_w // 2, top + box_h - 30
    max_forward = max(float(np.max(trajectory[:, 0])), 10.0)
    max_lateral = max(float(np.max(np.abs(trajectory[:, 1]))), 5.0)
    scale = min((box_h - 60) / max_forward, (box_w / 2 - 20) / max_lateral)
    points = [
        (origin_x - float(y) * scale, origin_y - float(x) * scale) for x, y in trajectory[:, :2]
    ]
    draw.line((origin_x, origin_y, origin_x, top + 18), fill=(120, 120, 120), width=2)
    draw.line((left + 15, origin_y, left + box_w - 15, origin_y), fill=(120, 120, 120), width=2)
    previous = (origin_x, origin_y)
    for index, point in enumerate(points):
        color = _time_color((index + 1) / len(points))
        draw.line((*previous, *point), fill=color, width=5)
        previous = point
    draw.ellipse((origin_x - 5, origin_y - 5, origin_x + 5, origin_y + 5), fill=(80, 180, 255))
    draw.text((left + 10, top + 8), "BEV: x forward, y left", fill="white")
    draw.text((origin_x + 5, top + 20), "x+", fill=(220, 220, 220))
    draw.text((left + 12, origin_y - 20), "y+", fill=(220, 220, 220))


def _render_frame(
    image: Image.Image,
    pixels: np.ndarray,
    depth: np.ndarray,
    trajectory: np.ndarray,
    future_times_s: np.ndarray,
    scene_token: str,
    elapsed_s: float,
    max_error_us: int,
    horizon_s: float,
) -> tuple[Image.Image, int]:
    frame = image.convert("RGB")
    overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = frame.size
    visible = (
        (depth > 0.1)
        & np.isfinite(pixels).all(axis=1)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    for index in range(1, len(pixels)):
        if not (visible[index - 1] and visible[index]):
            continue
        color = _time_color(index / (len(pixels) - 1))
        p0 = tuple(map(float, pixels[index - 1]))
        p1 = tuple(map(float, pixels[index]))
        draw.line((*p0, *p1), fill=(*color, 245), width=8)
    for index, time_s in enumerate(future_times_s):
        if not visible[index] or (index + 1) % 6:
            continue
        x, y = map(float, pixels[index])
        color = _time_color(index / max(len(pixels) - 1, 1))
        radius = 7
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(*color, 255))
        draw.text((x + 9, y - 14), f"{time_s:.1f}s", fill=(*color, 255))

    endpoint = trajectory[-1]
    draw.rounded_rectangle((14, 12, 740, 90), radius=10, fill=(0, 0, 0, 180))
    draw.text(
        (28, 22),
        f"CAM_FRONT | GT ego trajectory inside visual window | clip t={elapsed_s:.2f}s",
        fill="white",
    )
    draw.text(
        (28, 51),
        (
            f"scene ...{scene_token[-8:]} | endpoint "
            f"x={endpoint[0]:.1f}m y={endpoint[1]:+.1f}m yaw={math.degrees(endpoint[2]):+.1f}deg "
            f"| nearest LiDAR pose={max_error_us / 1000:.1f}ms"
        ),
        fill=(230, 230, 230),
    )
    draw.rounded_rectangle((14, height - 54, 445, height - 14), radius=8, fill=(0, 0, 0, 170))
    draw.text((27, height - 43), "near 0s", fill=_time_color(0.0))
    draw.line((105, height - 34, 340, height - 34), fill=_time_color(0.5), width=8)
    draw.text((355, height - 43), f"far {horizon_s:g}s", fill=_time_color(1.0))
    _draw_bev(draw, trajectory, width, height)
    return Image.alpha_composite(frame.convert("RGBA"), overlay).convert("RGB"), int(visible.sum())


def _encode_video(frames: list[Image.Image], output: Path, fps: float) -> None:
    if not frames:
        raise RuntimeError("No frames were rendered")
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
    assert process.stdin is not None
    assert process.stderr is not None
    for frame in frames:
        process.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
    process.stdin.close()
    error = process.stderr.read().decode("utf-8", errors="replace")
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}: {error}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_config = config["data"]
    data_root = args.data_root or Path(data_config["data_root"])
    horizon_s = float(data_config["future_horizon_s"])
    trajectory_hz = int(data_config["trajectory_hz"])
    if data_config["trajectory_pose_source"] != "lidar_sweeps":
        raise ValueError("Trajectory video currently requires trajectory_pose_source=lidar_sweeps")
    if data_config["trajectory_sampling"] != "nearest":
        raise ValueError("Trajectory video currently requires trajectory_sampling=nearest")
    _, entries = load_info_pickle(args.info)
    records = InfoSchemaAdapter().adapt(
        entries, scene_lookup=load_nuscenes_scene_lookup(data_root)
    )
    lidar_records, lidar_report = load_lidar_pose_records(
        data_root, {record.scene_token for record in records}
    )
    lidar_scenes: dict[str, list[LidarPoseRecord]] = {}
    for record in lidar_records:
        lidar_scenes.setdefault(record.scene_token, []).append(record)
    scenes = _group_scenes(records)
    max_error_us = round(float(data_config["max_trajectory_time_error_s"]) * 1_000_000)
    scene_token, scene_records, start_us, reference = _select_clip(
        scenes,
        args.scene_token,
        args.start_offset_s,
        args.duration_s,
        horizon_s,
        max_error_us,
    )
    end_us = start_us + round(args.duration_s * 1_000_000)
    clip_records = [item for item in scene_records if start_us <= item.timestamp_us < end_us]
    if not clip_records:
        raise RuntimeError("Selected clip contains no camera frames")
    future_times_s = np.arange(1, round(horizon_s * trajectory_hz) + 1) / float(
        trajectory_hz
    )
    scene_lidar_records = lidar_scenes[scene_token]
    pose_index = TemporalIndex(
        [item.timestamp_us for item in scene_lidar_records], scene_lidar_records
    )
    frames: list[Image.Image] = []
    visible_counts: list[int] = []
    consistency_errors: list[float] = []
    planarization_z_values: list[float] = []
    time_errors_us: list[int] = []
    endpoints: list[list[float]] = []
    for anchor in clip_records:
        anchor_match = pose_index.nearest(anchor.timestamp_us, max_error_us)
        if anchor_match is None:
            continue
        anchor_pose = anchor_match.value.ego_to_global
        future, actual_future_times_s, max_time_error_us = _future_poses(
            anchor.timestamp_us, pose_index, future_times_s, max_error_us
        )
        pixels, depth, trajectory, consistency_error, planarization_z = _project_trajectory(
            anchor, anchor_pose, future
        )
        with Image.open(resolve_data_path(data_root, anchor.cam_front_path)) as image:
            rendered, visible_count = _render_frame(
                image,
                pixels,
                depth,
                trajectory,
                actual_future_times_s,
                scene_token,
                (anchor.timestamp_us - start_us) / 1_000_000,
                max_time_error_us,
                horizon_s,
            )
        frames.append(rendered)
        visible_counts.append(visible_count)
        consistency_errors.append(consistency_error)
        planarization_z_values.append(planarization_z)
        time_errors_us.append(max_time_error_us)
        endpoints.append(trajectory[-1].astype(float).tolist())
    _encode_video(frames, args.output, args.fps)

    report_path = args.report or args.output.with_suffix(".json")
    report = {
        "output": str(args.output),
        "scene_token": scene_token,
        "automatic_reference_timestamp_us": reference.timestamp_us,
        "clip_start_timestamp_us": start_us,
        "duration_s": args.duration_s,
        "fps": args.fps,
        "num_frames": len(frames),
        "future_horizon_s": horizon_s,
        "future_points": len(future_times_s),
        "visible_projected_points_min": min(visible_counts),
        "visible_projected_points_mean": float(np.mean(visible_counts)),
        "visible_projected_points_max": max(visible_counts),
        "max_nearest_lidar_pose_distance_ms": max(time_errors_us) / 1000.0,
        "trajectory_pose_source": "lidar_sweeps",
        "trajectory_sampling": "nearest",
        "lidar_pose_source": lidar_report,
        "projection_parser_consistency_error_m": max(consistency_errors),
        "trajectory_planarization_z_abs_max_m": max(planarization_z_values),
        "endpoint_xyyaw_first": endpoints[0],
        "endpoint_xyyaw_last": endpoints[-1],
        "coordinate_convention": "anchor ego: x forward, y left, yaw CCW",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
