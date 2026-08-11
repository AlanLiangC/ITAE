"""NAVSIM manifest export built on the official scene filtering API."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .manifest import WindowRecord
from .trajectory import dense_trajectory_to_native_rate, navsim_2hz_to_10hz


@dataclass(frozen=True)
class NavsimExportConfig:
    data_root: Path
    split: str = "mini"
    num_history_frames: int = 1
    num_future_frames: int = 8
    frame_interval: int = 1
    visual_frame_indices: tuple[int, ...] = (0, 2, 4, 6, 8)
    max_time_error_s: float = 0.06
    has_route: bool = True

    def __post_init__(self) -> None:
        if self.num_history_frames != 1:
            raise ValueError("The 0/1/2/3/4s tokenizer contract requires one anchor history frame")
        if self.num_future_frames != 8:
            raise ValueError("The four-second NAVSIM contract requires eight 2Hz future frames")
        if self.visual_frame_indices != (0, 2, 4, 6, 8):
            raise ValueError("NAVSIM visual_frame_indices must be exactly [0,2,4,6,8]")
        if self.frame_interval <= 0:
            raise ValueError("frame_interval must be positive")
        if self.max_time_error_s <= 0:
            raise ValueError("max_time_error_s must be positive")

    @property
    def log_path(self) -> Path:
        return self.data_root / "navsim_logs" / self.split

    @property
    def sensor_path(self) -> Path:
        return self.data_root / "sensor_blobs" / self.split


def make_navsim_trajectory_from_dense(
    poses: np.ndarray,
    *,
    source_hz: int = 10,
    native_hz: int = 2,
):
    """Build the official 2Hz NAVSIM trajectory object from dense model output.

    This is the metric-facing adapter for a future planner. Importing NAVSIM is
    delayed so manifest/dataset users do not need the evaluation stack installed.
    """
    try:
        from navsim.common.dataclasses import Trajectory
        from nuplan.planning.simulation.trajectory.trajectory_sampling import (
            TrajectorySampling,
        )
    except ImportError as error:  # pragma: no cover - environment-dependent
        raise ImportError("NAVSIM trajectory conversion requires NAVSIM and nuplan") from error
    native = dense_trajectory_to_native_rate(
        poses, source_hz=source_hz, native_hz=native_hz
    )
    return Trajectory(
        native,
        TrajectorySampling(
            num_poses=len(native), interval_length=1.0 / float(native_hz)
        ),
    )


def split_navsim_logs(
    log_path: str | Path,
    *,
    train_fraction: float = 0.8,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Create a deterministic, log-disjoint mini train/validation split."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between zero and one")
    names = sorted(path.stem for path in Path(log_path).glob("*.pkl"))
    if len(names) < 2:
        raise ValueError(f"Need at least two NAVSIM logs under {log_path}")
    random.Random(seed).shuffle(names)
    boundary = min(max(round(len(names) * train_fraction), 1), len(names) - 1)
    train = sorted(names[:boundary])
    validation = sorted(names[boundary:])
    if set(train) & set(validation):
        raise AssertionError("NAVSIM log split is not disjoint")
    return train, validation


def _official_local_future_poses(frames: list[dict[str, Any]]) -> np.ndarray:
    """Match ``Scene.get_future_trajectory`` without loading maps or sensor pixels."""
    try:
        from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
            convert_absolute_to_relative_se2_array,
        )
        from nuplan.common.actor_state.state_representation import StateSE2
        from pyquaternion import Quaternion
    except ImportError as error:  # pragma: no cover - exercised in the NAVSIM environment
        raise ImportError(
            "NAVSIM export requires the editable NAVSIM package and nuplan-devkit"
        ) from error

    global_poses = []
    for frame in frames:
        translation = np.asarray(frame["ego2global_translation"], dtype=np.float64)
        yaw = Quaternion(*frame["ego2global_rotation"]).yaw_pitch_roll[0]
        global_poses.append([translation[0], translation[1], yaw])
    values = np.asarray(global_poses, dtype=np.float64)
    return convert_absolute_to_relative_se2_array(StateSE2(*values[0]), values[1:]).astype(
        np.float32
    )


def build_navsim_windows(
    config: NavsimExportConfig,
    *,
    log_names: list[str] | None = None,
    max_scenes: int | None = None,
) -> tuple[list[WindowRecord], dict[str, Any]]:
    """Build fixed-shape action windows from official NAVSIM log filtering."""
    try:
        from navsim.common.dataclasses import SceneFilter
        from navsim.common.dataloader import filter_scenes
    except ImportError as error:  # pragma: no cover - exercised in the NAVSIM environment
        raise ImportError(
            "NAVSIM export requires `pip install -e third_party/navsim --no-deps`"
        ) from error

    if not config.log_path.is_dir():
        raise FileNotFoundError(f"NAVSIM logs not found: {config.log_path}")
    if not config.sensor_path.is_dir():
        raise FileNotFoundError(f"NAVSIM sensor blobs not found: {config.sensor_path}")
    scene_filter = SceneFilter(
        num_history_frames=config.num_history_frames,
        num_future_frames=config.num_future_frames,
        frame_interval=config.frame_interval,
        has_route=config.has_route,
        max_scenes=max_scenes,
        log_names=log_names,
    )
    candidates, _ = filter_scenes(config.log_path, scene_filter)
    desired_image_times = np.asarray([0, 1, 2, 3, 4], dtype=np.float64)
    desired_native_times = np.arange(1, 9, dtype=np.float64) / 2.0
    max_error_us = round(config.max_time_error_s * 1_000_000)
    windows: list[WindowRecord] = []
    rejected = {
        "cross_scene": 0,
        "timestamp_order": 0,
        "timestamp_mismatch": 0,
        "missing_camera": 0,
        "missing_image": 0,
        "invalid_trajectory": 0,
    }
    max_image_error_us = 0
    max_trajectory_error_us = 0

    for token in sorted(candidates):
        frames = candidates[token]
        if len({str(frame["scene_token"]) for frame in frames}) != 1:
            rejected["cross_scene"] += 1
            continue
        timestamps = np.asarray([int(frame["timestamp"]) for frame in frames], dtype=np.int64)
        if np.any(np.diff(timestamps) <= 0):
            rejected["timestamp_order"] += 1
            continue
        anchor_timestamp = int(timestamps[0])
        frame_times = (timestamps - anchor_timestamp) / 1_000_000.0
        image_times = frame_times[list(config.visual_frame_indices)]
        native_times = frame_times[1:]
        image_errors = np.rint(np.abs(image_times - desired_image_times) * 1_000_000).astype(int)
        trajectory_errors = np.rint(
            np.abs(native_times - desired_native_times) * 1_000_000
        ).astype(int)
        if (
            image_errors.max(initial=0) > max_error_us
            or trajectory_errors.max(initial=0) > max_error_us
        ):
            rejected["timestamp_mismatch"] += 1
            continue

        image_paths: list[str] = []
        missing_camera = False
        for frame_index in config.visual_frame_indices:
            camera = frames[frame_index].get("cams", {}).get("CAM_F0")
            if not camera or not camera.get("data_path"):
                missing_camera = True
                break
            image_paths.append(str(config.sensor_path / str(camera["data_path"])))
        if missing_camera:
            rejected["missing_camera"] += 1
            continue
        if any(not Path(path).is_file() for path in image_paths):
            rejected["missing_image"] += 1
            continue
        try:
            native_trajectory = _official_local_future_poses(frames)
            trajectory, future_times = navsim_2hz_to_10hz(native_trajectory)
        except (AssertionError, KeyError, TypeError, ValueError):
            rejected["invalid_trajectory"] += 1
            continue
        if not np.isfinite(trajectory).all():
            rejected["invalid_trajectory"] += 1
            continue

        log_name = str(frames[0]["log_name"])
        scene_token = str(frames[0]["scene_token"])
        max_image_error_us = max(max_image_error_us, int(image_errors.max(initial=0)))
        max_trajectory_error_us = max(
            max_trajectory_error_us, int(trajectory_errors.max(initial=0))
        )
        windows.append(
            WindowRecord(
                sample_token=f"navsim:{frames[0]['token']}",
                scene_token=f"navsim:{scene_token}",
                group_token=f"navsim:{log_name}",
                anchor_timestamp_us=anchor_timestamp,
                image_paths=image_paths,
                image_timestamps_us=[
                    int(timestamps[index]) for index in config.visual_frame_indices
                ],
                frame_times_s=image_times.astype(float).tolist(),
                trajectory=trajectory.tolist(),
                future_times_s=future_times.tolist(),
                max_image_time_error_us=int(image_errors.max(initial=0)),
                max_trajectory_time_error_us=int(trajectory_errors.max(initial=0)),
                dataset_name="navsim",
                coordinate_frame="anchor_ego_x_forward_y_left",
                reference_point="rear_axle",
                native_trajectory_hz=2.0,
                native_trajectory=native_trajectory.tolist(),
                native_future_times_s=native_times.astype(float).tolist(),
                schema_version=2,
            )
        )

    report: dict[str, Any] = {
        "dataset_name": "navsim",
        "split": config.split,
        "log_names": sorted(log_names) if log_names is not None else None,
        "num_logs": (
            len(log_names)
            if log_names is not None
            else len(list(config.log_path.glob("*.pkl")))
        ),
        "num_candidates": len(candidates),
        "num_windows": len(windows),
        "rejected": rejected,
        "frame_interval": config.frame_interval,
        "frame_offsets_s": desired_image_times.tolist(),
        "native_trajectory_hz": 2,
        "target_trajectory_hz": 10,
        "future_points": 40,
        "max_image_time_error_us": max_image_error_us,
        "max_trajectory_time_error_us": max_trajectory_error_us,
    }
    if not windows:
        return windows, report
    endpoint_distances = [
        float(np.linalg.norm(np.asarray(window.trajectory[-1], dtype=np.float64)[:2]))
        for window in windows
    ]
    motion_buckets = Counter()
    for window, distance in zip(windows, endpoint_distances, strict=True):
        endpoint = np.asarray(window.trajectory[-1], dtype=np.float64)
        if distance < 2.0:
            bucket = "stationary"
        elif abs(endpoint[1]) > 2.0 or abs(endpoint[2]) > 0.15:
            bucket = "turn"
        else:
            bucket = "straight_slow" if distance < 20.0 else "straight_fast"
        motion_buckets[bucket] += 1
    knot_xy_errors = []
    knot_yaw_errors = []
    for window in windows:
        dense = np.asarray(window.trajectory, dtype=np.float64)
        native = np.asarray(window.native_trajectory, dtype=np.float64)
        knots = dense[[4, 9, 14, 19, 24, 29, 34, 39]]
        knot_xy_errors.append(
            float(np.linalg.norm(knots[:, :2] - native[:, :2], axis=-1).max())
        )
        yaw_delta = (knots[:, 2] - native[:, 2] + np.pi) % (2 * np.pi) - np.pi
        knot_yaw_errors.append(float(np.abs(yaw_delta).max()))
    report.update(
        {
            "windows_per_group": dict(
                sorted(Counter(window.group_token for window in windows).items())
            ),
            "motion_bucket_counts": dict(sorted(motion_buckets.items())),
            "endpoint_distance_m": {
                "min": min(endpoint_distances),
                "median": float(np.median(endpoint_distances)),
                "p95": float(np.percentile(endpoint_distances, 95)),
                "max": max(endpoint_distances),
            },
            "max_native_knot_xy_error_m": max(knot_xy_errors),
            "max_native_knot_yaw_error_rad": max(knot_yaw_errors),
        }
    )
    return windows, report
