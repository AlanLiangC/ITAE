"""Build leak-free configurable image/trajectory windows."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import interpolate_pose_at_timestamp, poses_to_local_trajectory
from .lidar import LidarPoseRecord
from .schema import FrameRecord, resolve_data_path
from .temporal import TemporalIndex


@dataclass(frozen=True)
class WindowRecord:
    """Serializable training window with configurable images and trajectory points."""

    sample_token: str
    scene_token: str
    anchor_timestamp_us: int
    image_paths: list[str]
    image_timestamps_us: list[int]
    frame_times_s: list[float]
    trajectory: list[list[float]]
    future_times_s: list[float]
    max_image_time_error_us: int
    max_trajectory_time_error_us: int
    ego_motion_states: list[list[float]] | None = None
    ego_motion_times_s: list[float] | None = None
    dataset_name: str = "nuscenes"
    group_token: str | None = None
    coordinate_frame: str = "anchor_ego_x_forward_y_left"
    reference_point: str = "native_ego"
    native_trajectory_hz: float | None = None
    native_trajectory: list[list[float]] | None = None
    native_future_times_s: list[float] | None = None
    schema_version: int = 1


class ManifestBuilder:
    """Create fixed-shape windows without crossing nuScenes scene boundaries."""

    def __init__(
        self,
        data_root: str | Path,
        frame_offsets_s: list[float] | None = None,
        horizon_s: float = 5.0,
        trajectory_hz: int = 12,
        anchor_stride_s: float = 0.5,
        max_time_error_s: float | None = 0.055,
        max_image_time_error_s: float | None = None,
        max_trajectory_time_error_s: float | None = None,
        max_pose_interpolation_gap_s: float = 0.75,
        image_source: str = "all",
        trajectory_pose_source: str = "frame_keyframes",
        trajectory_sampling: str = "interpolate",
        export_ego_motion_condition: bool = False,
    ) -> None:
        if trajectory_hz <= 0 or horizon_s <= 0:
            raise ValueError("trajectory_hz and horizon_s must be positive")
        if image_source not in {"all", "keyframe"}:
            raise ValueError("image_source must be 'all' or 'keyframe'")
        if trajectory_pose_source not in {"frame_keyframes", "lidar_sweeps"}:
            raise ValueError(
                "trajectory_pose_source must be 'frame_keyframes' or 'lidar_sweeps'"
            )
        if trajectory_sampling not in {"nearest", "interpolate"}:
            raise ValueError("trajectory_sampling must be 'nearest' or 'interpolate'")
        self.data_root = Path(data_root)
        self.frame_offsets_s = frame_offsets_s or [0, 1, 2, 3, 4, 5]
        num_future_points = round(horizon_s * trajectory_hz)
        if not np.isclose(num_future_points, horizon_s * trajectory_hz):
            raise ValueError("horizon_s * trajectory_hz must be an integer")
        self.requested_future_times_s = [
            step / trajectory_hz for step in range(1, num_future_points + 1)
        ]
        self.trajectory_hz = trajectory_hz
        self.anchor_stride_us = round(anchor_stride_s * 1_000_000)
        shared_error_s = 0.055 if max_time_error_s is None else max_time_error_s
        self.max_image_error_us = round(
            (shared_error_s if max_image_time_error_s is None else max_image_time_error_s)
            * 1_000_000
        )
        self.max_trajectory_error_us = round(
            (
                shared_error_s
                if max_trajectory_time_error_s is None
                else max_trajectory_time_error_s
            )
            * 1_000_000
        )
        self.max_pose_interpolation_gap_us = round(
            max_pose_interpolation_gap_s * 1_000_000
        )
        self.image_source = image_source
        self.trajectory_pose_source = trajectory_pose_source
        self.trajectory_sampling = trajectory_sampling
        self.export_ego_motion_condition = export_ego_motion_condition
        if self.export_ego_motion_condition and len(self.frame_offsets_s) < 2:
            raise ValueError("Ego-motion condition requires at least two condition frames")
        if self.export_ego_motion_condition and any(
            offset > 0 for offset in self.frame_offsets_s
        ):
            raise ValueError("Ego-motion condition cannot include future frame offsets")

    @classmethod
    def from_config(
        cls, config: dict[str, Any], data_root: str | Path | None = None
    ) -> ManifestBuilder:
        """Construct a builder from the YAML ``data`` section."""
        data = config["data"]
        return cls(
            data_root=data["data_root"] if data_root is None else data_root,
            frame_offsets_s=list(map(float, data["frame_offsets_s"])),
            horizon_s=float(data["future_horizon_s"]),
            trajectory_hz=int(data["trajectory_hz"]),
            anchor_stride_s=float(data["anchor_stride_s"]),
            max_time_error_s=None,
            max_image_time_error_s=float(data["max_image_time_error_s"]),
            max_trajectory_time_error_s=float(data["max_trajectory_time_error_s"]),
            max_pose_interpolation_gap_s=float(data["max_pose_interpolation_gap_s"]),
            image_source=str(data["image_source"]),
            trajectory_pose_source=str(data["trajectory_pose_source"]),
            trajectory_sampling=str(data["trajectory_sampling"]),
            export_ego_motion_condition=bool(
                config.get("ego_motion_condition", {}).get("enabled", False)
            ),
        )

    def build(
        self,
        records: list[FrameRecord],
        lidar_pose_records: list[LidarPoseRecord] | None = None,
    ) -> tuple[list[WindowRecord], dict[str, Any]]:
        scenes: dict[str, list[FrameRecord]] = {}
        for record in records:
            scenes.setdefault(record.scene_token, []).append(record)
        lidar_scenes: dict[str, list[LidarPoseRecord]] = {}
        for record in lidar_pose_records or []:
            lidar_scenes.setdefault(record.scene_token, []).append(record)

        windows: list[WindowRecord] = []
        rejected = {
            "image_time_mismatch": 0,
            "trajectory_time_mismatch": 0,
            "missing_image": 0,
            "duplicate_timestamp": 0,
            "duplicate_camera_timestamp": 0,
            "duplicate_trajectory_pose": 0,
            "anchor_lidar_mismatch": 0,
            "trajectory_interpolation_gap": 0,
            "ego_motion_time_mismatch": 0,
        }
        image_errors: list[int] = []
        trajectory_errors: list[int] = []

        for scene_token, scene_records in scenes.items():
            scene_records = sorted(scene_records, key=lambda item: item.timestamp_us)
            official_records = self._deduplicate_frames(
                [record for record in scene_records if not record.pose_interpolated],
                "timestamp_us",
                rejected,
                "duplicate_timestamp",
            )
            if not official_records:
                continue
            image_candidates = (
                official_records if self.image_source == "keyframe" else scene_records
            )
            camera_records = self._deduplicate_frames(
                sorted(image_candidates, key=lambda item: item.cam_front_timestamp_us),
                "cam_front_timestamp_us",
                rejected,
                "duplicate_camera_timestamp",
            )
            image_index = TemporalIndex(
                [record.cam_front_timestamp_us for record in camera_records], camera_records
            )

            if self.trajectory_pose_source == "lidar_sweeps":
                pose_records: Sequence[FrameRecord | LidarPoseRecord] = lidar_scenes.get(
                    scene_token, []
                )
                if not pose_records:
                    continue
            else:
                pose_records = official_records
            pose_records = sorted(pose_records, key=lambda item: item.timestamp_us)
            pose_timestamps_us = [record.timestamp_us for record in pose_records]
            poses = [record.ego_to_global for record in pose_records]
            pose_index = TemporalIndex(pose_timestamps_us, list(pose_records))
            lidar_keyframes_by_sample = {
                record.sample_token: record
                for record in pose_records
                if isinstance(record, LidarPoseRecord) and record.is_keyframe
            }

            last_anchor_us: int | None = None
            for camera_anchor in official_records:
                anchor_us = camera_anchor.timestamp_us
                if (
                    last_anchor_us is not None
                    and anchor_us - last_anchor_us < self.anchor_stride_us
                ):
                    continue
                last_anchor_us = anchor_us

                if self.trajectory_pose_source == "lidar_sweeps":
                    anchor_lidar = lidar_keyframes_by_sample.get(camera_anchor.sample_token)
                    if (
                        anchor_lidar is None
                        or abs(anchor_lidar.timestamp_us - anchor_us)
                        > self.max_trajectory_error_us
                    ):
                        rejected["anchor_lidar_mismatch"] += 1
                        continue
                    anchor_pose = anchor_lidar.ego_to_global
                else:
                    anchor_pose = camera_anchor.ego_to_global
                image_matches = [
                    image_index.nearest(
                        anchor_us + round(offset * 1_000_000), self.max_image_error_us
                    )
                    for offset in self.frame_offsets_s
                ]
                if any(match is None for match in image_matches):
                    rejected["image_time_mismatch"] += 1
                    continue

                trajectory_targets_us = [
                    anchor_us + round(offset * 1_000_000)
                    for offset in self.requested_future_times_s
                ]
                sampled = self._sample_trajectory(
                    trajectory_targets_us, pose_timestamps_us, poses, pose_index
                )
                if sampled is None:
                    rejected[
                        "trajectory_time_mismatch"
                        if self.trajectory_sampling == "nearest"
                        else "trajectory_interpolation_gap"
                    ] += 1
                    continue
                matched_trajectory_poses, matched_timestamps_us, current_trajectory_errors = sampled
                if any(
                    right <= left
                    for left, right in zip(
                        matched_timestamps_us, matched_timestamps_us[1:], strict=False
                    )
                ):
                    rejected["duplicate_trajectory_pose"] += 1
                    continue

                matched_images = [match.value for match in image_matches if match is not None]
                ego_motion_states = None
                ego_motion_times_s = None
                if self.export_ego_motion_condition:
                    ego_samples = []
                    for record in matched_images:
                        sample = interpolate_pose_at_timestamp(
                            record.cam_front_timestamp_us,
                            pose_timestamps_us,
                            poses,
                            self.max_pose_interpolation_gap_us,
                        )
                        if sample is None:
                            nearest = pose_index.nearest(
                                record.cam_front_timestamp_us,
                                self.max_trajectory_error_us,
                            )
                            sample = (
                                None
                                if nearest is None
                                else (nearest.value.ego_to_global, nearest.error_us)
                            )
                        ego_samples.append(sample)
                    if any(sample is None for sample in ego_samples):
                        rejected["ego_motion_time_mismatch"] += 1
                        continue
                    valid_ego_samples = [
                        sample for sample in ego_samples if sample is not None
                    ]
                    ego_motion_times_s = [
                        (record.cam_front_timestamp_us - anchor_us) / 1_000_000.0
                        for record in matched_images
                    ]
                    if any(
                        right <= left
                        for left, right in zip(
                            ego_motion_times_s,
                            ego_motion_times_s[1:],
                            strict=False,
                        )
                    ):
                        rejected["ego_motion_time_mismatch"] += 1
                        continue
                    ego_motion_states = self._ego_motion_states(
                        anchor_pose,
                        [sample[0] for sample in valid_ego_samples],
                        ego_motion_times_s,
                    ).tolist()
                resolved_paths = [
                    self._resolve_image_path(record.cam_front_path) for record in matched_images
                ]
                if any(not Path(path).is_file() for path in resolved_paths):
                    rejected["missing_image"] += 1
                    continue

                local_trajectory = poses_to_local_trajectory(
                    anchor_pose, matched_trajectory_poses
                )
                current_image_errors = [
                    match.error_us for match in image_matches if match is not None
                ]
                image_errors.extend(current_image_errors)
                trajectory_errors.extend(current_trajectory_errors)
                windows.append(
                    WindowRecord(
                        sample_token=camera_anchor.sample_token,
                        scene_token=scene_token,
                        anchor_timestamp_us=anchor_us,
                        image_paths=resolved_paths,
                        image_timestamps_us=[
                            record.cam_front_timestamp_us for record in matched_images
                        ],
                        frame_times_s=[
                            (record.cam_front_timestamp_us - anchor_us) / 1_000_000.0
                            for record in matched_images
                        ],
                        trajectory=local_trajectory.tolist(),
                        future_times_s=(
                            list(self.requested_future_times_s)
                            if self.trajectory_sampling == "interpolate"
                            else [
                                (timestamp_us - anchor_us) / 1_000_000.0
                                for timestamp_us in matched_timestamps_us
                            ]
                        ),
                        max_image_time_error_us=max(current_image_errors),
                        max_trajectory_time_error_us=max(current_trajectory_errors),
                        ego_motion_states=ego_motion_states,
                        ego_motion_times_s=ego_motion_times_s,
                        schema_version=2,
                    )
                )

        report = {
            "num_input_records": len(records),
            "num_lidar_pose_records": len(lidar_pose_records or []),
            "num_scenes": len(scenes),
            "num_windows": len(windows),
            "rejected": rejected,
            "image_source": self.image_source,
            "trajectory_pose_source": self.trajectory_pose_source,
            "trajectory_sampling": self.trajectory_sampling,
            "frame_offsets_s": self.frame_offsets_s,
            "trajectory_hz": self.trajectory_hz,
            "future_points": len(self.requested_future_times_s),
            "max_image_time_error_us": max(image_errors, default=None),
            "max_trajectory_time_error_us": max(trajectory_errors, default=None),
            "trajectory_xy_abs_max_m": self._trajectory_max(windows),
            "official_camera_keyframes": sum(not record.pose_interpolated for record in records),
            "ignored_interpolated_camera_records": (
                sum(record.pose_interpolated for record in records)
                if self.image_source == "keyframe"
                else 0
            ),
            "max_pose_interpolation_gap_us": self.max_pose_interpolation_gap_us,
            "ego_motion_condition": self.export_ego_motion_condition,
        }
        return windows, report

    @staticmethod
    def _deduplicate_frames(
        records: list[FrameRecord],
        timestamp_attribute: str,
        rejected: dict[str, int],
        rejection_key: str,
    ) -> list[FrameRecord]:
        deduplicated: list[FrameRecord] = []
        for record in records:
            if deduplicated and getattr(record, timestamp_attribute) == getattr(
                deduplicated[-1], timestamp_attribute
            ):
                rejected[rejection_key] += 1
                continue
            deduplicated.append(record)
        return deduplicated

    def _sample_trajectory(
        self,
        targets_us: list[int],
        timestamps_us: list[int],
        poses: list[np.ndarray],
        pose_index: TemporalIndex[FrameRecord | LidarPoseRecord],
    ) -> tuple[list[np.ndarray], list[int], list[int]] | None:
        if self.trajectory_sampling == "nearest":
            matches = [
                pose_index.nearest(target_us, self.max_trajectory_error_us)
                for target_us in targets_us
            ]
            if any(match is None for match in matches):
                return None
            valid = [match for match in matches if match is not None]
            return (
                [match.value.ego_to_global for match in valid],
                [match.value.timestamp_us for match in valid],
                [match.error_us for match in valid],
            )
        samples = [
            interpolate_pose_at_timestamp(
                target_us,
                timestamps_us,
                poses,
                max_gap_us=self.max_pose_interpolation_gap_us,
            )
            for target_us in targets_us
        ]
        if any(sample is None for sample in samples):
            return None
        valid_samples = [sample for sample in samples if sample is not None]
        return (
            [sample[0] for sample in valid_samples],
            targets_us,
            [sample[1] for sample in valid_samples],
        )

    def _resolve_image_path(self, path_value: str) -> str:
        return str(resolve_data_path(self.data_root, path_value))

    @staticmethod
    def _ego_motion_states(
        anchor_pose: np.ndarray,
        poses: list[np.ndarray],
        times_s: list[float],
    ) -> np.ndarray:
        """Return causal pose/velocity states aligned to condition LiDAR poses."""
        local = poses_to_local_trajectory(anchor_pose, poses).astype(np.float64)
        if len(local) < 2 or len(local) != len(times_s):
            raise ValueError("Ego-motion state requires at least two aligned poses")
        segment_states: list[list[float]] = []
        for index in range(1, len(local)):
            dt = float(times_s[index] - times_s[index - 1])
            if dt <= 0:
                raise ValueError("Ego-motion timestamps must be strictly increasing")
            delta_xy = local[index, :2] - local[index - 1, :2]
            yaw = float(local[index - 1, 2])
            cosine, sine = np.cos(yaw), np.sin(yaw)
            body_x = cosine * delta_xy[0] + sine * delta_xy[1]
            body_y = -sine * delta_xy[0] + cosine * delta_xy[1]
            segment_states.append(
                [body_x / dt, body_y / dt, (local[index, 2] - yaw) / dt]
            )
        velocities = np.asarray(
            [segment_states[0], *segment_states], dtype=np.float64
        )
        return np.concatenate([local, velocities], axis=-1).astype(np.float32)

    @staticmethod
    def _trajectory_max(windows: list[WindowRecord]) -> float | None:
        if not windows:
            return None
        return float(max(np.abs(np.asarray(window.trajectory)[:, :2]).max() for window in windows))


def save_manifest(windows: list[WindowRecord], path: str | Path) -> None:
    """Write one JSON object per line so large manifests remain streamable."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for window in windows:
            handle.write(json.dumps(asdict(window), ensure_ascii=False) + "\n")
    temporary.replace(destination)


def load_manifest(path: str | Path) -> list[WindowRecord]:
    """Load a JSONL manifest produced by :func:`save_manifest`."""
    windows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                windows.append(WindowRecord(**json.loads(line)))
            except (TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid manifest line {line_number}: {error}") from error
    return windows


def manifest_scene_tokens(path: str | Path) -> set[str]:
    """Read only scene tokens from a manifest for split-leakage checks."""
    scene_tokens: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                scene_tokens.add(str(payload["scene_token"]))
            except (KeyError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid manifest line {line_number}: {error}") from error
    return scene_tokens


def manifest_group_tokens(path: str | Path) -> set[str]:
    """Read namespaced split groups, falling back to legacy scene tokens."""
    groups: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                dataset = str(payload.get("dataset_name", "nuscenes"))
                group = str(payload.get("group_token") or payload["scene_token"])
                groups.add(group if ":" in group else f"{dataset}:{group}")
            except (KeyError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid manifest line {line_number}: {error}") from error
    return groups
