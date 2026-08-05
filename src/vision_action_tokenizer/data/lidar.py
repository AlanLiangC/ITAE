"""Load timestamped LIDAR_TOP ego poses from official nuScenes metadata tables."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ijson
import numpy as np

from .geometry import make_transform


@dataclass(frozen=True)
class LidarPoseRecord:
    """One measured ego pose at a LIDAR_TOP keyframe or sweep timestamp.

    nuScenes stores the ego pose and the LiDAR calibration separately.  The
    trajectory is formed from ``ego_to_global``; ``lidar_to_ego`` is retained so
    the sensor/ego transform is explicit and auditable rather than conflated with
    vehicle motion.
    """

    scene_token: str
    sample_token: str
    sample_data_token: str
    timestamp_us: int
    ego_to_global: np.ndarray
    lidar_to_ego: np.ndarray
    lidar_path: str
    is_keyframe: bool


def _stream_json_array(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("rb") as handle:
        yield from ijson.items(handle, "item")


def _version_directories(data_root: Path) -> list[Path]:
    if not data_root.is_dir():
        return []
    return sorted(
        (
            directory
            for directory in data_root.iterdir()
            if directory.is_dir() and directory.name.startswith("v1.0-")
        ),
        key=lambda path: ("trainval" not in path.name, path.name),
    )


def _select_version_directory(
    data_root: Path, scene_tokens: set[str]
) -> tuple[Path, dict[str, str]]:
    best: tuple[int, Path, dict[str, str]] | None = None
    for directory in _version_directories(data_root):
        sample_path = directory / "sample.json"
        if not sample_path.is_file():
            continue
        samples = json.loads(sample_path.read_text(encoding="utf-8"))
        sample_to_scene = {
            str(item["token"]): str(item["scene_token"])
            for item in samples
            if str(item["scene_token"]) in scene_tokens
        }
        matched_scenes = len(set(sample_to_scene.values()))
        if best is None or matched_scenes > best[0]:
            best = (matched_scenes, directory, sample_to_scene)
    if best is None or best[0] != len(scene_tokens):
        found = 0 if best is None else best[0]
        raise FileNotFoundError(
            f"No nuScenes metadata version contains all {len(scene_tokens)} requested scenes "
            f"(best match: {found}) under {data_root}"
        )
    return best[1], best[2]


def _lidar_calibrations(version_dir: Path) -> dict[str, np.ndarray]:
    sensors = json.loads((version_dir / "sensor.json").read_text(encoding="utf-8"))
    lidar_sensor_tokens = {
        str(item["token"]) for item in sensors if str(item.get("channel")) == "LIDAR_TOP"
    }
    calibrations = json.loads(
        (version_dir / "calibrated_sensor.json").read_text(encoding="utf-8")
    )
    return {
        str(item["token"]): make_transform(item["rotation"], item["translation"])
        for item in calibrations
        if str(item["sensor_token"]) in lidar_sensor_tokens
    }


def load_lidar_pose_records(
    data_root: str | Path, scene_tokens: Iterable[str]
) -> tuple[list[LidarPoseRecord], dict[str, Any]]:
    """Stream LIDAR_TOP keyframes/sweeps and their ego poses from nuScenes JSON.

    The large ``sample_data.json`` and ``ego_pose.json`` tables are parsed as
    streams, keeping memory bounded even for expanded metadata exports.
    """

    root = Path(data_root)
    requested_scenes = set(map(str, scene_tokens))
    if not requested_scenes:
        return [], {"num_records": 0, "num_scenes": 0}
    version_dir, sample_to_scene = _select_version_directory(root, requested_scenes)
    lidar_calibrations = _lidar_calibrations(version_dir)
    if not lidar_calibrations:
        raise RuntimeError(f"No LIDAR_TOP calibration found in {version_dir}")

    sample_data_rows: list[dict[str, Any]] = []
    ego_pose_tokens: set[str] = set()
    for item in _stream_json_array(version_dir / "sample_data.json"):
        calibration_token = str(item["calibrated_sensor_token"])
        sample_token = str(item["sample_token"])
        if calibration_token not in lidar_calibrations or sample_token not in sample_to_scene:
            continue
        normalized = {
            "scene_token": sample_to_scene[sample_token],
            "sample_token": sample_token,
            "sample_data_token": str(item["token"]),
            "ego_pose_token": str(item["ego_pose_token"]),
            "calibrated_sensor_token": calibration_token,
            "timestamp_us": int(item["timestamp"]),
            "filename": str(item["filename"]),
            "is_keyframe": bool(item["is_key_frame"]),
        }
        sample_data_rows.append(normalized)
        ego_pose_tokens.add(normalized["ego_pose_token"])

    ego_poses: dict[str, tuple[int, np.ndarray]] = {}
    for item in _stream_json_array(version_dir / "ego_pose.json"):
        token = str(item["token"])
        if token not in ego_pose_tokens:
            continue
        ego_poses[token] = (
            int(item["timestamp"]),
            make_transform(item["rotation"], item["translation"]),
        )

    missing_poses = ego_pose_tokens.difference(ego_poses)
    if missing_poses:
        raise RuntimeError(f"Missing {len(missing_poses)} LIDAR_TOP ego poses in {version_dir}")

    timestamp_mismatches = 0
    records: list[LidarPoseRecord] = []
    for item in sample_data_rows:
        pose_timestamp_us, ego_to_global = ego_poses[item["ego_pose_token"]]
        timestamp_mismatches += int(pose_timestamp_us != item["timestamp_us"])
        records.append(
            LidarPoseRecord(
                scene_token=item["scene_token"],
                sample_token=item["sample_token"],
                sample_data_token=item["sample_data_token"],
                timestamp_us=item["timestamp_us"],
                ego_to_global=ego_to_global,
                lidar_to_ego=lidar_calibrations[item["calibrated_sensor_token"]],
                lidar_path=item["filename"],
                is_keyframe=item["is_keyframe"],
            )
        )
    records.sort(key=lambda record: (record.scene_token, record.timestamp_us))

    intervals_ms: list[float] = []
    previous_by_scene: dict[str, int] = {}
    duplicate_timestamps = 0
    deduplicated: list[LidarPoseRecord] = []
    for record in records:
        previous = previous_by_scene.get(record.scene_token)
        if previous == record.timestamp_us:
            duplicate_timestamps += 1
            continue
        if previous is not None:
            intervals_ms.append((record.timestamp_us - previous) / 1000.0)
        previous_by_scene[record.scene_token] = record.timestamp_us
        deduplicated.append(record)
    interval_array = np.asarray(intervals_ms, dtype=np.float64)
    report = {
        "metadata_version": version_dir.name,
        "num_records": len(deduplicated),
        "num_scenes": len(requested_scenes),
        "num_keyframes": sum(record.is_keyframe for record in deduplicated),
        "num_sweeps": sum(not record.is_keyframe for record in deduplicated),
        "duplicate_timestamps": duplicate_timestamps,
        "ego_pose_timestamp_mismatches": timestamp_mismatches,
        "interval_ms_p01": (
            None if not intervals_ms else float(np.percentile(interval_array, 1))
        ),
        "interval_ms_median": None if not intervals_ms else float(np.median(interval_array)),
        "interval_ms_p99": (
            None if not intervals_ms else float(np.percentile(interval_array, 99))
        ),
    }
    return deduplicated, report
