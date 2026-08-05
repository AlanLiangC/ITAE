"""Adapters for MMDetection3D nuScenes info variants."""

from __future__ import annotations

import json
import pickle
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import make_transform, validate_transform


@dataclass(frozen=True)
class FrameRecord:
    """One interpolated ego pose paired with a CAM_FRONT frame.

    `ego_to_global` is the trajectory pose at `timestamp_us`.
    `camera_ego_to_global` is the ego pose at the actual camera exposure time.
    """

    scene_token: str
    sample_token: str
    timestamp_us: int
    cam_front_path: str
    cam_front_timestamp_us: int
    ego_to_global: np.ndarray
    camera_to_ego: np.ndarray | None = None
    camera_intrinsic: np.ndarray | None = None
    camera_ego_to_global: np.ndarray | None = None
    pose_interpolated: bool = False
    scene_inferred: bool = False


def load_info_pickle(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load a trusted MMDetection3D pickle and normalize its outer container.

    Pickle can execute arbitrary code. This function must only be used with files
    generated locally or obtained from a trusted source.
    """
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)  # noqa: S301 - trusted dataset artifact only.

    if isinstance(payload, list):
        return {}, payload
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported info payload type: {type(payload).__name__}")
    for key in ("data_list", "infos"):
        if key in payload:
            entries = payload[key]
            if not isinstance(entries, list):
                raise TypeError(f"Expected `{key}` to be a list")
            metadata = payload.get("metainfo", payload.get("metadata", {}))
            return metadata if isinstance(metadata, dict) else {}, entries
    raise KeyError("Could not find `data_list` or `infos` in info pickle")


def _first(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _matrix_or_pose(mapping: dict[str, Any], matrix_keys: tuple[str, ...]) -> np.ndarray | None:
    for key in matrix_keys:
        if key in mapping and mapping[key] is not None:
            return validate_transform(np.asarray(mapping[key]))
    rotation = _first(mapping, "ego2global_rotation", "ego_to_global_rotation")
    translation = _first(mapping, "ego2global_translation", "ego_to_global_translation")
    if rotation is not None and translation is not None:
        return make_transform(rotation, translation)
    return None


def _camera_to_ego(camera: dict[str, Any]) -> np.ndarray | None:
    direct = _first(camera, "cam2ego", "camera2ego", "sensor2ego")
    if direct is not None:
        return validate_transform(np.asarray(direct))
    rotation = _first(camera, "sensor2ego_rotation", "cam2ego_rotation")
    translation = _first(camera, "sensor2ego_translation", "cam2ego_translation")
    if rotation is not None and translation is not None:
        return make_transform(rotation, translation)
    return None


def _scene_from_interpolated_token(
    scene_lookup: dict[str, str], token: object | None
) -> str | None:
    """Resolve official and custom 12 Hz tokens to an official nuScenes scene.

    The interpolation export used by this project appends a one-character frame
    index to the originating 32-character nuScenes token.  The appended token is
    not present in the official JSON tables, but its 32-character prefix is.
    """
    if token is None:
        return None
    value = str(token)
    scene = scene_lookup.get(value)
    if scene is None and len(value) > 32:
        scene = scene_lookup.get(value[:32])
    return scene


def _is_suffixed_interpolated_token(token: str) -> bool:
    """Recognize this dataset's ``official_token + interpolation_index`` convention."""
    return (
        len(token) > 32
        and all(character in string.hexdigits for character in token[:32])
        and token[32:].isdigit()
    )


def resolve_data_path(data_root: str | Path, path_value: str | Path) -> Path:
    """Resolve an info path that may already include a ``data/nuscenes`` prefix."""
    root = Path(data_root)
    path = Path(path_value)
    if path.is_absolute():
        return path
    if path.is_file():
        return path
    rooted = root / path
    if rooted.is_file():
        return rooted
    # MMDetection3D infos commonly store paths such as
    # ``data/nuscenes/samples/...``.  When ``data_root`` already points at the
    # nuScenes directory, retain only the dataset-relative suffix.
    for marker in ("samples", "sweeps", "maps"):
        if marker in path.parts:
            candidate = root.joinpath(*path.parts[path.parts.index(marker) :])
            if candidate.is_file():
                return candidate
    return rooted


class InfoSchemaAdapter:
    """Convert old/new/custom MMDetection3D entries into `FrameRecord` objects."""

    def __init__(self, infer_scene_gap_s: float = 2.0) -> None:
        self.infer_scene_gap_us = int(infer_scene_gap_s * 1_000_000)

    def adapt(
        self,
        entries: list[dict[str, Any]],
        scene_lookup: dict[str, str] | None = None,
    ) -> list[FrameRecord]:
        records: list[FrameRecord] = []
        inferred_scene_index = 0
        previous_timestamp: int | None = None
        previous_explicit_scene: str | None = None

        for index, info in enumerate(entries):
            if not isinstance(info, dict):
                raise TypeError(f"Entry {index} is not a dictionary")
            camera = self._front_camera(info)
            timestamp = int(_first(camera, "timestamp", "cam_timestamp") or info["timestamp"])
            sample_token = str(_first(info, "token", "sample_token", "sample_idx") or index)
            camera_token = _first(camera, "sample_data_token", "token")
            explicit_scene = _first(info, "scene_token", "scene_name", "scene_id")
            if explicit_scene is None and scene_lookup:
                explicit_scene = _scene_from_interpolated_token(scene_lookup, sample_token)
                if explicit_scene is None and camera_token is not None:
                    explicit_scene = _scene_from_interpolated_token(scene_lookup, camera_token)
            inferred = explicit_scene is None

            if inferred:
                # Old MMDetection3D infos omit scene_token. Ordered infos can still be
                # separated conservatively at timestamp resets or large acquisition gaps.
                if previous_timestamp is not None and (
                    timestamp <= previous_timestamp
                    or timestamp - previous_timestamp > self.infer_scene_gap_us
                    or previous_explicit_scene is not None
                ):
                    inferred_scene_index += 1
                scene_token = f"inferred_scene_{inferred_scene_index:05d}"
            else:
                scene_token = str(explicit_scene)

            pose = _matrix_or_pose(info, ("ego2global", "ego_to_global"))
            if pose is None:
                pose = _matrix_or_pose(camera, ("ego2global", "ego_to_global"))
            if pose is None:
                raise KeyError(f"Entry {index} has no parsable ego-to-global pose")
            camera_pose = _matrix_or_pose(camera, ("ego2global", "ego_to_global"))

            path = _first(camera, "img_path", "data_path", "image_path", "filename")
            if path is None:
                raise KeyError(f"Entry {index} CAM_FRONT has no image path")

            intrinsic = _first(
                camera, "cam2img", "cam_intrinsic", "camera_intrinsic", "camera_intrinsics"
            )
            records.append(
                FrameRecord(
                    scene_token=scene_token,
                    sample_token=sample_token,
                    timestamp_us=int(_first(info, "timestamp") or timestamp),
                    cam_front_path=str(path),
                    cam_front_timestamp_us=timestamp,
                    ego_to_global=pose,
                    camera_to_ego=_camera_to_ego(camera),
                    camera_intrinsic=None
                    if intrinsic is None
                    else np.asarray(intrinsic, dtype=np.float64),
                    camera_ego_to_global=camera_pose,
                    pose_interpolated=_is_suffixed_interpolated_token(sample_token),
                    scene_inferred=inferred,
                )
            )
            previous_timestamp = timestamp
            previous_explicit_scene = None if inferred else scene_token
        return records

    @staticmethod
    def _front_camera(info: dict[str, Any]) -> dict[str, Any]:
        for container_key in ("images", "cams", "cameras"):
            container = info.get(container_key)
            if isinstance(container, dict) and isinstance(container.get("CAM_FRONT"), dict):
                return container["CAM_FRONT"]
        direct = info.get("CAM_FRONT")
        if isinstance(direct, dict):
            return direct
        # A custom front-only info may place image fields directly on the sample.
        if any(key in info for key in ("img_path", "data_path", "cam_front_path")):
            camera = dict(info)
            if "cam_front_path" in camera:
                camera["img_path"] = camera["cam_front_path"]
            return camera
        raise KeyError("Could not locate CAM_FRONT data in info entry")


def load_nuscenes_scene_lookup(data_root: str | Path) -> dict[str, str]:
    """Map official sample tokens to scene tokens from nuScenes tables.

    This is preferred over timestamp-gap inference for old MMDetection3D infos that
    omit `scene_token`. Custom tokens in this project retain the official sample
    token as a prefix, so scanning the multi-gigabyte sample-data table is neither
    necessary nor desirable. An empty mapping is returned when tables are absent.
    """
    root = Path(data_root)
    candidates = (
        [
            directory
            for directory in root.iterdir()
            if directory.is_dir() and directory.name.startswith("v1.0-")
        ]
        if root.is_dir()
        else []
    )
    lookup: dict[str, str] = {}
    for directory in candidates:
        sample_path = directory / "sample.json"
        if not sample_path.is_file():
            continue
        samples = json.loads(sample_path.read_text(encoding="utf-8"))
        sample_to_scene = {str(item["token"]): str(item["scene_token"]) for item in samples}
        lookup.update(sample_to_scene)
    return lookup


def inspect_records(
    records: list[FrameRecord], data_root: str | Path | None = None
) -> dict[str, Any]:
    """Compute a compact, JSON-serializable data audit report."""
    if not records:
        return {"num_records": 0}
    scene_counts: dict[str, int] = {}
    missing_paths = 0
    intervals_ms: list[float] = []
    pose_control_intervals_ms: list[float] = []
    by_scene: dict[str, list[FrameRecord]] = {}
    root = None if data_root is None else Path(data_root)
    for record in records:
        scene_counts[record.scene_token] = scene_counts.get(record.scene_token, 0) + 1
        by_scene.setdefault(record.scene_token, []).append(record)
        path = Path(record.cam_front_path)
        resolved = path if root is None else resolve_data_path(root, path)
        missing_paths += int(root is not None and not resolved.exists())
    for scene_records in by_scene.values():
        ordered = sorted(scene_records, key=lambda item: item.cam_front_timestamp_us)
        intervals_ms.extend(
            (b.cam_front_timestamp_us - a.cam_front_timestamp_us) / 1000.0
            for a, b in zip(ordered, ordered[1:])
        )
        pose_controls = sorted(
            (record for record in scene_records if not record.pose_interpolated),
            key=lambda item: item.timestamp_us,
        )
        pose_control_intervals_ms.extend(
            (right.timestamp_us - left.timestamp_us) / 1000.0
            for left, right in zip(pose_controls, pose_controls[1:])
        )
    intervals = np.asarray(intervals_ms, dtype=np.float64)
    pose_intervals = np.asarray(pose_control_intervals_ms, dtype=np.float64)
    return {
        "num_records": len(records),
        "num_scenes": len(scene_counts),
        "scene_inferred": any(record.scene_inferred for record in records),
        "missing_image_paths": missing_paths,
        "records_per_scene_min": min(scene_counts.values()),
        "records_per_scene_max": max(scene_counts.values()),
        "interval_ms_mean": None if intervals.size == 0 else float(intervals.mean()),
        "interval_ms_p95": None if intervals.size == 0 else float(np.percentile(intervals, 95)),
        "interval_ms_max": None if intervals.size == 0 else float(intervals.max()),
        "pose_control_records": sum(not record.pose_interpolated for record in records),
        "ignored_interpolated_pose_records": sum(record.pose_interpolated for record in records),
        "pose_control_interval_ms_p95": (
            None if pose_intervals.size == 0 else float(np.percentile(pose_intervals, 95))
        ),
        "pose_control_interval_ms_max": (
            None if pose_intervals.size == 0 else float(pose_intervals.max())
        ),
    }
