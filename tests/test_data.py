from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from vision_action_tokenizer.data.geometry import (
    interpolate_pose_at_timestamp,
    make_transform,
    poses_to_local_trajectory,
)
from vision_action_tokenizer.data.lidar import LidarPoseRecord
from vision_action_tokenizer.data.manifest import (
    ManifestBuilder,
    manifest_scene_tokens,
    save_manifest,
)
from vision_action_tokenizer.data.schema import FrameRecord, InfoSchemaAdapter
from vision_action_tokenizer.data.temporal import TemporalIndex


def test_pose_to_local_trajectory() -> None:
    anchor = make_transform([1, 0, 0, 0], [10, 20, 0])
    future = make_transform([1, 0, 0, 0], [13, 22, 0])
    trajectory = poses_to_local_trajectory(anchor, [future])
    np.testing.assert_allclose(trajectory[0], [3, 2, 0], atol=1e-6)


def test_pose_interpolation_uses_shortest_rotation() -> None:
    yaw_left = np.deg2rad(170.0)
    yaw_right = np.deg2rad(-170.0)
    left = make_transform([np.cos(yaw_left / 2), 0, 0, np.sin(yaw_left / 2)], [0, 0, 0])
    right = make_transform([np.cos(yaw_right / 2), 0, 0, np.sin(yaw_right / 2)], [10, 0, 0])
    sample = interpolate_pose_at_timestamp(50, [0, 100], [left, right], max_gap_us=100)
    assert sample is not None
    pose, nearest_error = sample
    np.testing.assert_allclose(pose[:3, 3], [5, 0, 0], atol=1e-6)
    assert abs(abs(np.arctan2(pose[1, 0], pose[0, 0])) - np.pi) < 1e-6
    assert nearest_error == 50


def test_temporal_nearest_respects_error() -> None:
    index = TemporalIndex([0, 100, 200], ["a", "b", "c"])
    assert index.nearest(145, 50).value == "b"
    assert index.nearest(145, 40) is None


def test_old_and_new_info_schema() -> None:
    old = {
        "token": "old",
        "scene_token": "scene",
        "timestamp": 1,
        "ego2global_rotation": [1, 0, 0, 0],
        "ego2global_translation": [0, 0, 0],
        "cams": {"CAM_FRONT": {"data_path": "old.jpg", "timestamp": 1}},
    }
    new = {
        "token": "new",
        "scene_token": "scene",
        "timestamp": 2,
        "ego2global": np.eye(4),
        "images": {"CAM_FRONT": {"img_path": "new.jpg", "timestamp": 2}},
    }
    records = InfoSchemaAdapter().adapt([old, new])
    assert [record.cam_front_path for record in records] == ["old.jpg", "new.jpg"]
    assert all(not record.scene_inferred for record in records)


def test_custom_suffixed_token_marks_unreliable_interpolated_pose() -> None:
    entry = {
        "token": "a" * 32 + "3",
        "scene_token": "scene",
        "timestamp": 1,
        "ego2global_rotation": [1, 0, 0, 0],
        "ego2global_translation": [0, 0, 0],
        "cams": {"CAM_FRONT": {"data_path": "frame.jpg", "timestamp": 1}},
    }
    record = InfoSchemaAdapter().adapt([entry])[0]
    assert record.pose_interpolated


def test_manifest_builds_six_images_and_sixty_points(tmp_path: Path) -> None:
    records = []
    # Seven seconds at 12 Hz provide multiple valid 5-second windows.
    for index in range(85):
        timestamp = round(index / 12 * 1_000_000)
        image_path = tmp_path / f"{index:03d}.jpg"
        Image.new("RGB", (32, 18)).save(image_path)
        pose = make_transform([1, 0, 0, 0], [index / 12 * 2.0, 0, 0])
        records.append(
            FrameRecord(
                scene_token="scene",
                sample_token=str(index),
                timestamp_us=timestamp,
                cam_front_path=str(image_path),
                cam_front_timestamp_us=timestamp,
                ego_to_global=pose,
            )
        )
    windows, report = ManifestBuilder(tmp_path, max_time_error_s=0.05).build(records)
    assert report["num_windows"] > 0
    assert len(windows[0].image_paths) == 6
    assert len(windows[0].trajectory) == 60
    np.testing.assert_allclose(windows[0].future_times_s, np.arange(1, 61) / 12, atol=1e-9)
    np.testing.assert_allclose(windows[0].trajectory[-1][:2], [10.0, 0.0], atol=1e-4)


def test_manifest_uses_keyframe_images_and_measured_lidar_poses(tmp_path: Path) -> None:
    camera_records = []
    for index in range(17):
        timestamp = round(index * 0.5 * 1_000_000)
        image_path = tmp_path / f"keyframe_{index:03d}.jpg"
        Image.new("RGB", (32, 18)).save(image_path)
        camera_records.append(
            FrameRecord(
                scene_token="scene",
                sample_token=f"sample-{index}",
                timestamp_us=timestamp,
                cam_front_path=str(image_path),
                cam_front_timestamp_us=timestamp,
                ego_to_global=make_transform([1, 0, 0, 0], [timestamp / 1e6, 0, 0]),
            )
        )
    lidar_records = []
    for index in range(161):
        timestamp = round(index * 0.05 * 1_000_000)
        lidar_records.append(
            LidarPoseRecord(
                scene_token="scene",
                sample_token=f"sample-{index // 10}",
                sample_data_token=f"lidar-{index}",
                timestamp_us=timestamp,
                ego_to_global=make_transform([1, 0, 0, 0], [timestamp / 1e6, 0, 0]),
                lidar_to_ego=np.eye(4),
                lidar_path=f"lidar-{index}.bin",
                is_keyframe=index % 10 == 0,
            )
        )
    builder = ManifestBuilder(
        tmp_path,
        frame_offsets_s=[0, 1, 2, 3, 4],
        horizon_s=4,
        trajectory_hz=10,
        anchor_stride_s=0,
        max_image_time_error_s=0.01,
        max_trajectory_time_error_s=0.01,
        image_source="keyframe",
        trajectory_pose_source="lidar_sweeps",
        trajectory_sampling="nearest",
    )
    windows, report = builder.build(camera_records, lidar_records)
    assert report["trajectory_sampling"] == "nearest"
    assert len(windows[0].image_paths) == 5
    assert len(windows[0].trajectory) == 40
    np.testing.assert_allclose(windows[0].frame_times_s, [0, 1, 2, 3, 4])
    np.testing.assert_allclose(windows[0].future_times_s, np.arange(1, 41) / 10)
    np.testing.assert_allclose(windows[0].trajectory[-1], [4, 0, 0], atol=1e-6)
    manifest_path = tmp_path / "windows.jsonl"
    save_manifest(windows, manifest_path)
    assert manifest_scene_tokens(manifest_path) == {"scene"}
