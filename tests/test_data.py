from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from vision_action_tokenizer.data.geometry import make_transform, poses_to_local_trajectory
from vision_action_tokenizer.data.manifest import ManifestBuilder
from vision_action_tokenizer.data.schema import FrameRecord, InfoSchemaAdapter
from vision_action_tokenizer.data.temporal import TemporalIndex


def test_pose_to_local_trajectory() -> None:
    anchor = make_transform([1, 0, 0, 0], [10, 20, 0])
    future = make_transform([1, 0, 0, 0], [13, 22, 0])
    trajectory = poses_to_local_trajectory(anchor, [future])
    np.testing.assert_allclose(trajectory[0], [3, 2, 0], atol=1e-6)


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
    np.testing.assert_allclose(windows[0].trajectory[-1][:2], [10.0, 0.0], atol=1e-4)

