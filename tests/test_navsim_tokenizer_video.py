from __future__ import annotations

import numpy as np

from tools.visualization.visualize_navsim_tokenizer_video import (
    _contiguous_runs,
    _project_ego_ground_to_camera,
)
from vision_action_tokenizer.data.manifest import WindowRecord


def _window(token: str, timestamp_us: int) -> WindowRecord:
    return WindowRecord(
        sample_token=f"navsim:{token}",
        scene_token="navsim:scene",
        group_token="navsim:log",
        anchor_timestamp_us=timestamp_us,
        image_paths=[],
        image_timestamps_us=[],
        frame_times_s=[0, 1, 2, 3, 4],
        trajectory=[[0.0, 0.0, 0.0]] * 40,
        future_times_s=[index / 10 for index in range(1, 41)],
        max_image_time_error_us=0,
        max_trajectory_time_error_us=0,
        dataset_name="navsim",
    )


def test_navsim_video_contiguous_runs_split_timestamp_gaps() -> None:
    indexed = [
        (0, _window("a", 0)),
        (1, _window("b", 500_000)),
        (2, _window("c", 1_500_000)),
        (3, _window("d", 2_000_000)),
    ]
    runs = _contiguous_runs(indexed)
    assert [[index for index, _ in run] for run in runs] == [[0, 1], [2, 3]]


def test_navsim_ground_projection_uses_official_sensor2lidar_convention() -> None:
    camera_to_lidar = np.array(
        [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
    )
    raw_frame = {
        "lidar2ego_rotation": [1.0, 0.0, 0.0, 0.0],
        "lidar2ego_translation": [0.0, 0.0, 0.0],
        "cams": {
            "CAM_F0": {
                "sensor2lidar_rotation": camera_to_lidar,
                "sensor2lidar_translation": [0.0, 0.0, 1.5],
                "cam_intrinsic": [
                    [1000.0, 0.0, 960.0],
                    [0.0, 1000.0, 560.0],
                    [0.0, 0.0, 1.0],
                ],
            }
        },
    }
    trajectory = np.array([[10.0, 0.0, 0.0], [20.0, -2.0, 0.0]])
    pixels, depth = _project_ego_ground_to_camera(
        trajectory, raw_frame, (0.0, 0.0)
    )
    np.testing.assert_allclose(depth, [10.0, 20.0], atol=1e-6)
    np.testing.assert_allclose(pixels[0], [960.0, 710.0], atol=1e-6)
    np.testing.assert_allclose(pixels[1], [1060.0, 635.0], atol=1e-6)
