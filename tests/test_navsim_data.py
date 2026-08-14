from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vision_action_tokenizer.data.manifest import WindowRecord
from vision_action_tokenizer.data.navsim import split_navsim_logs
from vision_action_tokenizer.data.trajectory import (
    dense_trajectory_to_native_rate,
    navsim_2hz_to_10hz,
    resample_se2_trajectory,
    shift_se2_reference_point,
)


def test_navsim_2hz_to_10hz_preserves_native_knots() -> None:
    times = np.arange(1, 9, dtype=np.float64) / 2.0
    native = np.stack([4.0 * times, 0.5 * times, 0.1 * times], axis=-1)
    dense, dense_times = navsim_2hz_to_10hz(native)
    assert dense.shape == (40, 3)
    np.testing.assert_allclose(dense_times, np.arange(1, 41) / 10.0)
    np.testing.assert_allclose(dense[[4, 9, 14, 19, 24, 29, 34, 39]], native, atol=1e-6)


def test_se2_resampling_unwraps_heading() -> None:
    source_times = np.array([0.0, 0.5, 1.0])
    source = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, np.deg2rad(170)], [2.0, 0.0, np.deg2rad(-170)]]
    )
    target = resample_se2_trajectory(source, source_times, np.arange(0.1, 1.01, 0.1))
    continuous = np.unwrap(target[:, 2])
    assert np.max(np.abs(np.diff(continuous))) < np.deg2rad(40)
    np.testing.assert_allclose(target[-1, 2], np.deg2rad(-170), atol=1e-6)


def test_se2_resampling_rejects_extrapolation_and_nonfinite() -> None:
    poses = np.zeros((2, 3))
    with pytest.raises(ValueError, match="extrapolation"):
        resample_se2_trajectory(poses, [0, 1], [0.5, 1.1])
    poses[1, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        resample_se2_trajectory(poses, [0, 1], [0.5, 1.0])


def test_dense_navsim_roundtrip_and_official_dataclass() -> None:
    times = np.arange(1, 9, dtype=np.float64) / 2.0
    native = np.stack([times**2, np.sin(times), 0.2 * times], axis=-1)
    dense, _ = navsim_2hz_to_10hz(native)
    np.testing.assert_allclose(dense_trajectory_to_native_rate(dense), native, atol=1e-6)

    pytest.importorskip("navsim")
    from navsim.evaluate.pdm_score import transform_trajectory
    from nuplan.common.actor_state.ego_state import EgoState
    from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

    from vision_action_tokenizer.data.navsim import make_navsim_trajectory_from_dense

    official = make_navsim_trajectory_from_dense(dense)
    assert official.poses.shape == (8, 3)
    assert official.trajectory_sampling.interval_length == 0.5
    np.testing.assert_allclose(official.poses, native, atol=1e-6)
    initial = EgoState.build_from_rear_axle(
        rear_axle_pose=StateSE2(0.0, 0.0, 0.0),
        rear_axle_velocity_2d=StateVector2D(0.0, 0.0),
        rear_axle_acceleration_2d=StateVector2D(0.0, 0.0),
        tire_steering_angle=0.0,
        time_point=TimePoint(0),
        vehicle_parameters=get_pacifica_parameters(),
    )
    official_interpolated = transform_trajectory(official, initial)
    official_states = official_interpolated.get_state_at_times(
        [TimePoint(round(time_s * 1_000_000)) for time_s in np.arange(1, 41) / 10]
    )
    official_dense = np.asarray(
        [
            [state.rear_axle.x, state.rear_axle.y, state.rear_axle.heading]
            for state in official_states
        ]
    )
    np.testing.assert_allclose(official_dense, dense, atol=1e-5)


def test_reference_point_shift_uses_rigid_body_motion() -> None:
    poses = np.array([[0.0, 0.0, np.pi / 2], [2.0, 0.0, 0.0]])
    shifted = shift_se2_reference_point(poses, (1.0, 0.0))
    np.testing.assert_allclose(shifted[0], [-1.0, 1.0, np.pi / 2], atol=1e-6)
    np.testing.assert_allclose(shifted[1], poses[1], atol=1e-6)


def test_navsim_log_split_is_deterministic_and_disjoint(tmp_path: Path) -> None:
    for index in range(10):
        (tmp_path / f"log-{index}.pkl").touch()
    first = split_navsim_logs(tmp_path, train_fraction=0.7, seed=42)
    second = split_navsim_logs(tmp_path, train_fraction=0.7, seed=42)
    assert first == second
    assert len(first[0]) == 7
    assert not set(first[0]) & set(first[1])


def test_trainval_export_streams_batches_with_stable_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools.data import build_navsim_manifest as exporter
    from vision_action_tokenizer.data.navsim import NavsimExportConfig

    def window(log_name: str) -> WindowRecord:
        native = [[0.5 * step, 0.0, 0.0] for step in range(1, 9)]
        dense = [[0.1 * step, 0.0, 0.0] for step in range(1, 41)]
        return WindowRecord(
            sample_token=f"navsim:{log_name}-sample",
            scene_token=f"navsim:{log_name}-scene",
            group_token=f"navsim:{log_name}",
            anchor_timestamp_us=0,
            image_paths=[],
            image_timestamps_us=[0, 1, 2, 3, 4],
            frame_times_s=[0, 1, 2, 3, 4],
            trajectory=dense,
            future_times_s=[step / 10 for step in range(1, 41)],
            max_image_time_error_us=0,
            max_trajectory_time_error_us=0,
            dataset_name="navsim",
            native_trajectory=native,
        )

    def fake_build(_config, *, log_names, max_scenes):
        selected = list(reversed(log_names))
        if max_scenes is not None:
            selected = selected[:max_scenes]
        windows = [window(name) for name in selected]
        return windows, {
            "num_candidates": len(windows),
            "rejected": {"cross_scene": 0},
            "max_image_time_error_us": 0,
            "max_trajectory_time_error_us": 0,
        }

    monkeypatch.setattr(exporter, "build_navsim_windows", fake_build)
    config = NavsimExportConfig(tmp_path, split="trainval")
    logs = ["log-c", "log-a", "log-b"]
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    first_report, first_groups = exporter._export_split(
        config, logs, first_path, log_batch_size=1, max_scenes=None
    )
    second_report, second_groups = exporter._export_split(
        config, logs, second_path, log_batch_size=2, max_scenes=None
    )
    assert first_path.read_bytes() == second_path.read_bytes()
    assert first_report["num_windows"] == 3
    assert second_report["num_candidates"] == 3
    assert first_groups == second_groups == {
        "navsim:log-a",
        "navsim:log-b",
        "navsim:log-c",
    }


def test_se2_resampling_matches_nuplan_interpolated_trajectory() -> None:
    pytest.importorskip("nuplan")
    from nuplan.common.actor_state.ego_state import EgoState
    from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
        InterpolatedTrajectory,
    )

    source_times = np.arange(9, dtype=np.float64) / 2.0
    source = np.stack(
        [source_times * 3.0, np.sin(source_times) * 2.0, source_times * 0.9], axis=-1
    )
    states = [
        EgoState.build_from_rear_axle(
            rear_axle_pose=StateSE2(*pose),
            rear_axle_velocity_2d=StateVector2D(0.0, 0.0),
            rear_axle_acceleration_2d=StateVector2D(0.0, 0.0),
            tire_steering_angle=0.0,
            time_point=TimePoint(round(time_s * 1_000_000)),
            vehicle_parameters=get_pacifica_parameters(),
        )
        for time_s, pose in zip(source_times, source, strict=True)
    ]
    official = InterpolatedTrajectory(states)
    target_times = np.arange(1, 41, dtype=np.float64) / 10.0
    official_states = official.get_state_at_times(
        [TimePoint(round(time_s * 1_000_000)) for time_s in target_times]
    )
    official_poses = np.asarray(
        [
            [state.rear_axle.x, state.rear_axle.y, state.rear_axle.heading]
            for state in official_states
        ]
    )
    project = resample_se2_trajectory(source, source_times, target_times)
    np.testing.assert_allclose(project, official_poses, atol=1e-5)
