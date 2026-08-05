from __future__ import annotations

import numpy as np

from vision_action_tokenizer.closed_loop import (
    KinematicReplayBackend,
    RecedingHorizonReplayBackend,
    ReplayScenario,
)


def make_scenario() -> ReplayScenario:
    timestamps = np.arange(25) / 12
    ego = np.stack([timestamps, np.zeros_like(timestamps), np.zeros_like(timestamps)], axis=-1)
    return ReplayScenario("straight", timestamps, ego)


def make_plan() -> np.ndarray:
    times = np.arange(1, 61) / 12
    return np.stack([times, np.zeros_like(times), np.zeros_like(times)], axis=-1)


def test_l0_and_l1_straight_rollout() -> None:
    for backend_type in (RecedingHorizonReplayBackend, KinematicReplayBackend):
        backend = backend_type(make_scenario(), execute_points=6)
        backend.reset()
        result = backend.step(make_plan())
        assert result.position_error_m < 1e-9
        assert not result.collision
        assert backend.get_metrics()["steps"] == 1
