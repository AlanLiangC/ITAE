"""Explicit L0/L1 rollout backends; neither is mislabeled as sensor closed-loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .data.geometry import compose_se2


@dataclass(frozen=True)
class Observation:
    scenario_id: str
    step_index: int
    timestamp_s: float
    ego_state_xyyaw: np.ndarray
    logged_ego_state_xyyaw: np.ndarray
    agents_xy: np.ndarray | None = None


@dataclass(frozen=True)
class StepResult:
    observation: Observation
    executed_relative_xyyaw: np.ndarray
    position_error_m: float
    yaw_error_rad: float
    collision: bool
    done: bool


class ClosedLoopBackend(Protocol):
    """Common planner rollout API reserved for replay, kinematic and future L2 backends."""

    def reset(self, scenario_id: str | None = None) -> Observation: ...

    def step(self, planned_trajectory: np.ndarray) -> StepResult: ...

    def get_metrics(self) -> dict[str, float]: ...


@dataclass(frozen=True)
class ReplayScenario:
    scenario_id: str
    timestamps_s: np.ndarray  # [N]
    ego_states_xyyaw: np.ndarray  # [N,3], global frame
    agents_xy: np.ndarray | None = None  # [N,M,2], NaN for unavailable agents

    def __post_init__(self) -> None:
        if self.ego_states_xyyaw.shape != (len(self.timestamps_s), 3):
            raise ValueError("ego_states_xyyaw must have shape [N,3]")
        if np.any(np.diff(self.timestamps_s) <= 0):
            raise ValueError("scenario timestamps must be strictly increasing")


class _BaseReplayBackend:
    def __init__(
        self,
        scenario: ReplayScenario,
        execute_points: int = 6,
        collision_radius_m: float = 2.0,
    ) -> None:
        self.scenario = scenario
        self.execute_points = execute_points
        self.collision_radius_m = collision_radius_m
        self.index = 0
        self.current_state = scenario.ego_states_xyyaw[0].copy()
        self.errors: list[float] = []
        self.yaw_errors: list[float] = []
        self.collisions = 0

    def reset(self, scenario_id: str | None = None) -> Observation:
        if scenario_id is not None and scenario_id != self.scenario.scenario_id:
            raise KeyError(f"Backend only contains scenario {self.scenario.scenario_id}")
        self.index = 0
        self.current_state = self.scenario.ego_states_xyyaw[0].copy()
        self.errors.clear()
        self.yaw_errors.clear()
        self.collisions = 0
        return self._observation()

    def _observation(self) -> Observation:
        agents = None if self.scenario.agents_xy is None else self.scenario.agents_xy[self.index]
        return Observation(
            scenario_id=self.scenario.scenario_id,
            step_index=self.index,
            timestamp_s=float(self.scenario.timestamps_s[self.index]),
            ego_state_xyyaw=self.current_state.copy(),
            logged_ego_state_xyyaw=self.scenario.ego_states_xyyaw[self.index].copy(),
            agents_xy=None if agents is None else agents.copy(),
        )

    def _collision(self) -> bool:
        if self.scenario.agents_xy is None:
            return False
        agents = self.scenario.agents_xy[self.index]
        valid = np.isfinite(agents).all(axis=-1)
        if not valid.any():
            return False
        distances = np.linalg.norm(agents[valid] - self.current_state[:2], axis=-1)
        return bool(np.any(distances < self.collision_radius_m))

    def get_metrics(self) -> dict[str, float]:
        return {
            "mean_position_error_m": float(np.mean(self.errors)) if self.errors else 0.0,
            "mean_yaw_error_rad": float(np.mean(self.yaw_errors)) if self.yaw_errors else 0.0,
            "collision_count": float(self.collisions),
            "steps": float(len(self.errors)),
        }

    @staticmethod
    def _validate_plan(planned_trajectory: np.ndarray, point_index: int) -> np.ndarray:
        plan = np.asarray(planned_trajectory, dtype=np.float64)
        if plan.ndim != 2 or plan.shape[1] != 3 or len(plan) <= point_index:
            raise ValueError("planned_trajectory must have shape [T,3] and cover execute_points")
        if not np.isfinite(plan).all():
            raise ValueError("planned_trajectory contains non-finite values")
        return plan


class RecedingHorizonReplayBackend(_BaseReplayBackend):
    """L0: advance to the next logged observation and measure the executed plan prefix."""

    def step(self, planned_trajectory: np.ndarray) -> StepResult:
        point_index = self.execute_points - 1
        plan = self._validate_plan(planned_trajectory, point_index)
        previous_logged = self.scenario.ego_states_xyyaw[self.index]
        self.index = min(self.index + self.execute_points, len(self.scenario.timestamps_s) - 1)
        logged_next = self.scenario.ego_states_xyyaw[self.index]
        predicted_global = compose_se2(previous_logged, plan[point_index])
        position_error = float(np.linalg.norm(predicted_global[:2] - logged_next[:2]))
        yaw_error = float(
            abs(
                np.arctan2(
                    np.sin(predicted_global[2] - logged_next[2]),
                    np.cos(predicted_global[2] - logged_next[2]),
                )
            )
        )
        # L0 intentionally snaps ego back to the log. This is why it is pseudo closed-loop.
        self.current_state = logged_next.copy()
        collision = self._collision()
        self.errors.append(position_error)
        self.yaw_errors.append(yaw_error)
        self.collisions += int(collision)
        return StepResult(
            self._observation(),
            plan[point_index].copy(),
            position_error,
            yaw_error,
            collision,
            self.index == len(self.scenario.timestamps_s) - 1,
        )


class KinematicReplayBackend(_BaseReplayBackend):
    """L1: execute the predicted prefix off-log while agents continue log replay."""

    def step(self, planned_trajectory: np.ndarray) -> StepResult:
        point_index = self.execute_points - 1
        plan = self._validate_plan(planned_trajectory, point_index)
        self.current_state = compose_se2(self.current_state, plan[point_index])
        self.index = min(self.index + self.execute_points, len(self.scenario.timestamps_s) - 1)
        logged = self.scenario.ego_states_xyyaw[self.index]
        position_error = float(np.linalg.norm(self.current_state[:2] - logged[:2]))
        yaw_error = float(
            abs(
                np.arctan2(
                    np.sin(self.current_state[2] - logged[2]),
                    np.cos(self.current_state[2] - logged[2]),
                )
            )
        )
        collision = self._collision()
        self.errors.append(position_error)
        self.yaw_errors.append(yaw_error)
        self.collisions += int(collision)
        return StepResult(
            self._observation(),
            plan[point_index].copy(),
            position_error,
            yaw_error,
            collision,
            self.index == len(self.scenario.timestamps_s) - 1,
        )
