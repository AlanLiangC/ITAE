"""Open-loop trajectory and comfort metrics."""

from __future__ import annotations

import torch
from torch import Tensor

from .data.geometry import angular_difference
from .models.decoder import trajectory_to_body_increments


def trajectory_derivatives(trajectory: Tensor, future_times: Tensor) -> dict[str, Tensor]:
    """Compute motion derivatives on the original trajectory sample grid."""
    if trajectory.shape[:2] != future_times.shape:
        raise ValueError("trajectory [B,T,3] and future_times [B,T] do not align")
    zero_time = torch.zeros_like(future_times[:, :1])
    delta_t = torch.diff(torch.cat([zero_time, future_times], dim=1), dim=1).clamp_min(1e-4)
    zero_xy = torch.zeros_like(trajectory[:, :1, :2])
    displacement = torch.diff(torch.cat([zero_xy, trajectory[..., :2]], dim=1), dim=1)
    speed = torch.linalg.vector_norm(displacement, dim=-1) / delta_t

    previous_speed = torch.cat([speed[:, :1], speed[:, :-1]], dim=1)
    acceleration = (speed - previous_speed) / delta_t
    previous_acceleration = torch.cat([acceleration[:, :1], acceleration[:, :-1]], dim=1)
    jerk = (acceleration - previous_acceleration) / delta_t

    zero_yaw = torch.zeros_like(trajectory[:, :1, 2])
    previous_yaw = torch.cat([zero_yaw, trajectory[:, :-1, 2]], dim=1)
    yaw_rate = angular_difference(trajectory[..., 2], previous_yaw) / delta_t
    return {
        "speed": speed,
        "acceleration": acceleration,
        "jerk": jerk,
        "yaw_rate": yaw_rate,
    }


def trajectory_metrics(
    prediction: Tensor,
    target: Tensor,
    future_times: Tensor,
    mask: Tensor | None = None,
    steps_per_token: int = 10,
) -> dict[str, Tensor]:
    """Return batch-averaged ADE/FDE/yaw and dynamics errors."""
    if steps_per_token <= 0:
        raise ValueError("steps_per_token must be positive")
    position_error = torch.linalg.vector_norm(prediction[..., :2] - target[..., :2], dim=-1)
    valid = torch.ones_like(position_error, dtype=torch.bool) if mask is None else mask.bool()
    ade = (position_error * valid).sum() / valid.sum().clamp_min(1)
    last_indices = valid.long().sum(dim=1).clamp_min(1) - 1
    fde = position_error.gather(1, last_indices.unsqueeze(1)).mean()
    yaw_error = angular_difference(prediction[..., 2], target[..., 2]).abs()
    yaw_mae = (yaw_error * valid).sum() / valid.sum().clamp_min(1)
    predicted_dynamics = trajectory_derivatives(prediction, future_times)
    target_dynamics = trajectory_derivatives(target, future_times)
    keyframe_indices = torch.arange(
        steps_per_token - 1, prediction.shape[1], steps_per_token, device=prediction.device
    )
    keyframe_valid = valid[:, keyframe_indices]
    keyframe_error = position_error[:, keyframe_indices]
    keyframe_ade = (keyframe_error * keyframe_valid).sum() / keyframe_valid.sum().clamp_min(1)
    predicted_increments = trajectory_to_body_increments(prediction)
    target_increments = trajectory_to_body_increments(target)
    increment_xy_error = torch.linalg.vector_norm(
        predicted_increments[..., :2] - target_increments[..., :2], dim=-1
    )
    increment_xy_mae = (increment_xy_error * valid).sum() / valid.sum().clamp_min(1)
    increment_yaw_error = angular_difference(
        predicted_increments[..., 2], target_increments[..., 2]
    ).abs()
    increment_yaw_mae = (increment_yaw_error * valid).sum() / valid.sum().clamp_min(1)
    metrics = {
        "metric/ade_m": ade,
        "metric/fde_m": fde,
        "metric/yaw_mae_rad": yaw_mae,
        "metric/keyframe_ade_m": keyframe_ade,
        "metric/increment_xy_mae_m": increment_xy_mae,
        "metric/increment_yaw_mae_rad": increment_yaw_mae,
    }
    for key in ("speed", "acceleration", "jerk", "yaw_rate"):
        error = (predicted_dynamics[key] - target_dynamics[key]).abs()
        metrics[f"metric/{key}_mae"] = (error * valid).sum() / valid.sum().clamp_min(1)
    return metrics
