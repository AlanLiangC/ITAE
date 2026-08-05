"""Open-loop trajectory and comfort metrics."""

from __future__ import annotations

import torch
from torch import Tensor

from .data.geometry import angular_difference
from .losses import trajectory_derivatives


def trajectory_metrics(
    prediction: Tensor,
    target: Tensor,
    future_times: Tensor,
    mask: Tensor | None = None,
) -> dict[str, Tensor]:
    """Return batch-averaged ADE/FDE/yaw and dynamics errors."""
    position_error = torch.linalg.vector_norm(prediction[..., :2] - target[..., :2], dim=-1)
    valid = torch.ones_like(position_error, dtype=torch.bool) if mask is None else mask.bool()
    ade = (position_error * valid).sum() / valid.sum().clamp_min(1)
    last_indices = valid.long().sum(dim=1).clamp_min(1) - 1
    fde = position_error.gather(1, last_indices.unsqueeze(1)).mean()
    yaw_error = angular_difference(prediction[..., 2], target[..., 2]).abs()
    yaw_mae = (yaw_error * valid).sum() / valid.sum().clamp_min(1)
    predicted_dynamics = trajectory_derivatives(prediction, future_times)
    target_dynamics = trajectory_derivatives(target, future_times)
    metrics = {"metric/ade_m": ade, "metric/fde_m": fde, "metric/yaw_mae_rad": yaw_mae}
    for key in ("speed", "acceleration", "jerk", "yaw_rate"):
        error = (predicted_dynamics[key] - target_dynamics[key]).abs()
        metrics[f"metric/{key}_mae"] = (error * valid).sum() / valid.sum().clamp_min(1)
    return metrics
