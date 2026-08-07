"""Small-data losses for interval action tokens and SE(2) reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from .models.decoder import trajectory_to_body_increments
from .models.tokenizer import TokenizerOutput


@dataclass(frozen=True)
class LossConfig:
    trajectory_weight: float = 1.0
    increment_weight: float = 0.5
    keyframe_weight: float = 0.5
    yaw_weight: float = 0.1
    body_velocity_weight: float = 0.0
    yaw_rate_weight: float = 0.0
    acceleration_weight: float = 0.0
    jerk_weight: float = 0.0
    boundary_continuity_weight: float = 0.0
    steps_per_token: int = 10


def _masked_mean(values: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return values.mean()
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    weights = mask.to(values.dtype).expand_as(values)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def trajectory_xy_loss(prediction: Tensor, target: Tensor, mask: Tensor | None) -> Tensor:
    distance = torch.sqrt((prediction[..., :2] - target[..., :2]).square().sum(-1) + 1e-6)
    return _masked_mean(distance, mask)


def periodic_yaw_loss(prediction: Tensor, target: Tensor, mask: Tensor | None) -> Tensor:
    return _masked_mean(1.0 - torch.cos(prediction[..., 2] - target[..., 2]), mask)


def _off_diagonal_cosine(latent: Tensor) -> Tensor:
    if latent.shape[0] < 2:
        return latent.new_zeros(())
    normalized = functional.normalize(latent.flatten(1), dim=-1)
    similarities = normalized @ normalized.transpose(0, 1)
    diagonal = torch.eye(latent.shape[0], dtype=torch.bool, device=latent.device)
    return similarities.masked_select(~diagonal).mean()


class TokenizerLoss(nn.Module):
    """Supervise the sole visual encoder at trajectory, increment and interval scales."""

    def __init__(self, config: LossConfig | None = None) -> None:
        super().__init__()
        self.config = config or LossConfig()
        if self.config.steps_per_token <= 0:
            raise ValueError("steps_per_token must be positive")

    def forward(
        self,
        output: TokenizerOutput,
        target_trajectory: Tensor,
        future_times: Tensor,
        trajectory_mask: Tensor | None = None,
        global_step: int = 0,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        del global_step
        if target_trajectory.shape != output.reconstruction.shape:
            raise ValueError("Target and reconstructed trajectory shapes do not match")
        if target_trajectory.shape[1] % self.config.steps_per_token:
            raise ValueError("Trajectory length must be divisible by steps_per_token")

        trajectory_xy = trajectory_xy_loss(
            output.reconstruction, target_trajectory, trajectory_mask
        )
        trajectory_yaw = periodic_yaw_loss(
            output.reconstruction, target_trajectory, trajectory_mask
        )
        target_increments = trajectory_to_body_increments(target_trajectory)
        zero_time = torch.zeros_like(future_times[:, :1])
        delta_t = torch.diff(torch.cat([zero_time, future_times], dim=1), dim=1)
        if torch.any(delta_t <= 0):
            raise ValueError("future_times must be strictly increasing and positive")
        increment_xy = _masked_mean(
            functional.smooth_l1_loss(
                output.predicted_increments[..., :2],
                target_increments[..., :2],
                reduction="none",
            ),
            trajectory_mask,
        )
        increment_yaw = _masked_mean(
            1.0
            - torch.cos(
                output.predicted_increments[..., 2] - target_increments[..., 2]
            ),
            trajectory_mask,
        )
        predicted_body_velocity = output.predicted_increments[..., :2] / delta_t.unsqueeze(-1)
        target_body_velocity = target_increments[..., :2] / delta_t.unsqueeze(-1)
        body_velocity = _masked_mean(
            functional.smooth_l1_loss(
                predicted_body_velocity, target_body_velocity, reduction="none"
            ),
            trajectory_mask,
        )
        predicted_yaw_rate = output.predicted_increments[..., 2] / delta_t
        target_yaw_rate = target_increments[..., 2] / delta_t
        yaw_rate = _masked_mean(
            functional.smooth_l1_loss(
                predicted_yaw_rate, target_yaw_rate, reduction="none"
            ),
            trajectory_mask,
        )

        acceleration_dt = delta_t[:, 1:].unsqueeze(-1)
        predicted_acceleration = torch.diff(predicted_body_velocity, dim=1) / acceleration_dt
        target_acceleration = torch.diff(target_body_velocity, dim=1) / acceleration_dt
        derivative_mask = (
            None
            if trajectory_mask is None
            else trajectory_mask[:, 1:] & trajectory_mask[:, :-1]
        )
        acceleration = _masked_mean(
            functional.smooth_l1_loss(
                predicted_acceleration, target_acceleration, reduction="none"
            ),
            derivative_mask,
        )
        if predicted_acceleration.shape[1] > 1:
            jerk_dt = delta_t[:, 2:].unsqueeze(-1)
            predicted_jerk = torch.diff(predicted_acceleration, dim=1) / jerk_dt
            target_jerk = torch.diff(target_acceleration, dim=1) / jerk_dt
            jerk_mask = (
                None
                if trajectory_mask is None
                else (
                    trajectory_mask[:, 2:]
                    & trajectory_mask[:, 1:-1]
                    & trajectory_mask[:, :-2]
                )
            )
            jerk = _masked_mean(
                functional.smooth_l1_loss(
                    predicted_jerk, target_jerk, reduction="none"
                ),
                jerk_mask,
            )
        else:
            jerk = acceleration.new_zeros(())

        boundary_left = torch.arange(
            self.config.steps_per_token - 1,
            target_trajectory.shape[1] - 1,
            self.config.steps_per_token,
            device=target_trajectory.device,
        )
        boundary_right = boundary_left + 1
        if len(boundary_left):
            boundary_velocity = functional.smooth_l1_loss(
                predicted_body_velocity[:, boundary_left],
                predicted_body_velocity[:, boundary_right],
            )
            boundary_yaw_rate = functional.smooth_l1_loss(
                predicted_yaw_rate[:, boundary_left],
                predicted_yaw_rate[:, boundary_right],
            )
            boundary_continuity = boundary_velocity + boundary_yaw_rate
        else:
            boundary_continuity = acceleration.new_zeros(())

        keyframe_indices = torch.arange(
            self.config.steps_per_token - 1,
            target_trajectory.shape[1],
            self.config.steps_per_token,
            device=target_trajectory.device,
        )
        keyframe_mask = (
            None if trajectory_mask is None else trajectory_mask[:, keyframe_indices]
        )
        keyframe_xy = trajectory_xy_loss(
            output.reconstruction[:, keyframe_indices],
            target_trajectory[:, keyframe_indices],
            keyframe_mask,
        )
        keyframe_yaw = periodic_yaw_loss(
            output.reconstruction[:, keyframe_indices],
            target_trajectory[:, keyframe_indices],
            keyframe_mask,
        )

        total = (
            self.config.trajectory_weight * trajectory_xy
            + self.config.yaw_weight * trajectory_yaw
            + self.config.increment_weight * (increment_xy + increment_yaw)
            + self.config.keyframe_weight * (keyframe_xy + keyframe_yaw)
            + self.config.body_velocity_weight * body_velocity
            + self.config.yaw_rate_weight * yaw_rate
            + self.config.acceleration_weight * acceleration
            + self.config.jerk_weight * jerk
            + self.config.boundary_continuity_weight * boundary_continuity
        )
        terms = {
            "loss/total": total.detach(),
            "loss/trajectory_xy": trajectory_xy.detach(),
            "loss/trajectory_yaw": trajectory_yaw.detach(),
            "loss/increment_xy": increment_xy.detach(),
            "loss/increment_yaw": increment_yaw.detach(),
            "loss/keyframe_xy": keyframe_xy.detach(),
            "loss/keyframe_yaw": keyframe_yaw.detach(),
            "loss/body_velocity": body_velocity.detach(),
            "loss/yaw_rate": yaw_rate.detach(),
            "loss/acceleration": acceleration.detach(),
            "loss/jerk": jerk.detach(),
            "loss/boundary_continuity": boundary_continuity.detach(),
            "action/batch_std": output.action_tokens.flatten(1)
            .std(dim=0, unbiased=False)
            .mean()
            .detach(),
            "action/offdiag_cosine": _off_diagonal_cosine(output.action_tokens).detach(),
            "action/abs_mean": output.action_tokens.abs().mean().detach(),
        }
        return total, terms
