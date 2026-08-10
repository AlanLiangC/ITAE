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
    residual_velocity_weight: float = 0.0
    residual_yaw_rate_weight: float = 0.0
    residual_mean_weight: float = 0.0
    residual_alignment_weight: float = 0.0
    residual_alignment_temperature: float = 0.1
    residual_alignment_warmup_steps: int = 0
    conditional_shuffle_weight: float = 0.0
    conditional_shuffle_margin: float = 0.01
    conditional_shuffle_warmup_steps: int = 0
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
        weights = {
            name: float(getattr(self.config, name))
            for name in (
                "residual_velocity_weight",
                "residual_yaw_rate_weight",
                "residual_mean_weight",
                "residual_alignment_weight",
                "conditional_shuffle_weight",
            )
        }
        if any(value < 0 for value in weights.values()):
            raise ValueError("Visual residual loss weights must be non-negative")
        if self.config.residual_alignment_temperature <= 0:
            raise ValueError("residual_alignment_temperature must be positive")
        if self.config.conditional_shuffle_margin < 0:
            raise ValueError("conditional_shuffle_margin must be non-negative")
        if min(
            self.config.residual_alignment_warmup_steps,
            self.config.conditional_shuffle_warmup_steps,
        ) < 0:
            raise ValueError("Visual residual warmup steps must be non-negative")

    @staticmethod
    def _warmup_factor(global_step: int, warmup_steps: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        return min(max(global_step, 0) / warmup_steps, 1.0)

    @staticmethod
    def _per_sample_residual_error(
        predicted_rates: Tensor,
        target_rates: Tensor,
        mask: Tensor | None,
    ) -> Tensor:
        xy = functional.smooth_l1_loss(
            predicted_rates[..., :2], target_rates[..., :2], reduction="none"
        ).mean(dim=-1)
        yaw = functional.smooth_l1_loss(
            predicted_rates[..., 2], target_rates[..., 2], reduction="none"
        )
        error = xy + 0.2 * yaw
        if mask is None:
            return error.mean(dim=1)
        valid = mask.to(error.dtype)
        return (error * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

    def forward(
        self,
        output: TokenizerOutput,
        target_trajectory: Tensor,
        future_times: Tensor,
        trajectory_mask: Tensor | None = None,
        global_step: int = 0,
        shuffled_output: TokenizerOutput | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
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

        residual_velocity = trajectory_xy.new_zeros(())
        residual_yaw_rate = trajectory_xy.new_zeros(())
        residual_mean = trajectory_xy.new_zeros(())
        residual_alignment = trajectory_xy.new_zeros(())
        residual_alignment_top1 = trajectory_xy.new_zeros(())
        conditional_shuffle = trajectory_xy.new_zeros(())
        conditional_shuffle_gap = trajectory_xy.new_zeros(())
        conditional_prediction_l2 = trajectory_xy.new_zeros(())
        visual_terms: dict[str, Tensor] = {}
        if output.residual_increments is not None:
            if output.base_increments is None or output.visual_residual_tokens is None:
                raise ValueError("Visual residual output is missing its frozen-base tensors")
            target_residual_increments = (
                target_increments - output.base_increments.detach()
            )
            predicted_residual_rates = (
                output.residual_increments / delta_t.unsqueeze(-1)
            )
            target_residual_rates = (
                target_residual_increments / delta_t.unsqueeze(-1)
            )
            residual_velocity = _masked_mean(
                functional.smooth_l1_loss(
                    predicted_residual_rates[..., :2],
                    target_residual_rates[..., :2],
                    reduction="none",
                ),
                trajectory_mask,
            )
            residual_yaw_rate = _masked_mean(
                functional.smooth_l1_loss(
                    predicted_residual_rates[..., 2],
                    target_residual_rates[..., 2],
                    reduction="none",
                ),
                trajectory_mask,
            )
            # A small batch-mean penalty prevents the adapter from recovering the
            # V3.1 failure mode where 99.8% of its energy was a shared correction.
            residual_mean = predicted_residual_rates.mean(dim=0).square().mean()

            batch = predicted_residual_rates.shape[0]
            steps = self.config.steps_per_token
            predicted_summary = predicted_residual_rates.reshape(
                batch, -1, steps, 3
            ).mean(dim=2)
            target_summary = target_residual_rates.reshape(
                batch, -1, steps, 3
            ).mean(dim=2)
            alignment_scale = predicted_summary.new_tensor([5.0, 2.0, 0.5])
            predicted_summary = (predicted_summary / alignment_scale).flatten(1)
            target_summary = (target_summary / alignment_scale).flatten(1)
            valid_alignment = target_summary.norm(dim=-1) > 1e-4
            if int(valid_alignment.sum()) >= 2:
                predicted_embedding = functional.normalize(
                    predicted_summary[valid_alignment], dim=-1, eps=0.05
                )
                target_embedding = functional.normalize(
                    target_summary[valid_alignment], dim=-1, eps=0.05
                )
                logits = (
                    predicted_embedding @ target_embedding.transpose(0, 1)
                ) / self.config.residual_alignment_temperature
                labels = torch.arange(logits.shape[0], device=logits.device)
                residual_alignment = 0.5 * (
                    functional.cross_entropy(logits, labels)
                    + functional.cross_entropy(logits.transpose(0, 1), labels)
                )
                residual_alignment_top1 = (
                    logits.argmax(dim=1) == labels
                ).float().mean()

            if shuffled_output is not None:
                if shuffled_output.residual_increments is None:
                    raise ValueError("Shuffled output has no visual residual increments")
                shuffled_rates = (
                    shuffled_output.residual_increments / delta_t.unsqueeze(-1)
                )
                normal_error = self._per_sample_residual_error(
                    predicted_residual_rates, target_residual_rates, trajectory_mask
                )
                shuffled_error = self._per_sample_residual_error(
                    shuffled_rates, target_residual_rates, trajectory_mask
                )
                conditional_shuffle_gap = (
                    shuffled_error - normal_error
                ).mean()
                conditional_shuffle = functional.relu(
                    self.config.conditional_shuffle_margin
                    + normal_error
                    - shuffled_error
                ).mean()
                conditional_prediction_l2 = torch.linalg.vector_norm(
                    output.reconstruction[..., :2]
                    - shuffled_output.reconstruction[..., :2],
                    dim=-1,
                ).mean()

            alignment_factor = self._warmup_factor(
                global_step, self.config.residual_alignment_warmup_steps
            )
            shuffle_factor = self._warmup_factor(
                global_step, self.config.conditional_shuffle_warmup_steps
            )
            visual_terms = {
                "loss/residual_velocity": residual_velocity.detach(),
                "loss/residual_yaw_rate": residual_yaw_rate.detach(),
                "loss/residual_mean": residual_mean.detach(),
                "loss/residual_alignment": residual_alignment.detach(),
                "loss/conditional_shuffle": conditional_shuffle.detach(),
                "residual/rate_abs_mean": predicted_residual_rates.abs()
                .mean()
                .detach(),
                "residual/token_batch_std": output.visual_residual_tokens.flatten(1)
                .std(dim=0, unbiased=False)
                .mean()
                .detach(),
                "residual/token_abs_mean": output.visual_residual_tokens.abs()
                .mean()
                .detach(),
                "alignment/top1": residual_alignment_top1.detach(),
                "alignment/weight_factor": trajectory_xy.new_tensor(
                    alignment_factor
                ),
                "condition/shuffle_error_gap": conditional_shuffle_gap.detach(),
                "condition/prediction_l2_m": conditional_prediction_l2.detach(),
                "condition/weight_factor": trajectory_xy.new_tensor(shuffle_factor),
            }
        else:
            alignment_factor = 0.0
            shuffle_factor = 0.0

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
            + self.config.residual_velocity_weight * residual_velocity
            + self.config.residual_yaw_rate_weight * residual_yaw_rate
            + self.config.residual_mean_weight * residual_mean
            + alignment_factor
            * self.config.residual_alignment_weight
            * residual_alignment
            + shuffle_factor
            * self.config.conditional_shuffle_weight
            * conditional_shuffle
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
        terms.update(visual_terms)
        return total, terms
