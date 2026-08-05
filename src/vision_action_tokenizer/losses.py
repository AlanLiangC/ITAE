"""Masked tokenizer losses for geometry, dynamics, physics and latent alignment."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from .data.geometry import angular_difference
from .models.tokenizer import TokenizerOutput


@dataclass(frozen=True)
class LossConfig:
    reconstruction_weight: float = 1.0
    dynamics_weight: float = 0.25
    physical_weight: float = 0.05
    kl_weight: float = 1e-4
    kl_free_bits: float = 0.05
    kl_warmup_steps: int = 10_000
    alignment_weight: float = 0.2
    visual_transition_weight: float = 0.1
    info_nce_temperature: float = 0.1
    max_accel_mps2: float = 5.0
    max_decel_mps2: float = 8.0
    max_jerk_mps3: float = 10.0
    max_yaw_rate_rps: float = 1.2


def _masked_mean(values: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return values.mean()
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    weights = mask.to(values.dtype).expand_as(values)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def trajectory_derivatives(trajectory: Tensor, future_times: Tensor) -> dict[str, Tensor]:
    """Compute speed, acceleration, jerk and yaw rate on the original T-point grid."""
    if trajectory.shape[:2] != future_times.shape:
        raise ValueError("trajectory [B,T,3] and future_times [B,T] do not align")
    zero_time = torch.zeros_like(future_times[:, :1])
    dt = torch.diff(torch.cat([zero_time, future_times], dim=1), dim=1).clamp_min(1e-4)
    zero_xy = torch.zeros_like(trajectory[:, :1, :2])
    displacement = torch.diff(torch.cat([zero_xy, trajectory[..., :2]], dim=1), dim=1)
    speed = torch.linalg.vector_norm(displacement, dim=-1) / dt

    speed_previous = torch.cat([speed[:, :1], speed[:, :-1]], dim=1)
    acceleration = (speed - speed_previous) / dt
    acceleration_previous = torch.cat([acceleration[:, :1], acceleration[:, :-1]], dim=1)
    jerk = (acceleration - acceleration_previous) / dt

    zero_yaw = torch.zeros_like(trajectory[:, :1, 2])
    yaw_previous = torch.cat([zero_yaw, trajectory[:, :-1, 2]], dim=1)
    yaw_rate = angular_difference(trajectory[..., 2], yaw_previous) / dt
    return {"speed": speed, "acceleration": acceleration, "jerk": jerk, "yaw_rate": yaw_rate}


def reconstruction_loss(prediction: Tensor, target: Tensor, mask: Tensor | None) -> Tensor:
    """Robust XY loss plus periodic yaw loss."""
    xy = torch.sqrt((prediction[..., :2] - target[..., :2]).square() + 1e-6)
    yaw = 1.0 - torch.cos(prediction[..., 2] - target[..., 2])
    return _masked_mean(xy, mask) + _masked_mean(yaw, mask)


def dynamics_loss(
    prediction: Tensor, target: Tensor, future_times: Tensor, mask: Tensor | None
) -> Tensor:
    pred = trajectory_derivatives(prediction, future_times)
    truth = trajectory_derivatives(target, future_times)
    return sum(
        _masked_mean(functional.smooth_l1_loss(pred[key], truth[key], reduction="none"), mask)
        for key in ("speed", "acceleration", "jerk", "yaw_rate")
    )


def physical_loss(
    prediction: Tensor, future_times: Tensor, mask: Tensor | None, config: LossConfig
) -> Tensor:
    derivatives = trajectory_derivatives(prediction, future_times)
    acceleration = derivatives["acceleration"]
    penalties = (
        functional.relu(acceleration - config.max_accel_mps2).square()
        + functional.relu(-acceleration - config.max_decel_mps2).square()
        + functional.relu(derivatives["jerk"].abs() - config.max_jerk_mps3).square()
        + functional.relu(derivatives["yaw_rate"].abs() - config.max_yaw_rate_rps).square()
    )
    return _masked_mean(penalties, mask)


def kl_loss(mean: Tensor, logvar: Tensor, free_bits: float) -> Tensor:
    """Diagonal Gaussian KL with free bits to reduce posterior collapse."""
    per_dimension = 0.5 * (mean.square() + logvar.exp() - logvar - 1.0)
    if free_bits > 0:
        per_dimension = per_dimension.clamp_min(free_bits)
    return per_dimension.mean()


def alignment_loss(visual: Tensor, trajectory: Tensor, temperature: float) -> Tensor:
    """Align modalities without allowing a constant collapsed representation."""
    visual_flat = functional.normalize(visual.flatten(1), dim=-1)
    trajectory_flat = functional.normalize(trajectory.flatten(1), dim=-1)
    cosine = 1.0 - (visual_flat * trajectory_flat).sum(dim=-1).mean()
    logits = visual_flat @ trajectory_flat.transpose(0, 1) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    contrastive = 0.5 * (
        functional.cross_entropy(logits, labels)
        + functional.cross_entropy(logits.transpose(0, 1), labels)
    )
    # VICReg-style variance floor prevents both branches from satisfying cosine loss
    # by mapping every driving behavior to the same action token.
    variance = functional.relu(1.0 - visual.flatten(0, 1).std(dim=0, unbiased=False)).mean()
    variance = (
        variance + functional.relu(1.0 - trajectory.flatten(0, 1).std(dim=0, unbiased=False)).mean()
    )
    return cosine + contrastive + 0.1 * variance


class TokenizerLoss(nn.Module):
    """Aggregate all tokenizer objectives and return individually logged terms."""

    def __init__(self, config: LossConfig | None = None) -> None:
        super().__init__()
        self.config = config or LossConfig()

    def forward(
        self,
        output: TokenizerOutput,
        target_trajectory: Tensor,
        future_times: Tensor,
        trajectory_mask: Tensor | None = None,
        global_step: int = 0,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        rec_vis = reconstruction_loss(output.reconstruction_vis, target_trajectory, trajectory_mask)
        rec_traj = reconstruction_loss(
            output.reconstruction_traj, target_trajectory, trajectory_mask
        )
        reconstruction = rec_vis + rec_traj
        dynamics = dynamics_loss(
            output.reconstruction_vis, target_trajectory, future_times, trajectory_mask
        ) + dynamics_loss(
            output.reconstruction_traj, target_trajectory, future_times, trajectory_mask
        )
        physical = physical_loss(
            output.reconstruction_vis, future_times, trajectory_mask, self.config
        ) + physical_loss(output.reconstruction_traj, future_times, trajectory_mask, self.config)
        kl = kl_loss(output.mean_vis, output.logvar_vis, self.config.kl_free_bits)
        alignment = alignment_loss(
            output.mean_vis, output.latent_traj, self.config.info_nce_temperature
        )
        visual_transition = (
            1.0
            - functional.cosine_similarity(
                output.predicted_transition,
                output.target_transition,
                dim=-1,
            ).mean()
        )
        kl_scale = min(1.0, global_step / max(self.config.kl_warmup_steps, 1))
        total = (
            self.config.reconstruction_weight * reconstruction
            + self.config.dynamics_weight * dynamics
            + self.config.physical_weight * physical
            + self.config.kl_weight * kl_scale * kl
            + self.config.alignment_weight * alignment
            + self.config.visual_transition_weight * visual_transition
        )
        terms = {
            "loss/total": total.detach(),
            "loss/reconstruction": reconstruction.detach(),
            "loss/reconstruction_vis": rec_vis.detach(),
            "loss/reconstruction_traj": rec_traj.detach(),
            "loss/dynamics": dynamics.detach(),
            "loss/physical": physical.detach(),
            "loss/kl": kl.detach(),
            "loss/kl_scale": total.new_tensor(kl_scale),
            "loss/alignment": alignment.detach(),
            "loss/visual_transition": visual_transition.detach(),
            "posterior/std": torch.exp(0.5 * output.logvar_vis).mean().detach(),
        }
        return total, terms
