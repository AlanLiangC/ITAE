"""Context-free trajectory decoders and differentiable vehicle integration."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from .common import MLP


def trajectory_to_body_increments(trajectory: Tensor) -> Tensor:
    """Convert anchor-frame `[x,y,yaw]` poses into body-frame SE(2) increments."""
    if trajectory.ndim != 3 or trajectory.shape[-1] != 3:
        raise ValueError(f"Expected trajectory [B,T,3], got {tuple(trajectory.shape)}")
    origin = torch.zeros_like(trajectory[:, :1])
    previous = torch.cat([origin, trajectory[:, :-1]], dim=1)
    global_delta = trajectory[..., :2] - previous[..., :2]
    previous_yaw = previous[..., 2]
    cosine = torch.cos(previous_yaw)
    sine = torch.sin(previous_yaw)
    body_dx = cosine * global_delta[..., 0] + sine * global_delta[..., 1]
    body_dy = -sine * global_delta[..., 0] + cosine * global_delta[..., 1]
    delta_yaw = torch.atan2(
        torch.sin(trajectory[..., 2] - previous_yaw),
        torch.cos(trajectory[..., 2] - previous_yaw),
    )
    return torch.stack([body_dx, body_dy, delta_yaw], dim=-1)


def integrate_se2_increments(increments: Tensor) -> Tensor:
    """Integrate body-frame `[dx,dy,dyaw]` increments into anchor-frame poses."""
    if increments.ndim != 3 or increments.shape[-1] != 3:
        raise ValueError(f"Expected increments [B,T,3], got {tuple(increments.shape)}")
    x = torch.zeros_like(increments[:, 0, 0])
    y = torch.zeros_like(x)
    yaw = torch.zeros_like(x)
    states = []
    for step in range(increments.shape[1]):
        dx, dy, delta_yaw = increments[:, step].unbind(dim=-1)
        cosine = torch.cos(yaw)
        sine = torch.sin(yaw)
        x = x + cosine * dx - sine * dy
        y = y + sine * dx + cosine * dy
        yaw = torch.atan2(torch.sin(yaw + delta_yaw), torch.cos(yaw + delta_yaw))
        states.append(torch.stack([x, y, yaw], dim=-1))
    return torch.stack(states, dim=1)


class SE2IncrementDecoder(nn.Module):
    """Decode interval tokens into body-frame SE(2) increments.

    ``displacement`` preserves the V1 decoder/checkpoint contract. ``velocity`` predicts
    metric body velocity and yaw rate, then multiplies by the measured step duration;
    this makes irregular LiDAR timestamps explicit and gives smoothness losses a stable
    physical unit.
    """

    def __init__(
        self,
        action_dim: int = 128,
        hidden_dim: int = 256,
        steps_per_token: int = 10,
        dropout: float = 0.0,
        parameterization: str = "displacement",
        initial_forward_speed_mps: float = 5.0,
        max_forward_speed_mps: float = 40.0,
        max_lateral_speed_mps: float = 8.0,
        max_yaw_rate_rps: float = 1.5,
    ) -> None:
        super().__init__()
        if steps_per_token <= 0:
            raise ValueError("steps_per_token must be positive")
        if parameterization not in {"displacement", "velocity"}:
            raise ValueError("parameterization must be `displacement` or `velocity`")
        if min(max_forward_speed_mps, max_lateral_speed_mps, max_yaw_rate_rps) <= 0:
            raise ValueError("Velocity and yaw-rate limits must be positive")
        self.steps_per_token = steps_per_token
        self.parameterization = parameterization
        self.max_forward_speed_mps = max_forward_speed_mps
        self.max_lateral_speed_mps = max_lateral_speed_mps
        self.max_yaw_rate_rps = max_yaw_rate_rps
        self.action_projection = nn.Sequential(
            nn.LayerNorm(action_dim), nn.Linear(action_dim, hidden_dim)
        )
        self.step_embedding = nn.Parameter(torch.randn(steps_per_token, hidden_dim) * 0.02)
        self.time_projection = MLP(2, hidden_dim, hidden_dim, dropout)
        self.decoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 4 if parameterization == "displacement" else 3),
        )
        output = self.decoder[-1]
        assert isinstance(output, nn.Linear)
        nn.init.normal_(output.weight, std=1e-3)
        nn.init.zeros_(output.bias)
        with torch.no_grad():
            if parameterization == "displacement":
                output.bias[3] = 1.0
            else:
                output.bias[0] = initial_forward_speed_mps

    def forward(self, action_tokens: Tensor, future_times: Tensor) -> tuple[Tensor, Tensor]:
        if action_tokens.ndim != 3 or future_times.ndim != 2:
            raise ValueError("Expected action_tokens [B,K,D] and future_times [B,T]")
        batch, intervals, _ = action_tokens.shape
        expected_steps = intervals * self.steps_per_token
        if future_times.shape != (batch, expected_steps):
            raise ValueError(
                f"Expected {expected_steps} future times for {intervals} action tokens, "
                f"got {tuple(future_times.shape)}"
            )
        zero = torch.zeros_like(future_times[:, :1])
        delta_t = torch.diff(torch.cat([zero, future_times], dim=1), dim=1)
        if torch.any(delta_t <= 0):
            raise ValueError("future_times must be strictly increasing and positive")
        delta_t = delta_t.reshape(batch, intervals, self.steps_per_token)
        fraction = torch.arange(
            1,
            self.steps_per_token + 1,
            dtype=future_times.dtype,
            device=future_times.device,
        ) / self.steps_per_token
        fraction = fraction.view(1, 1, self.steps_per_token).expand(batch, intervals, -1)
        time_features = torch.stack([delta_t, fraction], dim=-1)

        hidden = self.action_projection(action_tokens).unsqueeze(2)
        hidden = hidden + self.step_embedding.view(1, 1, self.steps_per_token, -1)
        hidden = hidden + self.time_projection(time_features)
        raw = self.decoder(hidden)
        if self.parameterization == "displacement":
            sin_cos = functional.normalize(raw[..., 2:4], dim=-1, eps=1e-6)
            delta_yaw = torch.atan2(sin_cos[..., 0], sin_cos[..., 1])
            increments = torch.cat([raw[..., :2], delta_yaw.unsqueeze(-1)], dim=-1)
        else:
            forward_velocity = self.max_forward_speed_mps * torch.tanh(
                raw[..., 0] / self.max_forward_speed_mps
            )
            lateral_velocity = self.max_lateral_speed_mps * torch.tanh(
                raw[..., 1] / self.max_lateral_speed_mps
            )
            yaw_rate = self.max_yaw_rate_rps * torch.tanh(
                raw[..., 2] / self.max_yaw_rate_rps
            )
            increments = torch.stack(
                [
                    forward_velocity * delta_t,
                    lateral_velocity * delta_t,
                    yaw_rate * delta_t,
                ],
                dim=-1,
            )
        increments = increments.reshape(batch, expected_steps, 3)
        return integrate_se2_increments(increments), increments


class ResidualVelocityDecoder(nn.Module):
    """Decode compact visual tokens into bounded body-rate corrections.

    The projection deliberately has no bias. A zero visual token therefore produces
    an exactly zero correction, while the random projection supplies an immediate
    gradient to a zero-initialized visual-token output layer.
    """

    def __init__(
        self,
        token_dim: int,
        steps_per_token: int = 10,
        max_forward_correction_mps: float = 5.0,
        max_lateral_correction_mps: float = 2.0,
        max_yaw_rate_correction_rps: float = 0.5,
    ) -> None:
        super().__init__()
        if token_dim <= 0 or steps_per_token <= 0:
            raise ValueError("Residual token dimensions must be positive")
        limits = torch.tensor(
            [
                max_forward_correction_mps,
                max_lateral_correction_mps,
                max_yaw_rate_correction_rps,
            ],
            dtype=torch.float32,
        )
        if torch.any(limits <= 0):
            raise ValueError("Residual velocity and yaw-rate limits must be positive")
        self.steps_per_token = steps_per_token
        self.register_buffer("limits", limits, persistent=True)
        self.output = nn.Linear(token_dim, steps_per_token * 3, bias=False)
        nn.init.normal_(self.output.weight, std=0.02)

    def forward(self, tokens: Tensor, future_times: Tensor) -> Tensor:
        if tokens.ndim != 3 or future_times.ndim != 2:
            raise ValueError("Expected residual tokens [B,K,D] and future_times [B,T]")
        batch, intervals, _ = tokens.shape
        expected_steps = intervals * self.steps_per_token
        if future_times.shape != (batch, expected_steps):
            raise ValueError(
                f"Expected {expected_steps} future times for {intervals} residual tokens, "
                f"got {tuple(future_times.shape)}"
            )
        zero = torch.zeros_like(future_times[:, :1])
        delta_t = torch.diff(torch.cat([zero, future_times], dim=1), dim=1)
        if torch.any(delta_t <= 0):
            raise ValueError("future_times must be strictly increasing and positive")
        raw_rates = self.output(tokens).reshape(batch, expected_steps, 3)
        limits = self.limits.to(dtype=raw_rates.dtype).view(1, 1, 3)
        rates = limits * torch.tanh(raw_rates / limits)
        return rates * delta_t.unsqueeze(-1)
