"""Context-free trajectory decoders and differentiable vehicle integration."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from .common import MLP, sinusoidal_time_embedding


class TrajectoryDecoder(nn.Module):
    """Decode action tokens to a local `[x,y,yaw]` trajectory without visual context.

    `future_times` is only a query grid in seconds. No current/future PE feature or
    scene condition enters this module, so the action latent must be sufficient.
    """

    def __init__(
        self,
        latent_dim: int,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        decoder_type: str = "kinematic",
        max_speed_mps: float = 35.0,
        max_accel_mps2: float = 8.0,
        max_yaw_rate_rps: float = 1.5,
    ) -> None:
        super().__init__()
        if decoder_type not in {"direct", "kinematic"}:
            raise ValueError("decoder_type must be `direct` or `kinematic`")
        self.decoder_type = decoder_type
        self.model_dim = model_dim
        self.max_speed_mps = max_speed_mps
        self.max_accel_mps2 = max_accel_mps2
        self.max_yaw_rate_rps = max_yaw_rate_rps
        self.memory_projection = nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, model_dim))
        self.time_projection = MLP(model_dim, model_dim, model_dim)
        layer = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers, nn.LayerNorm(model_dim))
        self.direct_head = nn.Linear(model_dim, 4)
        self.control_head = nn.Linear(model_dim, 2)
        self.initial_speed_head = MLP(model_dim, model_dim, 1)

    def forward(self, action_tokens: Tensor, future_times: Tensor) -> Tensor:
        if action_tokens.ndim != 3 or future_times.ndim != 2:
            raise ValueError("Expected action_tokens [B,K,Dz] and future_times [B,T]")
        memory = self.memory_projection(action_tokens)
        queries = self.time_projection(sinusoidal_time_embedding(future_times, self.model_dim))
        decoded = self.decoder(queries, memory)
        if self.decoder_type == "direct":
            raw = self.direct_head(decoded)
            sin_cos = functional.normalize(raw[..., 2:4], dim=-1, eps=1e-6)
            yaw = torch.atan2(sin_cos[..., 0], sin_cos[..., 1])
            return torch.cat([raw[..., :2], yaw.unsqueeze(-1)], dim=-1)

        controls = self.control_head(decoded)
        acceleration = torch.tanh(controls[..., 0]) * self.max_accel_mps2
        yaw_rate = torch.tanh(controls[..., 1]) * self.max_yaw_rate_rps
        initial_speed = torch.sigmoid(self.initial_speed_head(memory.mean(dim=1)).squeeze(-1))
        initial_speed = initial_speed * self.max_speed_mps
        return integrate_unicycle(initial_speed, acceleration, yaw_rate, future_times)


def integrate_unicycle(
    initial_speed: Tensor, acceleration: Tensor, yaw_rate: Tensor, future_times: Tensor
) -> Tensor:
    """Integrate controls to `[x,y,yaw]` using a midpoint unicycle scheme.

    Args:
        initial_speed: `[B]` in m/s, predicted from action tokens.
        acceleration: `[B,T]` in m/s^2.
        yaw_rate: `[B,T]` in rad/s.
        future_times: Strictly increasing `[B,T]` seconds after the anchor.
    """
    if not (acceleration.shape == yaw_rate.shape == future_times.shape):
        raise ValueError("acceleration, yaw_rate and future_times must share [B,T] shape")
    zeros = torch.zeros_like(future_times[:, :1])
    delta_t = torch.diff(torch.cat([zeros, future_times], dim=1), dim=1)
    if torch.any(delta_t <= 0):
        raise ValueError("future_times must be strictly increasing and positive")

    x = torch.zeros_like(initial_speed)
    y = torch.zeros_like(initial_speed)
    yaw = torch.zeros_like(initial_speed)
    speed = initial_speed
    states = []
    for step in range(future_times.shape[1]):
        dt = delta_t[:, step]
        next_speed = (speed + acceleration[:, step] * dt).clamp_min(0.0)
        next_yaw = yaw + yaw_rate[:, step] * dt
        mid_speed = 0.5 * (speed + next_speed)
        mid_yaw = 0.5 * (yaw + next_yaw)
        x = x + mid_speed * torch.cos(mid_yaw) * dt
        y = y + mid_speed * torch.sin(mid_yaw) * dt
        yaw = next_yaw
        speed = next_speed
        states.append(torch.stack([x, y, yaw], dim=-1))
    return torch.stack(states, dim=1)

