"""VGGT-Omega geometry-to-action tokenizer."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .common import sinusoidal_time_embedding
from .decoder import SE2IncrementDecoder


@dataclass
class TokenizerOutput:
    """Single-path visual action encoding and context-free reconstruction."""

    action_tokens: Tensor
    reconstruction: Tensor
    predicted_increments: Tensor


class IntervalActionEncoder(nn.Module):
    """Turn five CameraHead frame representations into four motion intervals."""

    def __init__(
        self,
        input_dim: int = 2048,
        frame_geometry_dim: int = 256,
        action_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.frame_geometry_dim = frame_geometry_dim
        self.frame_projection = nn.Sequential(
            nn.LayerNorm(input_dim * 2),
            nn.Linear(input_dim * 2, frame_geometry_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(frame_geometry_dim, frame_geometry_dim),
            nn.LayerNorm(frame_geometry_dim),
        )
        interval_input_dim = frame_geometry_dim * 4
        self.interval_projection = nn.Sequential(
            nn.LayerNorm(interval_input_dim),
            nn.Linear(interval_input_dim, frame_geometry_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(frame_geometry_dim, action_dim),
            nn.LayerNorm(action_dim),
        )

    def forward(
        self,
        camera_hidden: Tensor,
        register_hidden_mean: Tensor,
        frame_times: Tensor,
    ) -> Tensor:
        if camera_hidden.shape != register_hidden_mean.shape or camera_hidden.ndim != 3:
            raise ValueError(
                "camera_hidden and register_hidden_mean must share shape [B,F,C]"
            )
        if frame_times.shape != camera_hidden.shape[:2]:
            raise ValueError("frame_times must align with CameraHead features [B,F]")
        if camera_hidden.shape[1] < 2:
            raise ValueError("At least two frames are required to form an action interval")
        delta_t = torch.diff(frame_times, dim=1)
        if torch.any(delta_t <= 0):
            raise ValueError("frame_times must be strictly increasing")

        geometry = self.frame_projection(
            torch.cat([camera_hidden.float(), register_hidden_mean.float()], dim=-1)
        )
        time = sinusoidal_time_embedding(delta_t, self.frame_geometry_dim)
        left, right = geometry[:, :-1], geometry[:, 1:]
        interval = torch.cat([left, right, right - left, time], dim=-1)
        return self.interval_projection(interval)


class VisionActionTokenizer(nn.Module):
    """Encode VGGT CameraHead hidden tokens and decode a 10 Hz local trajectory."""

    def __init__(
        self,
        vggt_feature_dim: int = 2048,
        frame_geometry_dim: int = 256,
        action_token_dim: int = 128,
        num_action_tokens: int = 4,
        steps_per_token: int = 10,
        decoder_hidden_dim: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_action_tokens = num_action_tokens
        self.steps_per_token = steps_per_token
        self.encoder = IntervalActionEncoder(
            input_dim=vggt_feature_dim,
            frame_geometry_dim=frame_geometry_dim,
            action_dim=action_token_dim,
            dropout=dropout,
        )
        self.decoder = SE2IncrementDecoder(
            action_dim=action_token_dim,
            hidden_dim=decoder_hidden_dim,
            steps_per_token=steps_per_token,
            dropout=dropout,
        )

    def forward(
        self,
        camera_hidden: Tensor,
        register_hidden_mean: Tensor,
        frame_times: Tensor,
        future_times: Tensor,
    ) -> TokenizerOutput:
        if camera_hidden.shape[1] - 1 != self.num_action_tokens:
            raise ValueError(
                f"Expected {self.num_action_tokens + 1} frames, got {camera_hidden.shape[1]}"
            )
        action_tokens = self.encoder(camera_hidden, register_hidden_mean, frame_times)
        reconstruction, increments = self.decoder(action_tokens, future_times)
        return TokenizerOutput(
            action_tokens=action_tokens,
            reconstruction=reconstruction,
            predicted_increments=increments,
        )

    def decode(self, action_tokens: Tensor, future_times: Tensor) -> Tensor:
        """Decode `[B,4,D]` action tokens without any visual context."""
        reconstruction, _ = self.decoder(action_tokens, future_times)
        return reconstruction
