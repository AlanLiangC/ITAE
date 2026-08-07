"""VGGT-Omega geometry-to-action tokenizer."""

from __future__ import annotations

import math
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
        num_intervals: int = 4,
        interval_mixer_layers: int = 0,
        interval_mixer_heads: int = 4,
        register_pooling: str = "mean",
        register_summary_tokens: int = 4,
        register_pool_dim: int = 128,
    ) -> None:
        super().__init__()
        if register_pooling not in {"mean", "attention"}:
            raise ValueError("register_pooling must be `mean` or `attention`")
        if interval_mixer_layers < 0:
            raise ValueError("interval_mixer_layers must be non-negative")
        if interval_mixer_layers and action_dim % interval_mixer_heads:
            raise ValueError("action_dim must be divisible by interval_mixer_heads")
        self.frame_geometry_dim = frame_geometry_dim
        self.register_pooling = register_pooling
        if register_pooling == "mean":
            # Keep the V1 module names and shapes so its checkpoints remain loadable.
            self.frame_projection = nn.Sequential(
                nn.LayerNorm(input_dim * 2),
                nn.Linear(input_dim * 2, frame_geometry_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(frame_geometry_dim, frame_geometry_dim),
                nn.LayerNorm(frame_geometry_dim),
            )
        else:
            if register_summary_tokens <= 0 or register_pool_dim <= 0:
                raise ValueError("Register attention dimensions must be positive")
            self.camera_projection = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, frame_geometry_dim),
                nn.GELU(),
            )
            self.register_norm = nn.LayerNorm(input_dim)
            self.register_key = nn.Linear(input_dim, register_pool_dim)
            self.register_value = nn.Linear(input_dim, register_pool_dim)
            self.register_queries = nn.Parameter(
                torch.randn(register_summary_tokens, register_pool_dim) * 0.02
            )
            fusion_dim = frame_geometry_dim + register_summary_tokens * register_pool_dim
            self.frame_projection = nn.Sequential(
                nn.LayerNorm(fusion_dim),
                nn.Linear(fusion_dim, frame_geometry_dim),
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
        self.interval_position: nn.Parameter | None = None
        self.interval_mixer: nn.TransformerEncoder | None = None
        if interval_mixer_layers:
            self.interval_position = nn.Parameter(torch.randn(num_intervals, action_dim) * 0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=action_dim,
                nhead=interval_mixer_heads,
                dim_feedforward=action_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.interval_mixer = nn.TransformerEncoder(
                layer,
                num_layers=interval_mixer_layers,
                norm=nn.LayerNorm(action_dim),
                enable_nested_tensor=False,
            )

    def _frame_geometry(
        self,
        camera_hidden: Tensor,
        register_hidden_mean: Tensor | None,
        register_hidden: Tensor | None,
    ) -> Tensor:
        if self.register_pooling == "mean":
            if register_hidden_mean is None or camera_hidden.shape != register_hidden_mean.shape:
                raise ValueError(
                    "Mean pooling requires camera_hidden and register_hidden_mean [B,F,C]"
                )
            inputs = torch.cat(
                [camera_hidden.float(), register_hidden_mean.float()], dim=-1
            )
            return self.frame_projection(inputs)

        if register_hidden is None or register_hidden.ndim != 4:
            raise ValueError("Attention pooling requires register_hidden [B,F,R,C]")
        if register_hidden.shape[:2] != camera_hidden.shape[:2]:
            raise ValueError("register_hidden must align with camera_hidden [B,F]")
        registers = self.register_norm(register_hidden.float())
        keys = self.register_key(registers)
        values = self.register_value(registers)
        scores = torch.einsum("kd,bfrd->bfkr", self.register_queries, keys)
        scores = scores / math.sqrt(keys.shape[-1])
        summaries = torch.einsum("bfkr,bfrd->bfkd", scores.softmax(dim=-1), values)
        camera = self.camera_projection(camera_hidden.float())
        return self.frame_projection(torch.cat([camera, summaries.flatten(2)], dim=-1))

    def forward(
        self,
        camera_hidden: Tensor,
        register_hidden_mean: Tensor | None,
        frame_times: Tensor,
        register_hidden: Tensor | None = None,
    ) -> Tensor:
        if camera_hidden.ndim != 3:
            raise ValueError("camera_hidden must have shape [B,F,C]")
        if frame_times.shape != camera_hidden.shape[:2]:
            raise ValueError("frame_times must align with CameraHead features [B,F]")
        if camera_hidden.shape[1] < 2:
            raise ValueError("At least two frames are required to form an action interval")
        delta_t = torch.diff(frame_times, dim=1)
        if torch.any(delta_t <= 0):
            raise ValueError("frame_times must be strictly increasing")

        geometry = self._frame_geometry(
            camera_hidden, register_hidden_mean, register_hidden
        )
        time = sinusoidal_time_embedding(delta_t, self.frame_geometry_dim)
        left, right = geometry[:, :-1], geometry[:, 1:]
        interval = torch.cat([left, right, right - left, time], dim=-1)
        action_tokens = self.interval_projection(interval)
        if self.interval_mixer is not None:
            assert self.interval_position is not None
            if action_tokens.shape[1] != self.interval_position.shape[0]:
                raise ValueError("Action interval count does not match mixer positions")
            action_tokens = self.interval_mixer(
                action_tokens + self.interval_position.unsqueeze(0)
            )
        return action_tokens


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
        interval_mixer_layers: int = 0,
        interval_mixer_heads: int = 4,
        register_pooling: str = "mean",
        register_summary_tokens: int = 4,
        register_pool_dim: int = 128,
        decoder_parameterization: str = "displacement",
        initial_forward_speed_mps: float = 5.0,
        max_forward_speed_mps: float = 40.0,
        max_lateral_speed_mps: float = 8.0,
        max_yaw_rate_rps: float = 1.5,
    ) -> None:
        super().__init__()
        self.num_action_tokens = num_action_tokens
        self.steps_per_token = steps_per_token
        self.encoder = IntervalActionEncoder(
            input_dim=vggt_feature_dim,
            frame_geometry_dim=frame_geometry_dim,
            action_dim=action_token_dim,
            dropout=dropout,
            num_intervals=num_action_tokens,
            interval_mixer_layers=interval_mixer_layers,
            interval_mixer_heads=interval_mixer_heads,
            register_pooling=register_pooling,
            register_summary_tokens=register_summary_tokens,
            register_pool_dim=register_pool_dim,
        )
        self.decoder = SE2IncrementDecoder(
            action_dim=action_token_dim,
            hidden_dim=decoder_hidden_dim,
            steps_per_token=steps_per_token,
            dropout=dropout,
            parameterization=decoder_parameterization,
            initial_forward_speed_mps=initial_forward_speed_mps,
            max_forward_speed_mps=max_forward_speed_mps,
            max_lateral_speed_mps=max_lateral_speed_mps,
            max_yaw_rate_rps=max_yaw_rate_rps,
        )

    def forward(
        self,
        camera_hidden: Tensor,
        register_hidden_mean: Tensor,
        frame_times: Tensor,
        future_times: Tensor,
        register_hidden: Tensor | None = None,
    ) -> TokenizerOutput:
        if camera_hidden.shape[1] - 1 != self.num_action_tokens:
            raise ValueError(
                f"Expected {self.num_action_tokens + 1} frames, got {camera_hidden.shape[1]}"
            )
        action_tokens = self.encoder(
            camera_hidden,
            register_hidden_mean,
            frame_times,
            register_hidden=register_hidden,
        )
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
