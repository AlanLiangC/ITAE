"""Visual-transition and trajectory encoders for a shared action latent."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from .common import MLP, make_transformer_encoder, sinusoidal_time_embedding


class VisualTransitionEncoder(nn.Module):
    """Infer Gaussian action tokens from current-to-future visual transitions.

    Current PE tokens are used only here. They never enter the trajectory decoder.
    The default path differences matching spatial grid cells and preserves that
    ordered residual through fixed pooling. Legacy content matching remains an
    explicit ablation mode.
    """

    def __init__(
        self,
        model_dim: int,
        latent_dim: int,
        num_action_tokens: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        transition_mode: str = "spatial_difference",
    ) -> None:
        super().__init__()
        if transition_mode not in {"spatial_difference", "content_match"}:
            raise ValueError(
                "VisualTransitionEncoder transition_mode must be "
                "`spatial_difference` or `content_match`"
            )
        self.model_dim = model_dim
        self.num_action_tokens = num_action_tokens
        self.transition_mode = transition_mode
        self.time_projection = MLP(model_dim, model_dim, model_dim)
        self.current_match = (
            nn.MultiheadAttention(
                model_dim, num_heads, dropout=dropout, batch_first=True
            )
            if transition_mode == "content_match"
            else None
        )
        self.transition_norm = nn.LayerNorm(model_dim)
        self.transition_encoder = make_transformer_encoder(
            model_dim, num_heads, num_layers, dropout
        )
        if transition_mode == "content_match":
            self.action_queries = nn.Parameter(
                torch.randn(num_action_tokens, model_dim) * 0.02
            )
            self.action_attention = nn.MultiheadAttention(
                model_dim, num_heads, dropout=dropout, batch_first=True
            )
            self.pooled_action_projection = None
        else:
            self.register_parameter("action_queries", None)
            self.action_attention = None
            self.pooled_action_projection = MLP(
                model_dim, model_dim * 2, model_dim, dropout
            )
        self.posterior_norm = nn.LayerNorm(model_dim)
        self.to_mean = nn.Linear(model_dim, latent_dim)
        self.to_logvar = nn.Linear(model_dim, latent_dim)

    def forward(self, frame_tokens: Tensor, frame_times: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if frame_tokens.ndim != 4 or frame_times.ndim != 2:
            raise ValueError("Expected frame_tokens [B,F,R,D] and frame_times [B,F]")
        batch, frames, tokens_per_frame, dim = frame_tokens.shape
        if frames < 2 or dim != self.model_dim:
            raise ValueError("Visual transition encoder needs at least current + one future frame")

        time_embedding = self.time_projection(
            sinusoidal_time_embedding(frame_times, self.model_dim)
        ).unsqueeze(2)
        if self.transition_mode == "content_match":
            tokens = frame_tokens + time_embedding
            current = tokens[:, 0]
            future = tokens[:, 1:].reshape(batch, (frames - 1) * tokens_per_frame, dim)
            assert self.current_match is not None
            matched_current, _ = self.current_match(
                future, current, current, need_weights=False
            )
            transition = future - matched_current
            transition = self.transition_encoder(self.transition_norm(transition))
        else:
            # Grid token r has stable image coordinates across frames. Retaining current,
            # future and their signed difference avoids the content-query collapse observed
            # when every query converges to the same globally averaged scene descriptor.
            current = frame_tokens[:, :1].expand(-1, frames - 1, -1, -1)
            future = frame_tokens[:, 1:]
            delta = functional.layer_norm(future - current, (dim,))
            transition = delta + 0.1 * current + time_embedding[:, 1:]
            transition = transition.reshape(batch, (frames - 1) * tokens_per_frame, dim)
            # An explicit residual around the temporal encoder prevents it from erasing
            # the frozen PE motion delta during optimization.
            transition = transition + self.transition_encoder(
                self.transition_norm(transition)
            )

        horizon = frame_times[:, -1:].clamp_min(1e-3)
        fractions = torch.arange(self.num_action_tokens, device=frame_times.device) + 0.5
        fractions = fractions.to(frame_times.dtype) / self.num_action_tokens
        centers = horizon * fractions.unsqueeze(0)
        action_time = self.time_projection(
            sinusoidal_time_embedding(centers, self.model_dim)
        )
        if self.transition_mode == "content_match":
            assert self.action_queries is not None and self.action_attention is not None
            queries = self.action_queries.unsqueeze(0).expand(batch, -1, -1)
            queries = queries + action_time
            posterior, _ = self.action_attention(
                queries, transition, transition, need_weights=False
            )
        else:
            # Preserve the ordered transition sequence all the way into action tokens.
            # Learned content queries previously converged to identical global averages.
            pooled = functional.adaptive_avg_pool1d(
                transition.transpose(1, 2), self.num_action_tokens
            ).transpose(1, 2)
            assert self.pooled_action_projection is not None
            posterior = pooled + self.pooled_action_projection(pooled) + action_time
        posterior = self.posterior_norm(posterior)
        mean = self.to_mean(posterior)
        logvar = self.to_logvar(posterior).clamp(-10.0, 6.0)
        return mean, logvar, transition


class TrajectoryEncoder(nn.Module):
    """Encode a local `[x,y,yaw]` trajectory into deterministic action tokens."""

    def __init__(
        self,
        model_dim: int,
        latent_dim: int,
        num_action_tokens: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        position_scale_m: float = 50.0,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.num_action_tokens = num_action_tokens
        self.position_scale_m = position_scale_m
        self.input_projection = MLP(4, model_dim * 2, model_dim, dropout)
        self.time_projection = MLP(model_dim, model_dim, model_dim)
        self.encoder = make_transformer_encoder(model_dim, num_heads, num_layers, dropout)
        self.queries = nn.Parameter(torch.randn(num_action_tokens, model_dim) * 0.02)
        self.query_attention = nn.MultiheadAttention(
            model_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.output = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, latent_dim))

    def forward(
        self, trajectory: Tensor, future_times: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        if trajectory.ndim != 3 or trajectory.shape[-1] != 3:
            raise ValueError(f"Expected trajectory [B,T,3], got {tuple(trajectory.shape)}")
        yaw = trajectory[..., 2]
        inputs = torch.cat(
            [
                trajectory[..., :2] / self.position_scale_m,
                torch.sin(yaw).unsqueeze(-1),
                torch.cos(yaw).unsqueeze(-1),
            ],
            dim=-1,
        )
        tokens = self.input_projection(inputs)
        tokens = tokens + self.time_projection(
            sinusoidal_time_embedding(future_times, self.model_dim)
        )
        padding_mask = None if mask is None else ~mask.bool()
        memory = self.encoder(tokens, src_key_padding_mask=padding_mask)

        batch = trajectory.shape[0]
        horizon = future_times[:, -1:].clamp_min(1e-3)
        fractions = torch.arange(self.num_action_tokens, device=trajectory.device) + 0.5
        fractions = fractions.to(trajectory.dtype) / self.num_action_tokens
        centers = horizon * fractions.unsqueeze(0)
        queries = self.queries.unsqueeze(0).expand(batch, -1, -1)
        queries = queries + self.time_projection(sinusoidal_time_embedding(centers, self.model_dim))
        latent, _ = self.query_attention(
            queries, memory, memory, key_padding_mask=padding_mask, need_weights=False
        )
        return self.output(latent)
