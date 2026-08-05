"""Visual-transition and trajectory encoders for a shared action latent."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .common import MLP, make_transformer_encoder, sinusoidal_time_embedding


class VisualTransitionEncoder(nn.Module):
    """Infer Gaussian action tokens from current-to-future visual transitions.

    Current PE tokens are used only here. They never enter the trajectory decoder.
    Future tokens query the current tokens to estimate a matched baseline; the
    residual becomes transition memory, reducing static appearance leakage.
    """

    def __init__(
        self,
        model_dim: int,
        latent_dim: int,
        num_action_tokens: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.num_action_tokens = num_action_tokens
        self.time_projection = MLP(model_dim, model_dim, model_dim)
        self.current_match = nn.MultiheadAttention(
            model_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.transition_norm = nn.LayerNorm(model_dim)
        self.transition_encoder = make_transformer_encoder(
            model_dim, num_heads, num_layers, dropout
        )
        self.action_queries = nn.Parameter(torch.randn(num_action_tokens, model_dim) * 0.02)
        self.action_attention = nn.MultiheadAttention(
            model_dim, num_heads, dropout=dropout, batch_first=True
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
        tokens = frame_tokens + time_embedding
        current = tokens[:, 0]
        future = tokens[:, 1:].reshape(batch, (frames - 1) * tokens_per_frame, dim)

        # PE patch indices do not track the same world point after ego motion. Cross-attention
        # estimates a content-matched current baseline before forming the visual residual.
        matched_current, _ = self.current_match(future, current, current, need_weights=False)
        transition = self.transition_norm(future - matched_current)
        transition = self.transition_encoder(transition)

        horizon = frame_times[:, -1:].clamp_min(1e-3)
        fractions = torch.arange(self.num_action_tokens, device=frame_times.device) + 0.5
        fractions = fractions.to(frame_times.dtype) / self.num_action_tokens
        centers = horizon * fractions.unsqueeze(0)
        queries = self.action_queries.unsqueeze(0).expand(batch, -1, -1)
        queries = queries + self.time_projection(sinusoidal_time_embedding(centers, self.model_dim))
        posterior, _ = self.action_attention(queries, transition, transition, need_weights=False)
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
