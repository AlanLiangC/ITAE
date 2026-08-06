"""Complete vision-aligned continuous action tokenizer."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from .decoder import TrajectoryDecoder
from .encoders import TrajectoryEncoder, VisualTransitionEncoder
from .resampler import SpatialResampler


@dataclass
class TokenizerOutput:
    """Outputs required by reconstruction, alignment, KL and visual losses."""

    mean_vis: Tensor
    logvar_vis: Tensor
    latent_vis: Tensor
    latent_traj: Tensor
    reconstruction_vis: Tensor
    reconstruction_traj: Tensor
    predicted_transition: Tensor
    target_transition: Tensor


class FutureTransitionPredictor(nn.Module):
    """Predict pooled future PE changes from action tokens as an auxiliary target."""

    def __init__(
        self,
        latent_dim: int,
        model_dim: int,
        output_dim: int,
        num_future_frames: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.num_future_frames = num_future_frames
        self.latent_projection = nn.Linear(latent_dim, model_dim)
        self.frame_queries = nn.Parameter(torch.randn(num_future_frames, model_dim) * 0.02)
        self.attention = nn.MultiheadAttention(model_dim, num_heads, batch_first=True)
        self.output = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, output_dim))

    def forward(self, latent: Tensor) -> Tensor:
        memory = self.latent_projection(latent)
        queries = self.frame_queries.unsqueeze(0).expand(latent.shape[0], -1, -1)
        prediction, _ = self.attention(queries, memory, memory, need_weights=False)
        return self.output(prediction)


class VisionActionTokenizer(nn.Module):
    """Learn a visual action posterior aligned with a trajectory action encoder."""

    def __init__(
        self,
        pe_feature_dim: int = 768,
        model_dim: int = 512,
        latent_dim: int = 256,
        num_action_tokens: int = 10,
        resampled_tokens_per_frame: int = 32,
        num_heads: int = 8,
        encoder_layers: int = 4,
        decoder_layers: int = 4,
        dropout: float = 0.1,
        decoder_type: str = "kinematic",
        num_visual_frames: int = 6,
        max_speed_mps: float = 35.0,
        trajectory_position_scale_m: float = 50.0,
        resampler_type: str = "grid",
        visual_transition_mode: str = "spatial_difference",
    ) -> None:
        super().__init__()
        self.resampler = SpatialResampler(
            pe_feature_dim,
            model_dim,
            resampled_tokens_per_frame,
            num_heads,
            dropout,
            mode=resampler_type,
        )
        self.visual_encoder = VisualTransitionEncoder(
            model_dim,
            latent_dim,
            num_action_tokens,
            num_heads,
            encoder_layers,
            dropout,
            transition_mode=visual_transition_mode,
        )
        self.trajectory_encoder = TrajectoryEncoder(
            model_dim,
            latent_dim,
            num_action_tokens,
            num_heads,
            encoder_layers,
            dropout,
            trajectory_position_scale_m,
        )
        self.decoder = TrajectoryDecoder(
            latent_dim,
            model_dim,
            num_heads,
            decoder_layers,
            dropout,
            decoder_type,
            max_speed_mps=max_speed_mps,
        )
        self.transition_predictor = FutureTransitionPredictor(
            latent_dim, model_dim, pe_feature_dim, num_visual_frames - 1, num_heads
        )

    def forward(
        self,
        visual_features: Tensor,
        trajectory: Tensor,
        frame_times: Tensor,
        future_times: Tensor,
        trajectory_mask: Tensor | None = None,
        sample_posterior: bool = True,
    ) -> TokenizerOutput:
        frame_tokens = self.resampler(visual_features)
        mean_vis, logvar_vis, _ = self.visual_encoder(frame_tokens, frame_times)
        if sample_posterior:
            latent_vis = mean_vis + torch.exp(0.5 * logvar_vis) * torch.randn_like(mean_vis)
        else:
            latent_vis = mean_vis
        latent_traj = self.trajectory_encoder(trajectory, future_times, trajectory_mask)

        # Both paths share the same context-free decoder. This prevents current PE
        # features from becoming an inference-time shortcut around the action latent.
        reconstruction_vis = self.decoder(latent_vis, future_times)
        reconstruction_traj = self.decoder(latent_traj, future_times)
        predicted_transition = self.transition_predictor(latent_vis)
        # Use the frozen PE output for the auxiliary target. A target made from the
        # trainable resampler can move with the predictor and admit the trivial solution
        # where all frames and samples collapse to one vector.
        raw_global = visual_features.mean(dim=2)
        target_transition = raw_global[:, 1:] - raw_global[:, :1]
        target_transition = functional.layer_norm(
            target_transition, (target_transition.shape[-1],)
        )
        return TokenizerOutput(
            mean_vis=mean_vis,
            logvar_vis=logvar_vis,
            latent_vis=latent_vis,
            latent_traj=latent_traj,
            reconstruction_vis=reconstruction_vis,
            reconstruction_traj=reconstruction_traj,
            predicted_transition=predicted_transition,
            target_transition=target_transition.detach(),
        )

    def decode(self, action_tokens: Tensor, future_times: Tensor) -> Tensor:
        """Decode `[B,K,Dz]` action tokens without any visual context."""
        return self.decoder(action_tokens, future_times)
