"""A compact latent diffusion action expert for downstream integration."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from .common import MLP, sinusoidal_time_embedding


class LatentNormalizer(nn.Module):
    """Persist train-split statistics used by both expert and decoder integration."""

    def __init__(self, shape: tuple[int, int], epsilon: float = 1e-6) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("std", torch.ones(shape))
        self.register_buffer("fitted", torch.tensor(False))

    @torch.no_grad()
    def fit(self, latents: Tensor) -> None:
        if latents.ndim != 3 or latents.shape[1:] != self.mean.shape:
            raise ValueError(f"Expected [N,{tuple(self.mean.shape)}], got {tuple(latents.shape)}")
        self.mean.copy_(latents.mean(dim=0))
        self.std.copy_(latents.std(dim=0, unbiased=False).clamp_min(self.epsilon))
        self.fitted.fill_(True)

    def normalize(self, latent: Tensor) -> Tensor:
        return (latent - self.mean) / self.std

    def denormalize(self, latent: Tensor) -> Tensor:
        return latent * self.std + self.mean


class LatentDenoiser(nn.Module):
    """Predict DDPM noise on ordered action tokens conditioned on driving tokens."""

    def __init__(
        self,
        latent_dim: int = 256,
        condition_dim: int = 512,
        model_dim: int = 512,
        num_action_tokens: int = 10,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.latent_input = nn.Linear(latent_dim, model_dim)
        self.condition_input = nn.Linear(condition_dim, model_dim)
        self.slot_embedding = nn.Parameter(torch.randn(num_action_tokens, model_dim) * 0.02)
        self.time_mlp = MLP(model_dim, model_dim * 2, model_dim, dropout)
        layer = nn.TransformerDecoderLayer(
            model_dim,
            num_heads,
            model_dim * 4,
            dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(layer, num_layers, nn.LayerNorm(model_dim))
        self.output = nn.Linear(model_dim, latent_dim)

    def forward(self, noisy_latent: Tensor, timesteps: Tensor, condition_tokens: Tensor) -> Tensor:
        action = self.latent_input(noisy_latent) + self.slot_embedding.unsqueeze(0)
        time = self.time_mlp(
            sinusoidal_time_embedding(timesteps.to(noisy_latent.dtype), self.model_dim)
        ).unsqueeze(1)
        memory = self.condition_input(condition_tokens)
        return self.output(self.transformer(action + time, memory))


class GaussianLatentDiffusion(nn.Module):
    """DDPM training and deterministic DDIM sampling in action-token space."""

    def __init__(self, denoiser: LatentDenoiser, diffusion_steps: int = 1000) -> None:
        super().__init__()
        self.denoiser = denoiser
        betas = torch.linspace(1e-4, 0.02, diffusion_steps, dtype=torch.float64)
        alphas_cumulative = torch.cumprod(1.0 - betas, dim=0).float()
        self.register_buffer("alphas_cumulative", alphas_cumulative)

    def training_loss(self, clean_latent: Tensor, condition_tokens: Tensor) -> Tensor:
        batch = clean_latent.shape[0]
        timesteps = torch.randint(
            0, len(self.alphas_cumulative), (batch,), device=clean_latent.device
        )
        noise = torch.randn_like(clean_latent)
        alpha = self.alphas_cumulative[timesteps].view(batch, 1, 1)
        noisy = alpha.sqrt() * clean_latent + (1.0 - alpha).sqrt() * noise
        prediction = self.denoiser(noisy, timesteps, condition_tokens)
        return functional.mse_loss(prediction, noise)

    @torch.no_grad()
    def sample(
        self,
        condition_tokens: Tensor,
        latent_shape: tuple[int, int],
        sampling_steps: int = 20,
    ) -> Tensor:
        batch = condition_tokens.shape[0]
        latent = torch.randn((batch, *latent_shape), device=condition_tokens.device)
        schedule = torch.linspace(
            len(self.alphas_cumulative) - 1,
            0,
            sampling_steps,
            device=condition_tokens.device,
        ).long()
        for index, timestep in enumerate(schedule):
            t = torch.full((batch,), int(timestep), device=latent.device, dtype=torch.long)
            alpha = self.alphas_cumulative[timestep]
            predicted_noise = self.denoiser(latent, t, condition_tokens)
            predicted_clean = (latent - (1.0 - alpha).sqrt() * predicted_noise) / alpha.sqrt()
            if index == len(schedule) - 1:
                latent = predicted_clean
                break
            next_alpha = self.alphas_cumulative[schedule[index + 1]]
            latent = (
                next_alpha.sqrt() * predicted_clean + (1.0 - next_alpha).sqrt() * predicted_noise
            )
        return latent
