from __future__ import annotations

import torch

from vision_action_tokenizer.models.expert import GaussianLatentDiffusion, LatentDenoiser


def test_latent_diffusion_train_and_sample() -> None:
    denoiser = LatentDenoiser(
        latent_dim=8,
        condition_dim=12,
        model_dim=32,
        num_action_tokens=2,
        num_heads=4,
        num_layers=1,
        dropout=0,
    )
    diffusion = GaussianLatentDiffusion(denoiser, diffusion_steps=20)
    latent = torch.randn(3, 2, 8)
    condition = torch.randn(3, 5, 12)
    loss = diffusion.training_loss(latent, condition)
    loss.backward()
    sample = diffusion.sample(condition, (2, 8), sampling_steps=4)
    assert torch.isfinite(loss)
    assert sample.shape == latent.shape

