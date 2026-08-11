#!/usr/bin/env python3
"""Reference trainer for cached condition tokens -> visual action latents."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader, TensorDataset

from vision_action_tokenizer.models.expert import GaussianLatentDiffusion, LatentDenoiser


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents", required=True, type=Path)
    parser.add_argument("--conditions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    args = parser.parse_args()
    latent_payload = load_file(args.latents)
    condition_payload = load_file(args.conditions)
    latents = (latent_payload["latents"] - latent_payload["normalizer_mean"]) / latent_payload[
        "normalizer_std"
    ]
    conditions = condition_payload["conditions"]
    if len(latents) != len(conditions):
        raise ValueError("Latent and condition caches must use the same sample order")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    denoiser = LatentDenoiser(
        latent_dim=latents.shape[-1],
        condition_dim=conditions.shape[-1],
        num_action_tokens=latents.shape[1],
    ).to(device)
    diffusion = GaussianLatentDiffusion(denoiser).to(device)
    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=args.learning_rate)
    loader = DataLoader(
        TensorDataset(latents, conditions), batch_size=args.batch_size, shuffle=True
    )
    for epoch in range(args.epochs):
        total = 0.0
        for latent, condition in loader:
            loss = diffusion.training_loss(latent.to(device), condition.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
            optimizer.step()
            total += float(loss)
        print(f"epoch={epoch} diffusion_loss={total / max(len(loader), 1):.6f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": diffusion.state_dict()}, args.output)


if __name__ == "__main__":
    main()
