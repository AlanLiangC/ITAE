"""Reusable neural-network primitives."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def sinusoidal_time_embedding(times: Tensor, dim: int, max_period: float = 10_000.0) -> Tensor:
    """Embed arbitrary time values.

    Args:
        times: Seconds with shape `[...]`.
        dim: Output feature dimension.
    Returns:
        Tensor with shape `[..., dim]`.
    """
    half = dim // 2
    if half == 0:
        return times.unsqueeze(-1)
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=times.device, dtype=times.dtype)
        / max(half - 1, 1)
    )
    angles = times.unsqueeze(-1) * frequencies
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if dim % 2:
        embedding = torch.nn.functional.pad(embedding, (0, 1))
    return embedding


class MLP(nn.Module):
    """Two-layer projection with LayerNorm-friendly GELU activation."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.net(inputs)


def make_transformer_encoder(
    dim: int, num_heads: int, num_layers: int, dropout: float
) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=dim,
        nhead=num_heads,
        dim_feedforward=dim * 4,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=num_layers, norm=nn.LayerNorm(dim))

