"""Spatial token compression that preserves one token set per frame."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SpatialResampler(nn.Module):
    """Cross-attend learned queries to PE patch tokens.

    Converts `[B,F,N,C_pe]` to `[B,F,R,D]`, where R is small enough for temporal
    attention but remains larger than one global token.
    """

    def __init__(
        self, input_dim: int, model_dim: int, num_queries: int, num_heads: int, dropout: float
    ) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, model_dim)
        )
        self.queries = nn.Parameter(torch.randn(num_queries, model_dim) * 0.02)
        self.attention = nn.MultiheadAttention(
            model_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.output_norm = nn.LayerNorm(model_dim)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 4:
            raise ValueError(f"Expected PE features [B,F,N,C], got {tuple(features.shape)}")
        batch, frames, patches, channels = features.shape
        memory = self.input_projection(features.reshape(batch * frames, patches, channels))
        queries = self.queries.unsqueeze(0).expand(batch * frames, -1, -1)
        output, _ = self.attention(queries, memory, memory, need_weights=False)
        return self.output_norm(output).reshape(batch, frames, output.shape[1], output.shape[2])
