"""Spatial token compression that preserves one token set per frame."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


class SpatialResampler(nn.Module):
    """Compress PE patches while optionally retaining their spatial ordering.

    Converts `[B,F,N,C_pe]` to `[B,F,R,D]`, where R is small enough for temporal
    attention but remains larger than one global token. ``grid`` mode is intended
    for visual motion: unlike learned content queries, output token index ``r``
    always represents the same image-grid cell in every frame.
    """

    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        num_queries: int,
        num_heads: int,
        dropout: float,
        mode: str = "grid",
    ) -> None:
        super().__init__()
        if mode not in {"grid", "query"}:
            raise ValueError("SpatialResampler mode must be `grid` or `query`")
        if num_queries <= 0:
            raise ValueError("num_queries must be positive")
        self.mode = mode
        self.num_queries = num_queries
        self.input_projection = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, model_dim)
        )
        if mode == "query":
            self.queries = nn.Parameter(torch.randn(num_queries, model_dim) * 0.02)
            self.attention = nn.MultiheadAttention(
                model_dim, num_heads, dropout=dropout, batch_first=True
            )
        else:
            self.register_parameter("queries", None)
            self.attention = None
        self.spatial_embedding = nn.Parameter(torch.randn(num_queries, model_dim) * 0.02)
        self.output_norm = nn.LayerNorm(model_dim)

    @staticmethod
    def _factor_grid(token_count: int) -> tuple[int, int]:
        """Return the least elongated integer grid whose area is token_count."""
        rows = int(math.sqrt(token_count))
        while rows > 1 and token_count % rows:
            rows -= 1
        return rows, token_count // rows

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 4:
            raise ValueError(f"Expected PE features [B,F,N,C], got {tuple(features.shape)}")
        batch, frames, patches, channels = features.shape
        memory = self.input_projection(features.reshape(batch * frames, patches, channels))
        if self.mode == "query":
            assert self.queries is not None and self.attention is not None
            queries = self.queries.unsqueeze(0).expand(batch * frames, -1, -1)
            output, _ = self.attention(queries, memory, memory, need_weights=False)
        else:
            source_side = math.isqrt(patches)
            if source_side * source_side != patches:
                raise ValueError(
                    "grid resampling requires PE patches to form a square grid; "
                    f"got N={patches}"
                )
            target_rows, target_columns = self._factor_grid(self.num_queries)
            spatial = memory.transpose(1, 2).reshape(
                batch * frames, memory.shape[-1], source_side, source_side
            )
            output = functional.adaptive_avg_pool2d(
                spatial, (target_rows, target_columns)
            ).flatten(2).transpose(1, 2)
        output = output + self.spatial_embedding.unsqueeze(0)
        return self.output_norm(output).reshape(batch, frames, output.shape[1], output.shape[2])
