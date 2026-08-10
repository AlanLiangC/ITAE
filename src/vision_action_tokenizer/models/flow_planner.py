"""Conditional Transformer velocity field shared by raw and token planners."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .common import MLP, sinusoidal_time_embedding


class ConditionalFlowPlanner(nn.Module):
    """Predict a rectified-flow velocity over an ordered target sequence."""

    def __init__(
        self,
        target_dim: int,
        target_slots: int,
        condition_dim: int,
        condition_tokens: int,
        model_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if target_dim <= 0 or target_slots <= 0:
            raise ValueError("Planner target shape must be positive")
        if condition_dim <= 0 or condition_tokens <= 0:
            raise ValueError("Planner condition shape must be positive")
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        self.target_dim = target_dim
        self.target_slots = target_slots
        self.condition_dim = condition_dim
        self.condition_tokens = condition_tokens
        self.model_dim = model_dim
        # Construct every shape-invariant module first. With the same process seed,
        # raw/token experiments then receive bitwise-identical shared-core weights.
        self.condition_input = nn.Sequential(
            nn.LayerNorm(condition_dim), nn.Linear(condition_dim, model_dim)
        )
        self.condition_position = nn.Parameter(
            torch.randn(condition_tokens, model_dim) * 0.02
        )
        self.slot_time_mlp = MLP(model_dim, model_dim * 2, model_dim, dropout)
        self.flow_time_mlp = MLP(model_dim, model_dim * 4, model_dim, dropout)
        layer = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(
            layer, num_layers=num_layers, norm=nn.LayerNorm(model_dim)
        )
        self.target_input = nn.Linear(target_dim, model_dim)
        self.target_slot_embedding = nn.Parameter(
            torch.randn(target_slots, model_dim) * 0.02
        )
        self.output = nn.Linear(model_dim, target_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        state: Tensor,
        flow_time: Tensor,
        condition_tokens: Tensor,
        condition_mask: Tensor | None,
        slot_times: Tensor,
    ) -> Tensor:
        batch = state.shape[0]
        if state.shape != (batch, self.target_slots, self.target_dim):
            raise ValueError(
                f"Expected planner state [B,{self.target_slots},{self.target_dim}], "
                f"got {tuple(state.shape)}"
            )
        if condition_tokens.shape != (
            batch,
            self.condition_tokens,
            self.condition_dim,
        ):
            raise ValueError(
                "Planner condition shape mismatch: "
                f"got {tuple(condition_tokens.shape)}"
            )
        if flow_time.shape != (batch,):
            raise ValueError("flow_time must have shape [B]")
        if slot_times.shape != (batch, self.target_slots):
            raise ValueError("slot_times must have shape [B,target_slots]")
        if condition_mask is not None and condition_mask.shape != (
            batch,
            self.condition_tokens,
        ):
            raise ValueError("condition_mask shape does not match condition tokens")

        dtype = state.dtype
        target = self.target_input(state)
        target = target + self.target_slot_embedding.unsqueeze(0)
        target = target + self.slot_time_mlp(
            sinusoidal_time_embedding(slot_times.to(dtype), self.model_dim)
        )
        flow_embedding = self.flow_time_mlp(
            sinusoidal_time_embedding(flow_time.to(dtype), self.model_dim)
        ).unsqueeze(1)
        memory = self.condition_input(condition_tokens.to(dtype))
        memory = memory + self.condition_position.unsqueeze(0)
        hidden = self.transformer(
            target + flow_embedding,
            memory,
            memory_key_padding_mask=(None if condition_mask is None else ~condition_mask.bool()),
        )
        return self.output(hidden)


def build_flow_planner(config: dict, condition_shape: tuple[int, int]) -> ConditionalFlowPlanner:
    planner = config["planner"]
    target_shape = tuple(map(int, planner["target_shape"]))
    model = planner["model"]
    return ConditionalFlowPlanner(
        target_dim=target_shape[1],
        target_slots=target_shape[0],
        condition_dim=condition_shape[1],
        condition_tokens=condition_shape[0],
        model_dim=int(model.get("dim", 256)),
        num_heads=int(model.get("heads", 8)),
        num_layers=int(model.get("layers", 8)),
        mlp_ratio=int(model.get("mlp_ratio", 4)),
        dropout=float(model.get("dropout", 0.1)),
    )
