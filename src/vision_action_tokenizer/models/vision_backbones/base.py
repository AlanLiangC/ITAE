"""Shared interfaces for current-frame planner vision conditions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import Tensor, nn


@dataclass(frozen=True)
class VisionCondition:
    """Backbone-independent visual tokens consumed by the planner."""

    tokens: Tensor
    token_mask: Tensor
    grid_size: tuple[int, int] | None


class PlannerVisionBackbone(nn.Module):
    """Base class implemented by every selectable planner image encoder."""

    feature_dim: int
    output_token_count: int

    def preprocessing_metadata(self) -> dict[str, Any]:
        raise NotImplementedError

    def forward(self, images: Tensor) -> VisionCondition:
        raise NotImplementedError
