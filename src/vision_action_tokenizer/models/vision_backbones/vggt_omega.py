"""Single-frame VGGT-Omega adapter for the generic planner interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ..vggt_omega import OmegaCameraFeatureExtractor
from .base import PlannerVisionBackbone, VisionCondition


class SingleFrameVGGTOmegaBackbone(PlannerVisionBackbone):
    """Expose one camera token and the sixteen register tokens from an F=1 pass."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        expected_sha256: str | None,
        image_resolution: int = 512,
        resize_mode: str = "max_size",
        patch_size: int = 16,
    ) -> None:
        super().__init__()
        self.extractor = OmegaCameraFeatureExtractor(
            checkpoint_path, expected_sha256, freeze=True
        )
        self.feature_dim = 2048
        self.output_token_count = 17
        self.image_resolution = image_resolution
        self.resize_mode = resize_mode
        self.patch_size = patch_size
        self.checkpoint_sha256 = expected_sha256

    def preprocessing_metadata(self) -> dict[str, Any]:
        return {
            "backbone_type": "vggt_omega",
            "checkpoint_sha256": self.checkpoint_sha256,
            "image_resolution": self.image_resolution,
            "resize_mode": self.resize_mode,
            "patch_size": self.patch_size,
            "feature_dim": self.feature_dim,
            "pool_grid": None,
        }

    def forward(self, images: Tensor) -> VisionCondition:
        features = self.extractor(images.unsqueeze(1))
        tokens = torch.cat(
            [features.camera_hidden[:, 0, None], features.register_hidden[:, 0]], dim=1
        )
        mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        return VisionCondition(tokens=tokens, token_mask=mask, grid_size=None)
