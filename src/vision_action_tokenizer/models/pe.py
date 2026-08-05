"""Lazy wrapper around Meta Perception Encoder (PE)."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


class PEFeatureExtractor(nn.Module):
    """Extract spatial patch tokens from six images using a frozen PE model.

    Input shape is `[B, F, 3, H, W]`; output shape is `[B, F, P, C_pe]`.
    The fixed 2D pooling is deliberately applied before caching to keep storage
    practical while preserving an `pool_size x pool_size` spatial layout.
    """

    def __init__(
        self,
        model_name: str = "PE-Spatial-B16-512",
        layer_idx: int | None = None,
        pool_size: int = 8,
        freeze: bool = True,
        model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.layer_idx = layer_idx
        self.pool_size = pool_size
        self.freeze = freeze
        self.model = model if model is not None else self._load_official_model(model_name)
        if freeze:
            self.model.requires_grad_(False)
            self.model.eval()

    @staticmethod
    def _load_official_model(model_name: str) -> nn.Module:
        try:
            import core.vision_encoder.pe as pe  # type: ignore[import-not-found]
        except ImportError as error:
            raise ImportError(
                "Meta perception_models is not installed. Follow README section 2 or inject "
                "a compatible PE model when testing."
            ) from error
        return pe.VisionTransformer.from_config(model_name, pretrained=True)

    def train(self, mode: bool = True) -> "PEFeatureExtractor":
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 5:
            raise ValueError(f"Expected images [B,F,3,H,W], got {tuple(images.shape)}")
        batch, frames, channels, height, width = images.shape
        flattened = images.reshape(batch * frames, channels, height, width)
        context: Any = torch.no_grad() if self.freeze else torch.enable_grad()
        with context:
            kwargs: dict[str, Any] = {"strip_cls_token": True}
            if self.layer_idx is not None:
                kwargs["layer_idx"] = self.layer_idx
            features = self.model.forward_features(flattened, **kwargs)
        if features.ndim != 3:
            raise ValueError(f"PE forward_features must return [BF,P,C], got {features.shape}")
        features = self._spatial_pool(features)
        return features.reshape(batch, frames, features.shape[1], features.shape[2])

    def _spatial_pool(self, features: Tensor) -> Tensor:
        side = int(math.isqrt(features.shape[1]))
        if side * side != features.shape[1]:
            raise ValueError(
                f"PE patch count {features.shape[1]} is not square after stripping CLS token"
            )
        spatial = features.transpose(1, 2).reshape(features.shape[0], features.shape[2], side, side)
        pooled = functional.adaptive_avg_pool2d(spatial, (self.pool_size, self.pool_size))
        return pooled.flatten(2).transpose(1, 2).contiguous()


class VisionActionTrainingModel(nn.Module):
    """Compose the frozen PE teacher and trainable tokenizer for online training."""

    def __init__(self, pe_extractor: PEFeatureExtractor, tokenizer: nn.Module) -> None:
        super().__init__()
        self.pe_extractor = pe_extractor
        self.tokenizer = tokenizer

    def forward(
        self,
        images: Tensor,
        trajectory: Tensor,
        frame_times: Tensor,
        future_times: Tensor,
        trajectory_mask: Tensor | None = None,
        sample_posterior: bool = True,
    ) -> Any:
        visual_features = self.pe_extractor(images)
        return self.tokenizer(
            visual_features=visual_features,
            trajectory=trajectory,
            frame_times=frame_times,
            future_times=future_times,
            trajectory_mask=trajectory_mask,
            sample_posterior=sample_posterior,
        )


class CachedVisionActionTrainingModel(nn.Module):
    """Training wrapper used when frozen PE features have been cached offline."""

    def __init__(self, tokenizer: nn.Module) -> None:
        super().__init__()
        self.tokenizer = tokenizer

    def forward(
        self,
        visual_features: Tensor,
        trajectory: Tensor,
        frame_times: Tensor,
        future_times: Tensor,
        trajectory_mask: Tensor | None = None,
        sample_posterior: bool = True,
    ) -> Any:
        return self.tokenizer(
            visual_features=visual_features,
            trajectory=trajectory,
            frame_times=frame_times,
            future_times=future_times,
            trajectory_mask=trajectory_mask,
            sample_posterior=sample_posterior,
        )
