"""PE-Spatial adapter which preserves an ordered patch grid."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image, ImageOps
from torch import Tensor

from ...config import stable_hash
from ..vggt_omega import file_sha256
from .base import PlannerVisionBackbone, VisionCondition


class PESpatialTransform:
    """Deterministic PE image transform with an explicit resize policy."""

    def __init__(self, image_size: int = 512, resize_mode: str = "squash") -> None:
        if image_size <= 0:
            raise ValueError("PE image_size must be positive")
        if resize_mode not in {"squash", "letterbox"}:
            raise ValueError("PE resize_mode must be `squash` or `letterbox`")
        self.image_size = image_size
        self.resize_mode = resize_mode

    def __call__(self, image: Image.Image) -> Tensor:
        image = image.convert("RGB")
        if self.resize_mode == "squash":
            image = image.resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
        else:
            image.thumbnail(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
            left = (self.image_size - image.width) // 2
            top = (self.image_size - image.height) // 2
            image = ImageOps.expand(
                image,
                border=(
                    left,
                    top,
                    self.image_size - image.width - left,
                    self.image_size - image.height - top,
                ),
                fill=(127, 127, 127),
            )
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return tensor.sub_(0.5).div_(0.5)


class PESpatialBackbone(PlannerVisionBackbone):
    """Load a local PE checkpoint and return spatially pooled patch tokens."""

    def __init__(
        self,
        model_name: str,
        checkpoint_path: str | Path,
        expected_sha256: str | None,
        source_path: str | Path | None,
        image_size: int = 512,
        resize_mode: str = "squash",
        layer_idx: int = -1,
        strip_cls_token: bool = True,
        pool_grid: tuple[int, int] = (8, 8),
        freeze: bool = True,
    ) -> None:
        super().__init__()
        if pool_grid[0] <= 0 or pool_grid[1] <= 0:
            raise ValueError("PE pool_grid entries must be positive")
        checkpoint = Path(checkpoint_path)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"PE checkpoint does not exist: {checkpoint}")
        actual_sha = file_sha256(checkpoint)
        if expected_sha256 is not None and actual_sha != expected_sha256:
            raise ValueError(
                f"PE checkpoint SHA256 mismatch: {actual_sha} != {expected_sha256}"
            )
        if source_path is not None:
            source = Path(source_path).resolve()
            if not source.is_dir():
                raise FileNotFoundError(f"PE source_path does not exist: {source}")
            if str(source) not in sys.path:
                sys.path.insert(0, str(source))
        try:
            import core.vision_encoder.pe as pe
        except ImportError as error:
            raise ImportError(
                "Install facebookresearch/perception_models or set "
                "vision_condition.source_path"
            ) from error

        model = pe.VisionTransformer.from_config(model_name, pretrained=False)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        if "state_dict" in state:
            state = state["state_dict"]
        elif "weights" in state:
            state = state["weights"]
        state = {str(key).removeprefix("module."): value for key, value in state.items()}
        if any(key.startswith("visual.") for key in state):
            state = {
                key.removeprefix("visual."): value
                for key, value in state.items()
                if key.startswith("visual.")
            }
        model.load_state_dict(state, strict=True)
        self.model = model
        self.model_name = model_name
        self.checkpoint_path = str(checkpoint)
        self.checkpoint_sha256 = actual_sha
        self.source_path = None if source_path is None else str(Path(source_path))
        self.image_size = image_size
        self.resize_mode = resize_mode
        self.layer_idx = layer_idx
        self.strip_cls_token = strip_cls_token
        self.pool_grid = pool_grid
        self.feature_dim = int(model.width)
        self.patch_size = int(model.patch_size)
        self.input_grid = (image_size // self.patch_size, image_size // self.patch_size)
        self.output_token_count = pool_grid[0] * pool_grid[1]
        self.freeze = freeze
        if freeze:
            self.model.requires_grad_(False)
            self.model.eval()

    def train(self, mode: bool = True) -> PESpatialBackbone:
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    def preprocessing_metadata(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "backbone_type": "pe_spatial",
            "model_name": self.model_name,
            "checkpoint_sha256": self.checkpoint_sha256,
            "image_size": self.image_size,
            "resize_mode": self.resize_mode,
            "normalization": {"mean": [0.5] * 3, "std": [0.5] * 3},
            "layer_idx": self.layer_idx,
            "strip_cls_token": self.strip_cls_token,
            "input_grid": list(self.input_grid),
            "pool_grid": list(self.pool_grid),
            "feature_dim": self.feature_dim,
        }
        payload["preprocessing_hash"] = stable_hash(payload)
        return payload

    def forward(self, images: Tensor) -> VisionCondition:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected PE images [B,3,H,W], got {tuple(images.shape)}")
        if images.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(
                f"Expected PE images at {self.image_size}x{self.image_size}, "
                f"got {tuple(images.shape[-2:])}"
            )
        context: Any = torch.no_grad() if self.freeze else torch.enable_grad()
        with context:
            features = self.model.forward_features(
                images,
                norm=True,
                layer_idx=self.layer_idx,
                strip_cls_token=self.strip_cls_token,
            )
        expected_patches = self.input_grid[0] * self.input_grid[1]
        if features.shape[1:] != (expected_patches, self.feature_dim):
            raise ValueError(
                "Unexpected PE feature shape: "
                f"{tuple(features.shape)} != [B,{expected_patches},{self.feature_dim}]"
            )
        feature_grid = features.transpose(1, 2).reshape(
            images.shape[0], self.feature_dim, *self.input_grid
        )
        pooled = functional.adaptive_avg_pool2d(feature_grid, self.pool_grid)
        tokens = pooled.flatten(2).transpose(1, 2).contiguous()
        mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        return VisionCondition(tokens=tokens, token_mask=mask, grid_size=self.pool_grid)
