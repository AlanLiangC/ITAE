"""Frozen VGGT-Omega CameraHead feature extraction and training wrappers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class OmegaCameraFeatures:
    """Geometry-focused outputs after the pretrained CameraHead trunk."""

    camera_hidden: Tensor
    register_hidden_mean: Tensor
    pose_enc: Tensor


class OmegaCameraFeatureExtractor(nn.Module):
    """Run frozen Aggregator + CameraHead and expose its hidden tokens.

    The official ``camera_and_register_tokens`` prediction is the CameraHead input.
    This wrapper additionally executes the four pretrained CameraHead trunk blocks and
    returns their normalized hidden representation.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        expected_sha256: str | None = None,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        checkpoint = Path(checkpoint_path)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"VGGT-Omega checkpoint does not exist: {checkpoint}")
        if expected_sha256 is not None:
            actual = file_sha256(checkpoint)
            if actual != expected_sha256:
                raise ValueError(
                    f"VGGT-Omega checkpoint SHA256 mismatch: {actual} != {expected_sha256}"
                )
        try:
            from vggt_omega.models import VGGTOmega
        except ImportError as error:
            raise ImportError(
                "Install third_party/vggt-omega in the active Python 3.10+ environment"
            ) from error

        state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        selected = {
            key: value
            for key, value in state.items()
            if key.startswith(("aggregator.", "camera_head."))
        }
        if not selected:
            raise ValueError("Checkpoint contains no aggregator/camera_head parameters")
        self.model = VGGTOmega(
            enable_camera=True,
            enable_depth=False,
            enable_alignment=False,
        )
        self.model.load_state_dict(selected, strict=True, assign=True)
        self.checkpoint_path = str(checkpoint)
        self.freeze = freeze
        if freeze:
            self.model.requires_grad_(False)
            self.model.eval()

    def train(self, mode: bool = True) -> OmegaCameraFeatureExtractor:
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    def forward(self, images: Tensor) -> OmegaCameraFeatures:
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError(f"Expected images [B,F,3,H,W], got {tuple(images.shape)}")
        if images.shape[-2] % 16 or images.shape[-1] % 16:
            raise ValueError("VGGT-Omega image height/width must be divisible by patch size 16")

        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        grad_context: Any = torch.no_grad() if self.freeze else torch.enable_grad()
        with grad_context:
            with torch.autocast(device_type=images.device.type, dtype=amp_dtype):
                aggregated_tokens, patch_token_start = self.model.aggregator(images)
            final_tokens = aggregated_tokens[-1]
            if final_tokens is None:
                raise ValueError("VGGT-Omega Aggregator did not cache its final layer")
            head = self.model.camera_head
            if head is None:
                raise ValueError("VGGT-Omega CameraHead is disabled")
            special = final_tokens[:, :, :patch_token_start].float()
            special = head.token_norm(special)
            batch, frames, special_count, dim = special.shape
            special = special.reshape(batch, frames * special_count, dim)
            for block in head.trunk:
                special = block(special, None)
            hidden = special.reshape(batch, frames, special_count, dim)
            hidden = head.trunk_norm(hidden)
            camera_hidden = hidden[:, :, 0]
            register_hidden_mean = hidden[:, :, 1:].mean(dim=2)
            raw_pose = head.camera_branch(camera_hidden)
            pose_enc = torch.cat(
                [raw_pose[..., :7], torch.relu(raw_pose[..., 7:]) + 0.01], dim=-1
            )
        return OmegaCameraFeatures(
            camera_hidden=camera_hidden,
            register_hidden_mean=register_hidden_mean,
            pose_enc=pose_enc,
        )


class OnlineOmegaTrainingModel(nn.Module):
    """Compose the frozen 1B geometry backbone with the small action tokenizer."""

    def __init__(
        self, feature_extractor: OmegaCameraFeatureExtractor, tokenizer: nn.Module
    ) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

    def forward(
        self,
        images: Tensor,
        frame_times: Tensor,
        future_times: Tensor,
        **_: Any,
    ) -> Any:
        features = self.feature_extractor(images)
        return self.tokenizer(
            camera_hidden=features.camera_hidden,
            register_hidden_mean=features.register_hidden_mean,
            frame_times=frame_times,
            future_times=future_times,
        )


class CachedOmegaTrainingModel(nn.Module):
    """Small trainable graph operating on cached CameraHead hidden tokens."""

    def __init__(self, tokenizer: nn.Module) -> None:
        super().__init__()
        self.tokenizer = tokenizer

    def forward(
        self,
        camera_hidden: Tensor,
        register_hidden_mean: Tensor,
        frame_times: Tensor,
        future_times: Tensor,
        **_: Any,
    ) -> Any:
        return self.tokenizer(
            camera_hidden=camera_hidden,
            register_hidden_mean=register_hidden_mean,
            frame_times=frame_times,
            future_times=future_times,
        )
