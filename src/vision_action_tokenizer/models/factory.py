"""Build the VGGT-Omega interval action tokenizer consistently across tools."""

from __future__ import annotations

from typing import Any

from torch import nn

from .tokenizer import VisionActionTokenizer
from .vggt_omega import (
    CachedOmegaTrainingModel,
    OmegaCameraFeatureExtractor,
    OnlineOmegaTrainingModel,
)


def build_tokenizer(config: dict[str, Any]) -> VisionActionTokenizer:
    action = config["action_tokenizer"]
    backbone = config["vision_backbone"]
    if backbone.get("type") != "vggt_omega":
        raise ValueError("vision_backbone.type must be `vggt_omega`")
    if action.get("decoder_type") != "se2_increment":
        raise ValueError("action_tokenizer.decoder_type must be `se2_increment`")
    num_frames = int(action["num_frames"])
    num_tokens = int(action["num_action_tokens"])
    if num_tokens != num_frames - 1:
        raise ValueError("num_action_tokens must equal num_frames - 1")
    configured_frame_count = len(config["data"]["frame_offsets_s"])
    if configured_frame_count != num_frames:
        raise ValueError("action_tokenizer.num_frames must match data.frame_offsets_s")
    trajectory_steps = round(
        float(config["data"]["future_horizon_s"])
        * int(config["data"]["trajectory_hz"])
    )
    if num_tokens * int(action["steps_per_token"]) != trajectory_steps:
        raise ValueError(
            "num_action_tokens * steps_per_token must match horizon * trajectory_hz"
        )
    return VisionActionTokenizer(
        vggt_feature_dim=int(backbone.get("feature_dim", 2048)),
        frame_geometry_dim=int(action["frame_geometry_dim"]),
        action_token_dim=int(action["action_token_dim"]),
        num_action_tokens=num_tokens,
        steps_per_token=int(action["steps_per_token"]),
        decoder_hidden_dim=int(action["decoder_hidden_dim"]),
        dropout=float(action["dropout"]),
    )


def build_training_model(config: dict[str, Any], cached: bool = True) -> nn.Module:
    tokenizer = build_tokenizer(config)
    if cached:
        return CachedOmegaTrainingModel(tokenizer)
    backbone = config["vision_backbone"]
    if not bool(backbone.get("freeze_camera_trunk", True)):
        raise NotImplementedError(
            "CameraHead trunk fine-tuning requires the Phase-E online training path; "
            "the current closed-loop baseline keeps it frozen"
        )
    if not bool(backbone.get("freeze_aggregator", True)):
        raise NotImplementedError("The VGGT-Omega Aggregator must remain frozen")
    extractor = OmegaCameraFeatureExtractor(
        checkpoint_path=backbone["checkpoint_path"],
        expected_sha256=backbone.get("checkpoint_sha256"),
        freeze=True,
    )
    return OnlineOmegaTrainingModel(extractor, tokenizer)


def tokenizer_state_from_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Extract tokenizer weights from online, cached or tokenizer-only checkpoints."""
    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    prefixes = ("tokenizer.", "module.tokenizer.")
    for prefix in prefixes:
        selected = {
            key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)
        }
        if selected:
            return selected
    return state
