"""Build consistently configured model graphs for train/export tools."""

from __future__ import annotations

from typing import Any

from torch import nn

from .pe import CachedVisionActionTrainingModel, PEFeatureExtractor, VisionActionTrainingModel
from .tokenizer import VisionActionTokenizer


def build_tokenizer(config: dict[str, Any]) -> VisionActionTokenizer:
    model = config["model"]
    pe = config["pe"]
    return VisionActionTokenizer(
        pe_feature_dim=int(pe["feature_dim"]),
        model_dim=int(model["model_dim"]),
        latent_dim=int(model["latent_dim"]),
        num_action_tokens=int(model["num_action_tokens"]),
        resampled_tokens_per_frame=int(model["resampled_tokens_per_frame"]),
        num_heads=int(model["num_heads"]),
        encoder_layers=int(model["encoder_layers"]),
        decoder_layers=int(model["decoder_layers"]),
        dropout=float(model["dropout"]),
        decoder_type=str(model["decoder_type"]),
        num_visual_frames=len(config["data"]["frame_offsets_s"]),
        max_speed_mps=float(model["max_speed_mps"]),
        trajectory_position_scale_m=float(model.get("trajectory_position_scale_m", 50.0)),
        resampler_type=str(model.get("resampler_type", "grid")),
        visual_transition_mode=str(
            model.get("visual_transition_mode", "spatial_difference")
        ),
    )


def build_training_model(config: dict[str, Any], cached: bool = False) -> nn.Module:
    tokenizer = build_tokenizer(config)
    if cached:
        return CachedVisionActionTrainingModel(tokenizer)
    pe_config = config["pe"]
    extractor = PEFeatureExtractor(
        model_name=str(pe_config["model_name"]),
        checkpoint_path=pe_config.get("checkpoint_path"),
        layer_idx=pe_config.get("layer_idx"),
        pool_size=int(pe_config["pool_size"]),
        forward_batch_size=pe_config.get("forward_batch_size"),
        freeze=bool(pe_config.get("freeze", True)),
    )
    return VisionActionTrainingModel(extractor, tokenizer)


def tokenizer_state_from_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Extract tokenizer weights from online-PE, cached-PE or tokenizer-only checkpoints."""
    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    prefixes = ("tokenizer.", "module.tokenizer.")
    for prefix in prefixes:
        selected = {
            key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)
        }
        if selected:
            return selected
    return state
