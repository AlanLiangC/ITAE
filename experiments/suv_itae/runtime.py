"""Shared SUV-ITAE model loading and action-flow helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

ADAPTER_FORMAT = "suv_itae_action_adapter_v1"
ACTION_SHAPE = (4, 192)


def instantiate_suv_itae(
    model_config_path: str | Path,
    *,
    device: str | torch.device,
    model_dtype: torch.dtype = torch.bfloat16,
):
    """Instantiate SUV with a 4x192 action expert."""
    raw = OmegaConf.load(str(model_config_path))
    if not isinstance(raw, DictConfig):
        raw = OmegaConf.create(raw)
    model = instantiate(raw, model_dtype=model_dtype, device=str(device))
    if int(model.action_expert.action_dim) != ACTION_SHAPE[1]:
        raise ValueError(
            f"SUV-ITAE model must use action_dim={ACTION_SHAPE[1]}, "
            f"got {model.action_expert.action_dim}"
        )
    return model


def load_suv_base_for_itae(model, checkpoint_path: str | Path) -> dict[str, Any]:
    """Load all shape-compatible SUV weights and reinitialize the 3->192 edges."""
    payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if "mot" not in payload:
        raise ValueError(f"SUV base checkpoint has no `mot` state: {checkpoint_path}")
    current = model.mot.state_dict()
    compatible = {
        key: value
        for key, value in payload["mot"].items()
        if key in current and tuple(value.shape) == tuple(current[key].shape)
    }
    result = model.mot.load_state_dict(compatible, strict=False)
    allowed_missing = (
        "mixtures.action.action_encoder.weight",
        "mixtures.action.head.weight",
        "mixtures.action.head.bias",
    )
    unexpected_missing = [
        key for key in result.missing_keys if key not in allowed_missing
    ]
    if unexpected_missing:
        raise ValueError(
            "Base SUV partial load missed parameters other than the expected "
            f"3D action input/output projections: {unexpected_missing[:10]}"
        )
    if model.proprio_encoder is not None and "proprio_encoder" in payload:
        model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
    logger.info(
        "Loaded %d/%d shape-compatible MoT tensors from %s; initialized 192D action edges",
        len(compatible),
        len(current),
        checkpoint_path,
    )
    return payload


def load_adapter(model, checkpoint_path: str | Path) -> dict[str, Any]:
    payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if payload.get("format") != ADAPTER_FORMAT:
        raise ValueError(
            f"Unsupported SUV-ITAE adapter format in {checkpoint_path}: "
            f"{payload.get('format')!r}"
        )
    if tuple(payload.get("action_shape", ())) != ACTION_SHAPE:
        raise ValueError(f"Adapter action shape is not {ACTION_SHAPE}")
    model.action_expert.load_state_dict(payload["action_expert"], strict=True)
    return payload


@torch.no_grad()
def prepare_video_kv_cache(
    model,
    input_images: torch.Tensor,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    *,
    action_horizon: int = ACTION_SHAPE[0],
    tiled: bool = False,
) -> dict[str, Any]:
    """Encode current RGB frames once and prefill the frozen video expert."""
    if input_images.ndim != 4 or input_images.shape[1] != 3:
        raise ValueError(f"Expected images [B,3,H,W], got {tuple(input_images.shape)}")
    latent_items = [
        model._encode_input_image_latents_tensor(image.unsqueeze(0), tiled=tiled)
        for image in input_images
    ]
    condition_latents = torch.cat(latent_items, dim=0)
    condition_frames = int(condition_latents.shape[2])
    timestep_video = torch.zeros(
        (condition_latents.shape[0],),
        device=condition_latents.device,
        dtype=condition_latents.dtype,
    )
    video_pre = model.video_expert.pre_dit(
        x=condition_latents,
        timestep=timestep_video,
        context=context,
        context_mask=context_mask,
        action=None,
        fuse_vae_embedding_in_latents=bool(
            getattr(model.video_expert, "fuse_vae_embedding_in_latents", False)
        ),
        condition_latent_frames=condition_frames,
    )
    video_seq_len = int(video_pre["tokens"].shape[1])
    attention_mask = model._build_mot_attention_mask(
        video_seq_len=video_seq_len,
        action_seq_len=int(action_horizon),
        video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
        device=video_pre["tokens"].device,
        condition_video_frames=condition_frames,
    )
    cache = model.mot.prefill_video_cache(
        video_tokens=video_pre["tokens"],
        video_freqs=video_pre["freqs"],
        video_t_mod=video_pre["t_mod"],
        video_context_payload={
            "context": video_pre["context"],
            "mask": video_pre["context_mask"],
        },
        video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
    )
    return {
        "video_kv_cache": [
            {name: value.detach() for name, value in layer.items()} for layer in cache
        ],
        "attention_mask": attention_mask,
        "video_seq_len": video_seq_len,
    }


def predict_action_velocity(
    model,
    noisy_action: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    cache: dict[str, Any],
) -> torch.Tensor:
    """Differentiable action branch corresponding to SUV's cached inference path."""
    action_pre = model.action_expert.pre_dit(
        action_tokens=noisy_action,
        timestep=timestep,
        context=context,
        context_mask=context_mask,
    )
    tokens = model.mot.forward_action_with_video_cache(
        action_tokens=action_pre["tokens"],
        action_freqs=action_pre["freqs"],
        action_t_mod=action_pre["t_mod"],
        action_context_payload={
            "context": action_pre["context"],
            "mask": action_pre["context_mask"],
        },
        video_kv_cache=cache["video_kv_cache"],
        attention_mask=cache["attention_mask"],
        video_seq_len=int(cache["video_seq_len"]),
    )
    return model.action_expert.post_dit(tokens, action_pre)


@torch.no_grad()
def sample_action_tokens(
    model,
    input_images: torch.Tensor,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    *,
    num_inference_steps: int,
    sigma_shift: float,
    seeds: list[int],
) -> torch.Tensor:
    """Batch SUV action-only sampling while keeping one seed per clip frame."""
    batch = int(input_images.shape[0])
    if len(seeds) != batch:
        raise ValueError(f"Expected {batch} action seeds, got {len(seeds)}")
    if num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be positive")
    dtype = input_images.dtype
    noise = torch.stack(
        [
            torch.randn(
                ACTION_SHAPE,
                generator=torch.Generator(device="cpu").manual_seed(int(seed)),
                dtype=torch.float32,
            )
            for seed in seeds
        ]
    ).to(device=input_images.device, dtype=dtype)
    cache = prepare_video_kv_cache(
        model,
        input_images,
        context,
        context_mask,
        action_horizon=ACTION_SHAPE[0],
    )
    timesteps, deltas = model.infer_action_scheduler.build_inference_schedule(
        num_inference_steps=num_inference_steps,
        device=input_images.device,
        dtype=dtype,
        shift_override=sigma_shift,
    )
    action = noise
    for timestep, delta in zip(timesteps, deltas, strict=True):
        batch_timestep = timestep.expand(batch).to(
            device=input_images.device, dtype=dtype
        )
        velocity = predict_action_velocity(
            model,
            action,
            batch_timestep,
            context,
            context_mask,
            cache,
        )
        action = model.infer_action_scheduler.step(velocity, delta, action)
    return action.float()


def shifted_sigma(uniform_time: torch.Tensor, shift: float) -> torch.Tensor:
    return shift * uniform_time / (1.0 + (shift - 1.0) * uniform_time)
