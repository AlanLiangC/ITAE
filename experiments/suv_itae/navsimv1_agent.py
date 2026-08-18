"""NAVSIM v1 adapter that decodes SUV-predicted ITAE action tokens."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from suv.runtime import _mixed_precision_to_model_dtype, _normalize_mixed_precision

from experiments.navsimv1.pdm_agent import (
    NuPlanTrajectorySampling,
    SUVNavsimV1Agent,
    Trajectory,
    _build_history_condition_video,
)
from experiments.suv_itae.runtime import (
    ACTION_SHAPE,
    instantiate_suv_itae,
    load_adapter,
    load_suv_base_for_itae,
)
from vision_action_tokenizer.config import load_config
from vision_action_tokenizer.data.planner_dataset import PlannerTargetNormalizer
from vision_action_tokenizer.models.factory import (
    build_tokenizer,
    tokenizer_state_from_checkpoint,
)

logger = logging.getLogger(__name__)


class SUVITAENavsimV1Agent(SUVNavsimV1Agent):
    """Current-frame SUV policy with a frozen ITAE 4x192 trajectory decoder."""

    def __init__(
        self,
        checkpoint_path: str,
        adapter_checkpoint_path: str,
        action_tokenizer_config_path: str,
        action_tokenizer_checkpoint_path: str,
        model_config_path: str | None = None,
        action_horizon: int = 4,
        trajectory_hz: int = 10,
        **kwargs: Any,
    ) -> None:
        model_config = model_config_path or str(
            Path(__file__).resolve().parent / "config" / "model" / "suv_itae_navsim.yaml"
        )
        kwargs.update(
            model_config_path=model_config,
            visual_conditioning="current",
            slot_inference=False,
            prompt_mode="static",
            stagea_modality="rgb",
            action_dim=ACTION_SHAPE[1],
        )
        super().__init__(checkpoint_path=checkpoint_path, **kwargs)
        self.adapter_checkpoint_path = str(adapter_checkpoint_path)
        self.action_tokenizer_config_path = str(action_tokenizer_config_path)
        self.action_tokenizer_checkpoint_path = str(action_tokenizer_checkpoint_path)
        self.action_horizon = int(action_horizon)
        self.trajectory_hz = int(trajectory_hz)
        if self.action_horizon != ACTION_SHAPE[0]:
            raise ValueError(f"SUV-ITAE action_horizon must be {ACTION_SHAPE[0]}")
        if self.trajectory_hz != 10:
            raise ValueError("The supplied ITAE tokenizer was trained at 10 Hz")
        self.action_tokenizer = None
        self.action_normalizer = None

    def name(self) -> str:
        return "suv_itae_navsimv1_agent"

    def initialize(self) -> None:
        if self.model is not None:
            return
        device = self.device_name
        if device == "cuda":
            device = "cuda:0"
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA unavailable; falling back to CPU")
            device = "cpu"
        dtype = _mixed_precision_to_model_dtype(
            _normalize_mixed_precision(self.mixed_precision)
        )
        self.model = instantiate_suv_itae(
            self.model_config_path, device=device, model_dtype=dtype
        )
        load_suv_base_for_itae(self.model, self.checkpoint_path)
        adapter = load_adapter(self.model, self.adapter_checkpoint_path)

        tokenizer_config = load_config(self.action_tokenizer_config_path)
        tokenizer = build_tokenizer(tokenizer_config).to(device).eval()
        tokenizer_checkpoint = torch.load(
            self.action_tokenizer_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        tokenizer.load_state_dict(
            tokenizer_state_from_checkpoint(tokenizer_checkpoint), strict=True
        )
        tokenizer.requires_grad_(False)
        self.action_tokenizer = tokenizer

        normalizer = PlannerTargetNormalizer(ACTION_SHAPE).to(device)
        normalizer.load_state_dict(adapter["normalizer"], strict=True)
        normalizer.eval()
        self.action_normalizer = normalizer
        self.model.eval()
        logger.info(
            "Loaded SUV-ITAE adapter %s at step %s",
            self.adapter_checkpoint_path,
            adapter.get("global_step"),
        )

    @torch.no_grad()
    def compute_trajectory_for_modality(self, agent_input, stagea_modality=None):
        del stagea_modality
        self.initialize()
        assert self.model is not None
        assert self.action_tokenizer is not None
        assert self.action_normalizer is not None

        features = self.feature_builder.compute_features(agent_input)
        history_video = torch.as_tensor(features["video_front"], dtype=torch.float32)
        ego_status = torch.as_tensor(features["ego_status"], dtype=torch.float32)
        condition_video = _build_history_condition_video(
            history_video,
            frame_mode=self.video_frame_mode,
            padding_mode=self.history_padding_mode,
        )
        input_image = (
            condition_video.mul(2.0)
            .sub(1.0)
            .permute(1, 0, 2, 3)
            .unsqueeze(0)
            .contiguous()
        )
        prompt = self._build_prompt(ego_status, stagea_modality="rgb")
        context, context_mask = self._get_cached_text_context(prompt)
        prediction = self.model.infer_action(
            prompt=None,
            input_image=input_image,
            action_horizon=self.action_horizon,
            num_video_frames=int(condition_video.shape[0]) + self.num_future_frames,
            proprio=None,
            context=context,
            context_mask=context_mask,
            num_inference_steps=self.num_inference_steps,
            seed=self.seed,
            rand_device=self.rand_device,
            tiled=self.tiled,
        )["action"]
        device = next(self.action_tokenizer.parameters()).device
        normalized = prediction.to(device=device, dtype=torch.float32).unsqueeze(0)
        action_tokens = self.action_normalizer.denormalize(normalized)
        future_times = (
            torch.arange(1, 41, device=device, dtype=torch.float32)
            .div(float(self.trajectory_hz))
            .unsqueeze(0)
        )
        poses = self.action_tokenizer.decode(action_tokens, future_times)[0]
        poses = poses.detach().cpu().float().numpy().astype(np.float32)
        return Trajectory(
            poses,
            NuPlanTrajectorySampling(
                num_poses=int(poses.shape[0]),
                interval_length=1.0 / float(self.trajectory_hz),
            ),
        )
