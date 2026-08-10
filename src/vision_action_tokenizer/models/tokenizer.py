"""VGGT-Omega geometry-to-action tokenizer."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .common import sinusoidal_time_embedding
from .decoder import ResidualVelocityDecoder, SE2IncrementDecoder, integrate_se2_increments


@dataclass
class TokenizerOutput:
    """Single-path visual action encoding and context-free reconstruction."""

    action_tokens: Tensor
    reconstruction: Tensor
    predicted_increments: Tensor
    base_action_tokens: Tensor | None = None
    visual_residual_tokens: Tensor | None = None
    base_reconstruction: Tensor | None = None
    base_increments: Tensor | None = None
    residual_increments: Tensor | None = None


def _quaternion_to_matrix(quaternion: Tensor) -> Tensor:
    """Convert scalar-last XYZW quaternions to rotation matrices."""
    if quaternion.shape[-1] != 4:
        raise ValueError("Quaternion must have four scalar-last components")
    i, j, k, real = quaternion.unbind(dim=-1)
    scale = 2.0 / quaternion.square().sum(dim=-1).clamp_min(1e-8)
    matrix = torch.stack(
        [
            1 - scale * (j * j + k * k),
            scale * (i * j - k * real),
            scale * (i * k + j * real),
            scale * (i * j + k * real),
            1 - scale * (i * i + k * k),
            scale * (j * k - i * real),
            scale * (i * k - j * real),
            scale * (j * k + i * real),
            1 - scale * (i * i + j * j),
        ],
        dim=-1,
    )
    return matrix.reshape(*quaternion.shape[:-1], 3, 3)


def pose_motion_features(pose_enc: Tensor) -> Tensor:
    """Build scale-aware and scale-invariant motion cues from VGGT 9D poses.

    VGGT extrinsics are camera-from-world. Camera centers are expressed in the first
    camera coordinate system, then augmented with unit translation direction,
    log-distance and the first two columns of relative rotation. This exposes the
    reliable direction/rotation signal without pretending monocular translation is
    already metric.
    """
    if pose_enc.ndim != 3 or pose_enc.shape[-1] != 9:
        raise ValueError(f"Expected pose_enc [B,F,9], got {tuple(pose_enc.shape)}")
    translation = pose_enc[..., :3].float()
    rotation = _quaternion_to_matrix(pose_enc[..., 3:7].float())
    centers = -(rotation.transpose(-1, -2) @ translation.unsqueeze(-1)).squeeze(-1)
    relative_world = centers - centers[:, :1]
    anchor_rotation = rotation[:, :1]
    relative_center = (
        anchor_rotation @ relative_world.unsqueeze(-1)
    ).squeeze(-1)
    distance = torch.linalg.vector_norm(relative_center, dim=-1, keepdim=True)
    direction = relative_center / distance.clamp_min(1e-4)
    relative_rotation = rotation @ anchor_rotation.transpose(-1, -2)
    rotation_6d = relative_rotation[..., :, :2].reshape(*pose_enc.shape[:2], 6)
    return torch.cat(
        [relative_center, direction, torch.log1p(distance), rotation_6d], dim=-1
    )


class VisualMotionResidualEncoder(nn.Module):
    """Encode full registers and pretrained camera motion into correction tokens."""

    pose_feature_dim = 13

    def __init__(
        self,
        input_dim: int,
        register_token_count: int,
        register_projection_dim: int,
        frame_dim: int,
        token_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if min(
            input_dim,
            register_token_count,
            register_projection_dim,
            frame_dim,
            token_dim,
        ) <= 0:
            raise ValueError("Visual residual dimensions must be positive")
        self.register_token_count = register_token_count
        self.frame_dim = frame_dim
        self.register_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, register_projection_dim, bias=False),
            nn.GELU(),
        )
        self.register_fusion = nn.Sequential(
            nn.LayerNorm(register_token_count * register_projection_dim),
            nn.Linear(register_token_count * register_projection_dim, frame_dim),
            nn.GELU(),
        )
        self.pose_projection = nn.Sequential(
            nn.LayerNorm(self.pose_feature_dim),
            nn.Linear(self.pose_feature_dim, frame_dim),
            nn.GELU(),
        )
        self.frame_fusion = nn.Sequential(
            nn.LayerNorm(frame_dim * 2),
            nn.Linear(frame_dim * 2, frame_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.interval_fusion = nn.Sequential(
            nn.LayerNorm(frame_dim * 4),
            nn.Linear(frame_dim * 4, frame_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(frame_dim * 2, frame_dim),
            nn.GELU(),
        )
        self.output = nn.Linear(frame_dim, token_dim, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(
        self,
        register_hidden: Tensor,
        pose_enc: Tensor,
        frame_times: Tensor,
    ) -> Tensor:
        if register_hidden.ndim != 4:
            raise ValueError("Visual residual requires register_hidden [B,F,R,C]")
        if register_hidden.shape[2] != self.register_token_count:
            raise ValueError(
                "Visual residual register count mismatch: "
                f"{register_hidden.shape[2]} != {self.register_token_count}"
            )
        if pose_enc.shape[:2] != register_hidden.shape[:2]:
            raise ValueError("pose_enc must align with register_hidden frames")
        if frame_times.shape != register_hidden.shape[:2]:
            raise ValueError("frame_times must align with visual residual frames")
        delta_t = torch.diff(frame_times, dim=1)
        if torch.any(delta_t <= 0):
            raise ValueError("frame_times must be strictly increasing")

        # Center within each frame so the V2 mean-register path remains the sole
        # owner of the common feature; the new branch receives slot structure.
        registers = register_hidden.float()
        centered = registers - registers.mean(dim=2, keepdim=True)
        register_frame = self.register_projection(centered).flatten(2)
        register_frame = self.register_fusion(register_frame)
        pose_frame = self.pose_projection(pose_motion_features(pose_enc))
        frame = self.frame_fusion(torch.cat([register_frame, pose_frame], dim=-1))
        left, right = frame[:, :-1], frame[:, 1:]
        time = sinusoidal_time_embedding(delta_t, self.frame_dim)
        interval = self.interval_fusion(
            torch.cat([left, right, right - left, time], dim=-1)
        )
        return self.output(interval)


class IntervalActionEncoder(nn.Module):
    """Turn five CameraHead frame representations into four motion intervals."""

    def __init__(
        self,
        input_dim: int = 2048,
        frame_geometry_dim: int = 256,
        action_dim: int = 128,
        dropout: float = 0.0,
        num_intervals: int = 4,
        interval_mixer_layers: int = 0,
        interval_mixer_heads: int = 4,
        register_pooling: str = "mean",
        register_summary_tokens: int = 4,
        register_pool_dim: int = 128,
        register_token_count: int = 16,
        register_residual_dim: int = 32,
        register_residual_gate_init: float = 0.0,
        register_residual_zero_init: bool = False,
    ) -> None:
        super().__init__()
        if register_pooling not in {"mean", "attention", "mean_residual"}:
            raise ValueError(
                "register_pooling must be `mean`, `attention` or `mean_residual`"
            )
        if interval_mixer_layers < 0:
            raise ValueError("interval_mixer_layers must be non-negative")
        if interval_mixer_layers and action_dim % interval_mixer_heads:
            raise ValueError("action_dim must be divisible by interval_mixer_heads")
        self.frame_geometry_dim = frame_geometry_dim
        self.register_pooling = register_pooling
        if register_pooling in {"mean", "mean_residual"}:
            # Keep the V2-motion module names and shapes so it can initialize the
            # residual model exactly, before any full-register contribution is added.
            self.frame_projection = nn.Sequential(
                nn.LayerNorm(input_dim * 2),
                nn.Linear(input_dim * 2, frame_geometry_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(frame_geometry_dim, frame_geometry_dim),
                nn.LayerNorm(frame_geometry_dim),
            )
            if register_pooling == "mean_residual":
                if register_token_count <= 0 or register_residual_dim <= 0:
                    raise ValueError("Residual register dimensions must be positive")
                if not math.isfinite(register_residual_gate_init):
                    raise ValueError("register_residual_gate_init must be finite")
                self.register_token_count = register_token_count
                self.register_residual_projection = nn.Sequential(
                    nn.LayerNorm(input_dim),
                    nn.Linear(input_dim, register_residual_dim, bias=False),
                    nn.GELU(),
                )
                flattened_dim = register_token_count * register_residual_dim
                self.register_residual_fusion = nn.Sequential(
                    nn.LayerNorm(flattened_dim),
                    nn.Linear(flattened_dim, frame_geometry_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(frame_geometry_dim, frame_geometry_dim),
                    nn.LayerNorm(frame_geometry_dim),
                )
                self.register_residual_gate = nn.Parameter(
                    torch.full(
                        (frame_geometry_dim,), float(register_residual_gate_init)
                    )
                )
                if register_residual_zero_init:
                    self.register_residual_output: nn.Module = nn.Linear(
                        frame_geometry_dim, frame_geometry_dim
                    )
                    output = self.register_residual_output
                    assert isinstance(output, nn.Linear)
                    nn.init.zeros_(output.weight)
                    nn.init.zeros_(output.bias)
                else:
                    self.register_residual_output = nn.Identity()
        else:
            if register_summary_tokens <= 0 or register_pool_dim <= 0:
                raise ValueError("Register attention dimensions must be positive")
            self.camera_projection = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, frame_geometry_dim),
                nn.GELU(),
            )
            self.register_norm = nn.LayerNorm(input_dim)
            self.register_key = nn.Linear(input_dim, register_pool_dim)
            self.register_value = nn.Linear(input_dim, register_pool_dim)
            self.register_queries = nn.Parameter(
                torch.randn(register_summary_tokens, register_pool_dim) * 0.02
            )
            fusion_dim = frame_geometry_dim + register_summary_tokens * register_pool_dim
            self.frame_projection = nn.Sequential(
                nn.LayerNorm(fusion_dim),
                nn.Linear(fusion_dim, frame_geometry_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(frame_geometry_dim, frame_geometry_dim),
                nn.LayerNorm(frame_geometry_dim),
            )
        interval_input_dim = frame_geometry_dim * 4
        self.interval_projection = nn.Sequential(
            nn.LayerNorm(interval_input_dim),
            nn.Linear(interval_input_dim, frame_geometry_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(frame_geometry_dim, action_dim),
            nn.LayerNorm(action_dim),
        )
        self.interval_position: nn.Parameter | None = None
        self.interval_mixer: nn.TransformerEncoder | None = None
        if interval_mixer_layers:
            self.interval_position = nn.Parameter(torch.randn(num_intervals, action_dim) * 0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=action_dim,
                nhead=interval_mixer_heads,
                dim_feedforward=action_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.interval_mixer = nn.TransformerEncoder(
                layer,
                num_layers=interval_mixer_layers,
                norm=nn.LayerNorm(action_dim),
                enable_nested_tensor=False,
            )

    def _frame_geometry(
        self,
        camera_hidden: Tensor,
        register_hidden_mean: Tensor | None,
        register_hidden: Tensor | None,
    ) -> Tensor:
        if self.register_pooling in {"mean", "mean_residual"}:
            if register_hidden_mean is None or camera_hidden.shape != register_hidden_mean.shape:
                raise ValueError(
                    "Mean pooling requires camera_hidden and register_hidden_mean [B,F,C]"
                )
            inputs = torch.cat(
                [camera_hidden.float(), register_hidden_mean.float()], dim=-1
            )
            geometry = self.frame_projection(inputs)
            if self.register_pooling == "mean":
                return geometry
            if register_hidden is None or register_hidden.ndim != 4:
                raise ValueError(
                    "Mean-residual pooling requires register_hidden [B,F,R,C]"
                )
            expected_shape = (
                camera_hidden.shape[0],
                camera_hidden.shape[1],
                self.register_token_count,
                camera_hidden.shape[2],
            )
            if tuple(register_hidden.shape) != expected_shape:
                raise ValueError(
                    "register_hidden shape does not match mean-residual configuration: "
                    f"{tuple(register_hidden.shape)} != {expected_shape}"
                )
            centered = register_hidden.float() - register_hidden_mean.float().unsqueeze(2)
            residual = self.register_residual_projection(centered).flatten(2)
            residual = self.register_residual_fusion(residual)
            residual = self.register_residual_output(residual)
            gate = torch.tanh(self.register_residual_gate).view(1, 1, -1)
            return geometry + gate * residual

        if register_hidden is None or register_hidden.ndim != 4:
            raise ValueError("Attention pooling requires register_hidden [B,F,R,C]")
        if register_hidden.shape[:2] != camera_hidden.shape[:2]:
            raise ValueError("register_hidden must align with camera_hidden [B,F]")
        registers = self.register_norm(register_hidden.float())
        keys = self.register_key(registers)
        values = self.register_value(registers)
        scores = torch.einsum("kd,bfrd->bfkr", self.register_queries, keys)
        scores = scores / math.sqrt(keys.shape[-1])
        summaries = torch.einsum("bfkr,bfrd->bfkd", scores.softmax(dim=-1), values)
        camera = self.camera_projection(camera_hidden.float())
        return self.frame_projection(torch.cat([camera, summaries.flatten(2)], dim=-1))

    def forward(
        self,
        camera_hidden: Tensor,
        register_hidden_mean: Tensor | None,
        frame_times: Tensor,
        register_hidden: Tensor | None = None,
    ) -> Tensor:
        if camera_hidden.ndim != 3:
            raise ValueError("camera_hidden must have shape [B,F,C]")
        if frame_times.shape != camera_hidden.shape[:2]:
            raise ValueError("frame_times must align with CameraHead features [B,F]")
        if camera_hidden.shape[1] < 2:
            raise ValueError("At least two frames are required to form an action interval")
        delta_t = torch.diff(frame_times, dim=1)
        if torch.any(delta_t <= 0):
            raise ValueError("frame_times must be strictly increasing")

        geometry = self._frame_geometry(
            camera_hidden, register_hidden_mean, register_hidden
        )
        time = sinusoidal_time_embedding(delta_t, self.frame_geometry_dim)
        left, right = geometry[:, :-1], geometry[:, 1:]
        interval = torch.cat([left, right, right - left, time], dim=-1)
        action_tokens = self.interval_projection(interval)
        if self.interval_mixer is not None:
            assert self.interval_position is not None
            if action_tokens.shape[1] != self.interval_position.shape[0]:
                raise ValueError("Action interval count does not match mixer positions")
            action_tokens = self.interval_mixer(
                action_tokens + self.interval_position.unsqueeze(0)
            )
        return action_tokens


class VisionActionTokenizer(nn.Module):
    """Encode VGGT CameraHead hidden tokens and decode a 10 Hz local trajectory."""

    def __init__(
        self,
        vggt_feature_dim: int = 2048,
        frame_geometry_dim: int = 256,
        action_token_dim: int = 128,
        num_action_tokens: int = 4,
        steps_per_token: int = 10,
        decoder_hidden_dim: int = 256,
        dropout: float = 0.0,
        interval_mixer_layers: int = 0,
        interval_mixer_heads: int = 4,
        register_pooling: str = "mean",
        register_summary_tokens: int = 4,
        register_pool_dim: int = 128,
        register_token_count: int = 16,
        register_residual_dim: int = 32,
        register_residual_gate_init: float = 0.0,
        register_residual_zero_init: bool = False,
        decoder_parameterization: str = "displacement",
        initial_forward_speed_mps: float = 5.0,
        max_forward_speed_mps: float = 40.0,
        max_lateral_speed_mps: float = 8.0,
        max_yaw_rate_rps: float = 1.5,
        visual_residual_token_dim: int = 0,
        visual_residual_frame_dim: int = 128,
        visual_residual_register_dim: int = 32,
        visual_residual_max_forward_mps: float = 5.0,
        visual_residual_max_lateral_mps: float = 2.0,
        visual_residual_max_yaw_rate_rps: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_action_tokens = num_action_tokens
        self.steps_per_token = steps_per_token
        self.base_action_token_dim = action_token_dim
        self.visual_residual_token_dim = visual_residual_token_dim
        self.encoder = IntervalActionEncoder(
            input_dim=vggt_feature_dim,
            frame_geometry_dim=frame_geometry_dim,
            action_dim=action_token_dim,
            dropout=dropout,
            num_intervals=num_action_tokens,
            interval_mixer_layers=interval_mixer_layers,
            interval_mixer_heads=interval_mixer_heads,
            register_pooling=register_pooling,
            register_summary_tokens=register_summary_tokens,
            register_pool_dim=register_pool_dim,
            register_token_count=register_token_count,
            register_residual_dim=register_residual_dim,
            register_residual_gate_init=register_residual_gate_init,
            register_residual_zero_init=register_residual_zero_init,
        )
        self.decoder = SE2IncrementDecoder(
            action_dim=action_token_dim,
            hidden_dim=decoder_hidden_dim,
            steps_per_token=steps_per_token,
            dropout=dropout,
            parameterization=decoder_parameterization,
            initial_forward_speed_mps=initial_forward_speed_mps,
            max_forward_speed_mps=max_forward_speed_mps,
            max_lateral_speed_mps=max_lateral_speed_mps,
            max_yaw_rate_rps=max_yaw_rate_rps,
        )
        self.visual_residual_encoder: VisualMotionResidualEncoder | None = None
        self.visual_residual_decoder: ResidualVelocityDecoder | None = None
        if visual_residual_token_dim:
            if register_pooling != "mean":
                raise ValueError(
                    "Output-side visual residual requires the checkpoint-compatible mean path"
                )
            self.visual_residual_encoder = VisualMotionResidualEncoder(
                input_dim=vggt_feature_dim,
                register_token_count=register_token_count,
                register_projection_dim=visual_residual_register_dim,
                frame_dim=visual_residual_frame_dim,
                token_dim=visual_residual_token_dim,
                dropout=dropout,
            )
            self.visual_residual_decoder = ResidualVelocityDecoder(
                token_dim=visual_residual_token_dim,
                steps_per_token=steps_per_token,
                max_forward_correction_mps=visual_residual_max_forward_mps,
                max_lateral_correction_mps=visual_residual_max_lateral_mps,
                max_yaw_rate_correction_rps=visual_residual_max_yaw_rate_rps,
            )

    @property
    def output_action_token_dim(self) -> int:
        return self.base_action_token_dim + self.visual_residual_token_dim

    def forward(
        self,
        camera_hidden: Tensor,
        register_hidden_mean: Tensor,
        frame_times: Tensor,
        future_times: Tensor,
        register_hidden: Tensor | None = None,
        pose_enc: Tensor | None = None,
        disable_visual_residual: bool = False,
    ) -> TokenizerOutput:
        if camera_hidden.shape[1] - 1 != self.num_action_tokens:
            raise ValueError(
                f"Expected {self.num_action_tokens + 1} frames, got {camera_hidden.shape[1]}"
            )
        base_action_tokens = self.encoder(
            camera_hidden,
            register_hidden_mean,
            frame_times,
            register_hidden=register_hidden,
        )
        base_reconstruction, base_increments = self.decoder(
            base_action_tokens, future_times
        )
        if self.visual_residual_encoder is None:
            return TokenizerOutput(
                action_tokens=base_action_tokens,
                reconstruction=base_reconstruction,
                predicted_increments=base_increments,
            )
        if register_hidden is None or pose_enc is None:
            raise ValueError(
                "Output-side visual residual requires full register_hidden and pose_enc"
            )
        visual_tokens = self.visual_residual_encoder(
            register_hidden, pose_enc, frame_times
        )
        if disable_visual_residual:
            visual_tokens = torch.zeros_like(visual_tokens)
        assert self.visual_residual_decoder is not None
        residual_increments = self.visual_residual_decoder(
            visual_tokens, future_times
        )
        increments = base_increments + residual_increments
        reconstruction = integrate_se2_increments(increments)
        action_tokens = torch.cat([base_action_tokens, visual_tokens], dim=-1)
        return TokenizerOutput(
            action_tokens=action_tokens,
            reconstruction=reconstruction,
            predicted_increments=increments,
            base_action_tokens=base_action_tokens,
            visual_residual_tokens=visual_tokens,
            base_reconstruction=base_reconstruction,
            base_increments=base_increments,
            residual_increments=residual_increments,
        )

    def decode(self, action_tokens: Tensor, future_times: Tensor) -> Tensor:
        """Decode `[B,4,D]` action tokens without any visual context."""
        if self.visual_residual_decoder is None:
            reconstruction, _ = self.decoder(action_tokens, future_times)
            return reconstruction
        if action_tokens.shape[-1] != self.output_action_token_dim:
            raise ValueError(
                "Expected concatenated motion/residual action dimension "
                f"{self.output_action_token_dim}, got {action_tokens.shape[-1]}"
            )
        base_tokens = action_tokens[..., : self.base_action_token_dim]
        visual_tokens = action_tokens[..., self.base_action_token_dim :]
        _, base_increments = self.decoder(base_tokens, future_times)
        residual_increments = self.visual_residual_decoder(
            visual_tokens, future_times
        )
        return integrate_se2_increments(base_increments + residual_increments)
