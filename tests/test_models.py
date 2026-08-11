from __future__ import annotations

import pytest
import torch

from tools.analysis.interpolate_tokenizer_checkpoints import interpolate_model_states
from vision_action_tokenizer.losses import LossConfig, TokenizerLoss, trajectory_xy_loss
from vision_action_tokenizer.models.decoder import (
    ResidualVelocityDecoder,
    SE2IncrementDecoder,
    integrate_se2_increments,
    trajectory_to_body_increments,
)
from vision_action_tokenizer.models.tokenizer import (
    IntervalActionEncoder,
    VisionActionTokenizer,
    pose_motion_features,
)
from vision_action_tokenizer.trainer import (
    _reduce_sample_weighted_metrics,
    _validate_multi_source_resume_config,
)


def test_joint_resume_rejects_different_source_provenance() -> None:
    current = {
        "data": {"sources": {"navsim": {}}, "sampling": {"strategy": "balanced"}},
        "data_runtime": {"sources": {"navsim": {"manifest": "new"}}},
        "action_tokenizer": {"num_action_tokens": 4},
        "loss": {"trajectory_weight": 1.0},
        "vision_backbone": {
            "checkpoint_sha256": "sha",
            "cache_token_mode": "camera_register_tokens",
            "feature_dim": 2048,
        },
    }
    saved = {
        **current,
        "data_runtime": {"sources": {"navsim": {"manifest": "old"}}},
    }
    with pytest.raises(ValueError, match="provenance/config mismatch"):
        _validate_multi_source_resume_config(current, saved)

    _validate_multi_source_resume_config(current, current)


def test_se2_increment_round_trip() -> None:
    increments = torch.randn(3, 40, 3) * torch.tensor([0.5, 0.1, 0.03])
    trajectory = integrate_se2_increments(increments)
    recovered = trajectory_to_body_increments(trajectory)
    assert torch.allclose(recovered, increments, atol=1e-5)


def test_se2_increment_round_trip_across_pi_boundary() -> None:
    increments = torch.zeros(1, 4, 3)
    increments[0, :, 0] = 1.0
    increments[0, :, 2] = torch.tensor([3.10, 0.10, -0.20, -3.00])
    recovered = trajectory_to_body_increments(integrate_se2_increments(increments))
    assert torch.allclose(recovered, increments, atol=1e-5)


def test_interval_encoder_uses_measured_frame_times() -> None:
    torch.manual_seed(1)
    encoder = IntervalActionEncoder(input_dim=8, frame_geometry_dim=8, action_dim=4)
    camera = torch.randn(1, 5, 8)
    registers = torch.randn(1, 5, 8)
    regular = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0]])
    irregular = torch.tensor([[0.0, 0.8, 2.1, 2.9, 4.0]])
    assert not torch.allclose(
        encoder(camera, registers, regular), encoder(camera, registers, irregular)
    )


def test_masked_trajectory_loss_ignores_invalid_points() -> None:
    target = torch.zeros(1, 3, 3)
    prediction = target.clone()
    prediction[0, 2, :2] = 1000.0
    mask = torch.tensor([[True, True, False]])
    masked = trajectory_xy_loss(prediction, target, mask)
    clean = trajectory_xy_loss(prediction[:, :2], target[:, :2], None)
    assert torch.allclose(masked, clean)


def test_se2_decoder_shape_and_finite_output() -> None:
    decoder = SE2IncrementDecoder(action_dim=16, hidden_dim=32, steps_per_token=10)
    action = torch.randn(2, 4, 16)
    times = torch.arange(1, 41).float().unsqueeze(0).repeat(2, 1) / 10
    trajectory, increments = decoder(action, times)
    assert trajectory.shape == (2, 40, 3)
    assert increments.shape == (2, 40, 3)
    assert torch.isfinite(trajectory).all()


def test_velocity_decoder_uses_measured_lidar_timestep() -> None:
    decoder = SE2IncrementDecoder(
        action_dim=16,
        hidden_dim=32,
        steps_per_token=10,
        parameterization="velocity",
        initial_forward_speed_mps=5.0,
    )
    with torch.no_grad():
        decoder.decoder[-1].weight.zero_()
    action = torch.zeros(2, 4, 16)
    regular = torch.arange(1, 41).float() / 10
    stretched = torch.arange(1, 41).float() / 8
    times = torch.stack([regular, stretched])
    _, increments = decoder(action, times)
    ratio = increments[1, :, 0] / increments[0, :, 0]
    assert torch.allclose(ratio, torch.full_like(ratio, 1.25), atol=1e-5)


def test_rich_register_attention_tokenizer_backward() -> None:
    tokenizer = VisionActionTokenizer(
        vggt_feature_dim=32,
        frame_geometry_dim=16,
        action_token_dim=8,
        num_action_tokens=4,
        steps_per_token=10,
        decoder_hidden_dim=32,
        interval_mixer_layers=1,
        interval_mixer_heads=2,
        register_pooling="attention",
        register_summary_tokens=2,
        register_pool_dim=8,
        decoder_parameterization="velocity",
    )
    camera = torch.randn(2, 5, 32)
    register_mean = torch.randn(2, 5, 32)
    registers = torch.randn(2, 5, 16, 32)
    frame_times = torch.arange(5).float().repeat(2, 1)
    future_times = torch.arange(1, 41).float().repeat(2, 1) / 10
    output = tokenizer(
        camera,
        register_mean,
        frame_times,
        future_times,
        register_hidden=registers,
    )
    target = torch.zeros_like(output.reconstruction)
    loss, terms = TokenizerLoss(
        LossConfig(
            body_velocity_weight=0.1,
            yaw_rate_weight=0.02,
            acceleration_weight=0.002,
            jerk_weight=0.00002,
            boundary_continuity_weight=0.01,
        )
    )(output, target, future_times)
    loss.backward()
    assert torch.isfinite(loss)
    assert "loss/body_velocity" in terms
    assert tokenizer.encoder.register_queries.grad is not None


def test_mean_residual_register_starts_exactly_from_mean_model() -> None:
    torch.manual_seed(5)
    common = {
        "vggt_feature_dim": 32,
        "frame_geometry_dim": 16,
        "action_token_dim": 8,
        "num_action_tokens": 4,
        "steps_per_token": 10,
        "decoder_hidden_dim": 32,
        "interval_mixer_layers": 1,
        "interval_mixer_heads": 2,
        "decoder_parameterization": "velocity",
    }
    mean_model = VisionActionTokenizer(**common, register_pooling="mean").eval()
    residual_model = VisionActionTokenizer(
        **common,
        register_pooling="mean_residual",
        register_token_count=16,
        register_residual_dim=8,
    ).eval()
    incompatible = residual_model.load_state_dict(mean_model.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert incompatible.missing_keys
    assert all(key.startswith("encoder.register_residual_") for key in incompatible.missing_keys)

    camera = torch.randn(2, 5, 32)
    register_mean = torch.randn(2, 5, 32)
    registers = torch.randn(2, 5, 16, 32)
    frame_times = torch.arange(5).float().repeat(2, 1)
    future_times = torch.arange(1, 41).float().repeat(2, 1) / 10
    with torch.no_grad():
        mean_output = mean_model(camera, register_mean, frame_times, future_times)
        residual_output = residual_model(
            camera,
            register_mean,
            frame_times,
            future_times,
            register_hidden=registers,
        )
    assert torch.equal(mean_output.action_tokens, residual_output.action_tokens)
    assert torch.equal(mean_output.reconstruction, residual_output.reconstruction)
    assert torch.count_nonzero(residual_model.encoder.register_residual_gate) == 0


def test_zero_output_residual_is_exact_but_receives_immediate_gradient() -> None:
    torch.manual_seed(7)
    common = {
        "vggt_feature_dim": 32,
        "frame_geometry_dim": 16,
        "action_token_dim": 8,
        "num_action_tokens": 4,
        "steps_per_token": 10,
        "decoder_hidden_dim": 32,
        "decoder_parameterization": "velocity",
    }
    mean_model = VisionActionTokenizer(**common, register_pooling="mean").eval()
    residual_model = VisionActionTokenizer(
        **common,
        register_pooling="mean_residual",
        register_token_count=16,
        register_residual_dim=8,
        register_residual_gate_init=3.0,
        register_residual_zero_init=True,
    ).eval()
    residual_model.load_state_dict(mean_model.state_dict(), strict=False)
    camera = torch.randn(2, 5, 32)
    register_mean = torch.randn(2, 5, 32)
    registers = torch.randn(2, 5, 16, 32)
    frame_times = torch.arange(5).float().repeat(2, 1)
    future_times = torch.arange(1, 41).float().repeat(2, 1) / 10

    mean_output = mean_model(camera, register_mean, frame_times, future_times)
    residual_output = residual_model(
        camera,
        register_mean,
        frame_times,
        future_times,
        register_hidden=registers,
    )
    assert torch.equal(mean_output.reconstruction, residual_output.reconstruction)
    target = torch.randn_like(residual_output.reconstruction)
    loss = (residual_output.reconstruction - target).square().mean()
    loss.backward()
    output = residual_model.encoder.register_residual_output
    assert isinstance(output, torch.nn.Linear)
    assert output.weight.grad is not None
    assert torch.count_nonzero(output.weight.grad) > 0


def test_output_side_visual_residual_preserves_motion_and_decode_contract() -> None:
    torch.manual_seed(11)
    common = {
        "vggt_feature_dim": 32,
        "frame_geometry_dim": 16,
        "action_token_dim": 8,
        "num_action_tokens": 4,
        "steps_per_token": 10,
        "decoder_hidden_dim": 32,
        "decoder_parameterization": "velocity",
        "register_pooling": "mean",
        "register_token_count": 4,
    }
    motion = VisionActionTokenizer(**common).eval()
    residual = VisionActionTokenizer(
        **common,
        visual_residual_token_dim=4,
        visual_residual_frame_dim=8,
        visual_residual_register_dim=4,
    ).eval()
    incompatible = residual.load_state_dict(motion.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert all(key.startswith("visual_residual_") for key in incompatible.missing_keys)

    camera = torch.randn(3, 5, 32)
    register_mean = torch.randn(3, 5, 32)
    registers = torch.randn(3, 5, 4, 32)
    pose = torch.randn(3, 5, 9)
    pose[..., 3:7] = torch.nn.functional.normalize(pose[..., 3:7], dim=-1)
    frame_times = torch.arange(5).float().repeat(3, 1)
    future_times = torch.arange(1, 41).float().repeat(3, 1) / 10
    motion_output = motion(camera, register_mean, frame_times, future_times)
    residual_output = residual(
        camera,
        register_mean,
        frame_times,
        future_times,
        register_hidden=registers,
        pose_enc=pose,
    )
    assert torch.equal(motion_output.reconstruction, residual_output.reconstruction)
    assert residual_output.action_tokens.shape == (3, 4, 12)
    assert residual_output.visual_residual_tokens is not None
    assert torch.count_nonzero(residual_output.visual_residual_tokens) == 0
    assert torch.equal(
        residual.decode(residual_output.action_tokens, future_times),
        residual_output.reconstruction,
    )

    target = torch.randn_like(residual_output.reconstruction)
    loss = (residual_output.reconstruction - target).square().mean()
    loss.backward()
    assert residual.visual_residual_encoder is not None
    assert residual.visual_residual_encoder.output.weight.grad is not None
    assert torch.count_nonzero(
        residual.visual_residual_encoder.output.weight.grad
    ) > 0


def test_visual_residual_losses_measure_condition_dependence() -> None:
    torch.manual_seed(13)
    tokenizer = VisionActionTokenizer(
        vggt_feature_dim=16,
        frame_geometry_dim=8,
        action_token_dim=8,
        num_action_tokens=4,
        steps_per_token=10,
        decoder_hidden_dim=16,
        decoder_parameterization="velocity",
        register_pooling="mean",
        register_token_count=4,
        visual_residual_token_dim=4,
        visual_residual_frame_dim=8,
        visual_residual_register_dim=4,
    )
    assert tokenizer.visual_residual_encoder is not None
    with torch.no_grad():
        tokenizer.visual_residual_encoder.output.weight.normal_(std=0.02)
    camera = torch.randn(4, 5, 16)
    register_mean = torch.randn(4, 5, 16)
    registers = torch.randn(4, 5, 4, 16)
    pose = torch.randn(4, 5, 9)
    pose[..., 3:7] = torch.nn.functional.normalize(pose[..., 3:7], dim=-1)
    frame_times = torch.arange(5).float().repeat(4, 1)
    future_times = torch.arange(1, 41).float().repeat(4, 1) / 10
    output = tokenizer(
        camera,
        register_mean,
        frame_times,
        future_times,
        register_hidden=registers,
        pose_enc=pose,
    )
    shuffled = tokenizer(
        camera,
        register_mean,
        frame_times,
        future_times,
        register_hidden=registers.roll(1, dims=0),
        pose_enc=pose.roll(1, dims=0),
    )
    target = torch.randn_like(output.reconstruction) * torch.tensor([1.0, 0.2, 0.1])
    loss, terms = TokenizerLoss(
        LossConfig(
            residual_velocity_weight=0.25,
            residual_yaw_rate_weight=0.02,
            residual_mean_weight=0.01,
            residual_alignment_weight=0.01,
            conditional_shuffle_weight=0.1,
            conditional_shuffle_margin=0.01,
        )
    )(
        output,
        target,
        future_times,
        torch.ones(4, 40, dtype=torch.bool),
        global_step=10,
        shuffled_output=shuffled,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert "loss/residual_velocity" in terms
    assert "loss/residual_alignment" in terms
    assert "condition/shuffle_error_gap" in terms
    assert tokenizer.visual_residual_encoder.output.weight.grad is not None


def test_pose_motion_features_and_residual_decoder_contract() -> None:
    pose = torch.zeros(2, 5, 9)
    pose[..., 6] = 1.0
    pose[:, :, 2] = torch.arange(5).float()
    features = pose_motion_features(pose)
    assert features.shape == (2, 5, 13)
    assert torch.isfinite(features).all()

    decoder = ResidualVelocityDecoder(token_dim=4, steps_per_token=10)
    tokens = torch.zeros(2, 4, 4)
    future_times = torch.arange(1, 41).float().repeat(2, 1) / 10
    assert torch.count_nonzero(decoder(tokens, future_times)) == 0


def test_sample_weighted_validation_metrics() -> None:
    totals = {"metric/ade_m": torch.tensor(2 * 1.0 + 1 * 3.0)}
    averaged = _reduce_sample_weighted_metrics(totals, sample_count=3, world_size=1)
    assert averaged["metric/ade_m"].item() == pytest.approx(5 / 3)


def test_checkpoint_interpolation_is_strict() -> None:
    left = {"weight": torch.tensor([0.0, 2.0]), "count": torch.tensor(3)}
    right = {"weight": torch.tensor([2.0, 4.0]), "count": torch.tensor(3)}
    result = interpolate_model_states(left, right, alpha=0.25)
    assert torch.equal(result["weight"], torch.tensor([0.5, 2.5]))
    assert result["count"].item() == 3
    with pytest.raises(ValueError, match="keys differ"):
        interpolate_model_states(left, {"other": torch.ones(1)}, alpha=0.5)
    with pytest.raises(ValueError, match="Non-floating"):
        interpolate_model_states(left, {**right, "count": torch.tensor(4)}, alpha=0.5)


def test_full_vggt_tokenizer_loss_backward() -> None:
    tokenizer = VisionActionTokenizer(
        vggt_feature_dim=32,
        frame_geometry_dim=16,
        action_token_dim=8,
        num_action_tokens=4,
        steps_per_token=10,
        decoder_hidden_dim=32,
    )
    camera = torch.randn(3, 5, 32)
    registers = torch.randn(3, 5, 32)
    frame_times = torch.arange(5).float().unsqueeze(0).repeat(3, 1)
    future_times = torch.arange(1, 41).float().unsqueeze(0).repeat(3, 1) / 10
    target_increments = torch.randn(3, 40, 3) * torch.tensor([0.3, 0.05, 0.02])
    trajectory = integrate_se2_increments(target_increments)
    mask = torch.ones(3, 40, dtype=torch.bool)
    output = tokenizer(camera, registers, frame_times, future_times)
    assert output.action_tokens.shape == (3, 4, 8)
    assert output.reconstruction.shape == (3, 40, 3)
    loss, terms = TokenizerLoss(LossConfig(steps_per_token=10))(
        output, trajectory, future_times, mask
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert "loss/increment_xy" in terms
    assert "action/offdiag_cosine" in terms
    assert tokenizer.encoder.frame_projection[1].weight.grad is not None
