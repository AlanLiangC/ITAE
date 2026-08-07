from __future__ import annotations

import torch

from vision_action_tokenizer.losses import LossConfig, TokenizerLoss, trajectory_xy_loss
from vision_action_tokenizer.models.decoder import (
    SE2IncrementDecoder,
    integrate_se2_increments,
    trajectory_to_body_increments,
)
from vision_action_tokenizer.models.tokenizer import IntervalActionEncoder, VisionActionTokenizer


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
