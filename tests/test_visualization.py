from __future__ import annotations

import numpy as np
import torch

from vision_action_tokenizer.visualization import (
    render_bev_trajectory_comparison,
    render_evaluation_diagnostic,
    render_vggt_evaluation_diagnostic,
    trajectory_time_color,
)


def test_bev_comparison_is_tensorboard_ready() -> None:
    times = torch.arange(1, 41, dtype=torch.float32) / 10
    target = torch.stack([3 * times, torch.sin(times), 0.1 * times], dim=-1)
    reconstruction_vis = target.clone()
    reconstruction_vis[:, 1] += 0.5
    reconstruction_traj = target.clone()
    reconstruction_traj[:, 0] *= 0.9
    image = render_bev_trajectory_comparison(
        target,
        reconstruction_vis,
        reconstruction_traj,
        times,
        sample_token="sample-token",
    )
    assert image.shape == (3, 430, 1200)
    assert image.dtype == torch.float32
    assert np.isfinite(image.numpy()).all()
    assert 0 <= float(image.min()) <= float(image.max()) <= 1
    assert trajectory_time_color(0) == (30, 245, 110)
    assert trajectory_time_color(1) == (255, 55, 45)


def test_evaluation_diagnostic_combines_camera_and_three_bev_panels() -> None:
    times = torch.arange(1, 41, dtype=torch.float32) / 10
    target = torch.stack([3 * times, torch.sin(times), 0.1 * times], dim=-1)
    camera_images = torch.linspace(-1, 1, 5 * 3 * 32 * 48).reshape(5, 3, 32, 48)
    image = render_evaluation_diagnostic(
        target,
        target + torch.tensor([0.5, 0.2, 0.0]),
        target + torch.tensor([-0.3, 0.1, 0.0]),
        times,
        camera_images=camera_images,
        frame_times=torch.arange(5, dtype=torch.float32),
        sample_token="sample-token",
    )
    assert image.shape == (3, 900, 1200)
    assert image.dtype == torch.float32
    assert torch.isfinite(image).all()
    assert 0 <= float(image.min()) <= float(image.max()) <= 1


def test_vggt_diagnostic_has_error_panel() -> None:
    times = torch.arange(1, 41, dtype=torch.float32) / 10
    target = torch.stack([2 * times, torch.sin(times), 0.1 * times], dim=-1)
    reconstruction = target + torch.tensor([0.4, -0.2, 0.02])
    increments = torch.zeros(40, 3)
    camera_images = torch.rand(5, 3, 32, 48)
    image = render_vggt_evaluation_diagnostic(
        target,
        reconstruction,
        increments,
        times,
        camera_images,
        torch.arange(5, dtype=torch.float32),
    )
    assert image.shape == (3, 900, 1200)
    assert torch.isfinite(image).all()
