from __future__ import annotations

import numpy as np
import torch

from vision_action_tokenizer.visualization import (
    render_bev_trajectory_comparison,
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
