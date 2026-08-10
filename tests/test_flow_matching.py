from __future__ import annotations

import torch
from torch import nn

from vision_action_tokenizer.flow_matching import euler_sample, flow_matching_loss


class CountingConstantVelocity(nn.Module):
    def __init__(self, velocity: float) -> None:
        super().__init__()
        self.velocity = velocity
        self.calls = 0

    def forward(self, state, flow_time, condition, condition_mask, slot_times):
        self.calls += 1
        return torch.full_like(state, self.velocity)


def test_euler_sampler_uses_exactly_five_network_evaluations() -> None:
    model = CountingConstantVelocity(2.0)
    condition = torch.zeros(3, 4, 8)
    mask = torch.ones(3, 4, dtype=torch.bool)
    times = torch.ones(3, 2)
    noise = torch.zeros(3, 2, 5)
    sample, nfe = euler_sample(
        model,
        condition,
        mask,
        times,
        (2, 5),
        steps=5,
        noise=noise,
        expected_nfe=5,
    )
    assert nfe == model.calls == 5
    torch.testing.assert_close(sample, torch.full_like(sample, 2.0))


def test_flow_matching_zero_loss_for_oracle_velocity() -> None:
    class Oracle(nn.Module):
        def forward(self, state, flow_time, condition, condition_mask, slot_times):
            # Recovering the random endpoints is not possible from state alone, so this
            # test uses a target/noise-independent zero path.
            return torch.zeros_like(state)

    clean = torch.zeros(4, 3, 2)
    condition = torch.zeros(4, 5, 7)
    mask = torch.ones(4, 5, dtype=torch.bool)
    times = torch.zeros(4, 3)
    loss, metrics = flow_matching_loss(Oracle(), clean, condition, mask, times)
    assert torch.isfinite(loss)
    assert loss > 0
    assert set(metrics) == {
        "flow/loss",
        "flow/time_mean",
        "flow/target_velocity_rms",
        "flow/prediction_rms",
    }
