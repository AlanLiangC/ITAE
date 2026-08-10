"""Rectified-flow objective and an exact-NFE Euler sampler."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


def flow_matching_loss(
    model: nn.Module,
    clean_target: Tensor,
    condition_tokens: Tensor,
    condition_mask: Tensor,
    slot_times: Tensor,
    generator: torch.Generator | None = None,
    time_generator: torch.Generator | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Train `v(x_t,t,c)` against the constant straight-path velocity `x1-x0`."""
    batch = clean_target.shape[0]
    time = torch.rand(
        (batch,),
        dtype=clean_target.dtype,
        device=clean_target.device,
        generator=time_generator if time_generator is not None else generator,
    )
    noise = torch.randn(
        clean_target.shape,
        dtype=clean_target.dtype,
        device=clean_target.device,
        generator=generator,
    )
    broadcast_time = time.view(batch, *([1] * (clean_target.ndim - 1)))
    interpolated = (1.0 - broadcast_time) * noise + broadcast_time * clean_target
    target_velocity = clean_target - noise
    prediction = model(
        interpolated, time, condition_tokens, condition_mask, slot_times
    )
    per_sample = functional.mse_loss(
        prediction.float(), target_velocity.float(), reduction="none"
    ).flatten(1).mean(dim=1)
    loss = per_sample.mean()
    return loss, {
        "flow/loss": loss.detach(),
        "flow/time_mean": time.mean().detach(),
        "flow/target_velocity_rms": target_velocity.float().square().mean().sqrt().detach(),
        "flow/prediction_rms": prediction.float().square().mean().sqrt().detach(),
    }


@torch.no_grad()
def euler_sample(
    model: nn.Module,
    condition_tokens: Tensor,
    condition_mask: Tensor,
    slot_times: Tensor,
    target_shape: tuple[int, int],
    steps: int = 5,
    noise: Tensor | None = None,
    generator: torch.Generator | None = None,
    expected_nfe: int | None = None,
) -> tuple[Tensor, int]:
    """Integrate noise at t=0 to data at t=1 using exactly `steps` model calls."""
    if steps <= 0:
        raise ValueError("Euler sampling steps must be positive")
    batch = condition_tokens.shape[0]
    if noise is None:
        noise = torch.randn(
            (batch, *target_shape),
            device=condition_tokens.device,
            dtype=condition_tokens.dtype,
            generator=generator,
        )
    elif noise.shape != (batch, *target_shape):
        raise ValueError("Supplied Euler noise has the wrong shape")
    state = noise
    delta = 1.0 / steps
    nfe = 0
    for index in range(steps):
        time = torch.full(
            (batch,), index / steps, dtype=state.dtype, device=state.device
        )
        velocity = model(state, time, condition_tokens, condition_mask, slot_times)
        state = state + delta * velocity
        nfe += 1
    if expected_nfe is not None and nfe != expected_nfe:
        raise RuntimeError(f"Sampler used NFE={nfe}, expected {expected_nfe}")
    return state, nfe
