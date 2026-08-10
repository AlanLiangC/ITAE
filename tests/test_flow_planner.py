from __future__ import annotations

import pytest
import torch

from vision_action_tokenizer.models.flow_planner import ConditionalFlowPlanner
from vision_action_tokenizer.planner_evaluator import planner_slot_times


@pytest.mark.parametrize("target_shape", [(40, 3), (4, 192)])
def test_flow_planner_preserves_target_shape(target_shape: tuple[int, int]) -> None:
    model = ConditionalFlowPlanner(
        target_dim=target_shape[1],
        target_slots=target_shape[0],
        condition_dim=32,
        condition_tokens=16,
        model_dim=64,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
    )
    output = model(
        torch.randn(2, *target_shape),
        torch.rand(2),
        torch.randn(2, 16, 32),
        torch.ones(2, 16, dtype=torch.bool),
        torch.rand(2, target_shape[0]),
    )
    assert output.shape == (2, *target_shape)
    # Zero output initialization gives both representations the same initial field.
    torch.testing.assert_close(output, torch.zeros_like(output))


def test_planner_slot_times_use_real_raw_times_and_fixed_token_centers() -> None:
    future = torch.arange(1, 41).float().view(1, 40) / 10
    torch.testing.assert_close(planner_slot_times("raw_trajectory", future, 40), future)
    token = planner_slot_times("v4_action_token", future, 4)
    torch.testing.assert_close(token, torch.tensor([[0.5, 1.5, 2.5, 3.5]]))


def test_raw_and_token_planners_initialize_shared_core_identically() -> None:
    arguments = {
        "condition_dim": 32,
        "condition_tokens": 16,
        "model_dim": 64,
        "num_heads": 4,
        "num_layers": 2,
        "dropout": 0.0,
    }
    torch.manual_seed(42)
    raw = ConditionalFlowPlanner(target_dim=3, target_slots=40, **arguments)
    torch.manual_seed(42)
    token = ConditionalFlowPlanner(target_dim=192, target_slots=4, **arguments)
    target_specific = ("target_input.", "target_slot_embedding", "output.")
    raw_shared = {
        key: value
        for key, value in raw.state_dict().items()
        if not key.startswith(target_specific)
    }
    token_shared = {
        key: value
        for key, value in token.state_dict().items()
        if not key.startswith(target_specific)
    }
    assert raw_shared.keys() == token_shared.keys()
    for key in raw_shared:
        torch.testing.assert_close(raw_shared[key], token_shared[key], rtol=0, atol=0)
