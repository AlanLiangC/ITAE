from __future__ import annotations

import torch

from vision_action_tokenizer.data.planner_dataset import PlannerTargetNormalizer
from vision_action_tokenizer.planner_evaluator import PlannerOutputDecoder


def test_raw_output_decoder_denormalizes_and_wraps_yaw() -> None:
    normalizer = PlannerTargetNormalizer((2, 3))
    targets = torch.tensor(
        [
            [[1.0, 2.0, 3.2], [3.0, 4.0, 3.3]],
            [[2.0, 1.0, 3.4], [4.0, 3.0, 3.5]],
        ]
    )
    normalizer.fit(targets)
    decoder = PlannerOutputDecoder("raw_trajectory", normalizer)
    normalized = normalizer.normalize(targets)
    output = decoder(normalized, torch.tensor([[0.1, 0.2], [0.1, 0.2]]))
    torch.testing.assert_close(output[..., :2], targets[..., :2])
    assert torch.all(output[..., 2] >= -torch.pi)
    assert torch.all(output[..., 2] < torch.pi)
