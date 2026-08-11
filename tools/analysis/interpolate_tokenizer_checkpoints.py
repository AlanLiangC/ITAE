#!/usr/bin/env python3
"""Create a single-model linear weight soup from compatible tokenizer checkpoints."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch import Tensor


def interpolate_model_states(
    left: dict[str, Tensor],
    right: dict[str, Tensor],
    alpha: float,
) -> dict[str, Tensor]:
    """Return ``(1-alpha) * left + alpha * right`` with strict compatibility."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if left.keys() != right.keys():
        missing = sorted(left.keys() - right.keys())
        unexpected = sorted(right.keys() - left.keys())
        raise ValueError(
            f"Checkpoint model keys differ: missing={missing}, unexpected={unexpected}"
        )
    result = {}
    for key, left_value in left.items():
        right_value = right[key]
        if left_value.shape != right_value.shape or left_value.dtype != right_value.dtype:
            raise ValueError(f"Checkpoint tensor mismatch for {key}")
        if left_value.is_floating_point():
            result[key] = torch.lerp(left_value, right_value, alpha)
        else:
            if not torch.equal(left_value, right_value):
                raise ValueError(f"Non-floating checkpoint tensor differs for {key}")
            result[key] = left_value.clone()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True, type=Path)
    parser.add_argument("--right", required=True, type=Path)
    parser.add_argument("--alpha", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    left = torch.load(args.left, map_location="cpu", weights_only=False)
    right = torch.load(args.right, map_location="cpu", weights_only=False)
    if "model" not in left or "model" not in right:
        raise ValueError("Both inputs must be tokenizer training checkpoints")
    model = interpolate_model_states(left["model"], right["model"], args.alpha)
    payload = {
        "model": model,
        "config": right.get("config", left.get("config")),
        "interpolation": {
            "left": str(args.left),
            "right": str(args.right),
            "alpha": args.alpha,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, args.output)
    print(f"Saved interpolated tokenizer checkpoint to {args.output}", flush=True)


if __name__ == "__main__":
    main()
