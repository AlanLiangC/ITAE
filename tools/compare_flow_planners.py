#!/usr/bin/env python3
"""Build a paired raw-vs-token flow planner comparison report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.torch import load_file


def _load_history(path: Path) -> list[dict[str, float]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _auc(history: list[dict[str, float]]) -> float | None:
    if len(history) < 2:
        return None
    x = np.asarray([row["seen_samples"] for row in history], dtype=np.float64)
    y = np.asarray([row["metric/ade_m"] for row in history], dtype=np.float64)
    order = np.argsort(x)
    x, y = x[order], y[order]
    span = x[-1] - x[0]
    return None if span <= 0 else float(np.trapz(y, x) / span)


def _thresholds(history: list[dict[str, float]]) -> dict[str, int | None]:
    output: dict[str, int | None] = {}
    for threshold in (1.0, 0.75, 0.5):
        reached = [
            int(row["seen_samples"])
            for row in history
            if float(row["metric/ade_m"]) <= threshold
        ]
        output[str(threshold)] = min(reached) if reached else None
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-eval", required=True, type=Path)
    parser.add_argument("--token-eval", required=True, type=Path)
    parser.add_argument("--raw-history", type=Path)
    parser.add_argument("--token-history", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    raw_meta = json.loads((args.raw_eval / "metrics.json").read_text(encoding="utf-8"))
    token_meta = json.loads((args.token_eval / "metrics.json").read_text(encoding="utf-8"))
    if raw_meta["sample_tokens"] != token_meta["sample_tokens"]:
        raise ValueError("Raw/token evaluation sample order differs")
    if raw_meta["scene_tokens"] != token_meta["scene_tokens"]:
        raise ValueError("Raw/token evaluation scene order differs")
    if raw_meta.get("vision_condition_hash") != token_meta.get("vision_condition_hash"):
        raise ValueError("Raw/token evaluations used different vision conditions")
    if raw_meta.get("ego_motion_condition_hash") != token_meta.get(
        "ego_motion_condition_hash"
    ):
        raise ValueError("Raw/token evaluations used different ego-motion conditions")
    if raw_meta.get("planner_core_hash") != token_meta.get("planner_core_hash"):
        raise ValueError("Raw/token evaluations used different planner core configs")
    raw_parameters = int(raw_meta["parameter_count"])
    token_parameters = int(token_meta["parameter_count"])
    parameter_difference_ratio = abs(raw_parameters - token_parameters) / min(
        raw_parameters, token_parameters
    )
    if parameter_difference_ratio > 0.05:
        raise ValueError("Raw/token planner parameter counts differ by more than 5%")
    raw = load_file(str(args.raw_eval / "predictions.safetensors"))
    token = load_file(str(args.token_eval / "predictions.safetensors"))
    difference = token["ade_per_sample"].numpy() - raw["ade_per_sample"].numpy()
    scene_tokens = np.asarray(raw_meta["scene_tokens"])
    unique_scenes = np.unique(scene_tokens)
    scene_differences = np.asarray(
        [difference[scene_tokens == scene].mean() for scene in unique_scenes]
    )
    rng = np.random.default_rng(args.seed)
    bootstrap = np.empty(args.bootstrap_samples, dtype=np.float64)
    for index in range(args.bootstrap_samples):
        bootstrap[index] = rng.choice(
            scene_differences, size=len(scene_differences), replace=True
        ).mean()
    ci = np.quantile(bootstrap, [0.025, 0.975]).tolist()
    raw_history = _load_history(args.raw_history) if args.raw_history else []
    token_history = _load_history(args.token_history) if args.token_history else []
    report = {
        "difference_definition": "token ADE - raw ADE; negative favors token",
        "paired_samples": len(difference),
        "paired_scenes": len(unique_scenes),
        "raw_parameter_count": raw_parameters,
        "token_parameter_count": token_parameters,
        "parameter_difference_ratio": parameter_difference_ratio,
        "mean_ade_difference_m": float(difference.mean()),
        "scene_bootstrap_95_ci_m": ci,
        "raw_metrics": raw_meta["metrics"],
        "token_metrics": token_meta["metrics"],
        "convergence": {
            "raw_normalized_auc": _auc(raw_history),
            "token_normalized_auc": _auc(token_history),
            "raw_seen_samples_to_ade": _thresholds(raw_history),
            "token_seen_samples_to_ade": _thresholds(token_history),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
