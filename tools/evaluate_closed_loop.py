#!/usr/bin/env python3
"""Evaluate precomputed rolling plans with explicit L0 or L1 semantics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vision_action_tokenizer.closed_loop import (
    KinematicReplayBackend,
    RecedingHorizonReplayBackend,
    ReplayScenario,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, type=Path, help="NPZ with timestamps and ego_states")
    parser.add_argument("--plans", required=True, type=Path, help="NPY [replans,T,3]")
    parser.add_argument("--level", choices=("l0", "l1"), default="l1")
    parser.add_argument("--execute-points", type=int, default=6)
    args = parser.parse_args()
    payload = np.load(args.scenario)
    scenario = ReplayScenario(
        scenario_id=args.scenario.stem,
        timestamps_s=payload["timestamps"],
        ego_states_xyyaw=payload["ego_states"],
        agents_xy=payload["agents_xy"] if "agents_xy" in payload else None,
    )
    backend_type = RecedingHorizonReplayBackend if args.level == "l0" else KinematicReplayBackend
    backend = backend_type(scenario, execute_points=args.execute_points)
    backend.reset()
    for plan in np.load(args.plans):
        result = backend.step(plan)
        if result.done:
            break
    print(json.dumps(backend.get_metrics(), indent=2))


if __name__ == "__main__":
    main()

