#!/usr/bin/env python3
"""Audit NAVSIM native 2Hz knots against exported 10Hz trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vision_action_tokenizer.data.manifest import load_manifest
from vision_action_tokenizer.data.trajectory import (
    dense_trajectory_to_native_rate,
    navsim_2hz_to_10hz,
    wrap_angle,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    windows = load_manifest(args.manifest)
    if args.max_samples is not None:
        windows = windows[: args.max_samples]
    if not windows:
        raise ValueError("Manifest is empty")

    xy_errors: list[float] = []
    yaw_errors: list[float] = []
    regenerated_errors: list[float] = []
    roundtrip_errors: list[float] = []
    for window in windows:
        if window.dataset_name != "navsim" or window.native_trajectory is None:
            raise ValueError(f"Not a NAVSIM provenance record: {window.sample_token}")
        native = np.asarray(window.native_trajectory, dtype=np.float64)
        dense = np.asarray(window.trajectory, dtype=np.float64)
        knots = dense[[4, 9, 14, 19, 24, 29, 34, 39]]
        xy_errors.append(float(np.linalg.norm(knots[:, :2] - native[:, :2], axis=-1).max()))
        yaw_errors.append(float(np.abs(wrap_angle(knots[:, 2] - native[:, 2])).max()))
        regenerated, _ = navsim_2hz_to_10hz(native)
        regenerated_errors.append(float(np.abs(regenerated - dense).max()))
        selected_native = dense_trajectory_to_native_rate(dense)
        roundtrip, _ = navsim_2hz_to_10hz(selected_native)
        roundtrip_errors.append(float(np.abs(roundtrip - dense).max()))
    report = {
        "num_samples": len(windows),
        "max_native_knot_xy_error_m": max(xy_errors),
        "max_native_knot_yaw_error_rad": max(yaw_errors),
        "max_regenerated_value_error": max(regenerated_errors),
        "max_40_to_8_to_40_value_error": max(roundtrip_errors),
        "passed": max(xy_errors) < 1e-6
        and max(yaw_errors) < 1e-6
        and max(regenerated_errors) < 1e-6
        and max(roundtrip_errors) < 1e-6,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
