#!/usr/bin/env python3
"""Export log-disjoint NAVSIM action-tokenizer manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from vision_action_tokenizer.config import load_config
from vision_action_tokenizer.data.manifest import WindowRecord
from vision_action_tokenizer.data.navsim import (
    NavsimExportConfig,
    build_navsim_windows,
    split_navsim_logs,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _motion_bucket(window: WindowRecord, distance: float) -> str:
    endpoint = np.asarray(window.trajectory[-1], dtype=np.float64)
    if distance < 2.0:
        return "stationary"
    if abs(endpoint[1]) > 2.0 or abs(endpoint[2]) > 0.15:
        return "turn"
    return "straight_slow" if distance < 20.0 else "straight_fast"


class _SplitReportAccumulator:
    """Keep exact export statistics while window batches are released from memory."""

    def __init__(self, export: NavsimExportConfig, log_names: list[str]) -> None:
        self.export = export
        self.log_names = sorted(log_names)
        self.num_candidates = 0
        self.num_windows = 0
        self.rejected: Counter[str] = Counter()
        self.max_image_time_error_us = 0
        self.max_trajectory_time_error_us = 0
        self.windows_per_group: Counter[str] = Counter()
        self.motion_bucket_counts: Counter[str] = Counter()
        self.endpoint_distances: list[float] = []
        self.max_native_knot_xy_error_m = 0.0
        self.max_native_knot_yaw_error_rad = 0.0

    def update(self, windows: list[WindowRecord], batch_report: dict[str, Any]) -> None:
        self.num_candidates += int(batch_report["num_candidates"])
        self.num_windows += len(windows)
        self.rejected.update(batch_report["rejected"])
        self.max_image_time_error_us = max(
            self.max_image_time_error_us,
            int(batch_report["max_image_time_error_us"]),
        )
        self.max_trajectory_time_error_us = max(
            self.max_trajectory_time_error_us,
            int(batch_report["max_trajectory_time_error_us"]),
        )
        for window in windows:
            self.windows_per_group[str(window.group_token)] += 1
            endpoint = np.asarray(window.trajectory[-1], dtype=np.float64)
            distance = float(np.linalg.norm(endpoint[:2]))
            self.endpoint_distances.append(distance)
            self.motion_bucket_counts[_motion_bucket(window, distance)] += 1

            dense = np.asarray(window.trajectory, dtype=np.float64)
            native = np.asarray(window.native_trajectory, dtype=np.float64)
            knots = dense[[4, 9, 14, 19, 24, 29, 34, 39]]
            self.max_native_knot_xy_error_m = max(
                self.max_native_knot_xy_error_m,
                float(np.linalg.norm(knots[:, :2] - native[:, :2], axis=-1).max()),
            )
            yaw_delta = (knots[:, 2] - native[:, 2] + np.pi) % (2 * np.pi) - np.pi
            self.max_native_knot_yaw_error_rad = max(
                self.max_native_knot_yaw_error_rad,
                float(np.abs(yaw_delta).max()),
            )

    def finish(self, *, log_batch_size: int) -> dict[str, Any]:
        report: dict[str, Any] = {
            "dataset_name": "navsim",
            "split": self.export.split,
            "log_names": self.log_names,
            "num_logs": len(self.log_names),
            "log_batch_size": log_batch_size,
            "num_candidates": self.num_candidates,
            "num_windows": self.num_windows,
            "rejected": dict(sorted(self.rejected.items())),
            "frame_interval": self.export.frame_interval,
            "frame_offsets_s": [0.0, 1.0, 2.0, 3.0, 4.0],
            "native_trajectory_hz": 2,
            "target_trajectory_hz": 10,
            "future_points": 40,
            "max_image_time_error_us": self.max_image_time_error_us,
            "max_trajectory_time_error_us": self.max_trajectory_time_error_us,
        }
        if not self.endpoint_distances:
            return report
        distances = np.asarray(self.endpoint_distances, dtype=np.float64)
        report.update(
            {
                "windows_per_group": dict(sorted(self.windows_per_group.items())),
                "motion_bucket_counts": dict(sorted(self.motion_bucket_counts.items())),
                "endpoint_distance_m": {
                    "min": float(distances.min()),
                    "median": float(np.median(distances)),
                    "p95": float(np.percentile(distances, 95)),
                    "max": float(distances.max()),
                },
                "max_native_knot_xy_error_m": self.max_native_knot_xy_error_m,
                "max_native_knot_yaw_error_rad": self.max_native_knot_yaw_error_rad,
            }
        )
        return report


def _export_split(
    export: NavsimExportConfig,
    log_names: list[str],
    output: Path,
    *,
    log_batch_size: int,
    max_scenes: int | None,
) -> tuple[dict[str, Any], set[str]]:
    """Export one split atomically without retaining all log frames or windows."""
    if log_batch_size <= 0:
        raise ValueError("log_batch_size must be positive")
    ordered_logs = sorted(log_names)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    accumulator = _SplitReportAccumulator(export, ordered_logs)
    processed_batches = 0
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for start in range(0, len(ordered_logs), log_batch_size):
                remaining = (
                    None
                    if max_scenes is None
                    else max_scenes - accumulator.num_candidates
                )
                if remaining is not None and remaining <= 0:
                    break
                batch_logs = ordered_logs[start : start + log_batch_size]
                windows, batch_report = build_navsim_windows(
                    export,
                    log_names=batch_logs,
                    max_scenes=remaining,
                )
                # Stable across log_batch_size changes: logs and tokens are both ordered.
                windows.sort(key=lambda window: (str(window.group_token), window.sample_token))
                for window in windows:
                    handle.write(json.dumps(asdict(window), ensure_ascii=False) + "\n")
                accumulator.update(windows, batch_report)
                processed_batches += 1
                print(
                    f"export split={export.split} batches={processed_batches} "
                    f"logs={min(start + log_batch_size, len(ordered_logs))}/"
                    f"{len(ordered_logs)} "
                    f"candidates={accumulator.num_candidates} "
                    f"windows={accumulator.num_windows}",
                    flush=True,
                )
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    report = accumulator.finish(log_batch_size=log_batch_size)
    return report, set(accumulator.windows_per_group)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--train-output", required=True, type=Path)
    parser.add_argument("--val-output", required=True, type=Path)
    parser.add_argument("--report-output", required=True, type=Path)
    parser.add_argument("--max-scenes-per-split", type=int)
    parser.add_argument(
        "--log-batch-size",
        type=int,
        help="Logs loaded at once; defaults to navsim_export.log_batch_size or 8",
    )
    args = parser.parse_args()
    if args.max_scenes_per_split is not None and args.max_scenes_per_split <= 0:
        raise ValueError("--max-scenes-per-split must be positive")

    root_config = load_config(args.config)
    values = root_config["navsim_export"]
    data_root = Path(values["data_root"])
    export = NavsimExportConfig(
        data_root=data_root,
        split=str(values.get("split", "mini")),
        num_history_frames=int(values.get("num_history_frames", 1)),
        num_future_frames=int(values.get("num_future_frames", 8)),
        frame_interval=int(values.get("frame_interval", 1)),
        visual_frame_indices=tuple(map(int, values.get("visual_frame_indices", [0, 2, 4, 6, 8]))),
        max_time_error_s=float(values.get("max_time_error_s", 0.06)),
        has_route=bool(values.get("has_route", True)),
    )
    os.environ.setdefault("OPENSCENE_DATA_ROOT", str(data_root))
    os.environ.setdefault("NUPLAN_MAPS_ROOT", str(data_root / "maps"))
    os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
    train_logs, val_logs = split_navsim_logs(
        export.log_path,
        train_fraction=float(values.get("train_fraction", 0.8)),
        seed=int(values.get("split_seed", root_config.get("seed", 42))),
    )
    log_batch_size = (
        args.log_batch_size
        if args.log_batch_size is not None
        else int(values.get("log_batch_size", 8))
    )
    if log_batch_size <= 0:
        raise ValueError("log_batch_size must be positive")
    train_report, train_groups = _export_split(
        export,
        train_logs,
        args.train_output,
        log_batch_size=log_batch_size,
        max_scenes=args.max_scenes_per_split,
    )
    val_report, val_groups = _export_split(
        export,
        val_logs,
        args.val_output,
        log_batch_size=log_batch_size,
        max_scenes=args.max_scenes_per_split,
    )
    if not train_report["num_windows"] or not val_report["num_windows"]:
        raise ValueError("NAVSIM export produced an empty train or validation manifest")
    if train_groups & val_groups:
        raise ValueError("NAVSIM train/validation group leakage detected")
    report = {
        "schema_version": 2,
        "config": values,
        "train_logs": train_logs,
        "val_logs": val_logs,
        "train": train_report,
        "validation": val_report,
        "train_manifest": str(args.train_output),
        "val_manifest": str(args.val_output),
        "train_manifest_sha256": _sha256(args.train_output),
        "val_manifest_sha256": _sha256(args.val_output),
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "report_output": str(args.report_output),
                "train": {
                    "num_logs": train_report["num_logs"],
                    "num_candidates": train_report["num_candidates"],
                    "num_windows": train_report["num_windows"],
                    "manifest": str(args.train_output),
                    "manifest_sha256": report["train_manifest_sha256"],
                },
                "validation": {
                    "num_logs": val_report["num_logs"],
                    "num_candidates": val_report["num_candidates"],
                    "num_windows": val_report["num_windows"],
                    "manifest": str(args.val_output),
                    "manifest_sha256": report["val_manifest_sha256"],
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
