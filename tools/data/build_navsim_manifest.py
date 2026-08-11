#!/usr/bin/env python3
"""Export log-disjoint NAVSIM action-tokenizer manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from vision_action_tokenizer.config import load_config
from vision_action_tokenizer.data.manifest import save_manifest
from vision_action_tokenizer.data.navsim import (
    NavsimExportConfig,
    build_navsim_windows,
    split_navsim_logs,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--train-output", required=True, type=Path)
    parser.add_argument("--val-output", required=True, type=Path)
    parser.add_argument("--report-output", required=True, type=Path)
    parser.add_argument("--max-scenes-per-split", type=int)
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
    train_windows, train_report = build_navsim_windows(
        export, log_names=train_logs, max_scenes=args.max_scenes_per_split
    )
    val_windows, val_report = build_navsim_windows(
        export, log_names=val_logs, max_scenes=args.max_scenes_per_split
    )
    if not train_windows or not val_windows:
        raise ValueError("NAVSIM export produced an empty train or validation manifest")
    if {window.group_token for window in train_windows} & {
        window.group_token for window in val_windows
    }:
        raise ValueError("NAVSIM train/validation group leakage detected")

    save_manifest(train_windows, args.train_output)
    save_manifest(val_windows, args.val_output)
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
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
