#!/usr/bin/env python3
"""Build config-driven, timestamp-checked nuScenes training windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vision_action_tokenizer.config import load_config
from vision_action_tokenizer.data.lidar import load_lidar_pose_records
from vision_action_tokenizer.data.manifest import ManifestBuilder, save_manifest
from vision_action_tokenizer.data.schema import (
    InfoSchemaAdapter,
    load_info_pickle,
    load_nuscenes_scene_lookup,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--info", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, help="Optional override for config data_root")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_root = args.data_root or Path(config["data"]["data_root"])
    _, entries = load_info_pickle(args.info)
    records = InfoSchemaAdapter().adapt(
        entries, scene_lookup=load_nuscenes_scene_lookup(data_root)
    )
    builder = ManifestBuilder.from_config(config, data_root=data_root)
    lidar_records = None
    lidar_report = None
    if builder.trajectory_pose_source == "lidar_sweeps":
        lidar_records, lidar_report = load_lidar_pose_records(
            data_root, {record.scene_token for record in records}
        )
    windows, report = builder.build(records, lidar_pose_records=lidar_records)
    if lidar_report is not None:
        report["lidar_pose_source"] = lidar_report
    save_manifest(windows, args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not windows:
        raise SystemExit("No valid windows were produced; inspect the report and input schema.")


if __name__ == "__main__":
    main()
