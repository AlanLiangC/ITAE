#!/usr/bin/env python3
"""Build timestamp-checked 6-frame/60-point training windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vision_action_tokenizer.data.manifest import ManifestBuilder, save_manifest
from vision_action_tokenizer.data.schema import (
    InfoSchemaAdapter,
    load_info_pickle,
    load_nuscenes_scene_lookup,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--info", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--max-time-error", type=float, default=0.055)
    parser.add_argument("--anchor-stride", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, entries = load_info_pickle(args.info)
    records = InfoSchemaAdapter().adapt(
        entries, scene_lookup=load_nuscenes_scene_lookup(args.data_root)
    )
    builder = ManifestBuilder(
        args.data_root,
        frame_offsets_s=[0, 1, 2, 3, 4, 5],
        horizon_s=5.0,
        trajectory_hz=12,
        anchor_stride_s=args.anchor_stride,
        max_time_error_s=args.max_time_error,
    )
    windows, report = builder.build(records)
    save_manifest(windows, args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not windows:
        raise SystemExit("No valid windows were produced; inspect the report and input schema.")


if __name__ == "__main__":
    main()
