#!/usr/bin/env python3
"""Audit a nuScenes info pickle before any model training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vision_action_tokenizer.data.schema import (
    InfoSchemaAdapter,
    inspect_records,
    load_info_pickle,
    load_nuscenes_scene_lookup,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--info", required=True, type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata, entries = load_info_pickle(args.info)
    scene_lookup = load_nuscenes_scene_lookup(args.data_root) if args.data_root else None
    records = InfoSchemaAdapter().adapt(entries, scene_lookup=scene_lookup)
    report = inspect_records(records, args.data_root)
    report["outer_metadata_keys"] = sorted(metadata)
    report["first_info_keys"] = sorted(entries[0]) if entries else []
    report["scene_lookup_entries"] = len(scene_lookup or {})
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
