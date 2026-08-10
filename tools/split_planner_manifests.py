#!/usr/bin/env python3
"""Create deterministic scene-disjoint planner train/validation manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from vision_action_tokenizer.data.manifest import load_manifest, save_manifest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--final-manifest", required=True, type=Path)
    parser.add_argument(
        "--sample-universe-train-manifest",
        type=Path,
        help="Optional manifest whose sample tokens define the allowed train universe",
    )
    parser.add_argument(
        "--sample-universe-final-manifest",
        type=Path,
        help="Optional manifest whose sample tokens define the allowed final universe",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be between zero and one")

    windows = load_manifest(args.train_manifest)
    final_windows = load_manifest(args.final_manifest)
    input_train_count = len(windows)
    input_final_count = len(final_windows)
    if args.sample_universe_train_manifest is not None:
        allowed = {
            window.sample_token
            for window in load_manifest(args.sample_universe_train_manifest)
        }
        windows = [window for window in windows if window.sample_token in allowed]
    if args.sample_universe_final_manifest is not None:
        allowed = {
            window.sample_token
            for window in load_manifest(args.sample_universe_final_manifest)
        }
        final_windows = [
            window for window in final_windows if window.sample_token in allowed
        ]
    train_scenes = sorted({window.scene_token for window in windows})
    final_scenes = {window.scene_token for window in final_windows}
    overlap = set(train_scenes) & final_scenes
    if overlap:
        raise ValueError(f"Input train/final manifests share {len(overlap)} scenes")
    random.Random(args.seed).shuffle(train_scenes)
    validation_count = max(1, round(len(train_scenes) * args.validation_fraction))
    validation_scenes = set(train_scenes[:validation_count])
    planner_train = [w for w in windows if w.scene_token not in validation_scenes]
    planner_validation = [w for w in windows if w.scene_token in validation_scenes]
    if not planner_train or not planner_validation:
        raise ValueError("Planner split produced an empty partition")

    args.output.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": args.output / "planner_train.jsonl",
        "validation": args.output / "planner_validation.jsonl",
        "final_eval": args.output / "planner_final_eval.jsonl",
    }
    save_manifest(planner_train, paths["train"])
    save_manifest(planner_validation, paths["validation"])
    save_manifest(final_windows, paths["final_eval"])
    summary = {
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "source_train_manifest": str(args.train_manifest),
        "source_train_sha256": _sha256(args.train_manifest),
        "source_final_manifest": str(args.final_manifest),
        "source_final_sha256": _sha256(args.final_manifest),
        "sample_universe": {
            "train_manifest": (
                None
                if args.sample_universe_train_manifest is None
                else str(args.sample_universe_train_manifest)
            ),
            "final_manifest": (
                None
                if args.sample_universe_final_manifest is None
                else str(args.sample_universe_final_manifest)
            ),
            "filtered_train_samples": input_train_count - len(windows),
            "filtered_final_samples": input_final_count - len(final_windows),
        },
        "splits": {
            name: {
                "path": str(path),
                "sha256": _sha256(path),
                "samples": len(load_manifest(path)),
                "scenes": len({w.scene_token for w in load_manifest(path)}),
            }
            for name, path in paths.items()
        },
    }
    (args.output / "split_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
