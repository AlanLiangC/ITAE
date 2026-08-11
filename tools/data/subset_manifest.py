#!/usr/bin/env python3
"""Create a deterministic small manifest for cache/training smoke tests."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from vision_action_tokenizer.data.manifest import load_manifest, save_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--count", type=int)
    selection.add_argument(
        "--indices", type=str, help="Comma-separated deterministic source indices"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    windows = load_manifest(args.input)
    if args.indices is not None:
        indices = [int(value.strip()) for value in args.indices.split(",") if value.strip()]
        if not indices or len(set(indices)) != len(indices):
            raise ValueError("--indices must contain unique comma-separated integers")
        if min(indices) < 0 or max(indices) >= len(windows):
            raise IndexError("--indices contains an out-of-range manifest index")
    else:
        assert args.count is not None
        if args.count <= 0:
            raise ValueError("--count must be positive")
        if args.count > len(windows):
            raise ValueError(
                f"Requested {args.count} samples from a {len(windows)}-sample manifest"
            )
        indices = sorted(random.Random(args.seed).sample(range(len(windows)), args.count))
    save_manifest([windows[index] for index in indices], args.output)
    print(f"Wrote {len(indices)} samples to {args.output}")


if __name__ == "__main__":
    main()
