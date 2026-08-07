#!/usr/bin/env python3
"""Render image strips and local trajectories for manual data QA."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from vision_action_tokenizer.data.manifest import load_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", type=int, default=4)
    args = parser.parse_args()
    windows = load_manifest(args.manifest)[: args.count]
    if not windows:
        raise SystemExit("Manifest is empty")
    figure, axes = plt.subplots(len(windows), 7, figsize=(21, 3.5 * len(windows)))
    if len(windows) == 1:
        axes = axes[None, :]
    for row, window in enumerate(windows):
        pairs = zip(window.image_paths, window.frame_times_s, strict=True)
        for column, (path, time_s) in enumerate(pairs):
            with Image.open(path) as image:
                axes[row, column].imshow(image)
            axes[row, column].set_title(f"t={time_s:.1f}s")
            axes[row, column].axis("off")
        xy = list(zip(*[(point[0], point[1]) for point in window.trajectory], strict=True))
        axes[row, 6].plot(xy[1], xy[0])
        axes[row, 6].scatter([0], [0], marker="x")
        axes[row, 6].set_xlabel("y left [m]")
        axes[row, 6].set_ylabel("x forward [m]")
        axes[row, 6].axis("equal")
        axes[row, 6].grid(True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(args.output, dpi=160)


if __name__ == "__main__":
    main()
