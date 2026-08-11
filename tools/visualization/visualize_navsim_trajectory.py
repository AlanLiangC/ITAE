#!/usr/bin/env python3
"""Render a 2x2 NAVSIM camera/native/dense trajectory diagnostic."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from PIL import Image

from vision_action_tokenizer.data.manifest import load_manifest

GT_COLOR = "#35D07F"
NATIVE_COLOR = "#FFB347"


def _draw(window, frame_index: int, figure, axes) -> None:
    for axis in axes.flat:
        axis.clear()
    with Image.open(window.image_paths[frame_index]) as image:
        axes[0, 0].imshow(image.convert("RGB"))
    axes[0, 0].set_title(
        f"CAM_F0 t={window.frame_times_s[frame_index]:.3f}s | {window.sample_token}"
    )
    axes[0, 0].axis("off")

    thumbnails = []
    for path in window.image_paths:
        with Image.open(path) as image:
            thumbnails.append(np.asarray(image.convert("RGB").resize((320, 180))))
    axes[0, 1].imshow(np.concatenate(thumbnails, axis=1))
    axes[0, 1].set_title("Selected visual window: 0/1/2/3/4s")
    axes[0, 1].axis("off")

    dense = np.asarray(window.trajectory)
    native = np.asarray(window.native_trajectory)
    axes[1, 0].plot(dense[:, 1], dense[:, 0], color=GT_COLOR, linewidth=2, label="10Hz GT")
    axes[1, 0].scatter(
        native[:, 1], native[:, 0], color=NATIVE_COLOR, s=28, label="native 2Hz", zorder=3
    )
    axes[1, 0].scatter([0], [0], color="white", edgecolor="black", marker="o", zorder=4)
    axes[1, 0].set_xlabel("left y [m]")
    axes[1, 0].set_ylabel("forward x [m]")
    axes[1, 0].axis("equal")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend()

    times = np.asarray(window.future_times_s)
    axes[1, 1].plot(times, dense[:, 0], color="#4CA3FF", label="x")
    axes[1, 1].plot(times, dense[:, 1], color="#F06AA6", label="y")
    axes[1, 1].plot(times, dense[:, 2], color="#9B7EDE", label="yaw")
    for time_s in np.arange(0.5, 4.01, 0.5):
        axes[1, 1].axvline(time_s, color=NATIVE_COLOR, alpha=0.15)
    axes[1, 1].set_xlabel("future time [s]")
    axes[1, 1].set_title("Interpolated pose components")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend()
    figure.tight_layout()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--fps", type=int, default=2)
    args = parser.parse_args()
    windows = load_manifest(args.manifest)
    if not 0 <= args.index < len(windows):
        raise IndexError(f"--index must be in [0,{len(windows)})")
    window = windows[args.index]
    if window.dataset_name != "navsim" or window.native_trajectory is None:
        raise ValueError("Selected record lacks NAVSIM native trajectory provenance")
    figure, axes = plt.subplots(2, 2, figsize=(16, 9), facecolor="white")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".png":
        _draw(window, 0, figure, axes)
        figure.savefig(args.output, dpi=140)
    else:
        animation = FuncAnimation(
            figure,
            lambda frame_index: _draw(window, frame_index, figure, axes),
            frames=len(window.image_paths),
            interval=1000 / args.fps,
        )
        animation.save(args.output, fps=args.fps)
    plt.close(figure)
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
