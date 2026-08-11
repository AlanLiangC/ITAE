#!/usr/bin/env python3
"""Audit VGGT-Omega camera motion against LiDAR ego trajectories before training."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from vggt_omega.utils.pose_enc import encoding_to_camera

from vision_action_tokenizer.config import load_config
from vision_action_tokenizer.data.dataset import NuScenesWindowDataset, VGGTOmegaResize
from vision_action_tokenizer.models.vggt_omega import OmegaCameraFeatureExtractor


def _representative_indices(dataset: NuScenesWindowDataset) -> list[int]:
    best: dict[str, tuple[float, int]] = {
        "stationary": (math.inf, -1),
        "medium": (math.inf, -1),
        "long": (-math.inf, -1),
        "turn": (-math.inf, -1),
    }
    for index, window in enumerate(dataset.windows):
        x, y, _ = window.trajectory[-1]
        distance = math.hypot(x, y)
        if distance < best["stationary"][0]:
            best["stationary"] = (distance, index)
        medium_score = abs(distance - 20.0) + 0.2 * abs(y)
        if medium_score < best["medium"][0]:
            best["medium"] = (medium_score, index)
        if distance > best["long"][0]:
            best["long"] = (distance, index)
        turn_score = abs(y) if distance > 8.0 else -1.0
        if turn_score > best["turn"][0]:
            best["turn"] = (turn_score, index)
    return [best[name][1] for name in ("stationary", "medium", "long", "turn")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--calibration",
        type=Path,
        help="Apply a global scale JSON fitted on the train split instead of fitting here",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("VGGT-Omega motion audit requires CUDA")
    config = load_config(args.config)
    backbone = config["vision_backbone"]
    transform = VGGTOmegaResize(
        int(backbone["image_resolution"]),
        str(backbone["resize_mode"]),
        int(backbone["patch_size"]),
    )
    dataset = NuScenesWindowDataset(args.manifest, transform=transform, load_images=True)
    indices = _representative_indices(dataset)
    extractor = OmegaCameraFeatureExtractor(
        backbone["checkpoint_path"], backbone.get("checkpoint_sha256"), freeze=True
    ).cuda().eval()
    samples = []
    predicted_all = []
    target_all = []
    with torch.inference_mode():
        for index in indices:
            sample = dataset[index]
            images = sample["images"]
            assert isinstance(images, torch.Tensor)
            features = extractor(images.unsqueeze(0).cuda())
            extrinsics, _ = encoding_to_camera(
                features.pose_enc, tuple(images.shape[-2:])
            )
            rotation = extrinsics[0, :, :3, :3]
            translation = extrinsics[0, :, :3, 3]
            centers = -(
                rotation.transpose(-1, -2) @ translation.unsqueeze(-1)
            ).squeeze(-1)
            centers = centers - centers[:1]
            # OpenCV camera x points right and z points forward; ego y points left.
            predicted_xy = torch.stack([centers[:, 2], -centers[:, 0]], dim=-1).cpu().numpy()
            trajectory = np.asarray(sample["trajectory"], dtype=np.float32)
            future_times = np.asarray(sample["future_times"], dtype=np.float32)
            frame_times = np.asarray(sample["frame_times"], dtype=np.float32)
            target_xy = np.zeros((len(frame_times), 2), dtype=np.float32)
            for frame_index, frame_time in enumerate(frame_times[1:], start=1):
                nearest = int(np.abs(future_times - frame_time).argmin())
                target_xy[frame_index] = trajectory[nearest, :2]
            predicted_all.append(predicted_xy[1:])
            target_all.append(target_xy[1:])
            samples.append(
                {
                    "index": index,
                    "sample_token": sample["sample_token"],
                    "images": images,
                    "predicted_xy": predicted_xy,
                    "target_xy": target_xy,
                }
            )

    predicted_stack = np.concatenate(predicted_all, axis=0)
    target_stack = np.concatenate(target_all, axis=0)
    if args.calibration is None:
        denominator = float(np.square(predicted_stack).sum())
        global_scale = float(
            (predicted_stack * target_stack).sum() / max(denominator, 1e-9)
        )
        scale_source = f"fitted on {args.manifest}"
    else:
        calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
        global_scale = float(calibration["global_scale"])
        scale_source = f"loaded from {args.calibration}"
    calibrated = predicted_stack * global_scale
    keyframe_ade = float(np.linalg.norm(calibrated - target_stack, axis=-1).mean())
    direction_cosine = float(
        np.sum(predicted_stack * target_stack)
        / max(
            np.linalg.norm(predicted_stack) * np.linalg.norm(target_stack),
            1e-9,
        )
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(len(samples), 2, figsize=(12, 3.4 * len(samples)))
    report_samples = []
    for row, sample in enumerate(samples):
        images = sample.pop("images")
        contact = torch.cat([frame for frame in images], dim=2).permute(1, 2, 0).numpy()
        axes[row, 0].imshow(np.clip(contact, 0, 1))
        sample_suffix = str(sample["sample_token"])[-10:]
        axes[row, 0].set_title(f"index={sample['index']} sample=...{sample_suffix}")
        axes[row, 0].axis("off")
        predicted_xy = sample["predicted_xy"] * global_scale
        target_xy = sample["target_xy"]
        axes[row, 1].plot(target_xy[:, 1], target_xy[:, 0], "o-", label="LiDAR GT")
        axes[row, 1].plot(predicted_xy[:, 1], predicted_xy[:, 0], "o-", label="VGGT calibrated")
        axes[row, 1].set_aspect("equal", adjustable="datalim")
        axes[row, 1].set_xlabel("y left (m)")
        axes[row, 1].set_ylabel("x forward (m)")
        axes[row, 1].grid(True)
        axes[row, 1].legend()
        report_samples.append(
            {
                "index": sample["index"],
                "sample_token": sample["sample_token"],
                "predicted_xy_raw": sample["predicted_xy"].tolist(),
                "target_xy": target_xy.tolist(),
            }
        )
    figure.tight_layout()
    figure.savefig(args.output_dir / "camera_motion_audit.png", dpi=140)
    plt.close(figure)
    report = {
        "manifest": str(args.manifest),
        "indices": indices,
        "global_scale": global_scale,
        "scale_source": scale_source,
        "keyframe_ade_m": keyframe_ade,
        "translation_direction_cosine": direction_cosine,
        "note": "Scale is diagnostic-only and is never fitted per validation window.",
        "samples": report_samples,
    }
    (args.output_dir / "camera_motion_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if args.calibration is None:
        calibration = {
            "global_scale": global_scale,
            "fit_manifest": str(args.manifest),
            "camera_axes_to_ego_xy": ["camera_z", "-camera_x"],
        }
        (args.output_dir / "camera_motion_calibration.json").write_text(
            json.dumps(calibration, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
