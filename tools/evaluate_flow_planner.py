#!/usr/bin/env python3
"""Evaluate a raw/token flow planner with exact five-NFE sampling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from vision_action_tokenizer.config import load_config, stable_hash
from vision_action_tokenizer.data.planner_dataset import (
    PlannerDataset,
    PlannerTargetNormalizer,
    file_sha256,
)
from vision_action_tokenizer.models.factory import build_tokenizer, tokenizer_state_from_checkpoint
from vision_action_tokenizer.models.flow_planner import build_flow_planner
from vision_action_tokenizer.planner_evaluator import PlannerOutputDecoder, evaluate_planner
from vision_action_tokenizer.visualization import render_planner_diagnostic


def _cache(config: dict, section: str, split: str) -> str | None:
    value = config[section].get("cache")
    return value.get(split) if isinstance(value, dict) else value


def _decoder(
    config: dict, normalizer: PlannerTargetNormalizer, device: torch.device
) -> PlannerOutputDecoder:
    target_type = str(config["planner"]["target"])
    tokenizer = None
    if target_type == "v4_action_token":
        teacher = config["action_targets"]["teacher"]
        checkpoint_path = Path(teacher["checkpoint"])
        if teacher.get("checkpoint_sha256") not in {
            None,
            file_sha256(checkpoint_path),
        }:
            raise ValueError("V4 tokenizer checkpoint SHA256 mismatch")
        teacher_config = load_config(teacher["config"])
        tokenizer = build_tokenizer(teacher_config).to(device).eval()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        tokenizer.load_state_dict(tokenizer_state_from_checkpoint(checkpoint), strict=True)
        tokenizer.requires_grad_(False)
    return PlannerOutputDecoder(target_type, normalizer, tokenizer).to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", choices=["validation", "final_eval"], default="final_eval")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--vision-cache", type=Path)
    parser.add_argument("--action-targets", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.vision_cache is not None:
        config["vision_condition"]["cache"][args.split] = str(args.vision_cache)
    if args.action_targets is not None:
        config["action_targets"]["cache"][args.split] = str(args.action_targets)
    vision = config["vision_condition"]
    target_type = str(config["planner"]["target"])
    dataset = PlannerDataset(
        args.manifest,
        target_type=target_type,
        vision_cache=_cache(config, "vision_condition", args.split),
        action_target_cache=(
            _cache(config, "action_targets", args.split)
            if target_type == "v4_action_token"
            else None
        ),
        expected_vision_metadata={
            "backbone_type": vision["type"],
            "model_name": vision.get("model_name", vision["type"]),
            "checkpoint_sha256": vision.get("checkpoint_sha256"),
        },
    )
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be positive")
        dataset = torch.utils.data.Subset(dataset, range(min(args.max_samples, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=int(config["evaluation"].get("batch_size", 64)),
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 4)),
        pin_memory=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    target_shape = tuple(map(int, config["planner"]["target_shape"]))
    base_dataset = dataset.dataset if isinstance(dataset, torch.utils.data.Subset) else dataset
    condition_shape = tuple(map(int, base_dataset.vision_cache.index["condition_shape"]))
    model = build_flow_planner(config, condition_shape).to(device).eval()
    state_key = "model" if args.no_ema else "ema_model"
    model.load_state_dict(checkpoint[state_key], strict=True)
    normalizer = PlannerTargetNormalizer(
        target_shape, epsilon=float(config["planner"].get("normalizer_epsilon", 1e-4))
    ).to(device)
    normalizer.load_state_dict(checkpoint["normalizer"], strict=True)
    decoder = _decoder(config, normalizer, device)
    evaluation = evaluate_planner(
        model,
        loader,
        decoder,
        device,
        target_shape,
        inference_steps=int(config["evaluation"].get("inference_steps", 5)),
        expected_nfe=int(config["evaluation"].get("expected_nfe", 5)),
        seed=int(config["evaluation"].get("noise_seed", 12345)),
        keep_predictions=True,
    )
    assert evaluation.predictions is not None
    assert evaluation.targets is not None
    assert evaluation.future_times is not None
    assert evaluation.sample_tokens is not None
    position_error = torch.linalg.vector_norm(
        evaluation.predictions[..., :2] - evaluation.targets[..., :2], dim=-1
    )
    per_sample_ade = position_error.mean(dim=1)
    per_sample_fde = position_error[:, -1]
    endpoint = evaluation.targets[:, -1]
    distance = torch.linalg.vector_norm(endpoint[:, :2], dim=-1)
    stationary = distance < 2.0
    turn = (~stationary) & ((endpoint[:, 1].abs() > 2.0) | (endpoint[:, 2].abs() > 0.15))
    straight_slow = (~stationary) & (~turn) & (distance < 20.0)
    straight_fast = (~stationary) & (~turn) & (distance >= 20.0)
    bucket_metrics: dict[str, dict[str, float]] = {}
    for name, mask in {
        "stationary": stationary,
        "turn": turn,
        "straight_slow": straight_slow,
        "straight_fast": straight_fast,
    }.items():
        bucket_metrics[name] = {
            "count": int(mask.sum()),
            "ade_m": float(per_sample_ade[mask].mean()) if bool(mask.any()) else float("nan"),
            "fde_m": float(per_sample_fde[mask].mean()) if bool(mask.any()) else float("nan"),
        }
    base_windows = base_dataset.windows
    if isinstance(dataset, torch.utils.data.Subset):
        base_windows = [base_windows[index] for index in dataset.indices]
    scene_tokens = [window.scene_token for window in base_windows]

    args.output.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output / "predictions.safetensors"
    save_file(
        {
            "prediction": evaluation.predictions.contiguous(),
            "target": evaluation.targets.contiguous(),
            "future_times": evaluation.future_times.contiguous(),
            "ade_per_sample": per_sample_ade.contiguous(),
            "fde_per_sample": per_sample_fde.contiguous(),
        },
        str(predictions_path),
        metadata={"target_type": target_type, "nfe": "5"},
    )
    report = {
        "target_type": target_type,
        "config": str(args.config),
        "manifest": str(args.manifest),
        "manifest_sha256": file_sha256(args.manifest),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "state": state_key,
        "config_hash": stable_hash(config),
        "vision_condition_hash": stable_hash(config["vision_condition"]),
        "planner_core_hash": stable_hash(config["planner"]["model"]),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "condition_shape": list(condition_shape),
        "target_shape": list(target_shape),
        "sample_tokens": evaluation.sample_tokens,
        "scene_tokens": scene_tokens,
        "metrics": evaluation.metrics,
        "motion_buckets": bucket_metrics,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    writer = SummaryWriter(str(args.output / "tensorboard"))
    for key, value in evaluation.metrics.items():
        writer.add_scalar(f"evaluation/{key}", value, 0)
    item_count = min(
        int(config.get("tensorboard", {}).get("evaluation_visualization_items", 8)),
        len(evaluation.predictions),
    )
    for index in range(item_count):
        with Image.open(base_windows[index].image_paths[0]) as image:
            rgb = torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.uint8).copy())
            rgb = rgb.permute(2, 0, 1).float() / 255.0
        diagnostic = render_planner_diagnostic(
            rgb,
            evaluation.targets[index],
            evaluation.predictions[index],
            evaluation.future_times[index],
            evaluation.sample_tokens[index],
        )
        writer.add_image(f"evaluation/items/{index:02d}", diagnostic, 0)
    writer.close()
    print(json.dumps(evaluation.metrics, indent=2))
    print(f"Saved evaluation to {args.output}")


if __name__ == "__main__":
    main()
