#!/usr/bin/env python3
"""Train a current-frame conditional rectified-flow planner."""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torch.utils.tensorboard import SummaryWriter

from vision_action_tokenizer.config import (
    load_config,
    resolve_resume_checkpoint,
    seed_everything,
    stable_hash,
)
from vision_action_tokenizer.data.manifest import manifest_scene_tokens
from vision_action_tokenizer.data.planner_dataset import (
    PlannerDataset,
    PlannerTargetNormalizer,
    file_sha256,
)
from vision_action_tokenizer.distributed import cleanup_distributed, initialize_distributed
from vision_action_tokenizer.flow_matching import flow_matching_loss
from vision_action_tokenizer.models.factory import build_tokenizer, tokenizer_state_from_checkpoint
from vision_action_tokenizer.models.flow_planner import build_flow_planner
from vision_action_tokenizer.planner_evaluator import (
    PlannerOutputDecoder,
    evaluate_planner,
    planner_slot_times,
)
from vision_action_tokenizer.visualization import render_planner_diagnostic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--val-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--overfit-samples", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-every-steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--train-vision-cache", type=Path)
    parser.add_argument("--val-vision-cache", type=Path)
    parser.add_argument("--train-action-targets", type=Path)
    parser.add_argument("--val-action-targets", type=Path)
    return parser.parse_args()


def cache_for_split(config: dict[str, Any], section: str, split: str) -> str | None:
    value = config[section].get("cache")
    if isinstance(value, dict):
        return value.get(split)
    return value


def make_dataset(
    config: dict[str, Any], manifest: Path, split: str
) -> PlannerDataset:
    vision = config["vision_condition"]
    if str(vision.get("mode", "cached")) != "cached":
        raise NotImplementedError(
            "Formal planner training currently requires mode=cached; use the cache tool "
            "for any registered backbone"
        )
    target_type = str(config["planner"]["target"])
    ego = config.get("ego_motion_condition", {})
    ego_enabled = bool(ego.get("enabled", False))
    return PlannerDataset(
        manifest,
        target_type=target_type,
        vision_cache=cache_for_split(config, "vision_condition", split),
        action_target_cache=(
            cache_for_split(config, "action_targets", split)
            if target_type == "v4_action_token"
            else None
        ),
        expected_vision_metadata={
            "backbone_type": vision["type"],
            "model_name": vision.get("model_name", vision["type"]),
            "checkpoint_sha256": vision.get("checkpoint_sha256"),
            "frame_indices": vision.get("frame_indices"),
            "condition_frame_offsets_s": vision.get("frame_offsets_s"),
            "current_frame_index": vision.get("current_frame_index"),
            "ego_motion_shape": (
                [int(ego["num_tokens"]), int(ego["state_dim"])]
                if ego_enabled
                else None
            ),
            "ego_motion_state_fields": ego.get("state_fields") if ego_enabled else None,
            "ego_motion_scales": ego.get("scales") if ego_enabled else None,
        },
    )


def build_output_decoder(
    config: dict[str, Any], normalizer: PlannerTargetNormalizer, device: torch.device
) -> PlannerOutputDecoder:
    target_type = str(config["planner"]["target"])
    tokenizer = None
    if target_type == "v4_action_token":
        teacher = config["action_targets"]["teacher"]
        tokenizer_config_path = Path(teacher["config"])
        checkpoint_path = Path(teacher["checkpoint"])
        expected_sha = teacher.get("checkpoint_sha256")
        if expected_sha is not None and file_sha256(checkpoint_path) != expected_sha:
            raise ValueError("Configured V4 tokenizer checkpoint SHA256 mismatch")
        tokenizer_config = load_config(tokenizer_config_path)
        tokenizer = build_tokenizer(tokenizer_config).to(device).eval()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        tokenizer.load_state_dict(tokenizer_state_from_checkpoint(checkpoint), strict=True)
        tokenizer.requires_grad_(False)
    return PlannerOutputDecoder(target_type, normalizer, tokenizer).to(device)


@torch.no_grad()
def update_ema(ema: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    ema_state = ema.state_dict()
    model_state = model.state_dict()
    for key, ema_value in ema_state.items():
        source = model_state[key].detach()
        if torch.is_floating_point(ema_value):
            ema_value.mul_(decay).add_(source, alpha=1.0 - decay)
        else:
            ema_value.copy_(source)


def main() -> None:
    args = parse_args()
    context = initialize_distributed()
    try:
        config = load_config(args.config)
        if args.seed is not None:
            config["seed"] = args.seed
        if args.train_vision_cache is not None:
            config["vision_condition"]["cache"]["train"] = str(
                args.train_vision_cache
            )
        if args.val_vision_cache is not None:
            config["vision_condition"]["cache"]["validation"] = str(
                args.val_vision_cache
            )
        if args.train_action_targets is not None:
            config["action_targets"]["cache"]["train"] = str(
                args.train_action_targets
            )
        if args.val_action_targets is not None:
            config["action_targets"]["cache"]["validation"] = str(
                args.val_action_targets
            )
        if args.batch_size is not None:
            config["train"]["batch_size"] = args.batch_size
            config["evaluation"]["batch_size"] = args.batch_size
        if args.eval_every_steps is not None:
            config["train"]["eval_every_steps"] = args.eval_every_steps
        seed = int(config.get("seed", 42))
        seed_everything(seed + context.rank)
        train_scenes = manifest_scene_tokens(args.train_manifest)
        val_scenes = manifest_scene_tokens(args.val_manifest)
        overlap = train_scenes & val_scenes
        if overlap and args.overfit_samples is None:
            raise ValueError(f"Planner train/validation share {len(overlap)} scenes")
        if args.overfit_samples is not None and args.overfit_samples <= 0:
            raise ValueError("--overfit-samples must be positive")

        train_base = make_dataset(config, args.train_manifest, "train")
        val_base = make_dataset(config, args.val_manifest, "validation")
        train_dataset: Any = train_base
        val_dataset: Any = val_base
        fit_targets = train_base.all_targets()
        if args.overfit_samples is not None:
            count = min(args.overfit_samples, len(train_base))
            indices = list(range(count))
            train_dataset = Subset(train_base, indices)
            val_dataset = Subset(train_base, indices)
            fit_targets = fit_targets[:count]
        target_shape = tuple(map(int, config["planner"]["target_shape"]))
        if tuple(fit_targets.shape[1:]) != target_shape:
            raise ValueError(
                f"Configured target shape {target_shape} != data {tuple(fit_targets.shape[1:])}"
            )
        normalizer = PlannerTargetNormalizer(
            target_shape, epsilon=float(config["planner"].get("normalizer_epsilon", 1e-4))
        ).to(context.device)
        normalizer.fit(fit_targets.to(context.device))

        vision_index = train_base.vision_cache.index  # type: ignore[union-attr]
        condition_shape = tuple(map(int, vision_index["condition_shape"]))
        model = build_flow_planner(config, condition_shape).to(context.device)
        ema_model = copy.deepcopy(model).eval().requires_grad_(False)
        train_model: torch.nn.Module = model
        if context.world_size > 1:
            train_model = DistributedDataParallel(
                model,
                device_ids=[context.local_rank] if context.device.type == "cuda" else None,
            )
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if context.is_main:
            print(
                f"Planner target={config['planner']['target']} target_shape={target_shape} "
                f"condition_shape={condition_shape} parameters={parameter_count:,}",
                flush=True,
            )

        train_config = config["train"]
        batch_size = int(train_config["batch_size"])
        train_sampler = (
            DistributedSampler(
                train_dataset, context.world_size, context.rank, shuffle=True, seed=seed
            )
            if context.world_size > 1
            else None
        )
        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=int(config["data"].get("num_workers", 4)),
            pin_memory=True,
            drop_last=True,
            persistent_workers=int(config["data"].get("num_workers", 4)) > 0,
            generator=generator if train_sampler is None else None,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(config["evaluation"].get("batch_size", batch_size)),
            shuffle=False,
            num_workers=int(config["data"].get("num_workers", 4)),
            pin_memory=True,
            persistent_workers=int(config["data"].get("num_workers", 4)) > 0,
        )
        if len(train_loader) == 0:
            raise ValueError("Training loader is empty; reduce batch size")

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(train_config["learning_rate"]),
            betas=tuple(map(float, train_config.get("betas", [0.9, 0.95]))),
            weight_decay=float(train_config.get("weight_decay", 0.01)),
        )
        max_steps = int(args.max_steps or train_config["max_steps"])
        warmup_steps = int(train_config.get("warmup_steps", 1000))

        def lr_factor(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
            minimum = float(train_config.get("min_lr_ratio", 0.05))
            return minimum + 0.5 * (1.0 - minimum) * (
                1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0))
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
        precision = str(train_config.get("precision", "bf16"))
        amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        use_amp = context.device.type == "cuda" and precision in {"bf16", "fp16"}
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp and precision == "fp16")
        decoder = build_output_decoder(config, normalizer, context.device)
        noise_generator = torch.Generator(device=context.device).manual_seed(
            seed + 10_000 + context.rank
        )
        time_generator = torch.Generator(device=context.device).manual_seed(
            seed + 20_000 + context.rank
        )

        args.output.mkdir(parents=True, exist_ok=True)
        resolved_config = copy.deepcopy(config)
        resolved_config["runtime"] = {
            "train_manifest": str(args.train_manifest),
            "train_manifest_sha256": file_sha256(args.train_manifest),
            "val_manifest": str(args.val_manifest),
            "val_manifest_sha256": file_sha256(args.val_manifest),
            "parameter_count": parameter_count,
            "condition_shape": list(condition_shape),
            "target_shape": list(target_shape),
            "config_hash": stable_hash(config),
        }
        if context.is_main:
            (args.output / "resolved_config.json").write_text(
                json.dumps(resolved_config, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        writer = (
            SummaryWriter(
                log_dir=str(args.output / "tensorboard"),
                flush_secs=int(config.get("tensorboard", {}).get("flush_secs", 30)),
            )
            if context.is_main and bool(config.get("tensorboard", {}).get("enabled", True))
            else None
        )
        history_path = args.output / "training_history.jsonl"

        start_step = 0
        best_ade = float("inf")
        checkpoint_path = resolve_resume_checkpoint(
            config, args.output, cli_resume=args.resume, no_resume=args.no_resume
        )
        if checkpoint_path is not None:
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            model.load_state_dict(checkpoint["model"], strict=True)
            ema_model.load_state_dict(checkpoint["ema_model"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            normalizer.load_state_dict(checkpoint["normalizer"], strict=True)
            if "noise_generator_state" in checkpoint:
                noise_generator.set_state(checkpoint["noise_generator_state"])
            if "time_generator_state" in checkpoint:
                time_generator.set_state(checkpoint["time_generator_state"])
            start_step = int(checkpoint["global_step"])
            best_ade = float(checkpoint.get("best_ade", best_ade))
            if context.is_main:
                print(f"Resumed {checkpoint_path} at step {start_step}", flush=True)

        def save_checkpoint(path: Path, step: int) -> None:
            torch.save(
                {
                    "model": model.state_dict(),
                    "ema_model": ema_model.state_dict(),
                    "normalizer": normalizer.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "global_step": step,
                    "best_ade": best_ade,
                    "config": resolved_config,
                    "noise_generator_state": noise_generator.get_state(),
                    "time_generator_state": time_generator.get_state(),
                },
                path,
            )

        log_every = int(train_config.get("log_every", 20))
        eval_every = int(train_config.get("eval_every_steps", 500))
        save_every = int(train_config.get("save_every_steps", eval_every))
        ema_decay = float(train_config.get("ema_decay", 0.999))
        grad_clip = float(train_config.get("grad_clip_norm", 1.0))
        evaluation_config = config["evaluation"]
        eval_max_batches = evaluation_config.get("max_validation_batches")
        visualization_every = int(
            config.get("tensorboard", {}).get(
                "evaluation_visualization_every_steps", eval_every
            )
        )
        global_step = start_step
        epoch = 0
        started = time.time()
        optimizer.zero_grad(set_to_none=True)
        while global_step < max_steps:
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            for batch in train_loader:
                if global_step >= max_steps:
                    break
                train_model.train()
                target = normalizer.normalize(batch["target"].to(context.device).float())
                condition = batch["condition_tokens"].to(context.device).float()
                condition_mask = batch["condition_mask"].to(context.device)
                condition_times = batch.get("condition_times")
                if condition_times is not None:
                    condition_times = condition_times.to(context.device).float()
                ego_motion = batch.get("ego_motion")
                ego_motion_times = batch.get("ego_motion_times")
                if ego_motion is not None:
                    assert ego_motion_times is not None
                    ego_motion = ego_motion.to(context.device).float()
                    ego_motion_times = ego_motion_times.to(context.device).float()
                future_times = batch["future_times"].to(context.device).float()
                slot_times = planner_slot_times(
                    str(config["planner"]["target"]), future_times, target_shape[0]
                )
                with torch.autocast(
                    device_type=context.device.type,
                    dtype=amp_dtype,
                    enabled=use_amp,
                ):
                    loss, train_metrics = flow_matching_loss(
                        train_model,
                        target,
                        condition,
                        condition_mask,
                        slot_times,
                        condition_times=condition_times,
                        ego_motion=ego_motion,
                        ego_motion_times=ego_motion_times,
                        generator=noise_generator,
                        time_generator=time_generator,
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                update_ema(ema_model, model, ema_decay)
                global_step += 1

                if context.is_main and global_step % log_every == 0:
                    elapsed = time.time() - started
                    values = {
                        **{key: float(value) for key, value in train_metrics.items()},
                        "train/grad_norm": float(grad_norm),
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/seen_samples": global_step * batch_size * context.world_size,
                        "train/steps_per_second": global_step / max(elapsed, 1e-6),
                    }
                    print(
                        f"step={global_step}/{max_steps} loss={values['flow/loss']:.6f} "
                        f"lr={values['train/lr']:.2e} grad={values['train/grad_norm']:.3f}",
                        flush=True,
                    )
                    if writer is not None:
                        for key, value in values.items():
                            writer.add_scalar(key, value, global_step)

                should_evaluate = global_step % eval_every == 0 or global_step == max_steps
                if should_evaluate:
                    if context.world_size > 1:
                        dist.barrier()
                    if context.is_main:
                        keep_visualizations = (
                            writer is not None and global_step % visualization_every == 0
                        )
                        evaluation = evaluate_planner(
                            ema_model,
                            val_loader,
                            decoder,
                            context.device,
                            target_shape,
                            inference_steps=int(evaluation_config.get("inference_steps", 5)),
                            expected_nfe=int(evaluation_config.get("expected_nfe", 5)),
                            seed=int(evaluation_config.get("noise_seed", 12345)),
                            keep_predictions=keep_visualizations,
                            max_batches=(
                                None if eval_max_batches is None else int(eval_max_batches)
                            ),
                        )
                        ade = evaluation.metrics["metric/ade_m"]
                        record = {
                            "global_step": global_step,
                            "seen_samples": global_step * batch_size * context.world_size,
                            **evaluation.metrics,
                        }
                        with history_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(record) + "\n")
                        print(
                            f"validation step={global_step} ADE={ade:.6f} "
                            f"FDE={evaluation.metrics['metric/fde_m']:.6f} "
                            f"NFE={evaluation.metrics['sampler/nfe']:.0f}",
                            flush=True,
                        )
                        if writer is not None:
                            for key, value in evaluation.metrics.items():
                                writer.add_scalar(f"validation/{key}", value, global_step)
                        if keep_visualizations:
                            assert evaluation.predictions is not None
                            assert evaluation.targets is not None
                            assert evaluation.future_times is not None
                            assert evaluation.sample_tokens is not None
                            if isinstance(val_dataset, Subset):
                                visualization_windows = [
                                    val_dataset.dataset.windows[index]
                                    for index in val_dataset.indices
                                ]
                            else:
                                visualization_windows = val_dataset.windows
                            item_count = min(
                                int(
                                    config.get("tensorboard", {}).get(
                                        "evaluation_visualization_items", 8
                                    )
                                ),
                                len(evaluation.predictions),
                            )
                            for item_index in range(item_count):
                                window = visualization_windows[item_index]
                                frame_indices = list(
                                    map(
                                        int,
                                        config["vision_condition"].get(
                                            "frame_indices", [0]
                                        ),
                                    )
                                )
                                rgb_frames = []
                                for frame_index in frame_indices:
                                    with Image.open(
                                        window.image_paths[frame_index]
                                    ) as image:
                                        rgb = torch.from_numpy(
                                            np.asarray(
                                                image.convert("RGB"), dtype=np.uint8
                                            ).copy()
                                        )
                                        rgb_frames.append(
                                            rgb.permute(2, 0, 1).float() / 255.0
                                        )
                                diagnostic = render_planner_diagnostic(
                                    torch.stack(rgb_frames),
                                    evaluation.targets[item_index],
                                    evaluation.predictions[item_index],
                                    evaluation.future_times[item_index],
                                    evaluation.sample_tokens[item_index],
                                    frame_times=torch.tensor(
                                        [
                                            window.frame_times_s[index]
                                            for index in frame_indices
                                        ]
                                    ),
                                    ego_motion=(
                                        None
                                        if window.ego_motion_states is None
                                        else torch.tensor(window.ego_motion_states)
                                    ),
                                )
                                writer.add_image(
                                    f"validation/items/{item_index:02d}",
                                    diagnostic,
                                    global_step,
                                )
                        if ade < best_ade:
                            best_ade = ade
                            save_checkpoint(args.output / "best.pt", global_step)
                    if context.world_size > 1:
                        dist.barrier()
                if context.is_main and (
                    global_step % save_every == 0 or global_step == max_steps
                ):
                    save_checkpoint(args.output / "last.pt", global_step)
            epoch += 1
        if writer is not None:
            writer.close()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
