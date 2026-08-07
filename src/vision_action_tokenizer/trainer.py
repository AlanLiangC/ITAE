"""DDP-compatible tokenizer training loop with strict checkpoints."""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from PIL import Image
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from .data.dataset import VGGTOmegaResize
from .distributed import DistributedContext, reduce_metrics
from .losses import TokenizerLoss
from .metrics import trajectory_metrics
from .visualization import render_vggt_evaluation_diagnostic


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def _checkpoint_model_state(model: nn.Module) -> dict[str, Tensor]:
    """Exclude the frozen 1B backbone already identified by path and SHA256."""
    state = model.state_dict()
    feature_extractor = getattr(model, "feature_extractor", None)
    if feature_extractor is not None and getattr(feature_extractor, "freeze", False):
        state = {
            key: value
            for key, value in state.items()
            if not key.startswith("feature_extractor.model.")
        }
    return state


def _load_camera_window(paths_json: str, config: dict[str, Any]) -> Tensor:
    paths = json.loads(paths_json)
    backbone = config["vision_backbone"]
    transform = VGGTOmegaResize(
        image_resolution=int(backbone["image_resolution"]),
        mode=str(backbone["resize_mode"]),
        patch_size=int(backbone["patch_size"]),
    )
    frames = []
    for path in paths:
        with Image.open(path) as image:
            frames.append(transform(image))
    return torch.stack(frames)


class TokenizerTrainer:
    """Train the cached or online VGGT-Omega action tokenizer."""

    def __init__(
        self,
        model: nn.Module,
        loss_module: TokenizerLoss,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        context: DistributedContext,
        output_dir: str | Path,
        precision: str = "bf16",
        grad_clip_norm: float = 1.0,
        config: dict[str, Any] | None = None,
    ) -> None:
        if precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16 or bf16")
        self.model = model
        self.loss_module = loss_module
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.context = context
        self.output_dir = Path(output_dir)
        self.precision = precision
        self.grad_clip_norm = grad_clip_norm
        self.config = config or {}
        self.global_step = 0
        self.best_ade = math.inf
        train_config = self.config.get("train", {})
        self.early_stopping_patience = int(
            train_config.get("early_stopping_patience", 0)
        )
        self.early_stopping_min_delta = float(
            train_config.get("early_stopping_min_delta", 0.0)
        )
        self.epochs_without_improvement = 0
        if self.early_stopping_patience < 0:
            raise ValueError("train.early_stopping_patience must be non-negative")
        if self.early_stopping_min_delta < 0:
            raise ValueError("train.early_stopping_min_delta must be non-negative")
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=precision == "fp16" and context.device.type == "cuda"
        )
        tensorboard_config = self.config.get("tensorboard", {})
        self.eval_visualization_items = int(
            tensorboard_config.get("evaluation_visualization_items", 0)
        )
        self.eval_visualization_every = int(
            tensorboard_config.get("evaluation_visualization_every_epochs", 1)
        )
        self.eval_visualization_include_images = bool(
            tensorboard_config.get("evaluation_visualization_include_images", True)
        )
        self.eval_visualization_distinct_scenes = bool(
            tensorboard_config.get("evaluation_visualization_distinct_scenes", True)
        )
        if self.eval_visualization_items < 0:
            raise ValueError("tensorboard.evaluation_visualization_items must be non-negative")
        if self.eval_visualization_every <= 0:
            raise ValueError(
                "tensorboard.evaluation_visualization_every_epochs must be positive"
            )
        self.writer: SummaryWriter | None = None
        if context.is_main:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "resolved_config.json").write_text(
                json.dumps(self.config, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
            )
            if bool(tensorboard_config.get("enabled", False)):
                configured_log_dir = tensorboard_config.get("log_dir")
                log_dir = (
                    self.output_dir / "tensorboard"
                    if configured_log_dir in (None, "")
                    else Path(configured_log_dir)
                )
                self.writer = SummaryWriter(
                    log_dir=str(log_dir),
                    flush_secs=int(tensorboard_config.get("flush_secs", 30)),
                )

    def fit(
        self,
        train_loader: Any,
        val_loader: Any,
        epochs: int,
        log_every: int = 20,
        start_epoch: int = 0,
    ) -> None:
        try:
            for epoch in range(start_epoch, epochs):
                sampler = getattr(train_loader, "sampler", None)
                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)
                self._train_epoch(train_loader, epoch, log_every)
                metrics = self.evaluate(val_loader, epoch=epoch)
                ade = float(metrics.get("metric/ade_m", torch.tensor(math.inf)))
                is_best = ade < self.best_ade - self.early_stopping_min_delta
                if is_best:
                    self.best_ade = ade
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1
                if self.context.is_main:
                    self._print_metrics(f"val epoch={epoch}", metrics)
                    self._write_scalars("validation", metrics, self.global_step)
                    if self.writer is not None:
                        self.writer.add_scalar("progress/epoch", epoch, self.global_step)
                    self.save_checkpoint("last.pt", epoch)
                    if is_best:
                        self.save_checkpoint("best.pt", epoch)
                should_stop = (
                    self.early_stopping_patience > 0
                    and self.epochs_without_improvement >= self.early_stopping_patience
                )
                if self.context.world_size > 1:
                    stop_tensor = torch.tensor(
                        int(should_stop), device=self.context.device, dtype=torch.int32
                    )
                    dist.broadcast(stop_tensor, src=0)
                    should_stop = bool(stop_tensor.item())
                if should_stop:
                    if self.context.is_main:
                        print(
                            "Early stopping: validation ADE did not improve by "
                            f"{self.early_stopping_min_delta:g} m for "
                            f"{self.epochs_without_improvement} epochs; "
                            f"best={self.best_ade:.5f} m",
                            flush=True,
                        )
                    break
        finally:
            if self.writer is not None:
                self.writer.flush()
                self.writer.close()

    def _autocast(self) -> Any:
        dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
        enabled = self.precision != "fp32" and self.context.device.type == "cuda"
        return torch.autocast(self.context.device.type, dtype=dtype, enabled=enabled)

    def _train_epoch(self, loader: Any, epoch: int, log_every: int) -> None:
        self.model.train()
        running: dict[str, float] = defaultdict(float)
        for batch_index, raw_batch in enumerate(loader):
            batch = _to_device(raw_batch, self.context.device)
            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                model_inputs = {
                    "frame_times": batch["frame_times"],
                    "future_times": batch["future_times"],
                }
                if "camera_hidden" in batch:
                    model_inputs["camera_hidden"] = batch["camera_hidden"]
                    model_inputs["register_hidden_mean"] = batch["register_hidden_mean"]
                    if "register_hidden" in batch:
                        model_inputs["register_hidden"] = batch["register_hidden"]
                else:
                    model_inputs["images"] = batch["images"]
                output = self.model(**model_inputs)
                loss, terms = self.loss_module(
                    output,
                    batch["trajectory"],
                    batch["future_times"],
                    batch["trajectory_mask"],
                    self.global_step,
                )
            if not torch.isfinite(loss):
                tokens = batch.get("sample_token", ["unknown"])
                raise FloatingPointError(f"Non-finite loss at samples {tokens}")
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()
            self.global_step += 1
            for key, value in terms.items():
                running[key] += float(value)
            if self.context.is_main:
                step_metrics: dict[str, Any] = dict(terms)
                step_metrics["optimization/lr"] = self.optimizer.param_groups[0]["lr"]
                step_metrics["optimization/grad_norm"] = grad_norm.detach()
                self._write_scalars("train", step_metrics, self.global_step)
            if self.context.is_main and (batch_index + 1) % log_every == 0:
                averaged = {key: value / log_every for key, value in running.items()}
                averaged["lr"] = self.optimizer.param_groups[0]["lr"]
                self._print_metrics(f"train epoch={epoch} step={self.global_step}", averaged)
                running.clear()

    @torch.no_grad()
    def evaluate(self, loader: Any, epoch: int | None = None) -> dict[str, Tensor]:
        self.model.eval()
        totals: dict[str, Tensor] = {}
        batches = 0
        visualization_count = 0
        visualized_scenes: set[str] = set()
        render_visualizations = (
            self.writer is not None
            and self.eval_visualization_items > 0
            and (epoch is None or epoch % self.eval_visualization_every == 0)
        )
        for raw_batch in loader:
            batch = _to_device(raw_batch, self.context.device)
            with self._autocast():
                model_inputs = {
                    "frame_times": batch["frame_times"],
                    "future_times": batch["future_times"],
                }
                if "camera_hidden" in batch:
                    model_inputs["camera_hidden"] = batch["camera_hidden"]
                    model_inputs["register_hidden_mean"] = batch["register_hidden_mean"]
                    if "register_hidden" in batch:
                        model_inputs["register_hidden"] = batch["register_hidden"]
                else:
                    model_inputs["images"] = batch["images"]
                output = self.model(**model_inputs)
                _, terms = self.loss_module(
                    output,
                    batch["trajectory"],
                    batch["future_times"],
                    batch["trajectory_mask"],
                    self.global_step,
                )
                terms.update(
                    trajectory_metrics(
                        output.reconstruction,
                        batch["trajectory"],
                        batch["future_times"],
                        batch["trajectory_mask"],
                        steps_per_token=int(
                            self.config["action_tokenizer"]["steps_per_token"]
                        ),
                    )
                )
            if render_visualizations and visualization_count < self.eval_visualization_items:
                batch_size = batch["trajectory"].shape[0]
                sample_tokens = batch.get("sample_token", [""] * batch_size)
                scene_tokens = batch.get("scene_token", [""] * batch_size)
                for item_index in range(batch_size):
                    if visualization_count >= self.eval_visualization_items:
                        break
                    scene_token = str(scene_tokens[item_index])
                    if (
                        self.eval_visualization_distinct_scenes
                        and scene_token in visualized_scenes
                    ):
                        continue
                    camera_images = None
                    if self.eval_visualization_include_images:
                        if "images" in batch:
                            camera_images = batch["images"][item_index]
                        else:
                            camera_images = _load_camera_window(
                                batch["image_paths_json"][item_index], self.config
                            )
                    image = render_vggt_evaluation_diagnostic(
                        batch["trajectory"][item_index],
                        output.reconstruction[item_index],
                        output.predicted_increments[item_index],
                        batch["future_times"][item_index],
                        camera_images=camera_images,
                        frame_times=batch["frame_times"][item_index],
                        sample_token=str(sample_tokens[item_index]),
                        mask=batch["trajectory_mask"][item_index],
                    )
                    assert self.writer is not None
                    self.writer.add_image(
                        f"evaluation/vggt_diagnostic_2x2/item_{visualization_count:03d}",
                        image,
                        self.global_step,
                    )
                    visualized_scenes.add(scene_token)
                    visualization_count += 1
            for key, value in terms.items():
                totals[key] = totals.get(key, torch.zeros_like(value)) + value
            batches += 1
        if batches == 0:
            raise ValueError("Validation loader is empty")
        averaged = {key: value / batches for key, value in totals.items()}
        return reduce_metrics(averaged, self.context.world_size)

    def save_checkpoint(self, filename: str, epoch: int) -> None:
        """Atomically save enough state for a strict training resume."""
        model = _unwrap(self.model)
        payload = {
            "model": _checkpoint_model_state(model),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": None if self.scheduler is None else self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": epoch,
            "global_step": self.global_step,
            "best_ade": self.best_ade,
            "epochs_without_improvement": self.epochs_without_improvement,
            "config": self.config,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        destination = self.output_dir / filename
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, destination)

    def load_checkpoint(self, path: str | Path, weights_only: bool = False) -> int:
        checkpoint = torch.load(path, map_location=self.context.device, weights_only=False)
        model = _unwrap(self.model)
        incompatible = model.load_state_dict(checkpoint["model"], strict=False)
        invalid_missing = [
            key
            for key in incompatible.missing_keys
            if not key.startswith("feature_extractor.model.")
        ]
        if invalid_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "Checkpoint model mismatch: "
                f"missing={invalid_missing}, unexpected={incompatible.unexpected_keys}"
            )
        if weights_only:
            return 0
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scheduler is not None and checkpoint["scheduler"] is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.scaler.load_state_dict(checkpoint["scaler"])
        self.global_step = int(checkpoint["global_step"])
        self.best_ade = float(checkpoint["best_ade"])
        self.epochs_without_improvement = int(
            checkpoint.get("epochs_without_improvement", 0)
        )
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if torch.cuda.is_available() and checkpoint["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_state"]])
        return int(checkpoint["epoch"]) + 1

    @staticmethod
    def _print_metrics(prefix: str, metrics: dict[str, Any]) -> None:
        rendered = " ".join(f"{key}={float(value):.5f}" for key, value in sorted(metrics.items()))
        print(f"{prefix} {rendered}", flush=True)

    def _write_scalars(self, prefix: str, metrics: dict[str, Any], step: int) -> None:
        if self.writer is None:
            return
        for key, value in sorted(metrics.items()):
            self.writer.add_scalar(f"{prefix}/{key}", float(value), step)
