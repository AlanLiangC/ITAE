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
from .distributed import DistributedContext
from .losses import TokenizerLoss
from .metrics import trajectory_metrics
from .models.tokenizer import TokenizerOutput
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


def _validate_multi_source_resume_config(
    current: dict[str, Any], saved: dict[str, Any]
) -> None:
    """Reject structurally compatible checkpoints from a different joint run."""
    if not current.get("data", {}).get("sources"):
        return
    sections = {
        "action_tokenizer": (
            current.get("action_tokenizer"),
            saved.get("action_tokenizer"),
        ),
        "loss": (current.get("loss"), saved.get("loss")),
        "data.sources": (
            current.get("data", {}).get("sources"),
            saved.get("data", {}).get("sources"),
        ),
        "data.sampling": (
            current.get("data", {}).get("sampling"),
            saved.get("data", {}).get("sampling"),
        ),
        "data_runtime.sources": (
            current.get("data_runtime", {}).get("sources"),
            saved.get("data_runtime", {}).get("sources"),
        ),
    }
    mismatches = [name for name, (left, right) in sections.items() if left != right]
    current_backbone = current.get("vision_backbone", {})
    saved_backbone = saved.get("vision_backbone", {})
    for key in ("checkpoint_sha256", "cache_token_mode", "feature_dim"):
        if current_backbone.get(key) != saved_backbone.get(key):
            mismatches.append(f"vision_backbone.{key}")
    if mismatches:
        raise ValueError(
            "Joint resume checkpoint provenance/config mismatch: "
            + ", ".join(mismatches)
        )


def is_visual_residual_parameter(name: str) -> bool:
    """Identify both legacy V3 and output-side V4 adapter parameters."""
    return "register_residual_" in name or "visual_residual_" in name


def _reduce_sample_weighted_metrics(
    weighted_totals: dict[str, Tensor],
    sample_count: int,
    world_size: int,
) -> dict[str, Tensor]:
    """Reduce metric sums and divide once by the global number of samples."""
    if sample_count <= 0:
        raise ValueError("Validation loader is empty")
    if not weighted_totals:
        return {}
    device = next(iter(weighted_totals.values())).device
    count = torch.tensor(float(sample_count), device=device)
    if world_size > 1:
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        for value in weighted_totals.values():
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return {key: value / count for key, value in weighted_totals.items()}


def _model_diagnostics(model: nn.Module) -> dict[str, Tensor]:
    unwrapped = _unwrap(model)
    tokenizer = getattr(unwrapped, "tokenizer", unwrapped)
    encoder = getattr(tokenizer, "encoder", None)
    gate = getattr(encoder, "register_residual_gate", None)
    if not isinstance(gate, Tensor):
        return {}
    activated = torch.tanh(gate.detach())
    return {
        "register/gate_abs_mean": activated.abs().mean(),
        "register/gate_abs_max": activated.abs().max(),
    }


def _slice_output(output: TokenizerOutput, selection: Tensor) -> TokenizerOutput:
    return TokenizerOutput(
        **{
            name: None if value is None else value[selection]
            for name, value in vars(output).items()
        }
    )


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


def _trajectory_sampling_labels(
    trajectory: Tensor,
    future_times: Tensor,
    *,
    steps_per_interval: int,
    speed_trend_threshold_mps: float,
) -> tuple[str, str]:
    """Classify one sampled trajectory for source-sampler diagnostics."""
    endpoint = trajectory[-1]
    distance = float(torch.linalg.vector_norm(endpoint[:2]))
    if distance < 2.0:
        return "stationary", "stationary"
    motion = (
        "turn"
        if abs(float(endpoint[1])) > 2.0 or abs(float(endpoint[2])) > 0.15
        else ("straight_slow" if distance < 20.0 else "straight_fast")
    )
    previous_xy = torch.cat(
        [torch.zeros_like(trajectory[:1, :2]), trajectory[:-1, :2]], dim=0
    )
    delta_t = torch.diff(torch.cat([torch.zeros_like(future_times[:1]), future_times]))
    speed = torch.linalg.vector_norm(trajectory[:, :2] - previous_xy, dim=-1) / delta_t
    change = float(
        speed[-steps_per_interval:].mean() - speed[:steps_per_interval].mean()
    )
    if change > speed_trend_threshold_mps:
        trend = "accelerating"
    elif change < -speed_trend_threshold_mps:
        trend = "decelerating"
    else:
        trend = "steady"
    return motion, trend


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
        self.best_trained_ade = math.inf
        train_config = self.config.get("train", {})
        self.checkpoint_metric = str(
            train_config.get("checkpoint_metric", "metric/ade_m")
        )
        self.validation_precision = str(
            train_config.get("validation_precision", precision)
        )
        if self.validation_precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("train.validation_precision must be fp32, fp16 or bf16")
        self.early_stopping_patience = int(
            train_config.get("early_stopping_patience", 0)
        )
        self.early_stopping_min_delta = float(
            train_config.get("early_stopping_min_delta", 0.0)
        )
        self.epochs_without_improvement = 0
        self.freeze_base_epochs = int(train_config.get("freeze_base_epochs", 0))
        self.freeze_base = bool(train_config.get("freeze_base", False))
        self._initial_requires_grad = {
            name: parameter.requires_grad
            for name, parameter in self.model.named_parameters()
        }
        self._base_frozen: bool | None = None
        if self.early_stopping_patience < 0:
            raise ValueError("train.early_stopping_patience must be non-negative")
        if self.early_stopping_min_delta < 0:
            raise ValueError("train.early_stopping_min_delta must be non-negative")
        if self.freeze_base_epochs < 0:
            raise ValueError("train.freeze_base_epochs must be non-negative")
        if (self.freeze_base or self.freeze_base_epochs) and context.world_size > 1:
            raise ValueError("Staged base freezing currently supports one process")
        if (self.freeze_base or self.freeze_base_epochs) and not any(
            is_visual_residual_parameter(name) for name in self._initial_requires_grad
        ):
            raise ValueError("Base freezing requires a visual residual adapter")
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=precision == "fp16" and context.device.type == "cuda"
        )
        tensorboard_config = self.config.get("tensorboard", {})
        self.eval_visualization_items = int(
            tensorboard_config.get("evaluation_visualization_items", 0)
        )
        self.eval_visualization_items_per_source = int(
            tensorboard_config.get("evaluation_visualization_items_per_source", 0)
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
        if self.eval_visualization_items_per_source < 0:
            raise ValueError(
                "tensorboard.evaluation_visualization_items_per_source must be non-negative"
            )
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
            if (
                self.early_stopping_patience > 0
                and self.epochs_without_improvement >= self.early_stopping_patience
            ):
                if self.context.is_main:
                    print(
                        "Resume checkpoint already satisfies early stopping; "
                        "no additional epoch will be trained",
                        flush=True,
                    )
                return
            for epoch in range(start_epoch, epochs):
                self._configure_trainable_parameters(epoch)
                sampler = getattr(train_loader, "sampler", None)
                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)
                self._train_epoch(train_loader, epoch, log_every)
                metrics = self.evaluate(val_loader, epoch=epoch)
                if self.checkpoint_metric not in metrics:
                    raise KeyError(
                        f"Configured checkpoint metric is missing: {self.checkpoint_metric}"
                    )
                ade = float(metrics[self.checkpoint_metric])
                is_best = ade < self.best_ade - self.early_stopping_min_delta
                if is_best:
                    self.best_ade = ade
                is_best_trained = (
                    ade < self.best_trained_ade - self.early_stopping_min_delta
                )
                if is_best_trained:
                    self.best_trained_ade = ade
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
                    if is_best_trained:
                        self.save_checkpoint("best_trained.pt", epoch)
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
                            f"best_trained={self.best_trained_ade:.5f} m, "
                            f"deployment_best={self.best_ade:.5f} m",
                            flush=True,
                        )
                    break
        finally:
            if self.writer is not None:
                self.writer.flush()
                self.writer.close()

    def _autocast(self, precision: str | None = None) -> Any:
        selected = self.precision if precision is None else precision
        dtype = torch.bfloat16 if selected == "bf16" else torch.float16
        enabled = selected != "fp32" and self.context.device.type == "cuda"
        return torch.autocast(self.context.device.type, dtype=dtype, enabled=enabled)

    def _configure_trainable_parameters(self, epoch: int) -> None:
        freeze_base = self.freeze_base or epoch < self.freeze_base_epochs
        for name, parameter in self.model.named_parameters():
            initially_trainable = self._initial_requires_grad[name]
            is_residual = is_visual_residual_parameter(name)
            parameter.requires_grad_(
                initially_trainable and (not freeze_base or is_residual)
            )
        if freeze_base != self._base_frozen:
            self._base_frozen = freeze_base
            if self.context.is_main:
                state = "frozen" if freeze_base else "trainable"
                print(f"Base tokenizer parameters are now {state} at epoch {epoch}", flush=True)

    @staticmethod
    def _model_inputs(batch: dict[str, Any]) -> dict[str, Tensor]:
        inputs = {
            "frame_times": batch["frame_times"],
            "future_times": batch["future_times"],
        }
        if "camera_hidden" in batch:
            inputs["camera_hidden"] = batch["camera_hidden"]
            inputs["register_hidden_mean"] = batch["register_hidden_mean"]
            if "register_hidden" in batch:
                inputs["register_hidden"] = batch["register_hidden"]
            if "pose_enc" in batch:
                inputs["pose_enc"] = batch["pose_enc"]
        else:
            inputs["images"] = batch["images"]
        return inputs

    def _shuffled_output(
        self,
        batch: dict[str, Any],
        model_inputs: dict[str, Tensor],
    ) -> Any | None:
        if self.loss_module.config.conditional_shuffle_weight <= 0:
            return None
        batch_size = int(batch["trajectory"].shape[0])
        if batch_size < 2:
            return None
        if "register_hidden" not in model_inputs or "pose_enc" not in model_inputs:
            raise ValueError(
                "Conditional shuffle loss requires cached full registers and pose_enc"
            )
        shuffled = dict(model_inputs)
        shuffled["register_hidden"] = model_inputs["register_hidden"].roll(1, dims=0)
        shuffled["pose_enc"] = model_inputs["pose_enc"].roll(1, dims=0)
        return self.model(**shuffled)

    def _train_epoch(self, loader: Any, epoch: int, log_every: int) -> None:
        self.model.train()
        if self._base_frozen:
            # Keep the warm-started motion path deterministic while the new adapter
            # learns its first non-zero correction.
            self.model.eval()
            unwrapped = _unwrap(self.model)
            tokenizer = getattr(unwrapped, "tokenizer", unwrapped)
            for name, module in tokenizer.named_modules():
                if "residual" in name:
                    module.train()
        running: dict[str, float] = defaultdict(float)
        source_seen: dict[str, int] = defaultdict(int)
        motion_seen: dict[str, int] = defaultdict(int)
        trend_seen: dict[str, int] = defaultdict(int)
        total_seen = 0
        for batch_index, raw_batch in enumerate(loader):
            batch = _to_device(raw_batch, self.context.device)
            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                model_inputs = self._model_inputs(batch)
                output = self.model(**model_inputs)
                shuffled_output = self._shuffled_output(batch, model_inputs)
                loss, terms = self.loss_module(
                    output,
                    batch["trajectory"],
                    batch["future_times"],
                    batch["trajectory_mask"],
                    self.global_step,
                    shuffled_output=shuffled_output,
                )
                terms.update(_model_diagnostics(self.model))
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
            for source_name in raw_batch.get("dataset_name", []):
                source_seen[str(source_name)] += 1
                total_seen += 1
            balancing = self.config.get("train", {}).get("motion_balancing", {})
            steps_per_interval = int(self.config["action_tokenizer"]["steps_per_token"])
            trend_threshold = float(balancing.get("speed_trend_threshold_mps", 0.5))
            for trajectory, future_times in zip(
                raw_batch["trajectory"], raw_batch["future_times"], strict=True
            ):
                motion, trend = _trajectory_sampling_labels(
                    trajectory,
                    future_times,
                    steps_per_interval=steps_per_interval,
                    speed_trend_threshold_mps=trend_threshold,
                )
                motion_seen[motion] += 1
                trend_seen[trend] += 1
            for key, value in terms.items():
                running[key] += float(value)
            if self.context.is_main:
                step_metrics: dict[str, Any] = dict(terms)
                step_metrics["optimization/lr"] = self.optimizer.param_groups[0]["lr"]
                for group_index, group in enumerate(self.optimizer.param_groups):
                    group_name = str(group.get("name", group_index))
                    step_metrics[f"optimization/lr_{group_name}"] = group["lr"]
                step_metrics["optimization/grad_norm"] = grad_norm.detach()
                step_metrics["progress/base_frozen"] = float(bool(self._base_frozen))
                for source_name, count in source_seen.items():
                    step_metrics[f"sampling/source_{source_name}_fraction"] = (
                        count / max(total_seen, 1)
                    )
                self._write_scalars("train", step_metrics, self.global_step)
            if self.context.is_main and (batch_index + 1) % log_every == 0:
                averaged = {key: value / log_every for key, value in running.items()}
                averaged["lr"] = self.optimizer.param_groups[0]["lr"]
                self._print_metrics(f"train epoch={epoch} step={self.global_step}", averaged)
                running.clear()

        labels = {
            "source": sorted(self.config.get("data", {}).get("sources", source_seen)),
            "motion": ["stationary", "straight_slow", "straight_fast", "turn"],
            "speed_trend": ["stationary", "steady", "accelerating", "decelerating"],
        }
        observed = {"source": source_seen, "motion": motion_seen, "speed_trend": trend_seen}
        sampling_metrics: dict[str, float] = {}
        for family, names in labels.items():
            counts = torch.tensor(
                [observed[family][name] for name in names],
                dtype=torch.float64,
                device=self.context.device,
            )
            if self.context.world_size > 1:
                dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            denominator = float(counts.sum().clamp_min(1.0))
            for name, count in zip(names, counts.tolist(), strict=True):
                sampling_metrics[f"sampling/actual_{family}_{name}_fraction"] = (
                    count / denominator
                )
        expectations = self.config.get("data_runtime", {}).get(
            "sampling_expectation", {}
        )
        for family, values in expectations.items():
            for name, value in values.items():
                sampling_metrics[f"sampling/expected_{family}_{name}_fraction"] = float(
                    value
                )
        if self.context.is_main:
            self._write_scalars("train", sampling_metrics, self.global_step)
            self._print_metrics(f"sampling epoch={epoch}", sampling_metrics)

    @torch.no_grad()
    def evaluate(self, loader: Any, epoch: int | None = None) -> dict[str, Tensor]:
        self.model.eval()
        totals: dict[str, Tensor] = {}
        source_totals: dict[str, dict[str, Tensor]] = defaultdict(dict)
        source_counts: dict[str, int] = defaultdict(int)
        sample_count = 0
        visualization_count = 0
        source_visualization_counts: dict[str, int] = defaultdict(int)
        visualized_scenes: set[str] = set()
        render_visualizations = (
            self.writer is not None
            and self.eval_visualization_items > 0
            and (epoch is None or epoch % self.eval_visualization_every == 0)
        )
        for raw_batch in loader:
            batch = _to_device(raw_batch, self.context.device)
            with self._autocast(self.validation_precision):
                model_inputs = self._model_inputs(batch)
                output = self.model(**model_inputs)
                shuffled_output = self._shuffled_output(batch, model_inputs)
                _, terms = self.loss_module(
                    output,
                    batch["trajectory"],
                    batch["future_times"],
                    batch["trajectory_mask"],
                    self.global_step,
                    shuffled_output=shuffled_output,
                )
                terms.update(_model_diagnostics(self.model))
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
                dataset_names = [str(name) for name in batch.get("dataset_name", [])]
                for source_name in sorted(set(dataset_names)):
                    selection = torch.tensor(
                        [name == source_name for name in dataset_names],
                        device=self.context.device,
                        dtype=torch.bool,
                    )
                    source_output = _slice_output(output, selection)
                    source_shuffled = (
                        None
                        if shuffled_output is None
                        else _slice_output(shuffled_output, selection)
                    )
                    _, source_terms = self.loss_module(
                        source_output,
                        batch["trajectory"][selection],
                        batch["future_times"][selection],
                        batch["trajectory_mask"][selection],
                        self.global_step,
                        shuffled_output=source_shuffled,
                    )
                    source_terms.update(
                        trajectory_metrics(
                            source_output.reconstruction,
                            batch["trajectory"][selection],
                            batch["future_times"][selection],
                            batch["trajectory_mask"][selection],
                            steps_per_token=int(
                                self.config["action_tokenizer"]["steps_per_token"]
                            ),
                        )
                    )
                    source_count = int(selection.sum())
                    source_counts[source_name] += source_count
                    for key, value in source_terms.items():
                        weighted = value.detach().float() * source_count
                        current = source_totals[source_name].get(key)
                        source_totals[source_name][key] = (
                            weighted if current is None else current + weighted
                        )
            if render_visualizations and visualization_count < self.eval_visualization_items:
                batch_size = batch["trajectory"].shape[0]
                sample_tokens = batch.get("sample_token", [""] * batch_size)
                scene_tokens = batch.get("scene_token", [""] * batch_size)
                dataset_names = batch.get("dataset_name", ["dataset"] * batch_size)
                for item_index in range(batch_size):
                    if visualization_count >= self.eval_visualization_items:
                        break
                    scene_token = str(scene_tokens[item_index])
                    source_name = str(dataset_names[item_index])
                    if (
                        self.eval_visualization_items_per_source > 0
                        and source_visualization_counts[source_name]
                        >= self.eval_visualization_items_per_source
                    ):
                        continue
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
                        f"evaluation/{source_name}/vggt_diagnostic_2x2/"
                        f"item_{source_visualization_counts[source_name]:03d}",
                        image,
                        self.global_step,
                    )
                    visualized_scenes.add(scene_token)
                    visualization_count += 1
                    source_visualization_counts[source_name] += 1
            batch_size = int(batch["trajectory"].shape[0])
            for key, value in terms.items():
                weighted = value.detach().float() * batch_size
                totals[key] = totals.get(key, torch.zeros_like(weighted)) + weighted
            sample_count += batch_size
        metrics = _reduce_sample_weighted_metrics(
            totals, sample_count, self.context.world_size
        )
        configured_sources = list(self.config.get("data", {}).get("sources", {}))
        all_sources = sorted(set(configured_sources) | set(source_counts))
        source_metrics: dict[str, dict[str, Tensor]] = {}
        for source_name in all_sources:
            current_totals = source_totals[source_name]
            device = self.context.device
            count = torch.tensor(float(source_counts[source_name]), device=device)
            if self.context.world_size > 1:
                dist.all_reduce(count, op=dist.ReduceOp.SUM)
            reduced_source_totals: dict[str, Tensor] = {}
            for key in sorted(totals):
                value = current_totals.get(key, torch.zeros_like(totals[key]))
                if self.context.world_size > 1:
                    dist.all_reduce(value, op=dist.ReduceOp.SUM)
                reduced_source_totals[key] = value
            if count.item() <= 0:
                continue
            averaged = {
                key: value / count for key, value in reduced_source_totals.items()
            }
            source_metrics[source_name] = averaged
            metrics.update(
                {f"source/{source_name}/{key}": value for key, value in averaged.items()}
            )
        if source_metrics:
            shared_keys = set.intersection(
                *(set(values) for values in source_metrics.values())
            )
            sampling = self.config.get("data", {}).get("sampling", {})
            strategy = str(sampling.get("strategy", "proportional"))
            configured_weights = sampling.get("source_weights", {})
            if strategy == "balanced":
                weights = {source_name: 1.0 for source_name in source_metrics}
            elif strategy == "weighted":
                weights = {
                    source_name: float(configured_weights[source_name])
                    for source_name in source_metrics
                }
            else:
                runtime_sources = self.config.get("data_runtime", {}).get("sources", {})
                weights = {
                    source_name: float(
                        runtime_sources.get(source_name, {}).get("num_train_windows", 1.0)
                    )
                    for source_name in source_metrics
                }
            weight_sum = sum(weights.values())
            if weight_sum <= 0:
                raise ValueError("Validation source weights must sum to a positive value")
            for key in shared_keys:
                metrics[f"macro/{key}"] = torch.stack(
                    [values[key] for values in source_metrics.values()]
                ).mean()
                metrics[f"weighted/{key}"] = sum(
                    source_metrics[source_name][key] * weights[source_name]
                    for source_name in source_metrics
                ) / weight_sum
        return metrics

    def establish_initial_baseline(self, loader: Any) -> dict[str, Tensor]:
        """Measure a warm start with the current precision and aggregation contract."""
        metrics = self.evaluate(loader, epoch=-1)
        if self.checkpoint_metric not in metrics:
            raise KeyError(f"Configured checkpoint metric is missing: {self.checkpoint_metric}")
        ade = float(metrics[self.checkpoint_metric])
        if not math.isfinite(ade):
            raise ValueError("Initial validation ADE is not finite")
        self.best_ade = ade
        self.best_trained_ade = math.inf
        self.epochs_without_improvement = 0
        if self.context.is_main:
            self._print_metrics("val initial", metrics)
            self._write_scalars("validation", metrics, self.global_step)
            if self.writer is not None:
                self.writer.add_scalar("progress/epoch", -1, self.global_step)
        return metrics

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
            "best_trained_ade": self.best_trained_ade,
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
        _validate_multi_source_resume_config(
            self.config, checkpoint.get("config", {})
        )
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
        # Older checkpoints predate the separate trained-model metric. Falling
        # back to best_ade preserves their early-stopping state on resume.
        self.best_trained_ade = float(
            checkpoint.get("best_trained_ade", checkpoint["best_ade"])
        )
        self.epochs_without_improvement = int(
            checkpoint.get("epochs_without_improvement", 0)
        )
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if torch.cuda.is_available() and checkpoint["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_state"]])
        return int(checkpoint["epoch"]) + 1

    def load_initial_weights(
        self,
        path: str | Path,
        allowed_missing_prefixes: tuple[str, ...] = (),
        inherit_best_metric: bool = True,
    ) -> None:
        """Warm-start model weights without importing optimizer or training state."""
        checkpoint = torch.load(path, map_location=self.context.device, weights_only=False)
        model = _unwrap(self.model)
        incompatible = model.load_state_dict(checkpoint["model"], strict=False)
        invalid_missing = [
            key
            for key in incompatible.missing_keys
            if not key.startswith(allowed_missing_prefixes)
            and not key.startswith("feature_extractor.model.")
        ]
        if invalid_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "Initial checkpoint model mismatch: "
                f"missing={invalid_missing}, unexpected={incompatible.unexpected_keys}"
            )
        if inherit_best_metric and "best_ade" in checkpoint:
            self.best_ade = float(checkpoint["best_ade"])
        if self.context.is_main:
            print(
                f"Initialized model weights from {path}; "
                f"new_parameters={incompatible.missing_keys}",
                flush=True,
            )

    @staticmethod
    def _print_metrics(prefix: str, metrics: dict[str, Any]) -> None:
        rendered = " ".join(f"{key}={float(value):.5f}" for key, value in sorted(metrics.items()))
        print(f"{prefix} {rendered}", flush=True)

    def _write_scalars(self, prefix: str, metrics: dict[str, Any], step: int) -> None:
        if self.writer is None:
            return
        for key, value in sorted(metrics.items()):
            self.writer.add_scalar(f"{prefix}/{key}", float(value), step)
