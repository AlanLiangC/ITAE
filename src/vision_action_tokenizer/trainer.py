"""DDP-compatible tokenizer training loop with strict checkpoints."""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel

from .distributed import DistributedContext, reduce_metrics
from .losses import TokenizerLoss
from .metrics import trajectory_metrics


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def _checkpoint_model_state(model: nn.Module) -> dict[str, Tensor]:
    """Exclude a frozen PE backbone that is already identified by config/checkpoint path."""
    state = model.state_dict()
    pe_extractor = getattr(model, "pe_extractor", None)
    if pe_extractor is not None and getattr(pe_extractor, "freeze", False):
        state = {
            key: value for key, value in state.items() if not key.startswith("pe_extractor.model.")
        }
    return state


class TokenizerTrainer:
    """Train the online-PE + tokenizer model and log auditable scalar terms."""

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
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=precision == "fp16" and context.device.type == "cuda"
        )
        if context.is_main:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "resolved_config.json").write_text(
                json.dumps(self.config, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
            )

    def fit(
        self,
        train_loader: Any,
        val_loader: Any,
        epochs: int,
        log_every: int = 20,
        start_epoch: int = 0,
    ) -> None:
        for epoch in range(start_epoch, epochs):
            sampler = getattr(train_loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            self._train_epoch(train_loader, epoch, log_every)
            metrics = self.evaluate(val_loader)
            if self.context.is_main:
                self._print_metrics(f"val epoch={epoch}", metrics)
                ade = metrics.get("metric/ade_m", torch.tensor(math.inf)).item()
                is_best = ade < self.best_ade
                if is_best:
                    self.best_ade = ade
                self.save_checkpoint("last.pt", epoch)
                if is_best:
                    self.save_checkpoint("best.pt", epoch)

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
                    "trajectory": batch["trajectory"],
                    "frame_times": batch["frame_times"],
                    "future_times": batch["future_times"],
                    "trajectory_mask": batch["trajectory_mask"],
                    "sample_posterior": True,
                }
                feature_key = "visual_features" if "visual_features" in batch else "images"
                model_inputs[feature_key] = batch[feature_key]
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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()
            self.global_step += 1
            for key, value in terms.items():
                running[key] += float(value)
            if self.context.is_main and (batch_index + 1) % log_every == 0:
                averaged = {key: value / log_every for key, value in running.items()}
                averaged["lr"] = self.optimizer.param_groups[0]["lr"]
                self._print_metrics(f"train epoch={epoch} step={self.global_step}", averaged)
                running.clear()

    @torch.no_grad()
    def evaluate(self, loader: Any) -> dict[str, Tensor]:
        self.model.eval()
        totals: dict[str, Tensor] = {}
        batches = 0
        for raw_batch in loader:
            batch = _to_device(raw_batch, self.context.device)
            with self._autocast():
                model_inputs = {
                    "trajectory": batch["trajectory"],
                    "frame_times": batch["frame_times"],
                    "future_times": batch["future_times"],
                    "trajectory_mask": batch["trajectory_mask"],
                    "sample_posterior": False,
                }
                feature_key = "visual_features" if "visual_features" in batch else "images"
                model_inputs[feature_key] = batch[feature_key]
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
                        output.reconstruction_vis,
                        batch["trajectory"],
                        batch["future_times"],
                        batch["trajectory_mask"],
                    )
                )
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
            key for key in incompatible.missing_keys if not key.startswith("pe_extractor.model.")
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
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if torch.cuda.is_available() and checkpoint["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_state"]])
        return int(checkpoint["epoch"]) + 1

    @staticmethod
    def _print_metrics(prefix: str, metrics: dict[str, Any]) -> None:
        rendered = " ".join(f"{key}={float(value):.5f}" for key, value in sorted(metrics.items()))
        print(f"{prefix} {rendered}", flush=True)
