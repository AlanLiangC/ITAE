"""Shared decoding and evaluation for raw/token flow planners."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .data.planner_dataset import PlannerTargetNormalizer, wrap_yaw
from .flow_matching import euler_sample
from .metrics import trajectory_metrics


class PlannerOutputDecoder(nn.Module):
    """Map normalized raw points or V4 action tokens into the common trajectory space."""

    def __init__(
        self,
        target_type: str,
        normalizer: PlannerTargetNormalizer,
        tokenizer: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.target_type = target_type
        self.normalizer = normalizer
        self.tokenizer = tokenizer
        if target_type == "v4_action_token" and tokenizer is None:
            raise ValueError("Token planner output decoding requires the frozen tokenizer")
        if target_type not in {"raw_trajectory", "v4_action_token"}:
            raise ValueError(f"Unsupported planner target type: {target_type!r}")

    def forward(self, normalized_output: Tensor, future_times: Tensor) -> Tensor:
        denormalized = self.normalizer.denormalize(normalized_output.float())
        if self.target_type == "raw_trajectory":
            return wrap_yaw(denormalized)
        assert self.tokenizer is not None
        return self.tokenizer.decode(denormalized, future_times.float())


@dataclass
class PlannerEvaluation:
    metrics: dict[str, float]
    predictions: Tensor | None = None
    targets: Tensor | None = None
    future_times: Tensor | None = None
    sample_tokens: list[str] | None = None


@torch.no_grad()
def evaluate_planner(
    model: nn.Module,
    loader: DataLoader,
    decoder: PlannerOutputDecoder,
    device: torch.device,
    target_shape: tuple[int, int],
    inference_steps: int = 5,
    expected_nfe: int = 5,
    seed: int = 12345,
    keep_predictions: bool = False,
    max_batches: int | None = None,
) -> PlannerEvaluation:
    model.eval()
    decoder.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    totals: dict[str, float] = {}
    count = 0
    all_predictions: list[Tensor] = []
    all_targets: list[Tensor] = []
    all_times: list[Tensor] = []
    sample_tokens: list[str] = []
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        condition = batch["condition_tokens"].to(device).float()
        condition_mask = batch["condition_mask"].to(device)
        future_times = batch["future_times"].to(device).float()
        slot_times = planner_slot_times(
            decoder.target_type, future_times, target_shape[0]
        )
        normalized, nfe = euler_sample(
            model,
            condition,
            condition_mask,
            slot_times,
            target_shape,
            steps=inference_steps,
            generator=generator,
            expected_nfe=expected_nfe,
        )
        if nfe != expected_nfe:
            raise RuntimeError("Unexpected planner NFE")
        prediction = decoder(normalized, future_times)
        target = batch["trajectory"].to(device).float()
        metrics = trajectory_metrics(
            prediction, target, future_times, batch["trajectory_mask"].to(device)
        )
        if decoder.target_type == "v4_action_token":
            teacher_tokens = decoder.normalizer.normalize(
                batch["target"].to(device).float()
            )
            metrics["token/normalized_mse"] = (
                normalized.float() - teacher_tokens
            ).square().mean()
            oracle = batch["oracle_trajectory"].to(device).float()
            oracle_metrics = trajectory_metrics(
                oracle, target, future_times, batch["trajectory_mask"].to(device)
            )
            for key, value in oracle_metrics.items():
                metrics[f"oracle/{key}"] = value
        batch_size = len(prediction)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value) * batch_size
        count += batch_size
        if keep_predictions:
            all_predictions.append(prediction.cpu())
            all_targets.append(target.cpu())
            all_times.append(future_times.cpu())
            sample_tokens.extend(batch["sample_token"])
    if count == 0:
        raise ValueError("Planner evaluation loader produced no samples")
    result = {key: value / count for key, value in totals.items()}
    if "oracle/metric/ade_m" in result:
        result["token/excess_ade_m"] = (
            result["metric/ade_m"] - result["oracle/metric/ade_m"]
        )
        result["token/excess_fde_m"] = (
            result["metric/fde_m"] - result["oracle/metric/fde_m"]
        )
    result["sampler/nfe"] = float(expected_nfe)
    result["sampler/samples"] = float(count)
    return PlannerEvaluation(
        metrics=result,
        predictions=torch.cat(all_predictions) if all_predictions else None,
        targets=torch.cat(all_targets) if all_targets else None,
        future_times=torch.cat(all_times) if all_times else None,
        sample_tokens=sample_tokens if keep_predictions else None,
    )


def planner_slot_times(target_type: str, future_times: Tensor, slots: int) -> Tensor:
    if target_type == "raw_trajectory":
        if future_times.shape[1] != slots:
            raise ValueError("Raw planner slots must match future trajectory timestamps")
        return future_times
    if target_type == "v4_action_token":
        if slots != 4:
            raise ValueError("V4 token planner requires four slots")
        return torch.tensor(
            [0.5, 1.5, 2.5, 3.5],
            dtype=future_times.dtype,
            device=future_times.device,
        ).expand(future_times.shape[0], -1)
    raise ValueError(f"Unsupported planner target type: {target_type!r}")
