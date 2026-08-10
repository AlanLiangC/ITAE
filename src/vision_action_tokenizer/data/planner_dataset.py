"""Datasets and normalization for current-frame flow planning."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from safetensors.torch import load_file
from torch import Tensor, nn
from torch.utils.data import Dataset

from .manifest import WindowRecord, load_manifest


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unwrap_yaw(trajectory: Tensor) -> Tensor:
    """Unwrap the yaw channel of `[... ,T,3]` trajectories."""
    if trajectory.shape[-1] != 3:
        raise ValueError("Expected trajectory with final dimension [x,y,yaw]")
    yaw = trajectory[..., 2]
    if yaw.shape[-1] <= 1:
        return trajectory.clone()
    raw_delta = torch.diff(yaw, dim=-1)
    wrapped_delta = torch.atan2(torch.sin(raw_delta), torch.cos(raw_delta))
    unwrapped = torch.cat(
        [yaw[..., :1], yaw[..., :1] + torch.cumsum(wrapped_delta, dim=-1)], dim=-1
    )
    output = trajectory.clone()
    output[..., 2] = unwrapped
    return output


def wrap_yaw(trajectory: Tensor) -> Tensor:
    output = trajectory.clone()
    output[..., 2] = torch.atan2(
        torch.sin(output[..., 2]), torch.cos(output[..., 2])
    )
    return output


class PlannerTargetNormalizer(nn.Module):
    """Train-split per-slot normalization shared by training and evaluation."""

    def __init__(self, shape: tuple[int, int], epsilon: float = 1e-4) -> None:
        super().__init__()
        if epsilon <= 0:
            raise ValueError("Normalizer epsilon must be positive")
        self.epsilon = float(epsilon)
        self.register_buffer("mean", torch.zeros(shape, dtype=torch.float32))
        self.register_buffer("std", torch.ones(shape, dtype=torch.float32))
        self.register_buffer("fitted", torch.tensor(False))

    @torch.no_grad()
    def fit(self, targets: Tensor) -> None:
        if targets.ndim != 3 or tuple(targets.shape[1:]) != tuple(self.mean.shape):
            raise ValueError(
                f"Expected targets [N,{tuple(self.mean.shape)}], got {tuple(targets.shape)}"
            )
        if not torch.isfinite(targets).all():
            raise ValueError("Cannot fit normalization on non-finite targets")
        self.mean.copy_(targets.float().mean(dim=0))
        self.std.copy_(
            targets.float().std(dim=0, unbiased=False).clamp_min(self.epsilon)
        )
        self.fitted.fill_(True)

    def _require_fitted(self) -> None:
        if not bool(self.fitted):
            raise RuntimeError("Planner target normalizer has not been fitted")

    def normalize(self, target: Tensor) -> Tensor:
        self._require_fitted()
        return (target - self.mean) / self.std

    def denormalize(self, target: Tensor) -> Tensor:
        self._require_fitted()
        return target * self.std + self.mean

    def metadata(self) -> dict[str, Any]:
        return {
            "shape": list(self.mean.shape),
            "epsilon": self.epsilon,
            "fitted": bool(self.fitted),
        }


class PlannerVisionCache:
    """Strict reader for sharded current-frame vision-condition caches."""

    def __init__(
        self,
        directory: str | Path,
        manifest_path: str | Path,
        expected_metadata: dict[str, object] | None = None,
    ) -> None:
        self.directory = Path(directory)
        index_path = self.directory / "index.json"
        if not index_path.is_file():
            raise FileNotFoundError(f"Planner vision cache index does not exist: {index_path}")
        self.index = json.loads(index_path.read_text(encoding="utf-8"))
        if self.index.get("cache_type") != "planner_vision_condition_v1":
            raise ValueError("Unsupported planner vision cache type")
        if not self.index.get("complete", False):
            raise ValueError("Planner vision cache is incomplete")
        if self.index.get("manifest_sha256") != file_sha256(manifest_path):
            raise ValueError("Planner vision cache was built from another manifest")
        for key, expected in (expected_metadata or {}).items():
            if expected is not None and self.index.get(key) != expected:
                raise ValueError(
                    f"Planner vision cache metadata mismatch for {key}: "
                    f"{self.index.get(key)!r} != {expected!r}"
                )
        self.shards = list(self.index["shards"])
        self.num_samples = int(self.index["num_samples"])
        self.token_shape = tuple(map(int, self.index["condition_shape"]))
        self._sample_to_shard: list[int] = [-1] * self.num_samples
        cursor = 0
        for shard_index, shard in enumerate(self.shards):
            start, end = int(shard["start"]), int(shard["end"])
            if start != cursor or not start < end <= self.num_samples:
                raise ValueError("Planner vision cache shards are not contiguous")
            self._sample_to_shard[start:end] = [shard_index] * (end - start)
            cursor = end
        if cursor != self.num_samples:
            raise ValueError("Planner vision cache does not cover every sample")
        self._loaded_shard_index: int | None = None
        self._loaded_tensors: dict[str, Tensor] | None = None
        self._verified_files: set[str] = set()

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        shard_index = self._sample_to_shard[index]
        shard = self.shards[shard_index]
        filename = str(shard["file"])
        if self._loaded_shard_index != shard_index:
            path = self.directory / filename
            if filename not in self._verified_files:
                if file_sha256(path) != shard["sha256"]:
                    raise ValueError(f"Planner vision cache shard is corrupt: {path}")
                self._verified_files.add(filename)
            tensors = load_file(str(path))
            if set(tensors) != {"condition_tokens", "condition_mask"}:
                raise ValueError("Planner vision cache shard tensor keys are invalid")
            if tuple(tensors["condition_tokens"].shape[1:]) != self.token_shape:
                raise ValueError("Planner vision cache condition shape mismatch")
            self._loaded_tensors = tensors
            self._loaded_shard_index = shard_index
        assert self._loaded_tensors is not None
        offset = index - int(shard["start"])
        return (
            self._loaded_tensors["condition_tokens"][offset],
            self._loaded_tensors["condition_mask"][offset].bool(),
        )


class ActionTargetCache:
    """Small deterministic cache of frozen V4 action-token teacher targets."""

    def __init__(self, path: str | Path, expected_sample_tokens: list[str]) -> None:
        self.path = Path(path)
        metadata_path = self.path.with_suffix(".json")
        if not self.path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"Action target cache is incomplete: {self.path}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("cache_type") != "v4_action_targets_v1":
            raise ValueError("Unsupported action target cache type")
        if self.metadata.get("sample_tokens") != expected_sample_tokens:
            raise ValueError("Action target cache sample order does not match manifest")
        payload = load_file(str(self.path))
        required = {"action_tokens", "oracle_trajectory", "future_times"}
        if set(payload) != required:
            raise ValueError(f"Action target cache keys must be {sorted(required)}")
        self.action_tokens = payload["action_tokens"].float()
        self.oracle_trajectory = payload["oracle_trajectory"].float()
        self.future_times = payload["future_times"].float()
        if len(self.action_tokens) != len(expected_sample_tokens):
            raise ValueError("Action target cache sample count mismatch")

    def __len__(self) -> int:
        return len(self.action_tokens)


class PlannerDataset(Dataset[dict[str, Tensor | str]]):
    """Pair one current-frame condition with a raw or tokenized planning target."""

    def __init__(
        self,
        manifest: str | Path | list[WindowRecord],
        target_type: str,
        vision_cache: str | Path | None = None,
        action_target_cache: str | Path | None = None,
        transform: Callable[[Image.Image], Tensor] | None = None,
        expected_vision_metadata: dict[str, object] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest) if isinstance(manifest, (str, Path)) else None
        self.windows = load_manifest(manifest) if self.manifest_path is not None else manifest
        if target_type not in {"raw_trajectory", "v4_action_token"}:
            raise ValueError(f"Unsupported planner target type: {target_type!r}")
        self.target_type = target_type
        self.transform = transform
        if vision_cache is None:
            if transform is None:
                raise ValueError("Online planner dataset requires an image transform")
            self.vision_cache = None
        else:
            if self.manifest_path is None:
                raise ValueError("Cached planner dataset requires a manifest path")
            self.vision_cache = PlannerVisionCache(
                vision_cache, self.manifest_path, expected_vision_metadata
            )
            if len(self.vision_cache) != len(self.windows):
                raise ValueError("Planner vision cache sample count does not match manifest")
        sample_tokens = [window.sample_token for window in self.windows]
        self.action_cache = None
        if target_type == "v4_action_token":
            if action_target_cache is None:
                raise ValueError("Token planner requires an action target cache")
            self.action_cache = ActionTargetCache(action_target_cache, sample_tokens)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        window = self.windows[index]
        trajectory = torch.tensor(window.trajectory, dtype=torch.float32)
        future_times = torch.tensor(window.future_times_s, dtype=torch.float32)
        if trajectory.shape != (40, 3) or future_times.shape != (40,):
            raise ValueError("Planner expects exactly 40 trajectory points over four seconds")
        if not torch.all(future_times[1:] > future_times[:-1]):
            raise ValueError("Planner future_times must be strictly increasing")
        sample: dict[str, Tensor | str] = {
            "sample_token": window.sample_token,
            "scene_token": window.scene_token,
            "current_image_path": window.image_paths[0],
            "trajectory": trajectory,
            "future_times": future_times,
            "trajectory_mask": torch.ones(40, dtype=torch.bool),
        }
        if self.target_type == "raw_trajectory":
            sample["target"] = unwrap_yaw(trajectory)
        else:
            assert self.action_cache is not None
            sample["target"] = self.action_cache.action_tokens[index]
            sample["oracle_trajectory"] = self.action_cache.oracle_trajectory[index]
            if not torch.allclose(
                future_times, self.action_cache.future_times[index], atol=1e-6, rtol=0
            ):
                raise ValueError("Action target cache future_times do not match manifest")
        if self.vision_cache is None:
            assert self.transform is not None
            with Image.open(window.image_paths[0]) as image:
                sample["current_image"] = self.transform(image)
        else:
            tokens, mask = self.vision_cache[index]
            sample["condition_tokens"] = tokens
            sample["condition_mask"] = mask
        return sample

    def all_targets(self) -> Tensor:
        if self.target_type == "v4_action_token":
            assert self.action_cache is not None
            return self.action_cache.action_tokens.clone()
        return torch.stack(
            [
                unwrap_yaw(torch.tensor(window.trajectory, dtype=torch.float32))
                for window in self.windows
            ]
        )
