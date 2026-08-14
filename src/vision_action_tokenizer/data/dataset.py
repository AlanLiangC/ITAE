"""PyTorch datasets for configurable vision/action windows."""

from __future__ import annotations

import hashlib
import json
from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from .manifest import WindowRecord, load_manifest
from .trajectory import shift_se2_reference_point


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class VGGTOmegaResize:
    """Match VGGT-Omega's RGB `[0,1]` preprocessing without double normalization."""

    def __init__(
        self,
        image_resolution: int = 512,
        mode: str = "max_size",
        patch_size: int = 16,
    ) -> None:
        if mode not in {"balanced", "max_size"}:
            raise ValueError("VGGT resize mode must be `balanced` or `max_size`")
        if image_resolution <= 0 or image_resolution % patch_size:
            raise ValueError("image_resolution must be a positive patch-size multiple")
        self.image_resolution = image_resolution
        self.mode = mode
        self.patch_size = patch_size

    def _round_to_patch(self, value: float) -> int:
        return max(self.patch_size, round(value / self.patch_size) * self.patch_size)

    @staticmethod
    def _crop_supported_aspect(image: Image.Image) -> Image.Image:
        width, height = image.size
        aspect = height / max(width, 1)
        if aspect < 0.5:
            crop_width = min(width, max(1, round(height / 0.5)))
            left = max((width - crop_width) // 2, 0)
            return image.crop((left, 0, left + crop_width, height))
        if aspect > 2.0:
            crop_height = min(height, max(1, round(width * 2.0)))
            top = max((height - crop_height) // 2, 0)
            return image.crop((0, top, width, top + crop_height))
        return image

    def __call__(self, image: Image.Image) -> Tensor:
        image = self._crop_supported_aspect(image.convert("RGB"))
        width, height = image.size
        aspect = height / max(width, 1)
        if self.mode == "max_size":
            if aspect >= 1.0:
                target_height = self.image_resolution
                target_width = self._round_to_patch(self.image_resolution / aspect)
            else:
                target_width = self.image_resolution
                target_height = self._round_to_patch(self.image_resolution * aspect)
        else:
            token_count = (self.image_resolution // self.patch_size) ** 2
            width_patches = max(1, round(np.sqrt(token_count / aspect)))
            height_patches = max(1, round(token_count / width_patches))
            target_width = width_patches * self.patch_size
            target_height = height_patches * self.patch_size
        image = image.resize((target_width, target_height), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()


Sample = dict[str, Tensor | str]


def configured_reference_point_offset(
    config: dict[str, Any], windows: list[WindowRecord]
) -> tuple[float, float]:
    """Resolve one dataset-specific reference offset for a manifest."""
    source_config = config.get("data", {}).get("sources")
    if not source_config:
        return (0.0, 0.0)
    dataset_names = {window.dataset_name for window in windows}
    if not dataset_names:
        raise ValueError("Cannot resolve a reference-point offset for an empty manifest")
    if len(dataset_names) != 1:
        raise ValueError(
            "A manifest must contain one dataset when resolving a reference-point offset"
        )
    dataset_name = next(iter(dataset_names))
    values = source_config.get(dataset_name, {}).get(
        "reference_point_offset_m", [0.0, 0.0]
    )
    offset = tuple(map(float, values))
    if len(offset) != 2 or not np.isfinite(offset).all():
        raise ValueError(
            f"{dataset_name}.reference_point_offset_m must contain two finite values"
        )
    return offset


class ActionWindowDataset(Dataset[Sample]):
    """Load images and a local `[T,3]` trajectory from a dataset-neutral manifest.

    All trajectories use meters/radians in the anchor ego frame. `trajectory_mask`
    is `[T]` and is currently all true; it remains explicit for future variable-length data.
    """

    def __init__(
        self,
        manifest: str | Path | list[WindowRecord],
        image_size: int = 512,
        transform: Callable[[Image.Image], Tensor] | None = None,
        load_images: bool = True,
        reference_point_offset_m: tuple[float, float] = (0.0, 0.0),
        image_cache_size: int = 0,
    ) -> None:
        self.windows = load_manifest(manifest) if isinstance(manifest, (str, Path)) else manifest
        self.transform = transform or VGGTOmegaResize(image_resolution=image_size)
        self.load_images = load_images
        self.image_cache_size = int(image_cache_size)
        if self.image_cache_size < 0:
            raise ValueError("image_cache_size must be non-negative")
        self._image_cache: OrderedDict[str, Tensor] = OrderedDict()
        self.reference_point_offset_m = tuple(map(float, reference_point_offset_m))
        if len(self.reference_point_offset_m) != 2:
            raise ValueError("reference_point_offset_m must contain [x, y]")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        window = self.windows[index]
        trajectory = torch.from_numpy(
            shift_se2_reference_point(window.trajectory, self.reference_point_offset_m)
        )
        sample: dict[str, Tensor | str] = {
            "trajectory": trajectory,
            "trajectory_mask": torch.ones(trajectory.shape[0], dtype=torch.bool),
            "frame_times": torch.tensor(window.frame_times_s, dtype=torch.float32),
            "future_times": torch.tensor(window.future_times_s, dtype=torch.float32),
            "sample_token": window.sample_token,
            "scene_token": window.scene_token,
            "group_token": window.group_token or window.scene_token,
            "dataset_name": window.dataset_name,
            "image_paths_json": json.dumps(window.image_paths, ensure_ascii=False),
        }
        if self.load_images:
            images = []
            for path in window.image_paths:
                cached = self._image_cache.pop(path, None)
                if cached is None:
                    with Image.open(path) as image:
                        cached = self.transform(image)
                if self.image_cache_size:
                    self._image_cache[path] = cached
                    if len(self._image_cache) > self.image_cache_size:
                        self._image_cache.popitem(last=False)
                images.append(cached)
            sample["images"] = torch.stack(images)
        return sample


# Backward-compatible name used by existing downstream code and old imports.
NuScenesWindowDataset = ActionWindowDataset


class MultiSourceActionDataset(Dataset[Sample]):
    """Concatenate source datasets while preserving an explicit source identity."""

    def __init__(self, sources: Mapping[str, Dataset[Sample]]) -> None:
        if not sources:
            raise ValueError("At least one action dataset source is required")
        self.sources = dict(sources)
        if any(not name or "/" in name for name in self.sources):
            raise ValueError("Dataset source names must be non-empty TensorBoard-safe labels")
        self.source_names = list(self.sources)
        self.cumulative_sizes: list[int] = []
        total = 0
        for dataset in self.sources.values():
            if len(dataset) == 0:
                raise ValueError("Action dataset sources must not be empty")
            total += len(dataset)
            self.cumulative_sizes.append(total)

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def source_and_local_index(self, index: int) -> tuple[str, int]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        source_index = bisect_right(self.cumulative_sizes, index)
        previous = 0 if source_index == 0 else self.cumulative_sizes[source_index - 1]
        return self.source_names[source_index], index - previous

    def __getitem__(self, index: int) -> Sample:
        source_name, local_index = self.source_and_local_index(index)
        sample = dict(self.sources[source_name][local_index])
        sample["dataset_name"] = source_name
        return sample

    @property
    def source_ranges(self) -> dict[str, range]:
        ranges: dict[str, range] = {}
        start = 0
        for name, end in zip(self.source_names, self.cumulative_sizes, strict=True):
            ranges[name] = range(start, end)
            start = end
        return ranges


class CachedVGGTOmegaFeatureDataset(Dataset[Sample]):
    """Attach strict cached CameraHead hidden tokens to a trajectory sample."""

    def __init__(
        self,
        base: ActionWindowDataset,
        cache_directory: str | Path,
        manifest_path: str | Path | None = None,
        expected_metadata: dict[str, object] | None = None,
        verify_checksums: bool = True,
    ) -> None:
        try:
            from safetensors import safe_open
        except ImportError as error:
            raise ImportError("Install safetensors to use VGGT-Omega feature caches") from error

        self.base = base
        self.cache_directory = Path(cache_directory)
        self.index = json.loads(
            (self.cache_directory / "index.json").read_text(encoding="utf-8")
        )
        self.shards = self.index["shards"]
        self._shard_ends = [int(shard["end"]) for shard in self.shards]
        if self._shard_ends != sorted(self._shard_ends):
            raise ValueError("VGGT-Omega cache shard ranges are not ordered")
        if self.index.get("cache_type") != "vggt_omega_camera_head_hidden_v1":
            raise ValueError("Not a supported VGGT-Omega CameraHead feature cache")
        if not self.index.get("complete", True):
            raise ValueError("VGGT-Omega cache is incomplete; resume feature extraction first")
        if int(self.index["num_samples"]) != len(base):
            raise ValueError("VGGT-Omega cache sample count does not match manifest")
        # Caches created before token_mode was recorded contain this exact mean-only set.
        token_mode = self.index.get("token_mode", "camera_register_mean")
        self.tensor_keys = ["camera_hidden", "register_hidden_mean", "pose_enc"]
        if token_mode == "camera_register_tokens":
            self.tensor_keys.insert(2, "register_hidden")
        elif token_mode != "camera_register_mean":
            raise ValueError(f"Unsupported VGGT-Omega cache token mode: {token_mode!r}")
        if manifest_path is not None:
            digest = _file_sha256(manifest_path)
            if self.index.get("manifest_sha256") != digest:
                raise ValueError("VGGT-Omega cache was built from a different manifest")
        for key, expected in (expected_metadata or {}).items():
            actual = token_mode if key == "token_mode" else self.index.get(key)
            if expected is not None and actual != expected:
                raise ValueError(
                    f"VGGT-Omega cache metadata mismatch for {key}: "
                    f"{actual!r} != {expected!r}"
                )
        self.verify_checksums = bool(verify_checksums)
        self._verified_files: set[str] = set()
        self._safe_open = safe_open

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        sample = self.base[index]
        shard_index = bisect_right(self._shard_ends, index)
        if shard_index >= len(self.shards):
            raise IndexError(f"Sample {index} is missing from VGGT-Omega cache")
        shard = self.shards[shard_index]
        if not int(shard["start"]) <= index < int(shard["end"]):
            raise IndexError(f"Sample {index} is missing from VGGT-Omega cache")
        shard_path = self.cache_directory / str(shard["file"])
        if str(shard["file"]) not in self._verified_files:
            expected_size = shard.get("size_bytes")
            if expected_size is not None and shard_path.stat().st_size != int(expected_size):
                raise ValueError(f"VGGT-Omega cache shard size mismatch: {shard_path}")
            if self.verify_checksums:
                actual_sha = _file_sha256(shard_path)
                if actual_sha != shard.get("sha256"):
                    raise ValueError(
                        f"VGGT-Omega cache shard checksum mismatch: {shard_path}"
                    )
            self._verified_files.add(str(shard["file"]))
        offset = index - int(shard["start"])
        with self._safe_open(str(shard_path), framework="pt", device="cpu") as tensors:
            missing = set(self.tensor_keys) - set(tensors.keys())
            if missing:
                raise ValueError(f"VGGT-Omega cache shard is missing tensors: {sorted(missing)}")
            for key in self.tensor_keys:
                expected_shape = tuple(self.index[f"{key}_shape"])
                tensor_slice = tensors.get_slice(key)
                actual_shape = tuple(tensor_slice.get_shape()[1:])
                if actual_shape != expected_shape:
                    raise ValueError(
                        f"VGGT-Omega cache tensor shape mismatch for {key}: "
                        f"{actual_shape} != {expected_shape}"
                    )
                sample[key] = tensor_slice[offset]
        return sample
