"""PyTorch datasets for configurable vision/action windows."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from .manifest import WindowRecord, load_manifest


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


class NuScenesWindowDataset(Dataset[Sample]):
    """Load images and a local `[T,3]` trajectory from a manifest.

    All trajectories use meters/radians in the anchor ego frame. `trajectory_mask`
    is `[T]` and is currently all true; it remains explicit for future variable-length data.
    """

    def __init__(
        self,
        manifest: str | Path | list[WindowRecord],
        image_size: int = 512,
        transform: Callable[[Image.Image], Tensor] | None = None,
        load_images: bool = True,
    ) -> None:
        self.windows = load_manifest(manifest) if isinstance(manifest, (str, Path)) else manifest
        self.transform = transform or VGGTOmegaResize(image_resolution=image_size)
        self.load_images = load_images

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        window = self.windows[index]
        trajectory = torch.tensor(window.trajectory, dtype=torch.float32)
        sample: dict[str, Tensor | str] = {
            "trajectory": trajectory,
            "trajectory_mask": torch.ones(trajectory.shape[0], dtype=torch.bool),
            "frame_times": torch.tensor(window.frame_times_s, dtype=torch.float32),
            "future_times": torch.tensor(window.future_times_s, dtype=torch.float32),
            "sample_token": window.sample_token,
            "scene_token": window.scene_token,
            "image_paths_json": json.dumps(window.image_paths, ensure_ascii=False),
        }
        if self.load_images:
            images = []
            for path in window.image_paths:
                with Image.open(path) as image:
                    images.append(self.transform(image))
            sample["images"] = torch.stack(images)
        return sample


class CachedVGGTOmegaFeatureDataset(Dataset[Sample]):
    """Attach strict cached CameraHead hidden tokens to a trajectory sample."""

    tensor_keys = ("camera_hidden", "register_hidden_mean", "pose_enc")

    def __init__(
        self,
        base: NuScenesWindowDataset,
        cache_directory: str | Path,
        manifest_path: str | Path | None = None,
        expected_metadata: dict[str, object] | None = None,
    ) -> None:
        try:
            from safetensors.torch import load_file
        except ImportError as error:
            raise ImportError("Install safetensors to use VGGT-Omega feature caches") from error

        self.base = base
        self.cache_directory = Path(cache_directory)
        self.index = json.loads(
            (self.cache_directory / "index.json").read_text(encoding="utf-8")
        )
        self.shards = self.index["shards"]
        if self.index.get("cache_type") != "vggt_omega_camera_head_hidden_v1":
            raise ValueError("Not a supported VGGT-Omega CameraHead feature cache")
        if not self.index.get("complete", True):
            raise ValueError("VGGT-Omega cache is incomplete; resume feature extraction first")
        if int(self.index["num_samples"]) != len(base):
            raise ValueError("VGGT-Omega cache sample count does not match manifest")
        if manifest_path is not None:
            digest = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
            if self.index.get("manifest_sha256") != digest:
                raise ValueError("VGGT-Omega cache was built from a different manifest")
        for key, expected in (expected_metadata or {}).items():
            if expected is not None and self.index.get(key) != expected:
                raise ValueError(
                    f"VGGT-Omega cache metadata mismatch for {key}: "
                    f"{self.index.get(key)!r} != {expected!r}"
                )
        self._loaded_file: str | None = None
        self._loaded_tensors: dict[str, Tensor] | None = None
        self._verified_files: set[str] = set()
        self._load_file = load_file

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        sample = self.base[index]
        shard = next((item for item in self.shards if item["start"] <= index < item["end"]), None)
        if shard is None:
            raise IndexError(f"Sample {index} is missing from VGGT-Omega cache")
        if self._loaded_file != shard["file"]:
            shard_path = self.cache_directory / str(shard["file"])
            if str(shard["file"]) not in self._verified_files:
                actual_sha = hashlib.sha256(shard_path.read_bytes()).hexdigest()
                if actual_sha != shard.get("sha256"):
                    raise ValueError(
                        f"VGGT-Omega cache shard checksum mismatch: {shard_path}"
                    )
                self._verified_files.add(str(shard["file"]))
            self._loaded_tensors = self._load_file(
                str(shard_path)
            )
            missing = set(self.tensor_keys) - set(self._loaded_tensors)
            if missing:
                raise ValueError(f"VGGT-Omega cache shard is missing tensors: {sorted(missing)}")
            for key in self.tensor_keys:
                expected_shape = tuple(self.index[f"{key}_shape"])
                actual_shape = tuple(self._loaded_tensors[key].shape[1:])
                if actual_shape != expected_shape:
                    raise ValueError(
                        f"VGGT-Omega cache tensor shape mismatch for {key}: "
                        f"{actual_shape} != {expected_shape}"
                    )
            self._loaded_file = shard["file"]
        assert self._loaded_tensors is not None
        offset = index - shard["start"]
        for key in self.tensor_keys:
            sample[key] = self._loaded_tensors[key][offset]
        return sample
