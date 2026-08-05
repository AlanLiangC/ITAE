"""PyTorch dataset for six-frame vision/action windows."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from .manifest import WindowRecord, load_manifest


class LetterboxNormalize:
    """Preserve the 16:9 front-camera FOV while producing a square PE input."""

    def __init__(
        self,
        image_size: int = 512,
        mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: tuple[float, float, float] = (0.5, 0.5, 0.5),
    ) -> None:
        self.image_size = image_size
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.fill = tuple(int(round(channel * 255)) for channel in self.mean)

    def __call__(self, image: Image.Image) -> Tensor:
        image = image.convert("RGB")
        scale = min(self.image_size / image.width, self.image_size / image.height)
        resized = image.resize(
            (round(image.width * scale), round(image.height * scale)), Image.Resampling.BILINEAR
        )
        canvas = Image.new("RGB", (self.image_size, self.image_size), self.fill)
        left = (self.image_size - resized.width) // 2
        top = (self.image_size - resized.height) // 2
        canvas.paste(resized, (left, top))
        array = np.asarray(canvas, dtype=np.float32) / 255.0
        array = (array - self.mean) / self.std
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()


Sample = dict[str, Union[Tensor, str]]


class NuScenesWindowDataset(Dataset[Sample]):
    """Load six images and a local `[60,3]` trajectory from a manifest.

    All trajectories use meters/radians in the anchor ego frame. `trajectory_mask`
    is `[T]` and is currently all true; it remains explicit for future variable-length data.
    """

    def __init__(
        self,
        manifest: str | Path | list[WindowRecord],
        image_size: int = 512,
        transform: LetterboxNormalize | None = None,
        load_images: bool = True,
    ) -> None:
        self.windows = load_manifest(manifest) if isinstance(manifest, (str, Path)) else manifest
        self.transform = transform or LetterboxNormalize(image_size=image_size)
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
        }
        if self.load_images:
            images = []
            for path in window.image_paths:
                with Image.open(path) as image:
                    images.append(self.transform(image))
            sample["images"] = torch.stack(images)  # [F=6, 3, H, W]
        return sample


class CachedPEFeatureDataset(Dataset[Sample]):
    """Attach cached `[F,P,C_pe]` features to a metadata-only base dataset."""

    def __init__(
        self,
        base: NuScenesWindowDataset,
        cache_directory: str | Path,
        manifest_path: str | Path | None = None,
        expected_metadata: dict[str, object] | None = None,
    ) -> None:
        import json

        try:
            from safetensors.torch import load_file
        except ImportError as error:
            raise ImportError("Install safetensors to use cached PE features") from error

        self.base = base
        self.cache_directory = Path(cache_directory)
        index = json.loads((self.cache_directory / "index.json").read_text(encoding="utf-8"))
        self.shards = index["shards"]
        if int(index["num_samples"]) != len(base):
            raise ValueError("PE cache sample count does not match manifest")
        if manifest_path is not None:
            digest = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
            if index.get("manifest_sha256") != digest:
                raise ValueError("PE cache was built from a different manifest")
        for key, expected in (expected_metadata or {}).items():
            if index.get(key) != expected:
                raise ValueError(
                    f"PE cache metadata mismatch for {key}: {index.get(key)!r} != {expected!r}"
                )
        self._loaded_file: str | None = None
        self._loaded_features: Tensor | None = None
        self._load_file = load_file

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        sample = self.base[index]
        shard = next((item for item in self.shards if item["start"] <= index < item["end"]), None)
        if shard is None:
            raise IndexError(f"Sample {index} is missing from PE cache")
        if self._loaded_file != shard["file"]:
            self._loaded_features = self._load_file(str(self.cache_directory / shard["file"]))[
                "features"
            ]
            self._loaded_file = shard["file"]
        assert self._loaded_features is not None
        sample["visual_features"] = self._loaded_features[index - shard["start"]]
        return sample
