"""Deterministic weighted sampling shared by single- and multi-source training."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence

import torch
from torch.utils.data import Sampler


class DeterministicDistributedWeightedSampler(Sampler[int]):
    """Draw one deterministic global weighted stream and shard it across ranks."""

    def __init__(
        self,
        weights: Sequence[float] | torch.Tensor,
        *,
        num_samples: int,
        seed: int = 0,
        num_replicas: int = 1,
        rank: int = 0,
    ) -> None:
        self.weights = torch.as_tensor(weights, dtype=torch.double, device="cpu")
        if self.weights.ndim != 1 or len(self.weights) == 0:
            raise ValueError("weights must be a non-empty one-dimensional sequence")
        if not torch.isfinite(self.weights).all() or torch.any(self.weights <= 0):
            raise ValueError("weights must be finite and positive")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if num_replicas <= 0 or not 0 <= rank < num_replicas:
            raise ValueError("Invalid distributed sampler rank configuration")
        self.requested_num_samples = int(num_samples)
        self.seed = int(seed)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.num_samples = math.ceil(self.requested_num_samples / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        global_indices = torch.multinomial(
            self.weights,
            self.total_size,
            replacement=True,
            generator=generator,
        )
        return iter(global_indices[self.rank : self.total_size : self.num_replicas].tolist())

    def __len__(self) -> int:
        return self.num_samples
