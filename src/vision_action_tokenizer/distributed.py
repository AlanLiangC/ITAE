"""Small DDP helpers with no dependency on a training framework."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def initialize_distributed() -> DistributedContext:
    """Initialize NCCL/Gloo from torchrun environment variables when present."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend)
    return DistributedContext(rank, local_rank, world_size, device)


def reduce_metrics(metrics: dict[str, Tensor], world_size: int) -> dict[str, Tensor]:
    """Average scalar metric tensors across workers."""
    if world_size == 1:
        return metrics
    reduced = {}
    for key, value in metrics.items():
        value = value.detach().clone()
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        reduced[key] = value / world_size
    return reduced


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()

