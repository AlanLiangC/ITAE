"""Timestamp-safe temporal lookup within a nuScenes scene."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class NearestResult(Generic[T]):
    value: T
    error_us: int


class TemporalIndex(Generic[T]):
    """Nearest-neighbor lookup over strictly increasing integer timestamps."""

    def __init__(self, timestamps_us: list[int], values: list[T]) -> None:
        if len(timestamps_us) != len(values) or not timestamps_us:
            raise ValueError("timestamps and values must have the same non-zero length")
        if any(right <= left for left, right in zip(timestamps_us, timestamps_us[1:])):
            raise ValueError("timestamps must be strictly increasing within a scene")
        self.timestamps_us = timestamps_us
        self.values = values

    def nearest(self, target_us: int, max_error_us: int) -> NearestResult[T] | None:
        position = bisect_left(self.timestamps_us, target_us)
        candidates = []
        if position < len(self.timestamps_us):
            candidates.append(position)
        if position > 0:
            candidates.append(position - 1)
        best = min(candidates, key=lambda idx: abs(self.timestamps_us[idx] - target_us))
        error = abs(self.timestamps_us[best] - target_us)
        if error > max_error_us:
            return None
        return NearestResult(self.values[best], error)
