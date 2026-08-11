"""Dataset-neutral SE(2) trajectory resampling utilities."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def wrap_angle(angle: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Wrap radians to ``[-pi, pi)`` without changing the input shape."""
    values = np.asarray(angle, dtype=np.float64)
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def resample_se2_trajectory(
    poses: npt.ArrayLike,
    source_times_s: npt.ArrayLike,
    target_times_s: npt.ArrayLike,
) -> npt.NDArray[np.float32]:
    """Linearly resample local ``[x, y, yaw]`` poses with continuous yaw.

    This matches the pose component of nuPlan's ``InterpolatedTrajectory``:
    translations are interpolated linearly and angular values are unwrapped before
    interpolation. Extrapolation is deliberately rejected.
    """
    source = np.asarray(poses, dtype=np.float64)
    source_times = np.asarray(source_times_s, dtype=np.float64)
    target_times = np.asarray(target_times_s, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"poses must have shape [T,3], got {source.shape}")
    if source_times.ndim != 1 or len(source_times) != len(source):
        raise ValueError("source_times_s must be one-dimensional and aligned with poses")
    if target_times.ndim != 1 or len(target_times) == 0:
        raise ValueError("target_times_s must be a non-empty one-dimensional array")
    if len(source) < 2:
        raise ValueError("At least two source poses are required")
    if not np.isfinite(source).all() or not np.isfinite(source_times).all():
        raise ValueError("Source trajectory contains non-finite values")
    if not np.isfinite(target_times).all():
        raise ValueError("Target timestamps contain non-finite values")
    if np.any(np.diff(source_times) <= 0) or np.any(np.diff(target_times) <= 0):
        raise ValueError("Source and target timestamps must be strictly increasing")
    tolerance = 1e-9
    if (
        target_times[0] < source_times[0] - tolerance
        or target_times[-1] > source_times[-1] + tolerance
    ):
        raise ValueError("Target timestamps require trajectory extrapolation")

    result = np.empty((len(target_times), 3), dtype=np.float64)
    result[:, 0] = np.interp(target_times, source_times, source[:, 0])
    result[:, 1] = np.interp(target_times, source_times, source[:, 1])
    continuous_yaw = np.unwrap(source[:, 2])
    result[:, 2] = wrap_angle(np.interp(target_times, source_times, continuous_yaw))
    return result.astype(np.float32)


def navsim_2hz_to_10hz(
    future_poses: npt.ArrayLike,
    *,
    horizon_s: float = 4.0,
    target_hz: int = 10,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float64]]:
    """Convert NAVSIM's eight 2Hz future poses to forty 10Hz poses."""
    native = np.asarray(future_poses, dtype=np.float64)
    native_count = round(horizon_s * 2)
    target_count = round(horizon_s * target_hz)
    if native.shape != (native_count, 3):
        raise ValueError(
            f"Expected {native_count} NAVSIM 2Hz future poses, got {native.shape}"
        )
    source_times = np.arange(native_count + 1, dtype=np.float64) / 2.0
    source_poses = np.concatenate([np.zeros((1, 3), dtype=np.float64), native], axis=0)
    target_times = np.arange(1, target_count + 1, dtype=np.float64) / float(target_hz)
    return resample_se2_trajectory(source_poses, source_times, target_times), target_times


def dense_trajectory_to_native_rate(
    poses: npt.ArrayLike,
    *,
    source_hz: int = 10,
    native_hz: int = 2,
) -> npt.NDArray[np.float32]:
    """Select native-rate future poses from an aligned dense trajectory.

    Both representations exclude the anchor pose and end at the same horizon. For
    the NAVSIM contract this selects dense indices ``4, 9, ..., 39``.
    """
    values = np.asarray(poses, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("poses must be a finite [T,3] array")
    if source_hz <= 0 or native_hz <= 0 or source_hz % native_hz:
        raise ValueError("source_hz must be an integer multiple of native_hz")
    stride = source_hz // native_hz
    if len(values) == 0 or len(values) % stride:
        raise ValueError("Dense trajectory length must align with the native sampling rate")
    return values[stride - 1 :: stride].astype(np.float32)


def shift_se2_reference_point(
    poses: npt.ArrayLike, offset_m: tuple[float, float]
) -> npt.NDArray[np.float32]:
    """Move relative poses from the native origin to a rigid ego-frame point."""
    values = np.asarray(poses, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("poses must be a finite [T,3] array")
    offset = np.asarray(offset_m, dtype=np.float64)
    if offset.shape != (2,) or not np.isfinite(offset).all():
        raise ValueError("offset_m must contain two finite values")
    if np.all(offset == 0):
        return values.astype(np.float32)
    cosine = np.cos(values[:, 2])
    sine = np.sin(values[:, 2])
    rotated_x = cosine * offset[0] - sine * offset[1]
    rotated_y = sine * offset[0] + cosine * offset[1]
    shifted = values.copy()
    shifted[:, 0] += rotated_x - offset[0]
    shifted[:, 1] += rotated_y - offset[1]
    return shifted.astype(np.float32)
