"""SE(2)/SE(3) geometry with nuScenes coordinate conventions."""

from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor


def quaternion_wxyz_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    """Convert a nuScenes quaternion `[w, x, y, z]` to a 3x3 rotation matrix."""
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError(f"Quaternion must have shape (4,), got {q.shape}")
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("Quaternion has invalid norm")
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def make_transform(rotation_wxyz: Sequence[float], translation_xyz: Sequence[float]) -> np.ndarray:
    """Build an ego-to-global homogeneous transform from quaternion and translation."""
    translation = np.asarray(translation_xyz, dtype=np.float64)
    if translation.shape != (3,):
        raise ValueError(f"Translation must have shape (3,), got {translation.shape}")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion_wxyz_to_matrix(rotation_wxyz)
    transform[:3, 3] = translation
    return transform


def validate_transform(transform: np.ndarray) -> np.ndarray:
    """Validate and return a 4x4 finite rigid transform as float64."""
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"Transform must have shape (4, 4), got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("Transform contains non-finite values")
    if not np.allclose(matrix[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-5):
        raise ValueError("Invalid homogeneous transform last row")
    return matrix


def _rotation_matrix_to_quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    """Convert a proper 3x3 rotation matrix to a normalized quaternion."""
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ]
            )
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ]
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    return quaternion / np.linalg.norm(quaternion)


def interpolate_transform(left: np.ndarray, right: np.ndarray, fraction: float) -> np.ndarray:
    """Interpolate two SE(3) transforms with linear translation and quaternion SLERP."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("Interpolation fraction must be in [0, 1]")
    left = validate_transform(left)
    right = validate_transform(right)
    left_q = _rotation_matrix_to_quaternion_wxyz(left[:3, :3])
    right_q = _rotation_matrix_to_quaternion_wxyz(right[:3, :3])
    dot = float(np.dot(left_q, right_q))
    if dot < 0.0:
        right_q = -right_q
        dot = -dot
    if dot > 0.9995:
        quaternion = left_q + fraction * (right_q - left_q)
        quaternion /= np.linalg.norm(quaternion)
    else:
        angle = math.acos(np.clip(dot, -1.0, 1.0))
        denominator = math.sin(angle)
        quaternion = (
            math.sin((1.0 - fraction) * angle) / denominator * left_q
            + math.sin(fraction * angle) / denominator * right_q
        )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion_wxyz_to_matrix(quaternion)
    transform[:3, 3] = (1.0 - fraction) * left[:3, 3] + fraction * right[:3, 3]
    return transform


def interpolate_pose_at_timestamp(
    target_us: int,
    timestamps_us: Sequence[int],
    poses: Sequence[np.ndarray],
    max_gap_us: int,
) -> tuple[np.ndarray, int] | None:
    """Interpolate an ego pose and return its nearest-source time error in microseconds."""
    if len(timestamps_us) != len(poses) or not timestamps_us:
        raise ValueError("timestamps and poses must have the same non-zero length")
    position = bisect_left(timestamps_us, target_us)
    if position < len(timestamps_us) and timestamps_us[position] == target_us:
        return validate_transform(poses[position]), 0
    if position == 0 or position == len(timestamps_us):
        return None
    left_time = timestamps_us[position - 1]
    right_time = timestamps_us[position]
    gap_us = right_time - left_time
    if gap_us <= 0 or gap_us > max_gap_us:
        return None
    fraction = (target_us - left_time) / gap_us
    pose = interpolate_transform(poses[position - 1], poses[position], fraction)
    return pose, min(target_us - left_time, right_time - target_us)


def poses_to_local_trajectory(
    anchor_ego_to_global: np.ndarray, poses: Sequence[np.ndarray]
) -> np.ndarray:
    """Convert future ego poses to `[x_forward, y_left, yaw_ccw]` in anchor ego frame."""
    anchor = validate_transform(anchor_ego_to_global)
    global_to_anchor = np.linalg.inv(anchor)
    trajectory = []
    for pose in poses:
        relative = global_to_anchor @ validate_transform(pose)
        yaw = math.atan2(relative[1, 0], relative[0, 0])
        trajectory.append([relative[0, 3], relative[1, 3], yaw])
    result = np.asarray(trajectory, dtype=np.float64)
    if len(result):
        result[:, 2] = np.unwrap(result[:, 2])
    return result.astype(np.float32)


def wrap_angle(angle: Tensor) -> Tensor:
    """Wrap radians to `[-pi, pi)` without breaking autograd."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def angular_difference(a: Tensor, b: Tensor) -> Tensor:
    """Return the shortest signed angular difference `a - b` in radians."""
    return wrap_angle(a - b)


def compose_se2(state_xyyaw: np.ndarray, relative_xyyaw: np.ndarray) -> np.ndarray:
    """Compose a global SE(2) state with a relative ego-frame state."""
    x, y, yaw = np.asarray(state_xyyaw, dtype=np.float64)
    dx, dy, dyaw = np.asarray(relative_xyyaw, dtype=np.float64)
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            x + cos_yaw * dx - sin_yaw * dy,
            y + sin_yaw * dx + cos_yaw * dy,
            math.atan2(math.sin(yaw + dyaw), math.cos(yaw + dyaw)),
        ],
        dtype=np.float64,
    )
