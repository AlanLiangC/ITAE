"""TensorBoard-friendly BEV rendering for trajectory reconstructions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor


def trajectory_time_color(fraction: float) -> tuple[int, int, int]:
    """Match the green-to-yellow-to-red trajectory palette used by the video tool."""
    fraction = min(max(float(fraction), 0.0), 1.0)
    return (
        round(30 + 225 * fraction),
        round(245 - 190 * fraction),
        round(110 - 65 * fraction),
    )


def _to_numpy(value: Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _draw_dashed_path(
    draw: ImageDraw.ImageDraw, points: Sequence[tuple[float, float]], fill: tuple[int, int, int]
) -> None:
    for index, (left, right) in enumerate(zip(points, points[1:])):
        if index % 2 == 0:
            draw.line((*left, *right), fill=fill, width=3)


def _draw_time_colored_path(
    draw: ImageDraw.ImageDraw, points: Sequence[tuple[float, float]]
) -> None:
    for index, (left, right) in enumerate(zip(points, points[1:])):
        fraction = index / max(len(points) - 2, 1)
        draw.line((*left, *right), fill=trajectory_time_color(fraction), width=6)
    for index in range(1, len(points), max((len(points) - 1) // 8, 1)):
        x, y = points[index]
        color = trajectory_time_color((index - 1) / max(len(points) - 2, 1))
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)


def render_bev_trajectory_comparison(
    target: Tensor | np.ndarray,
    reconstruction_vis: Tensor | np.ndarray,
    reconstruction_traj: Tensor | np.ndarray,
    future_times: Tensor | np.ndarray,
    sample_token: str = "",
    mask: Tensor | np.ndarray | None = None,
    width: int = 1200,
    height: int = 430,
) -> Tensor:
    """Render GT and both reconstruction branches as a `[3,H,W]` float image.

    All trajectories use the anchor ego convention: x points forward, y points left.
    The three panels share identical metric bounds so geometric errors remain comparable.
    """

    trajectories = [
        _to_numpy(target),
        _to_numpy(reconstruction_vis),
        _to_numpy(reconstruction_traj),
    ]
    times = _to_numpy(future_times).reshape(-1)
    if any(trajectory.ndim != 2 or trajectory.shape[1] < 2 for trajectory in trajectories):
        raise ValueError("Expected each trajectory to have shape [T,3] or [T,>=2]")
    if any(len(trajectory) != len(times) for trajectory in trajectories):
        raise ValueError("Trajectories and future_times must have the same length")
    if mask is not None:
        valid = _to_numpy(mask).astype(bool).reshape(-1)
        if len(valid) != len(times):
            raise ValueError("mask and future_times must have the same length")
        trajectories = [trajectory[valid] for trajectory in trajectories]
        times = times[valid]
    if len(times) == 0:
        raise ValueError("Cannot render an empty trajectory")

    all_xy = np.concatenate([trajectory[:, :2] for trajectory in trajectories], axis=0)
    all_xy = np.vstack([all_xy, np.zeros((1, 2), dtype=np.float32)])
    x_min, y_min = np.min(all_xy, axis=0)
    x_max, y_max = np.max(all_xy, axis=0)
    x_span = max(float(x_max - x_min), 10.0)
    y_span = max(float(y_max - y_min), 10.0)
    x_center = 0.5 * float(x_min + x_max)
    y_center = 0.5 * float(y_min + y_max)
    x_min, x_max = x_center - 0.6 * x_span, x_center + 0.6 * x_span
    y_min, y_max = y_center - 0.6 * y_span, y_center + 0.6 * y_span

    scale = 2
    canvas = Image.new("RGB", (width * scale, height * scale), (15, 18, 24))
    draw = ImageDraw.Draw(canvas)
    try:
        header_font = ImageFont.truetype("DejaVuSans.ttf", 15 * scale)
        title_font = ImageFont.truetype("DejaVuSans.ttf", 13 * scale)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 11 * scale)
    except OSError:
        header_font = ImageFont.load_default(size=15 * scale)
        title_font = ImageFont.load_default(size=13 * scale)
        small_font = ImageFont.load_default(size=11 * scale)
    header_h = 54 * scale
    gap = 14 * scale
    outer = 18 * scale
    panel_w = (width * scale - 2 * outer - 2 * gap) / 3
    panel_top = header_h + 10 * scale
    panel_bottom = height * scale - 22 * scale
    panel_titles = ("GT", "Visual-latent reconstruction", "Trajectory-latent reconstruction")

    draw.text(
        (outer, 12 * scale),
        (
            "BEV trajectory reconstruction | x forward, y left | "
            f"samples={times[0]:.2f}..{times[-1]:.2f}s | sample=...{sample_token[-12:]}"
        ),
        fill=(235, 238, 245),
        font=header_font,
    )

    def project(xy: np.ndarray, panel_left: float) -> list[tuple[float, float]]:
        plot_left = panel_left + 24 * scale
        plot_right = panel_left + panel_w - 18 * scale
        plot_top = panel_top + 30 * scale
        plot_bottom = panel_bottom - 18 * scale
        u = plot_left + (y_max - xy[:, 1]) / (y_max - y_min) * (plot_right - plot_left)
        v = plot_top + (x_max - xy[:, 0]) / (x_max - x_min) * (plot_bottom - plot_top)
        return list(zip(u.astype(float), v.astype(float)))

    endpoints = [trajectory[-1, :2] for trajectory in trajectories]
    for panel_index, title in enumerate(panel_titles):
        left = outer + panel_index * (panel_w + gap)
        right = left + panel_w
        draw.rounded_rectangle(
            (left, panel_top, right, panel_bottom),
            radius=12 * scale,
            fill=(27, 32, 42),
            outline=(67, 76, 92),
            width=2,
        )
        draw.text(
            (left + 14 * scale, panel_top + 8 * scale),
            title,
            fill=(235, 238, 245),
            font=title_font,
        )
        origin = project(np.zeros((1, 2), dtype=np.float32), left)[0]
        draw.line((left + 15 * scale, origin[1], right - 15 * scale, origin[1]), fill=(65, 72, 84))
        draw.line(
            (origin[0], panel_top + 30 * scale, origin[0], panel_bottom - 10 * scale),
            fill=(65, 72, 84),
        )
        draw.text(
            (origin[0] + 4 * scale, panel_top + 31 * scale),
            "x+",
            fill=(135, 143, 158),
            font=small_font,
        )
        draw.text(
            (left + 17 * scale, origin[1] - 16 * scale),
            "y+",
            fill=(135, 143, 158),
            font=small_font,
        )
        target_points = [origin, *project(trajectories[0][:, :2], left)]
        if panel_index:
            _draw_dashed_path(draw, target_points, fill=(170, 175, 185))
        prediction = trajectories[panel_index]
        prediction_points = [origin, *project(prediction[:, :2], left)]
        _draw_time_colored_path(draw, prediction_points)
        endpoint_error = float(np.linalg.norm(endpoints[panel_index] - endpoints[0]))
        draw.text(
            (left + 14 * scale, panel_bottom - 17 * scale),
            f"FDE={endpoint_error:.2f}m" if panel_index else "time: green -> yellow -> red",
            fill=(205, 210, 220),
            font=small_font,
        )

    canvas = canvas.resize((width, height), Image.Resampling.LANCZOS)
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()
