"""TensorBoard-friendly BEV rendering for trajectory reconstructions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
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
    for index, (left, right) in enumerate(zip(points, points[1:], strict=False)):
        if index % 2 == 0:
            draw.line((*left, *right), fill=fill, width=3)


def _draw_time_colored_path(
    draw: ImageDraw.ImageDraw, points: Sequence[tuple[float, float]]
) -> None:
    for index, (left, right) in enumerate(zip(points, points[1:], strict=False)):
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
        return list(zip(u.astype(float), v.astype(float), strict=True))

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


def _camera_frame_to_image(frame: Tensor | np.ndarray) -> Image.Image:
    array = _to_numpy(frame)
    if array.ndim != 3:
        raise ValueError(f"Expected camera frame [3,H,W] or [H,W,3], got {array.shape}")
    if array.shape[0] in {1, 3, 4}:
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] != 3:
        raise ValueError(f"Expected an RGB camera frame, got {array.shape}")
    # Accept both VGGT RGB [0,1] and legacy normalized tensors for general use.
    if float(array.min()) < -0.01:
        array = array * 0.5 + 0.5
    array = np.clip(array, 0.0, 1.0)
    return Image.fromarray(np.round(array * 255).astype(np.uint8), mode="RGB")


def render_evaluation_diagnostic(
    target: Tensor | np.ndarray,
    reconstruction_vis: Tensor | np.ndarray,
    reconstruction_traj: Tensor | np.ndarray,
    future_times: Tensor | np.ndarray,
    camera_images: Tensor | np.ndarray | None,
    frame_times: Tensor | np.ndarray | None,
    sample_token: str = "",
    mask: Tensor | np.ndarray | None = None,
    width: int = 1200,
    height: int = 900,
) -> Tensor:
    """Render a TensorBoard-ready 2x2 camera/trajectory diagnostic page.

    The top-left panel contains every CAM_FRONT input frame. The remaining panels
    show GT, visual-latent reconstruction and trajectory-latent reconstruction in
    BEV with shared metric bounds.
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

    frames = None if camera_images is None else _to_numpy(camera_images)
    frame_time_values = None if frame_times is None else _to_numpy(frame_times).reshape(-1)
    if frames is not None:
        if frames.ndim != 4:
            raise ValueError(f"Expected camera_images [F,3,H,W], got {frames.shape}")
        if frame_time_values is not None and len(frame_time_values) != len(frames):
            raise ValueError("camera_images and frame_times must have the same frame count")

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
        small_font = ImageFont.truetype("DejaVuSans.ttf", 10 * scale)
    except OSError:
        header_font = ImageFont.load_default(size=15 * scale)
        title_font = ImageFont.load_default(size=13 * scale)
        small_font = ImageFont.load_default(size=10 * scale)

    outer = 18 * scale
    gap = 14 * scale
    header_h = 52 * scale
    panel_w = (width * scale - 2 * outer - gap) / 2
    panel_h = (height * scale - header_h - outer - gap) / 2
    panel_positions = (
        (outer, header_h),
        (outer + panel_w + gap, header_h),
        (outer, header_h + panel_h + gap),
        (outer + panel_w + gap, header_h + panel_h + gap),
    )
    draw.text(
        (outer, 12 * scale),
        (
            "Evaluation diagnostic | CAM_FRONT + shared-scale BEV | "
            f"trajectory={times[0]:.2f}..{times[-1]:.2f}s | sample=...{sample_token[-12:]}"
        ),
        fill=(235, 238, 245),
        font=header_font,
    )

    def panel_box(position: tuple[float, float], title: str) -> tuple[float, float, float, float]:
        left, top = position
        right, bottom = left + panel_w, top + panel_h
        draw.rounded_rectangle(
            (left, top, right, bottom),
            radius=12 * scale,
            fill=(27, 32, 42),
            outline=(67, 76, 92),
            width=2,
        )
        draw.text(
            (left + 14 * scale, top + 8 * scale),
            title,
            fill=(235, 238, 245),
            font=title_font,
        )
        return left, top, right, bottom

    camera_left, camera_top, camera_right, camera_bottom = panel_box(
        panel_positions[0], "CAM_FRONT input window"
    )
    if frames is None:
        draw.text(
            (camera_left + 18 * scale, camera_top + 52 * scale),
            "Input images unavailable (enable evaluation_visualization_include_images).",
            fill=(190, 196, 208),
            font=small_font,
        )
    else:
        columns = 3
        rows = 2
        inner_gap = 8 * scale
        content_left = camera_left + 12 * scale
        content_top = camera_top + 38 * scale
        content_width = camera_right - camera_left - 24 * scale
        content_height = camera_bottom - content_top - 10 * scale
        cell_width = (content_width - (columns - 1) * inner_gap) / columns
        cell_height = (content_height - (rows - 1) * inner_gap) / rows
        for frame_index, frame in enumerate(frames[: columns * rows]):
            row, column = divmod(frame_index, columns)
            cell_left = content_left + column * (cell_width + inner_gap)
            cell_top = content_top + row * (cell_height + inner_gap)
            label_height = 17 * scale
            thumbnail = ImageOps.contain(
                _camera_frame_to_image(frame),
                (max(1, round(cell_width)), max(1, round(cell_height - label_height))),
                Image.Resampling.LANCZOS,
            )
            paste_x = round(cell_left + (cell_width - thumbnail.width) / 2)
            paste_y = round(cell_top + label_height)
            canvas.paste(thumbnail, (paste_x, paste_y))
            relative_time = (
                float(frame_time_values[frame_index])
                if frame_time_values is not None
                else float(frame_index)
            )
            draw.text(
                (cell_left + 2 * scale, cell_top),
                f"frame {frame_index} | t={relative_time:.2f}s",
                fill=(205, 210, 220),
                font=small_font,
            )

    panel_titles = (
        "GT trajectory",
        "Visual-latent reconstruction",
        "Trajectory-latent reconstruction",
    )
    endpoints = [trajectory[-1, :2] for trajectory in trajectories]
    ade_errors = [
        float(np.linalg.norm(trajectory[:, :2] - trajectories[0][:, :2], axis=-1).mean())
        for trajectory in trajectories
    ]
    for trajectory_index, (position, title) in enumerate(
        zip(panel_positions[1:], panel_titles, strict=True)
    ):
        left, top, right, bottom = panel_box(position, title)
        plot_left = left + 26 * scale
        plot_right = right - 20 * scale
        plot_top = top + 38 * scale
        plot_bottom = bottom - 25 * scale

        def project(
            xy: np.ndarray,
            left_bound: float = plot_left,
            right_bound: float = plot_right,
            top_bound: float = plot_top,
            bottom_bound: float = plot_bottom,
        ) -> list[tuple[float, float]]:
            u = left_bound + (y_max - xy[:, 1]) / (y_max - y_min) * (
                right_bound - left_bound
            )
            v = top_bound + (x_max - xy[:, 0]) / (x_max - x_min) * (
                bottom_bound - top_bound
            )
            return list(zip(u.astype(float), v.astype(float), strict=True))

        origin = project(np.zeros((1, 2), dtype=np.float32))[0]
        draw.line((plot_left, origin[1], plot_right, origin[1]), fill=(65, 72, 84), width=2)
        draw.line((origin[0], plot_top, origin[0], plot_bottom), fill=(65, 72, 84), width=2)
        draw.text((origin[0] + 4 * scale, plot_top), "x+", fill=(135, 143, 158), font=small_font)
        draw.text((plot_left, origin[1] - 15 * scale), "y+", fill=(135, 143, 158), font=small_font)
        target_points = [origin, *project(trajectories[0][:, :2])]
        if trajectory_index:
            _draw_dashed_path(draw, target_points, fill=(170, 175, 185))
        prediction_points = [origin, *project(trajectories[trajectory_index][:, :2])]
        _draw_time_colored_path(draw, prediction_points)
        endpoint_error = float(np.linalg.norm(endpoints[trajectory_index] - endpoints[0]))
        footer = (
            "time: green -> yellow -> red"
            if trajectory_index == 0
            else (
                f"ADE={ade_errors[trajectory_index]:.2f}m | "
                f"FDE={endpoint_error:.2f}m | dashed=GT"
            )
        )
        draw.text(
            (left + 14 * scale, bottom - 19 * scale),
            footer,
            fill=(205, 210, 220),
            font=small_font,
        )

    canvas = canvas.resize((width, height), Image.Resampling.LANCZOS)
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def render_planner_diagnostic(
    current_image: Tensor | np.ndarray,
    target: Tensor | np.ndarray,
    prediction: Tensor | np.ndarray,
    future_times: Tensor | np.ndarray,
    sample_token: str = "",
    width: int = 1200,
    height: int = 800,
    frame_times: Tensor | np.ndarray | None = None,
    ego_motion: Tensor | np.ndarray | None = None,
) -> Tensor:
    """Render condition RGB frames, GT, prediction and overlay in a 2x2 page."""
    frame_array = _to_numpy(current_image)
    if frame_array.ndim == 3:
        frame_array = frame_array[None]
    if frame_array.ndim != 4:
        raise ValueError("Planner condition images must have shape [F,3,H,W]")
    images = [_camera_frame_to_image(frame) for frame in frame_array]
    time_values = (
        np.arange(len(images), dtype=np.float32)
        if frame_times is None
        else _to_numpy(frame_times).reshape(-1)
    )
    if len(time_values) != len(images):
        raise ValueError("frame_times must align with planner condition images")
    ego_values = None if ego_motion is None else _to_numpy(ego_motion)
    if ego_values is not None and ego_values.shape != (len(images), 6):
        raise ValueError("ego_motion must have shape [condition_frames,6]")
    target_np = _to_numpy(target)
    prediction_np = _to_numpy(prediction)
    times = _to_numpy(future_times).reshape(-1)
    if target_np.shape != prediction_np.shape or target_np.shape != (len(times), 3):
        raise ValueError("Planner visualization expects aligned [T,3] trajectories")

    scale = 2
    canvas = Image.new("RGB", (width * scale, height * scale), (15, 18, 24))
    draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.truetype("DejaVuSans.ttf", 16 * scale)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 11 * scale)
    except OSError:
        title_font = ImageFont.load_default(size=16 * scale)
        small_font = ImageFont.load_default(size=11 * scale)
    margin = 18 * scale
    gap = 14 * scale
    header = 42 * scale
    panel_w = (width * scale - 2 * margin - gap) / 2
    panel_h = (height * scale - header - margin - gap) / 2
    panels = [
        (margin, header),
        (margin + panel_w + gap, header),
        (margin, header + panel_h + gap),
        (margin + panel_w + gap, header + panel_h + gap),
    ]
    draw.text(
        (margin, 10 * scale),
        f"Flow planner evaluation | sample=...{sample_token[-16:]}",
        fill=(235, 238, 245),
        font=title_font,
    )
    for left, top in panels:
        draw.rounded_rectangle(
            (left, top, left + panel_w, top + panel_h),
            radius=10 * scale,
            fill=(27, 32, 42),
            outline=(67, 76, 92),
            width=2,
        )

    draw.text(
        (panels[0][0] + 12 * scale, panels[0][1] + 8 * scale),
        f"CAM_FRONT condition ({len(images)} frame{'s' if len(images) != 1 else ''})",
        fill=(235, 238, 245),
        font=small_font,
    )
    inner_gap = 7 * scale
    content_width = panel_w - 24 * scale
    cell_width = (content_width - (len(images) - 1) * inner_gap) / len(images)
    cell_height = panel_h - 53 * scale
    for frame_index, (image, relative_time) in enumerate(
        zip(images, time_values, strict=True)
    ):
        cell_left = panels[0][0] + 12 * scale + frame_index * (
            cell_width + inner_gap
        )
        thumbnail = ImageOps.contain(
            image,
            (max(1, round(cell_width)), max(1, round(cell_height - 18 * scale))),
            Image.Resampling.LANCZOS,
        )
        paste_x = round(cell_left + (cell_width - thumbnail.width) / 2)
        paste_y = round(panels[0][1] + 48 * scale)
        canvas.paste(thumbnail, (paste_x, paste_y))
        label = f"t={float(relative_time):+.2f}s"
        if ego_values is not None:
            speed = float(np.linalg.norm(ego_values[frame_index, 3:5]))
            label += f" | v{speed:.1f} r{ego_values[frame_index, 5]:+.2f}"
        draw.text(
            (cell_left, panels[0][1] + 31 * scale),
            label,
            fill=(205, 210, 220),
            font=small_font,
        )

    all_xy = np.vstack([np.zeros((1, 2)), target_np[:, :2], prediction_np[:, :2]])
    x_min, y_min = all_xy.min(axis=0)
    x_max, y_max = all_xy.max(axis=0)
    x_span = max(float(x_max - x_min), 10.0)
    y_span = max(float(y_max - y_min), 10.0)
    x_center, y_center = 0.5 * (x_min + x_max), 0.5 * (y_min + y_max)
    x_min, x_max = x_center - 0.6 * x_span, x_center + 0.6 * x_span
    y_min, y_max = y_center - 0.6 * y_span, y_center + 0.6 * y_span

    def project(xy: np.ndarray, panel: tuple[float, float]) -> list[tuple[float, float]]:
        left, top = panel
        plot_left, plot_right = left + 28 * scale, left + panel_w - 20 * scale
        plot_top, plot_bottom = top + 38 * scale, top + panel_h - 20 * scale
        u = plot_left + (y_max - xy[:, 1]) / (y_max - y_min) * (plot_right - plot_left)
        v = plot_top + (x_max - xy[:, 0]) / (x_max - x_min) * (plot_bottom - plot_top)
        return list(zip(u.astype(float), v.astype(float), strict=True))

    entries = [
        ("Ground truth", target_np, False),
        ("5-NFE prediction", prediction_np, False),
        ("Overlay: dashed GT + colored prediction", prediction_np, True),
    ]
    origin_np = np.zeros((1, 2), dtype=np.float32)
    for panel, (title, trajectory, overlay) in zip(panels[1:], entries, strict=True):
        draw.text(
            (panel[0] + 12 * scale, panel[1] + 8 * scale),
            title,
            fill=(235, 238, 245),
            font=small_font,
        )
        origin = project(origin_np, panel)[0]
        draw.line(
            (panel[0] + 18 * scale, origin[1], panel[0] + panel_w - 18 * scale, origin[1]),
            fill=(65, 72, 84),
        )
        if overlay:
            _draw_dashed_path(
                draw, [origin, *project(target_np[:, :2], panel)], (205, 210, 220)
            )
        _draw_time_colored_path(
            draw, [origin, *project(trajectory[:, :2], panel)]
        )
    fde = float(np.linalg.norm(prediction_np[-1, :2] - target_np[-1, :2]))
    draw.text(
        (panels[3][0] + 12 * scale, panels[3][1] + panel_h - 18 * scale),
        f"FDE={fde:.2f}m | green -> yellow -> red",
        fill=(205, 210, 220),
        font=small_font,
    )
    canvas = canvas.resize((width, height), Image.Resampling.LANCZOS)
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def render_vggt_evaluation_diagnostic(
    target: Tensor | np.ndarray,
    reconstruction: Tensor | np.ndarray,
    predicted_increments: Tensor | np.ndarray,
    future_times: Tensor | np.ndarray,
    camera_images: Tensor | np.ndarray | None,
    frame_times: Tensor | np.ndarray | None,
    sample_token: str = "",
    mask: Tensor | np.ndarray | None = None,
    width: int = 1200,
    height: int = 900,
) -> Tensor:
    """Render CAM_FRONT, GT, VGGT reconstruction and temporal error in a 2x2 page."""
    base = render_evaluation_diagnostic(
        target,
        reconstruction,
        reconstruction,
        future_times,
        camera_images,
        frame_times,
        sample_token=sample_token,
        mask=mask,
        width=width,
        height=height,
    )
    canvas = Image.fromarray(
        np.round(base.permute(1, 2, 0).numpy() * 255).astype(np.uint8), mode="RGB"
    )
    draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.truetype("DejaVuSans.ttf", 13)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 10)
    except OSError:
        title_font = ImageFont.load_default(size=13)
        small_font = ImageFont.load_default(size=10)

    left, top, right, bottom = 607, 473, 1182, 882
    draw.rounded_rectangle(
        (left, top, right, bottom),
        radius=12,
        fill=(27, 32, 42),
        outline=(67, 76, 92),
        width=1,
    )
    draw.text(
        (left + 14, top + 8),
        "Error / SE(2) increment diagnostics",
        fill=(235, 238, 245),
        font=title_font,
    )

    target_np = _to_numpy(target)
    reconstruction_np = _to_numpy(reconstruction)
    increments_np = _to_numpy(predicted_increments)
    times_np = _to_numpy(future_times).reshape(-1)
    if mask is not None:
        valid = _to_numpy(mask).astype(bool).reshape(-1)
        target_np = target_np[valid]
        reconstruction_np = reconstruction_np[valid]
        increments_np = increments_np[valid]
        times_np = times_np[valid]
    position_error = np.linalg.norm(
        reconstruction_np[:, :2] - target_np[:, :2], axis=-1
    )
    origin = np.zeros((1, 3), dtype=np.float32)
    previous = np.concatenate([origin, target_np[:-1]], axis=0)
    global_delta = target_np[:, :2] - previous[:, :2]
    cosine = np.cos(previous[:, 2])
    sine = np.sin(previous[:, 2])
    target_body_xy = np.stack(
        [
            cosine * global_delta[:, 0] + sine * global_delta[:, 1],
            -sine * global_delta[:, 0] + cosine * global_delta[:, 1],
        ],
        axis=-1,
    )
    increment_error = np.linalg.norm(increments_np[:, :2] - target_body_xy, axis=-1)
    ade = float(position_error.mean())
    fde = float(position_error[-1])
    increment_mae = float(increment_error.mean())
    draw.text(
        (left + 14, top + 31),
        f"ADE={ade:.2f}m | FDE={fde:.2f}m | increment MAE={increment_mae:.3f}m",
        fill=(205, 210, 220),
        font=small_font,
    )

    def draw_curve(
        values: np.ndarray,
        bounds: tuple[float, float, float, float],
        color: tuple[int, int, int],
        label: str,
    ) -> None:
        plot_left, plot_top, plot_right, plot_bottom = bounds
        maximum = max(float(values.max()), 1e-3)
        draw.rectangle(bounds, outline=(65, 72, 84), width=1)
        points = []
        for index, value in enumerate(values):
            x = plot_left + index / max(len(values) - 1, 1) * (plot_right - plot_left)
            y = plot_bottom - float(value) / maximum * (plot_bottom - plot_top)
            points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill=color, width=2)
        draw.text(
            (plot_left + 4, plot_top + 3),
            f"{label} | max={maximum:.3f}",
            fill=color,
            font=small_font,
        )

    draw_curve(
        position_error,
        (left + 18, top + 61, right - 18, top + 205),
        (255, 120, 80),
        "position error (m)",
    )
    draw_curve(
        increment_error,
        (left + 18, top + 222, right - 18, bottom - 22),
        (80, 210, 160),
        "body increment error (m)",
    )
    draw.text(
        (right - 105, bottom - 17),
        f"t={times_np[-1]:.2f}s",
        fill=(150, 157, 170),
        font=small_font,
    )
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()
