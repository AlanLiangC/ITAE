"""NAVSIM trajectory videos used by SUV-ITAE validation."""

from __future__ import annotations

import pickle
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from pyquaternion import Quaternion

GT_COLOR = (55, 220, 115)
PRED_COLOR = (255, 145, 45)


@lru_cache(maxsize=8)
def _load_log_frames(log_path: str) -> dict[str, dict]:
    with Path(log_path).open("rb") as handle:
        frames = pickle.load(handle)  # noqa: S301 - trusted local NAVSIM data
    if not isinstance(frames, list):
        raise ValueError(f"Expected a list of frames in {log_path}")
    return {str(frame["token"]): frame for frame in frames}


def load_navsim_frame(
    data_root: str | Path,
    split: str,
    group_token: str,
    sample_token: str,
) -> dict:
    """Load one raw NAVSIM frame, including its CAM_F0 calibration."""
    log_name = group_token.removeprefix("navsim:")
    raw_token = sample_token.removeprefix("navsim:")
    log_path = Path(data_root) / "navsim_logs" / split / f"{log_name}.pkl"
    if not log_path.is_file():
        raise FileNotFoundError(f"NAVSIM log is missing: {log_path}")
    frames = _load_log_frames(str(log_path.resolve()))
    if raw_token not in frames:
        raise KeyError(f"Sample {raw_token} is absent from {log_path}")
    return frames[raw_token]


def project_ego_ground_to_camera(
    trajectory: np.ndarray, raw_frame: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Project rear-axle ego-frame ground points into NAVSIM CAM_F0."""
    camera = raw_frame["cams"]["CAM_F0"]
    sensor_to_lidar_rotation = np.asarray(
        camera["sensor2lidar_rotation"], dtype=np.float64
    )
    sensor_to_lidar_translation = np.asarray(
        camera["sensor2lidar_translation"], dtype=np.float64
    )
    intrinsic = np.asarray(camera["cam_intrinsic"], dtype=np.float64)
    lidar_to_ego = np.eye(4, dtype=np.float64)
    lidar_to_ego[:3, :3] = Quaternion(
        raw_frame["lidar2ego_rotation"]
    ).rotation_matrix
    lidar_to_ego[:3, 3] = np.asarray(
        raw_frame["lidar2ego_translation"], dtype=np.float64
    )

    points_ego = np.column_stack(
        [trajectory[:, :2], np.zeros(len(trajectory)), np.ones(len(trajectory))]
    )
    points_lidar = (np.linalg.inv(lidar_to_ego) @ points_ego.T).T[:, :3]
    lidar_to_camera_rotation = np.linalg.inv(sensor_to_lidar_rotation)
    lidar_to_camera_translation = (
        sensor_to_lidar_translation @ lidar_to_camera_rotation.T
    )
    points_camera = (
        lidar_to_camera_rotation @ points_lidar.T
    ).T - lidar_to_camera_translation
    homogeneous = (intrinsic @ points_camera.T).T
    depth = points_camera[:, 2]
    pixels = np.full((len(trajectory), 2), np.nan, dtype=np.float64)
    valid = depth > 1e-3
    pixels[valid] = homogeneous[valid, :2] / depth[valid, None]
    return pixels, depth


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _draw_path(
    draw: ImageDraw.ImageDraw,
    points: np.ndarray,
    visible: np.ndarray,
    end: int,
    color: tuple[int, int, int, int],
    *,
    width: int,
    dashed: bool = False,
) -> None:
    for index in range(1, min(end + 1, len(points))):
        if visible[index - 1] and visible[index] and (not dashed or index % 2):
            draw.line(
                (*points[index - 1].tolist(), *points[index].tolist()),
                fill=color,
                width=width,
            )
    if 0 <= end < len(points) and visible[end]:
        x, y = points[end]
        radius = max(width, 4)
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius), fill=color
        )


def _bev_projection(
    gt: np.ndarray,
    prediction: np.ndarray,
    box: tuple[int, int, int, int],
    limits: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    left, top, right, bottom = box
    x_min, x_limit, y_limit = limits

    def project(xy: np.ndarray) -> np.ndarray:
        u = left + (y_limit - xy[:, 1]) / (2 * y_limit) * (right - left)
        v = top + (x_limit - xy[:, 0]) / (x_limit - x_min) * (bottom - top)
        return np.column_stack((u, v))

    origin = tuple(project(np.zeros((1, 2)))[0].tolist())
    return project(gt[:, :2]), project(prediction[:, :2]), origin


def render_trajectory_clip(
    image_paths: list[str | Path],
    raw_frames: list[dict],
    gt: np.ndarray,
    prediction: np.ndarray,
    future_times_s: np.ndarray,
    *,
    sample_tokens: list[str],
    anchor_timestamps_us: list[int],
    clip_label: str,
    global_step: int,
    frame_size: tuple[int, int] = (432, 768),
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Render a real clip where the model predicts independently at every frame."""
    gt = np.asarray(gt, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    times = np.asarray(future_times_s, dtype=np.float64)
    clip_frames = len(image_paths)
    if (
        gt.shape != prediction.shape
        or gt.ndim != 3
        or gt.shape[0] != clip_frames
        or gt.shape[2] < 3
    ):
        raise ValueError(
            f"Expected matching [clip,T,3] trajectories, got {gt.shape}/{prediction.shape}"
        )
    if times.ndim == 1:
        times = np.repeat(times[None], clip_frames, axis=0)
    if times.shape != gt.shape[:2]:
        raise ValueError("future_times_s must align with [clip,T] trajectories")
    if not (
        len(raw_frames)
        == len(sample_tokens)
        == len(anchor_timestamps_us)
        == clip_frames
    ):
        raise ValueError("Clip images, raw frames, tokens and timestamps must align")
    height, width = map(int, frame_size)
    if min(height, width) <= 0 or height % 2 or width % 2:
        raise ValueError("Video height and width must be positive even integers")

    camera_width = int(width * 0.64)
    header_height = 58
    bev_box = (camera_width + 24, header_height + 45, width - 22, height - 40)
    all_xy = np.vstack(
        (np.zeros((1, 2)), gt[..., :2].reshape(-1, 2), prediction[..., :2].reshape(-1, 2))
    )
    x_min = min(float(np.min(all_xy[:, 0])), -2.0)
    x_max = max(float(np.max(all_xy[:, 0])), 8.0)
    x_padding = max((x_max - x_min) * 0.08, 1.0)
    bev_limits = (
        x_min - x_padding,
        x_max + x_padding,
        max(float(np.max(np.abs(all_xy[:, 1]))) * 1.15, 5.0),
    )

    xy_error = np.linalg.norm(prediction[..., :2] - gt[..., :2], axis=2)
    yaw_error = np.abs(
        (prediction[..., 2] - gt[..., 2] + np.pi) % (2 * np.pi) - np.pi
    )
    metrics = {
        "mean_ade_m": float(xy_error.mean()),
        "mean_fde_m": float(xy_error[:, -1].mean()),
        "max_fde_m": float(xy_error[:, -1].max()),
        "mean_yaw_mae_deg": float(np.degrees(yaw_error.mean())),
    }

    rendered: list[np.ndarray] = []
    frame_reports: list[dict[str, float | int | str]] = []
    first_timestamp = int(anchor_timestamps_us[0])
    for frame_index in range(clip_frames):
        with Image.open(image_paths[frame_index]) as source:
            camera = source.convert("RGB")
        raw_width, raw_height = camera.size
        region_height = height - header_height
        scale = min(camera_width / raw_width, region_height / raw_height)
        display_width = int(round(raw_width * scale))
        display_height = int(round(raw_height * scale))
        camera_left = (camera_width - display_width) // 2
        camera_top = header_height + (region_height - display_height) // 2
        camera_resized = camera.resize(
            (display_width, display_height), Image.Resampling.BILINEAR
        )
        frame_gt = gt[frame_index]
        frame_prediction = prediction[frame_index]
        gt_pixels, gt_depth = project_ego_ground_to_camera(
            frame_gt, raw_frames[frame_index]
        )
        pred_pixels, pred_depth = project_ego_ground_to_camera(
            frame_prediction, raw_frames[frame_index]
        )
        gt_visible = (
            (gt_depth > 0.1)
            & np.isfinite(gt_pixels).all(axis=1)
            & (gt_pixels[:, 0] >= 0)
            & (gt_pixels[:, 0] < raw_width)
            & (gt_pixels[:, 1] >= 0)
            & (gt_pixels[:, 1] < raw_height)
        )
        pred_visible = (
            (pred_depth > 0.1)
            & np.isfinite(pred_pixels).all(axis=1)
            & (pred_pixels[:, 0] >= 0)
            & (pred_pixels[:, 0] < raw_width)
            & (pred_pixels[:, 1] >= 0)
            & (pred_pixels[:, 1] < raw_height)
        )
        pixel_offset = np.asarray([camera_left, camera_top], dtype=np.float64)
        gt_pixels = gt_pixels * scale + pixel_offset
        pred_pixels = pred_pixels * scale + pixel_offset
        gt_bev, pred_bev, origin = _bev_projection(
            frame_gt, frame_prediction, bev_box, bev_limits
        )

        canvas = Image.new("RGB", (width, height), (18, 21, 28))
        canvas.paste(camera_resized, (camera_left, camera_top))
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        _draw_path(
            draw,
            gt_pixels,
            gt_visible,
            len(frame_gt) - 1,
            (*GT_COLOR, 255),
            width=6,
            dashed=True,
        )
        _draw_path(
            draw,
            pred_pixels,
            pred_visible,
            len(frame_prediction) - 1,
            (*PRED_COLOR, 255),
            width=5,
        )
        for point_index in range(9, len(frame_prediction), 10):
            if pred_visible[point_index]:
                x, y = pred_pixels[point_index]
                draw.ellipse(
                    (x - 3, y - 3, x + 3, y + 3), fill=(*PRED_COLOR, 255)
                )

        left, top, right, bottom = bev_box
        draw.rounded_rectangle(
            (camera_width + 10, header_height + 8, width - 10, height - 12),
            radius=10,
            fill=(12, 15, 21, 245),
            outline=(75, 84, 100, 255),
            width=2,
        )
        draw.line((left, origin[1], right, origin[1]), fill=(62, 68, 80, 255), width=1)
        draw.line((origin[0], top, origin[0], bottom), fill=(62, 68, 80, 255), width=1)
        gt_bev = np.vstack((np.asarray(origin), gt_bev))
        pred_bev = np.vstack((np.asarray(origin), pred_bev))
        bev_visible = np.ones(len(gt_bev), dtype=bool)
        _draw_path(
            draw,
            gt_bev,
            bev_visible,
            len(gt_bev) - 1,
            (*GT_COLOR, 255),
            width=5,
            dashed=True,
        )
        _draw_path(
            draw,
            pred_bev,
            bev_visible,
            len(pred_bev) - 1,
            (*PRED_COLOR, 255),
            width=5,
        )
        draw.ellipse(
            (origin[0] - 4, origin[1] - 4, origin[0] + 4, origin[1] + 4),
            fill=(75, 175, 255, 255),
        )

        draw.rectangle((0, 0, width, header_height), fill=(8, 10, 15, 240))
        draw.text(
            (14, 7),
            (
                f"SUV-ITAE step {global_step:,} | {clip_label} clip "
                f"t={(int(anchor_timestamps_us[frame_index]) - first_timestamp) / 1e6:.1f}s"
            ),
            fill="white",
            font=_font(20),
        )
        frame_ade = float(xy_error[frame_index].mean())
        frame_fde = float(xy_error[frame_index, -1])
        draw.text((14, 33), "GT", fill=(*GT_COLOR, 255), font=_font(13))
        draw.text(
            (38, 33), "prediction", fill=(*PRED_COLOR, 255), font=_font(13)
        )
        draw.text(
            (115, 33),
            (
                f"| frame {frame_index + 1}/{clip_frames}  "
                f"ADE={frame_ade:.2f}m  FDE={frame_fde:.2f}m"
            ),
            fill=(215, 221, 230, 255),
            font=_font(13),
        )
        draw.text(
            (camera_width + 23, header_height + 17),
            "BEV: x forward / y left",
            fill="white",
            font=_font(15),
        )
        draw.text(
            (camera_width + 23, height - 33),
            sample_tokens[frame_index].removeprefix("navsim:"),
            fill=(175, 184, 198, 255),
            font=_font(11),
        )
        frame = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
        rendered.append(np.asarray(frame, dtype=np.uint8))
        frame_reports.append(
            {
                "frame_index": frame_index,
                "sample_token": sample_tokens[frame_index],
                "anchor_timestamp_us": int(anchor_timestamps_us[frame_index]),
                "ade_m": frame_ade,
                "fde_m": frame_fde,
                "yaw_mae_deg": float(np.degrees(yaw_error[frame_index].mean())),
                "gt_visible_points": int(gt_visible.sum()),
                "prediction_visible_points": int(pred_visible.sum()),
            }
        )

    array = np.stack(rendered, axis=0)
    video = torch.from_numpy(array.copy()).permute(0, 3, 1, 2)
    metrics["frames"] = frame_reports
    metrics["clip_duration_s"] = (
        int(anchor_timestamps_us[-1]) - first_timestamp
    ) / 1e6
    return video, metrics


def encode_mp4(video: torch.Tensor, output: str | Path, fps: float) -> None:
    """Encode a uint8 ``[T,C,H,W]`` tensor through system ffmpeg."""
    if video.ndim != 4 or video.shape[1] != 3 or video.dtype != torch.uint8:
        raise ValueError("video must be a uint8 [T,3,H,W] tensor")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _, _, height, width = video.shape
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        try:
            from imageio_ffmpeg import get_ffmpeg_exe

            ffmpeg = get_ffmpeg_exe()
        except ImportError as error:
            raise RuntimeError(
                "MP4 export needs ffmpeg or the imageio-ffmpeg package"
            ) from error
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        str(float(fps)),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None and process.stderr is not None
    frames = video.permute(0, 2, 3, 1).contiguous().numpy()
    try:
        process.stdin.write(frames.tobytes())
        process.stdin.close()
        error = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait()
    except BrokenPipeError:
        error = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait()
    if return_code:
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}: {error}")
