from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import save_file
from vggt_omega.utils.load_fn import load_and_preprocess_images

from vision_action_tokenizer.data.dataset import (
    ActionWindowDataset,
    CachedVGGTOmegaFeatureDataset,
    MultiSourceActionDataset,
    NuScenesWindowDataset,
    VGGTOmegaResize,
)
from vision_action_tokenizer.data.geometry import (
    interpolate_pose_at_timestamp,
    make_transform,
    poses_to_local_trajectory,
)
from vision_action_tokenizer.data.lidar import LidarPoseRecord
from vision_action_tokenizer.data.manifest import (
    ManifestBuilder,
    WindowRecord,
    manifest_scene_tokens,
    save_manifest,
)
from vision_action_tokenizer.data.sampler import DeterministicDistributedWeightedSampler
from vision_action_tokenizer.data.schema import FrameRecord, InfoSchemaAdapter
from vision_action_tokenizer.data.temporal import TemporalIndex


def test_pose_to_local_trajectory() -> None:
    anchor = make_transform([1, 0, 0, 0], [10, 20, 0])
    future = make_transform([1, 0, 0, 0], [13, 22, 0])
    trajectory = poses_to_local_trajectory(anchor, [future])
    np.testing.assert_allclose(trajectory[0], [3, 2, 0], atol=1e-6)


def test_vggt_transform_matches_official_max_size(tmp_path: Path) -> None:
    image_path = tmp_path / "front.png"
    array = np.arange(90 * 160 * 3, dtype=np.uint8).reshape(90, 160, 3)
    Image.fromarray(array).save(image_path)
    official = load_and_preprocess_images(
        [str(image_path)], mode="max_size", image_resolution=512, patch_size=16
    )[0]
    with Image.open(image_path) as image:
        project = VGGTOmegaResize(512, "max_size", 16)(image)
    assert project.shape == (3, 288, 512)
    assert torch.allclose(project, official)


def test_action_window_image_lru_avoids_repeated_decode(tmp_path: Path) -> None:
    image_path = tmp_path / "shared.png"
    Image.new("RGB", (16, 16)).save(image_path)
    transform_calls = []

    def transform(_image: Image.Image) -> torch.Tensor:
        transform_calls.append(1)
        return torch.zeros(3, 16, 16)

    records = [
        WindowRecord(
            sample_token=f"sample-{index}",
            scene_token="scene",
            anchor_timestamp_us=index,
            image_paths=[str(image_path)],
            image_timestamps_us=[index],
            frame_times_s=[0.0],
            trajectory=[[0.0, 0.0, 0.0]],
            future_times_s=[0.1],
            max_image_time_error_us=0,
            max_trajectory_time_error_us=0,
        )
        for index in range(2)
    ]
    dataset = ActionWindowDataset(
        records, transform=transform, image_cache_size=1
    )
    dataset[0]
    dataset[1]
    assert len(transform_calls) == 1


def test_vggt_cache_rejects_incomplete_and_corrupt_shards(tmp_path: Path) -> None:
    record = WindowRecord(
        sample_token="sample",
        scene_token="scene",
        anchor_timestamp_us=0,
        image_paths=[],
        image_timestamps_us=[],
        frame_times_s=[0, 1, 2, 3, 4],
        trajectory=[[0.0, 0.0, 0.0]] * 40,
        future_times_s=[step / 10 for step in range(1, 41)],
        max_image_time_error_us=0,
        max_trajectory_time_error_us=0,
    )
    base = NuScenesWindowDataset([record], load_images=False)
    shard = tmp_path / "features_00000.safetensors"
    save_file(
        {
            "camera_hidden": torch.zeros(1, 5, 8),
            "register_hidden_mean": torch.zeros(1, 5, 8),
            "pose_enc": torch.zeros(1, 5, 9),
        },
        shard,
    )
    index = {
        "cache_type": "vggt_omega_camera_head_hidden_v1",
        "num_samples": 1,
        "complete": False,
        "shards": [
            {
                "file": shard.name,
                "start": 0,
                "end": 1,
                "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
            }
        ],
    }
    (tmp_path / "index.json").write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        CachedVGGTOmegaFeatureDataset(base, tmp_path)

    index["complete"] = True
    (tmp_path / "index.json").write_text(json.dumps(index), encoding="utf-8")
    shard.write_bytes(shard.read_bytes() + b"corrupt")
    dataset = CachedVGGTOmegaFeatureDataset(base, tmp_path)
    with pytest.raises(ValueError, match="checksum"):
        dataset[0]


def test_vggt_rich_register_cache_returns_all_tokens(tmp_path: Path) -> None:
    record = WindowRecord(
        sample_token="sample",
        scene_token="scene",
        anchor_timestamp_us=0,
        image_paths=[],
        image_timestamps_us=[],
        frame_times_s=[0, 1, 2, 3, 4],
        trajectory=[[0.0, 0.0, 0.0]] * 40,
        future_times_s=[step / 10 for step in range(1, 41)],
        max_image_time_error_us=0,
        max_trajectory_time_error_us=0,
    )
    base = NuScenesWindowDataset([record], load_images=False)
    tensors = {
        "camera_hidden": torch.zeros(1, 5, 8),
        "register_hidden_mean": torch.zeros(1, 5, 8),
        "register_hidden": torch.zeros(1, 5, 16, 8),
        "pose_enc": torch.zeros(1, 5, 9),
    }
    shard = tmp_path / "features_00000.safetensors"
    save_file(tensors, shard)
    index = {
        "cache_type": "vggt_omega_camera_head_hidden_v1",
        "token_mode": "camera_register_tokens",
        "num_samples": 1,
        "complete": True,
        "camera_hidden_shape": [5, 8],
        "register_hidden_mean_shape": [5, 8],
        "register_hidden_shape": [5, 16, 8],
        "pose_enc_shape": [5, 9],
        "shards": [
            {
                "file": shard.name,
                "start": 0,
                "end": 1,
                "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
            }
        ],
    }
    (tmp_path / "index.json").write_text(json.dumps(index), encoding="utf-8")
    sample = CachedVGGTOmegaFeatureDataset(base, tmp_path)[0]
    assert sample["register_hidden"].shape == (5, 16, 8)


def test_vggt_cache_uses_binary_shard_lookup_and_sample_slices(tmp_path: Path) -> None:
    records = [
        WindowRecord(
            sample_token=f"sample-{index}",
            scene_token="scene",
            anchor_timestamp_us=index,
            image_paths=[],
            image_timestamps_us=[],
            frame_times_s=[0, 1, 2, 3, 4],
            trajectory=[[0.0, 0.0, 0.0]] * 40,
            future_times_s=[step / 10 for step in range(1, 41)],
            max_image_time_error_us=0,
            max_trajectory_time_error_us=0,
        )
        for index in range(2)
    ]
    base = ActionWindowDataset(records, load_images=False)
    shards = []
    for index in range(2):
        path = tmp_path / f"features_{index:05d}.safetensors"
        save_file(
            {
                "camera_hidden": torch.full((1, 5, 8), float(index)),
                "register_hidden_mean": torch.zeros(1, 5, 8),
                "pose_enc": torch.zeros(1, 5, 9),
            },
            path,
        )
        shards.append(
            {
                "file": path.name,
                "start": index,
                "end": index + 1,
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    index = {
        "cache_type": "vggt_omega_camera_head_hidden_v1",
        "num_samples": 2,
        "complete": True,
        "camera_hidden_shape": [5, 8],
        "register_hidden_mean_shape": [5, 8],
        "pose_enc_shape": [5, 9],
        "shards": shards,
    }
    (tmp_path / "index.json").write_text(json.dumps(index), encoding="utf-8")
    dataset = CachedVGGTOmegaFeatureDataset(
        base, tmp_path, verify_checksums=False
    )
    assert torch.all(dataset[0]["camera_hidden"] == 0)
    assert torch.all(dataset[1]["camera_hidden"] == 1)


def test_streaming_sample_token_hash_matches_stable_hash() -> None:
    from tools.features.cache_vggt_omega_features import (
        _partition_bounds,
        _sample_token_order_sha256,
    )
    from vision_action_tokenizer.config import stable_hash

    records = [
        WindowRecord(
            sample_token=token,
            scene_token="scene",
            anchor_timestamp_us=0,
            image_paths=[],
            image_timestamps_us=[],
            frame_times_s=[],
            trajectory=[],
            future_times_s=[],
            max_image_time_error_us=0,
            max_trajectory_time_error_us=0,
        )
        for token in ("navsim:first", "navsim:测试")
    ]
    assert _sample_token_order_sha256(records, 2) == stable_hash(
        [record.sample_token for record in records]
    )
    assert [_partition_bounds(10, 3, index) for index in range(3)] == [
        (0, 3),
        (3, 6),
        (6, 10),
    ]


def test_merge_partitioned_vggt_caches(tmp_path: Path) -> None:
    from tools.features.merge_vggt_omega_feature_caches import merge_feature_caches
    from vision_action_tokenizer.models.vggt_omega import file_sha256

    records = [
        WindowRecord(
            sample_token=f"sample-{index}",
            scene_token="scene",
            anchor_timestamp_us=index,
            image_paths=[],
            image_timestamps_us=[],
            frame_times_s=[],
            trajectory=[],
            future_times_s=[],
            max_image_time_error_us=0,
            max_trajectory_time_error_us=0,
        )
        for index in range(4)
    ]
    manifest = tmp_path / "manifest.jsonl"
    save_manifest(records, manifest)
    part_directories = []
    for part_index, range_start in enumerate((0, 2)):
        directory = tmp_path / f"part-{part_index}"
        directory.mkdir()
        shard = directory / "features_00000.safetensors"
        shard.write_bytes(f"part-{part_index}".encode())
        index = {
            "cache_type": "vggt_omega_camera_head_hidden_v1",
            "manifest_sha256": file_sha256(manifest),
            "manifest_num_samples": 4,
            "num_partitions": 2,
            "partition_index": part_index,
            "range_start": range_start,
            "range_end": range_start + 2,
            "expected_num_samples": 2,
            "num_samples": 2,
            "complete": True,
            "elapsed_seconds": 1.0,
            "sample_token_order_sha256": "partition-only",
            "shards": [
                {
                    "file": shard.name,
                    "start": 0,
                    "end": 2,
                    "sha256": file_sha256(shard),
                    "size_bytes": shard.stat().st_size,
                }
            ],
        }
        (directory / "index.json").write_text(json.dumps(index), encoding="utf-8")
        part_directories.append(directory)

    output = tmp_path / "merged"
    merged = merge_feature_caches(part_directories, manifest, output)
    assert merged["num_samples"] == 4
    assert [(shard["start"], shard["end"]) for shard in merged["shards"]] == [
        (0, 2),
        (2, 4),
    ]
    assert (output / "features_00000.safetensors").stat().st_ino == (
        part_directories[0] / "features_00000.safetensors"
    ).stat().st_ino


def test_multi_source_dataset_tags_source_and_sampler_is_deterministic() -> None:
    def record(token: str, dataset_name: str) -> WindowRecord:
        return WindowRecord(
            sample_token=token,
            scene_token=f"scene-{token}",
            anchor_timestamp_us=0,
            image_paths=[],
            image_timestamps_us=[],
            frame_times_s=[0, 1, 2, 3, 4],
            trajectory=[[0.0, 0.0, 0.0]] * 40,
            future_times_s=[step / 10 for step in range(1, 41)],
            max_image_time_error_us=0,
            max_trajectory_time_error_us=0,
            dataset_name=dataset_name,
        )

    dataset = MultiSourceActionDataset(
        {
            "nuscenes": ActionWindowDataset([record("nusc", "legacy")], load_images=False),
            "navsim": ActionWindowDataset([record("nav", "navsim")], load_images=False),
        }
    )
    assert dataset[0]["dataset_name"] == "nuscenes"
    assert dataset[1]["dataset_name"] == "navsim"
    sampler = DeterministicDistributedWeightedSampler(
        [1.0, 3.0], num_samples=20, seed=7
    )
    sampler.set_epoch(2)
    first = list(sampler)
    sampler.set_epoch(2)
    assert list(sampler) == first
    sampler.set_epoch(3)
    assert list(sampler) != first

    rank_zero = DeterministicDistributedWeightedSampler(
        [1.0, 3.0], num_samples=20, seed=7, num_replicas=2, rank=0
    )
    rank_one = DeterministicDistributedWeightedSampler(
        [1.0, 3.0], num_samples=20, seed=7, num_replicas=2, rank=1
    )
    combined = [value for pair in zip(rank_zero, rank_one, strict=True) for value in pair]
    single = DeterministicDistributedWeightedSampler([1.0, 3.0], num_samples=20, seed=7)
    assert combined == list(single)


def test_pose_interpolation_uses_shortest_rotation() -> None:
    yaw_left = np.deg2rad(170.0)
    yaw_right = np.deg2rad(-170.0)
    left = make_transform([np.cos(yaw_left / 2), 0, 0, np.sin(yaw_left / 2)], [0, 0, 0])
    right = make_transform([np.cos(yaw_right / 2), 0, 0, np.sin(yaw_right / 2)], [10, 0, 0])
    sample = interpolate_pose_at_timestamp(50, [0, 100], [left, right], max_gap_us=100)
    assert sample is not None
    pose, nearest_error = sample
    np.testing.assert_allclose(pose[:3, 3], [5, 0, 0], atol=1e-6)
    assert abs(abs(np.arctan2(pose[1, 0], pose[0, 0])) - np.pi) < 1e-6
    assert nearest_error == 50


def test_temporal_nearest_respects_error() -> None:
    index = TemporalIndex([0, 100, 200], ["a", "b", "c"])
    assert index.nearest(145, 50).value == "b"
    assert index.nearest(145, 40) is None


def test_old_and_new_info_schema() -> None:
    old = {
        "token": "old",
        "scene_token": "scene",
        "timestamp": 1,
        "ego2global_rotation": [1, 0, 0, 0],
        "ego2global_translation": [0, 0, 0],
        "cams": {"CAM_FRONT": {"data_path": "old.jpg", "timestamp": 1}},
    }
    new = {
        "token": "new",
        "scene_token": "scene",
        "timestamp": 2,
        "ego2global": np.eye(4),
        "images": {"CAM_FRONT": {"img_path": "new.jpg", "timestamp": 2}},
    }
    records = InfoSchemaAdapter().adapt([old, new])
    assert [record.cam_front_path for record in records] == ["old.jpg", "new.jpg"]
    assert all(not record.scene_inferred for record in records)


def test_custom_suffixed_token_marks_unreliable_interpolated_pose() -> None:
    entry = {
        "token": "a" * 32 + "3",
        "scene_token": "scene",
        "timestamp": 1,
        "ego2global_rotation": [1, 0, 0, 0],
        "ego2global_translation": [0, 0, 0],
        "cams": {"CAM_FRONT": {"data_path": "frame.jpg", "timestamp": 1}},
    }
    record = InfoSchemaAdapter().adapt([entry])[0]
    assert record.pose_interpolated


def test_manifest_builds_six_images_and_sixty_points(tmp_path: Path) -> None:
    records = []
    # Seven seconds at 12 Hz provide multiple valid 5-second windows.
    for index in range(85):
        timestamp = round(index / 12 * 1_000_000)
        image_path = tmp_path / f"{index:03d}.jpg"
        Image.new("RGB", (32, 18)).save(image_path)
        pose = make_transform([1, 0, 0, 0], [index / 12 * 2.0, 0, 0])
        records.append(
            FrameRecord(
                scene_token="scene",
                sample_token=str(index),
                timestamp_us=timestamp,
                cam_front_path=str(image_path),
                cam_front_timestamp_us=timestamp,
                ego_to_global=pose,
            )
        )
    windows, report = ManifestBuilder(tmp_path, max_time_error_s=0.05).build(records)
    assert report["num_windows"] > 0
    assert len(windows[0].image_paths) == 6
    assert len(windows[0].trajectory) == 60
    np.testing.assert_allclose(windows[0].future_times_s, np.arange(1, 61) / 12, atol=1e-9)
    np.testing.assert_allclose(windows[0].trajectory[-1][:2], [10.0, 0.0], atol=1e-4)


def test_manifest_uses_keyframe_images_and_measured_lidar_poses(tmp_path: Path) -> None:
    camera_records = []
    for index in range(17):
        timestamp = round(index * 0.5 * 1_000_000)
        image_path = tmp_path / f"keyframe_{index:03d}.jpg"
        Image.new("RGB", (32, 18)).save(image_path)
        camera_records.append(
            FrameRecord(
                scene_token="scene",
                sample_token=f"sample-{index}",
                timestamp_us=timestamp,
                cam_front_path=str(image_path),
                cam_front_timestamp_us=timestamp,
                ego_to_global=make_transform([1, 0, 0, 0], [timestamp / 1e6, 0, 0]),
            )
        )
    lidar_records = []
    for index in range(161):
        timestamp = round(index * 0.05 * 1_000_000)
        lidar_records.append(
            LidarPoseRecord(
                scene_token="scene",
                sample_token=f"sample-{index // 10}",
                sample_data_token=f"lidar-{index}",
                timestamp_us=timestamp,
                ego_to_global=make_transform([1, 0, 0, 0], [timestamp / 1e6, 0, 0]),
                lidar_to_ego=np.eye(4),
                lidar_path=f"lidar-{index}.bin",
                is_keyframe=index % 10 == 0,
            )
        )
    builder = ManifestBuilder(
        tmp_path,
        frame_offsets_s=[0, 1, 2, 3, 4],
        horizon_s=4,
        trajectory_hz=10,
        anchor_stride_s=0,
        max_image_time_error_s=0.01,
        max_trajectory_time_error_s=0.01,
        image_source="keyframe",
        trajectory_pose_source="lidar_sweeps",
        trajectory_sampling="nearest",
    )
    windows, report = builder.build(camera_records, lidar_records)
    assert report["trajectory_sampling"] == "nearest"
    assert len(windows[0].image_paths) == 5
    assert len(windows[0].trajectory) == 40
    np.testing.assert_allclose(windows[0].frame_times_s, [0, 1, 2, 3, 4])
    np.testing.assert_allclose(windows[0].future_times_s, np.arange(1, 41) / 10)
    np.testing.assert_allclose(windows[0].trajectory[-1], [4, 0, 0], atol=1e-6)
    manifest_path = tmp_path / "windows.jsonl"
    save_manifest(windows, manifest_path)
    assert manifest_scene_tokens(manifest_path) == {"scene"}


def test_ego_motion_state_uses_only_aligned_history_poses() -> None:
    anchor = make_transform([1, 0, 0, 0], [10.0, 0.0, 0.0])
    poses = [
        make_transform([1, 0, 0, 0], [5.0, 0.0, 0.0]),
        make_transform([1, 0, 0, 0], [7.5, 0.0, 0.0]),
        make_transform([1, 0, 0, 0], [10.0, 0.0, 0.0]),
    ]
    states = ManifestBuilder._ego_motion_states(anchor, poses, [-1.0, -0.5, 0.0])
    np.testing.assert_allclose(states[:, :3], [[-5, 0, 0], [-2.5, 0, 0], [0, 0, 0]])
    np.testing.assert_allclose(states[:, 3:], [[5, 0, 0]] * 3)
