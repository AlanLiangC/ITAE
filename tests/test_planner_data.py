from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from vision_action_tokenizer.data.manifest import WindowRecord, save_manifest
from vision_action_tokenizer.data.planner_dataset import (
    PlannerDataset,
    PlannerTargetNormalizer,
    file_sha256,
    unwrap_yaw,
    wrap_yaw,
)


def _window(tmp_path: Path) -> WindowRecord:
    return WindowRecord(
        sample_token="sample",
        scene_token="scene",
        anchor_timestamp_us=0,
        image_paths=[str(tmp_path / f"{index}.jpg") for index in range(5)],
        image_timestamps_us=list(range(5)),
        frame_times_s=[0.0, 1.0, 2.0, 3.0, 4.0],
        trajectory=[[float(index), 0.0, 3.0 + 0.01 * index] for index in range(40)],
        future_times_s=[0.1 * (index + 1) for index in range(40)],
        max_image_time_error_us=0,
        max_trajectory_time_error_us=0,
    )


def test_yaw_unwrap_and_normalization_roundtrip() -> None:
    trajectory = torch.tensor([[[0.0, 0.0, 3.13], [1.0, 0.0, -3.13]]])
    unwrapped = unwrap_yaw(trajectory)
    assert unwrapped[0, 1, 2] > 3.13
    normalizer = PlannerTargetNormalizer((2, 3))
    normalizer.fit(unwrapped)
    recovered = normalizer.denormalize(normalizer.normalize(unwrapped))
    torch.testing.assert_close(recovered, unwrapped)
    wrapped = wrap_yaw(recovered)
    assert torch.all(wrapped[..., 2] < torch.pi)


def test_planner_dataset_reads_strict_condition_cache(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    save_manifest([_window(tmp_path)], manifest)
    cache = tmp_path / "cache"
    cache.mkdir()
    shard = cache / "features_00000.safetensors"
    save_file(
        {
            "condition_tokens": torch.ones(1, 4, 6),
            "condition_mask": torch.ones(1, 4, dtype=torch.bool),
        },
        str(shard),
    )
    (cache / "index.json").write_text(
        json.dumps(
            {
                "cache_type": "planner_vision_condition_v1",
                "complete": True,
                "manifest_sha256": file_sha256(manifest),
                "backbone_type": "test",
                "num_samples": 1,
                "condition_shape": [4, 6],
                "shards": [
                    {
                        "file": shard.name,
                        "start": 0,
                        "end": 1,
                        "sha256": file_sha256(shard),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    dataset = PlannerDataset(
        manifest,
        "raw_trajectory",
        vision_cache=cache,
        expected_vision_metadata={"backbone_type": "test"},
    )
    sample = dataset[0]
    assert sample["condition_tokens"].shape == (4, 6)
    assert "condition_times" not in sample
    assert sample["target"].shape == (40, 3)


def test_planner_dataset_reads_multiframe_condition_times(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    save_manifest([_window(tmp_path)], manifest)
    cache = tmp_path / "cache"
    cache.mkdir()
    shard = cache / "features_00000.safetensors"
    times = torch.tensor([[*([-1.0] * 2), *([-0.5] * 2), *([0.0] * 2)]])
    save_file(
        {
            "condition_tokens": torch.ones(1, 6, 4),
            "condition_mask": torch.ones(1, 6, dtype=torch.bool),
            "condition_times": times,
        },
        str(shard),
    )
    (cache / "index.json").write_text(
        json.dumps(
            {
                "cache_type": "planner_vision_condition_v2",
                "complete": True,
                "manifest_sha256": file_sha256(manifest),
                "num_samples": 1,
                "condition_shape": [6, 4],
                "frame_indices": [0, 1, 2],
                "condition_frame_offsets_s": [-1.0, -0.5, 0.0],
                "shards": [
                    {
                        "file": shard.name,
                        "start": 0,
                        "end": 1,
                        "sha256": file_sha256(shard),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    dataset = PlannerDataset(manifest, "raw_trajectory", vision_cache=cache)
    sample = dataset[0]
    torch.testing.assert_close(sample["condition_times"], times[0])


def test_planner_dataset_reads_exported_ego_motion_condition(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    save_manifest([_window(tmp_path)], manifest)
    cache = tmp_path / "cache"
    cache.mkdir()
    shard = cache / "features_00000.safetensors"
    save_file(
        {
            "condition_tokens": torch.ones(1, 6, 4),
            "condition_mask": torch.ones(1, 6, dtype=torch.bool),
            "condition_times": torch.tensor([[-1.0, -1.0, -0.5, -0.5, 0.0, 0.0]]),
            "ego_motion_states": torch.randn(1, 3, 6),
            "ego_motion_times": torch.tensor([[-1.0, -0.5, 0.0]]),
        },
        str(shard),
    )
    (cache / "index.json").write_text(
        json.dumps(
            {
                "cache_type": "planner_vision_condition_v3",
                "complete": True,
                "manifest_sha256": file_sha256(manifest),
                "num_samples": 1,
                "condition_shape": [6, 4],
                "ego_motion_shape": [3, 6],
                "causal_tolerance_s": 0.1,
                "shards": [
                    {
                        "file": shard.name,
                        "start": 0,
                        "end": 1,
                        "sha256": file_sha256(shard),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    sample = PlannerDataset(manifest, "raw_trajectory", vision_cache=cache)[0]
    assert sample["ego_motion"].shape == (3, 6)
    torch.testing.assert_close(
        sample["ego_motion_times"], torch.tensor([-1.0, -0.5, 0.0])
    )
