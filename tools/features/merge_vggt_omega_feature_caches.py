#!/usr/bin/env python3
"""Merge independently generated VGGT-Omega cache partitions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from vision_action_tokenizer.models.vggt_omega import file_sha256

_PARTITION_FIELDS = {
    "complete",
    "elapsed_seconds",
    "expected_num_samples",
    "manifest_num_samples",
    "num_partitions",
    "num_samples",
    "partition_index",
    "range_end",
    "range_start",
    "sample_token_order_sha256",
    "shards",
}


def _manifest_token_hash_and_count(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    digest.update(b"[")
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            token = json.loads(line)["sample_token"]
            if count:
                digest.update(b", ")
            digest.update(
                json.dumps(token, ensure_ascii=False, default=str).encode("utf-8")
            )
            count += 1
    digest.update(b"]")
    return digest.hexdigest(), count


def _copy_or_link(source: Path, destination: Path, mode: str) -> None:
    if mode == "hardlink":
        os.link(source, destination)
    elif mode == "symlink":
        destination.symlink_to(source.resolve())
    elif mode == "copy":
        shutil.copy2(source, destination)
    else:  # pragma: no cover - argparse and callers validate this
        raise ValueError(f"Unknown merge mode: {mode}")


def merge_feature_caches(
    part_directories: list[Path],
    manifest: Path,
    output: Path,
    *,
    mode: str = "hardlink",
    verify_checksums: bool = False,
) -> dict[str, Any]:
    if mode not in {"hardlink", "symlink", "copy"}:
        raise ValueError("mode must be hardlink, symlink, or copy")
    if len(part_directories) < 2:
        raise ValueError("At least two partition directories are required")
    manifest_sha = file_sha256(manifest)
    token_hash, manifest_count = _manifest_token_hash_and_count(manifest)

    parts: list[tuple[Path, dict[str, Any]]] = []
    for directory in part_directories:
        index_path = directory / "index.json"
        if not index_path.is_file():
            raise FileNotFoundError(f"Partition index is missing: {index_path}")
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if not index.get("complete"):
            raise ValueError(f"Cache partition is incomplete: {directory}")
        if index.get("manifest_sha256") != manifest_sha:
            raise ValueError(f"Manifest checksum mismatch in partition: {directory}")
        required = {
            "manifest_num_samples",
            "num_partitions",
            "partition_index",
            "range_start",
            "range_end",
        }
        if not required <= set(index):
            raise ValueError(f"Not a partitioned cache: {directory}")
        parts.append((directory, index))
    parts.sort(key=lambda item: int(item[1]["partition_index"]))

    expected_partitions = int(parts[0][1]["num_partitions"])
    if len(parts) != expected_partitions:
        raise ValueError(f"Expected {expected_partitions} partitions, got {len(parts)}")
    baseline = {
        key: value
        for key, value in parts[0][1].items()
        if key not in _PARTITION_FIELDS
    }
    cursor = 0
    for expected_index, (directory, index) in enumerate(parts):
        comparable = {
            key: value for key, value in index.items() if key not in _PARTITION_FIELDS
        }
        if comparable != baseline:
            raise ValueError(f"Cache metadata mismatch in partition: {directory}")
        if int(index["num_partitions"]) != expected_partitions:
            raise ValueError(f"num_partitions mismatch in partition: {directory}")
        if int(index["partition_index"]) != expected_index:
            raise ValueError("Cache partition indices are not contiguous")
        if int(index["manifest_num_samples"]) != manifest_count:
            raise ValueError(f"Manifest sample count mismatch in partition: {directory}")
        if int(index["range_start"]) != cursor:
            raise ValueError("Cache partition ranges have a gap or overlap")
        partition_size = int(index["range_end"]) - int(index["range_start"])
        if int(index["num_samples"]) != partition_size:
            raise ValueError(f"Cache partition has the wrong sample count: {directory}")
        cursor = int(index["range_end"])
    if cursor != manifest_count:
        raise ValueError(f"Partition ranges end at {cursor}, manifest has {manifest_count}")

    if output.exists():
        index_path = output / "index.json"
        if index_path.is_file():
            existing = json.loads(index_path.read_text(encoding="utf-8"))
            if (
                existing.get("complete")
                and existing.get("manifest_sha256") == manifest_sha
                and int(existing.get("num_samples", -1)) == manifest_count
            ):
                print(f"Merged cache already complete: {output}", flush=True)
                return existing
        raise FileExistsError(f"Merge output already exists: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.merge-{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Temporary merge directory already exists: {temporary}")
    temporary.mkdir()
    merged_shards: list[dict[str, object]] = []
    try:
        shard_index = 0
        for directory, index in parts:
            local_cursor = 0
            global_start = int(index["range_start"])
            for shard in index["shards"]:
                local_start = int(shard["start"])
                local_end = int(shard["end"])
                if local_start != local_cursor or local_end <= local_start:
                    raise ValueError(f"Non-contiguous shards in partition: {directory}")
                source = directory / str(shard["file"])
                if not source.is_file():
                    raise FileNotFoundError(f"Cache shard is missing: {source}")
                expected_size = shard.get("size_bytes")
                if expected_size is not None and source.stat().st_size != int(expected_size):
                    raise ValueError(f"Cache shard size mismatch: {source}")
                if verify_checksums and file_sha256(source) != shard.get("sha256"):
                    raise ValueError(f"Cache shard checksum mismatch: {source}")
                filename = f"features_{shard_index:05d}.safetensors"
                destination = temporary / filename
                _copy_or_link(source, destination, mode)
                merged_shards.append(
                    {
                        "file": filename,
                        "start": global_start + local_start,
                        "end": global_start + local_end,
                        "sha256": shard["sha256"],
                        "size_bytes": source.stat().st_size,
                    }
                )
                shard_index += 1
                local_cursor = local_end
            if local_cursor != int(index["num_samples"]):
                raise ValueError(f"Shard range is incomplete in partition: {directory}")

        merged = {
            **baseline,
            "sample_token_order_sha256": token_hash,
            "expected_num_samples": manifest_count,
            "num_samples": manifest_count,
            "complete": True,
            "elapsed_seconds": max(
                float(index.get("elapsed_seconds", 0.0)) for _, index in parts
            ),
            "shards": merged_shards,
        }
        (temporary / "index.json").write_text(
            json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(
        f"Merged {len(parts)} partitions and {manifest_count} samples into {output}",
        flush=True,
    )
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", type=Path, nargs="+", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("hardlink", "symlink", "copy"), default="hardlink"
    )
    parser.add_argument("--verify-checksums", action="store_true")
    args = parser.parse_args()
    merge_feature_caches(
        args.parts,
        args.manifest,
        args.output,
        mode=args.mode,
        verify_checksums=args.verify_checksums,
    )


if __name__ == "__main__":
    main()
