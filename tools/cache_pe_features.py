#!/usr/bin/env python3
"""Cache fixed-pooled frozen PE patch tokens in safetensors shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from vision_action_tokenizer.config import load_config
from vision_action_tokenizer.data.dataset import NuScenesWindowDataset
from vision_action_tokenizer.models.pe import PEFeatureExtractor


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=128)
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = NuScenesWindowDataset(args.manifest, int(config["data"]["image_size"]))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    pe_config = config["pe"]
    extractor = PEFeatureExtractor(
        model_name=pe_config["model_name"],
        layer_idx=pe_config.get("layer_idx"),
        pool_size=int(pe_config["pool_size"]),
        freeze=True,
    ).to(device)
    args.output.mkdir(parents=True, exist_ok=True)
    pending: list[torch.Tensor] = []
    count = 0
    shard_index = 0
    shards = []

    def flush() -> None:
        nonlocal pending, shard_index
        if not pending:
            return
        features = torch.cat(pending, dim=0).contiguous()
        start = count - features.shape[0]
        filename = f"features_{shard_index:05d}.safetensors"
        save_file(
            {"features": features},
            args.output / filename,
            metadata={
                "model_name": str(pe_config["model_name"]),
                "layer_idx": str(pe_config.get("layer_idx")),
                "pool_size": str(pe_config["pool_size"]),
            },
        )
        shards.append({"file": filename, "start": start, "end": count})
        shard_index += 1
        pending = []

    with torch.inference_mode():
        for batch in loader:
            features = extractor(batch["images"].to(device, non_blocking=True)).half().cpu()
            for sample in features:
                pending.append(sample.unsqueeze(0))
                count += 1
                if len(pending) == args.shard_size:
                    flush()
        flush()
    index = {
        "num_samples": count,
        "manifest_sha256": file_sha256(args.manifest),
        "model_name": pe_config["model_name"],
        "layer_idx": pe_config.get("layer_idx"),
        "pool_size": pe_config["pool_size"],
        "shards": shards,
    }
    (args.output / "index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

