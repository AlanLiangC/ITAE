"""Configuration loading and reproducibility helpers."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def _merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_config(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path, _seen: set[Path] | None = None) -> dict[str, Any]:
    """Load YAML with an optional relative ``_base_`` config and deep overrides."""
    config_path = Path(path).resolve()
    seen = set() if _seen is None else set(_seen)
    if config_path in seen:
        raise ValueError(f"Recursive config inheritance detected at {config_path}")
    seen.add(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in config {path}, got {type(config).__name__}")
    base_path = config.pop("_base_", None)
    if base_path is None:
        return config
    if not isinstance(base_path, (str, Path)):
        raise TypeError("_base_ must be a relative or absolute YAML path")
    resolved_base = Path(base_path)
    if not resolved_base.is_absolute():
        resolved_base = config_path.parent / resolved_base
    return _merge_config(load_config(resolved_base, seen), config)


def stable_hash(value: Any) -> str:
    """Return a deterministic SHA256 hash for JSON-serializable metadata."""
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_resume_checkpoint(
    config: dict[str, Any],
    output_dir: str | Path,
    cli_resume: str | Path | None = None,
    no_resume: bool = False,
) -> Path | None:
    """Resolve CLI/config/automatic checkpoint selection for training resume.

    Priority is explicit CLI path, ``--no-resume``, then ``train.resume``. The
    config value accepts ``auto``/``latest``, null/false/``none``, or a path. A
    relative filename such as ``last.pt`` is also searched inside ``output_dir``.
    """

    output = Path(output_dir)
    if cli_resume is not None:
        checkpoint = Path(cli_resume)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Explicit --resume checkpoint does not exist: {checkpoint}")
        return checkpoint
    if no_resume:
        return None

    setting = config.get("train", {}).get("resume", "auto")
    if setting is None or setting is False:
        return None
    if setting is True:
        setting = "auto"
    if not isinstance(setting, (str, Path)):
        raise TypeError("train.resume must be auto/latest, null/false/none, or a path")
    normalized = str(setting).strip()
    if normalized.lower() in {"", "none", "never", "off", "false"}:
        return None
    if normalized.lower() in {"auto", "latest"}:
        last_checkpoint = output / "last.pt"
        if last_checkpoint.is_file():
            return last_checkpoint
        candidates = [path for path in output.glob("*.pt") if path.is_file()]
        if not candidates:
            return None
        return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))

    checkpoint = Path(normalized)
    if not checkpoint.is_absolute() and not checkpoint.is_file():
        checkpoint = output / checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Configured train.resume checkpoint does not exist: {checkpoint}")
    return checkpoint


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch on every distributed worker."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
