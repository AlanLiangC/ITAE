from __future__ import annotations

import os
from pathlib import Path

import pytest

from vision_action_tokenizer.config import resolve_resume_checkpoint


def test_auto_resume_selects_latest_checkpoint(tmp_path: Path) -> None:
    last = tmp_path / "last.pt"
    best = tmp_path / "best.pt"
    last.touch()
    best.touch()
    os.utime(last, ns=(1_000, 1_000))
    os.utime(best, ns=(2_000, 2_000))
    config = {"train": {"resume": "auto"}}
    assert resolve_resume_checkpoint(config, tmp_path) == best


def test_resume_priority_and_fresh_override(tmp_path: Path) -> None:
    configured = tmp_path / "configured.pt"
    explicit = tmp_path / "explicit.pt"
    configured.touch()
    explicit.touch()
    config = {"train": {"resume": "configured.pt"}}
    assert resolve_resume_checkpoint(config, tmp_path) == configured
    assert resolve_resume_checkpoint(config, tmp_path, cli_resume=explicit) == explicit
    assert resolve_resume_checkpoint(config, tmp_path, no_resume=True) is None


def test_auto_resume_starts_fresh_when_output_is_empty(tmp_path: Path) -> None:
    assert resolve_resume_checkpoint({"train": {"resume": "latest"}}, tmp_path) is None
    assert resolve_resume_checkpoint({"train": {"resume": None}}, tmp_path) is None


def test_explicit_missing_resume_checkpoint_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="train.resume"):
        resolve_resume_checkpoint({"train": {"resume": "missing.pt"}}, tmp_path)
