from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tools.suv.evaluate_navsim_v1 import (
    PAPER_COLUMNS,
    PUBLISHED_NAVSIM_V1,
    _detect_map_version,
    _gpu_ids,
    summarize_results,
)


def test_gpu_ids_are_ordered_and_unique() -> None:
    assert _gpu_ids("3, 1") == ["3", "1"]
    with pytest.raises(ValueError, match="unique"):
        _gpu_ids("0,0")
    with pytest.raises(ValueError, match="At least one"):
        _gpu_ids(" , ")


def test_detect_map_version_uses_available_metadata(tmp_path: Path) -> None:
    (tmp_path / "nuplan-maps-v1.0.json").touch()
    assert _detect_map_version(tmp_path, None) == "nuplan-maps-v1.0"
    assert _detect_map_version(tmp_path, "custom-map") == "custom-map"


def test_suv_summary_compares_percentage_points(tmp_path: Path) -> None:
    path = tmp_path / "scores.csv"
    fieldnames = ["token", "valid", *PAPER_COLUMNS.values()]
    rows = []
    for token, valid, offset in (("token-a", True, 0.0), ("token-b", True, 0.02)):
        row = {"token": token, "valid": valid}
        row.update(
            {
                column: PUBLISHED_NAVSIM_V1[label] / 100.0 + offset
                for label, column in PAPER_COLUMNS.items()
            }
        )
        rows.append(row)
    average = {"token": "average", "valid": True}
    average.update(
        {
            column: PUBLISHED_NAVSIM_V1[label] / 100.0 + 0.01
            for label, column in PAPER_COLUMNS.items()
        }
    )
    rows.append(average)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize_results(path)
    assert summary["num_samples"] == 2
    assert summary["num_valid"] == 2
    for values in summary["comparison_percentage_points"].values():
        assert values["delta"] == pytest.approx(1.0)
