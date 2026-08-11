#!/usr/bin/env python3
"""Aggregate independent raw-vs-token comparison JSON files across seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparisons", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if len(args.comparisons) < 2:
        raise ValueError("Seed summary requires at least two comparison files")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.comparisons]
    metric_names = sorted(
        set(reports[0]["raw_metrics"]) & set(reports[0]["token_metrics"])
    )
    summary: dict[str, object] = {
        "comparisons": [str(path) for path in args.comparisons],
        "num_seeds": len(reports),
        "metrics": {},
    }
    for metric in metric_names:
        raw = np.asarray([report["raw_metrics"][metric] for report in reports])
        token = np.asarray([report["token_metrics"][metric] for report in reports])
        summary["metrics"][metric] = {  # type: ignore[index]
            "raw_mean": float(raw.mean()),
            "raw_std": float(raw.std(ddof=1)),
            "token_mean": float(token.mean()),
            "token_std": float(token.std(ddof=1)),
            "token_minus_raw_mean": float((token - raw).mean()),
            "token_minus_raw_std": float((token - raw).std(ddof=1)),
        }
    raw_auc = [report["convergence"]["raw_normalized_auc"] for report in reports]
    token_auc = [report["convergence"]["token_normalized_auc"] for report in reports]
    if all(value is not None for value in [*raw_auc, *token_auc]):
        raw_array = np.asarray(raw_auc, dtype=np.float64)
        token_array = np.asarray(token_auc, dtype=np.float64)
        summary["convergence_auc"] = {
            "raw_mean": float(raw_array.mean()),
            "raw_std": float(raw_array.std(ddof=1)),
            "token_mean": float(token_array.mean()),
            "token_std": float(token_array.std(ddof=1)),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
