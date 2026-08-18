#!/usr/bin/env python3
"""Run and summarize SUV's official NAVSIM v1 PDMS evaluation from ITAE."""

from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_ROOT = Path(
    "/inspire/hdd/project/spatiotemporal-intelligence-research/ky26298/Projects/"
    "pure_checkpoints/SUV_ckpt"
)
DEFAULT_DATA_ROOT = Path("/inspire/hdd/global_public/public_datas/NAVSIM")
DEFAULT_RUN_ROOT = REPOSITORY_ROOT / "output" / "suv" / "navsim_v1"
PUBLISHED_NAVSIM_V1 = {
    "NC": 99.1,
    "DAC": 97.8,
    "TTC": 96.7,
    "Comfort": 100.0,
    "EP": 84.6,
    "PDMS": 90.8,
}
PAPER_COLUMNS = {
    "NC": "no_at_fault_collisions",
    "DAC": "drivable_area_compliance",
    "TTC": "time_to_collision_within_bound",
    "Comfort": "comfort",
    "EP": "ego_progress",
    "PDMS": "score",
}


@dataclass(frozen=True)
class RuntimePaths:
    checkpoint_root: Path
    data_root: Path
    map_root: Path
    metric_cache: Path
    text_cache: Path
    output_dir: Path
    map_version: str


def _absolute(path: Path) -> Path:
    return path.expanduser().resolve()


def _detect_map_version(map_root: Path, requested: str | None) -> str:
    if requested:
        return requested
    versions = sorted(path.stem for path in map_root.glob("nuplan-maps-v*.json"))
    if not versions:
        return "nuplan-maps-v1.0"
    return versions[-1]


def _runtime_paths(args: argparse.Namespace) -> RuntimePaths:
    run_root = _absolute(Path(args.run_root))
    data_root = _absolute(Path(args.data_root))
    map_root = _absolute(Path(args.map_root)) if args.map_root else data_root / "maps"
    return RuntimePaths(
        checkpoint_root=_absolute(Path(args.checkpoint_root)),
        data_root=data_root,
        map_root=map_root,
        metric_cache=(
            _absolute(Path(args.metric_cache))
            if args.metric_cache
            else run_root / "metric_cache"
        ),
        text_cache=(
            _absolute(Path(args.text_cache))
            if args.text_cache
            else run_root / "text_embeddings"
        ),
        output_dir=(
            _absolute(Path(args.output_dir))
            if args.output_dir
            else run_root / "evaluation"
        ),
        map_version=_detect_map_version(map_root, args.map_version),
    )


def _gpu_ids(value: str) -> list[str]:
    devices = [item.strip() for item in value.split(",") if item.strip()]
    if not devices:
        raise ValueError("At least one GPU ID is required")
    if len(set(devices)) != len(devices):
        raise ValueError("GPU IDs must be unique")
    return devices


def _metric_cache_count(path: Path) -> int:
    metadata = path / "metadata"
    if not metadata.is_dir():
        return 0
    csv_paths = sorted(metadata.glob("*.csv"))
    if not csv_paths:
        return 0
    with csv_paths[-1].open(encoding="utf-8") as handle:
        return max(sum(1 for line in handle if line.strip()) - 1, 0)


def _navsim_v1_api_error() -> str | None:
    try:
        dataclasses = importlib.import_module("navsim.common.dataclasses")
        score_module = importlib.import_module("navsim.evaluate.pdm_score")
    except Exception as error:  # pragma: no cover - environment-specific
        return f"NAVSIM import failed: {error}"
    if not hasattr(dataclasses, "PDMResults"):
        return "NAVSIM is not v1.x: navsim.common.dataclasses.PDMResults is missing"
    parameters = inspect.signature(score_module.pdm_score).parameters
    if "traffic_agents_policy" in parameters:
        return "NAVSIM is not v1.x: pdm_score has the v2 traffic_agents_policy API"
    return None


def collect_doctor_report(paths: RuntimePaths, *, require_metric_cache: bool) -> dict[str, Any]:
    wan_root = paths.checkpoint_root / "Wan2.2-TI2V-5B"
    required_files = [
        paths.checkpoint_root / "suv_navsim.pt",
        wan_root / "Wan2.2_VAE.pth",
        wan_root / "models_t5_umt5-xxl-enc-bf16.pth",
        wan_root / "diffusion_pytorch_model.safetensors.index.json",
        wan_root / "google" / "umt5-xxl" / "tokenizer.json",
    ]
    missing_files = [str(path) for path in required_files if not path.is_file()]
    log_path = paths.data_root / "navsim_logs" / "test"
    sensor_path = paths.data_root / "sensor_blobs" / "test"
    map_file = paths.map_root / f"{paths.map_version}.json"
    imports: dict[str, str] = {}
    module_paths: dict[str, str] = {}
    for module_name in (
        "torch",
        "hydra",
        "omegaconf",
        "transformers",
        "pytorch_lightning",
        "suv",
        "navsim",
    ):
        try:
            module = importlib.import_module(module_name)
            imports[module_name] = str(getattr(module, "__version__", "importable"))
            module_file = getattr(module, "__file__", None)
            if module_file:
                module_paths[module_name] = str(Path(module_file).resolve())
        except Exception as error:  # pragma: no cover - environment-specific
            imports[module_name] = f"ERROR: {error}"
    metric_count = _metric_cache_count(paths.metric_cache)
    errors = []
    if missing_files:
        errors.append(f"missing checkpoint files: {missing_files}")
    for label, path in (
        ("NAVSIM test logs", log_path),
        ("NAVSIM test sensor blobs", sensor_path),
        ("NuPlan map root", paths.map_root),
    ):
        if not path.is_dir():
            errors.append(f"{label} missing: {path}")
    if not map_file.is_file():
        errors.append(f"map version metadata missing: {map_file}")
    import_errors = [
        f"{name}: {value}"
        for name, value in imports.items()
        if value.startswith("ERROR:")
    ]
    errors.extend(import_errors)
    navsim_module_path = module_paths.get("navsim")
    if navsim_module_path:
        navsim_root = Path(navsim_module_path).parent
        required_navsim_configs = [
            navsim_root
            / "planning"
            / "script"
            / "config"
            / "metric_caching"
            / "default_metric_caching.yaml",
            navsim_root
            / "planning"
            / "script"
            / "config"
            / "pdm_scoring"
            / "default_run_pdm_score.yaml",
        ]
        missing_configs = [
            str(path) for path in required_navsim_configs if not path.is_file()
        ]
        if missing_configs:
            errors.append(
                "NAVSIM v1 Hydra configs are missing; install v1.1 in editable mode: "
                f"{missing_configs}"
            )
    navsim_error = _navsim_v1_api_error()
    if navsim_error:
        errors.append(navsim_error)
    if require_metric_cache and metric_count == 0:
        errors.append(f"NAVSIM v1 metric cache missing or empty: {paths.metric_cache}")
    return {
        "ok": not errors,
        "errors": errors,
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "imports": imports,
        "module_paths": module_paths,
        "paths": {key: str(value) for key, value in asdict(paths).items()},
        "test_log_files": len(list(log_path.glob("*.pkl"))) if log_path.is_dir() else 0,
        "metric_cache_entries": metric_count,
        "text_cache_exists": paths.text_cache.is_dir(),
    }


def _base_environment(paths: RuntimePaths, gpu_ids: list[str]) -> dict[str, str]:
    environment = os.environ.copy()
    python_paths = [str(REPOSITORY_ROOT / "third_party" / "SUV" / "src"), str(REPOSITORY_ROOT)]
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment.update(
        {
            "CKPT_LOCAL_DIR": str(paths.checkpoint_root),
            "CUDA_VISIBLE_DEVICES": ",".join(gpu_ids),
            "DIFFSYNTH_SKIP_DOWNLOAD": "true",
            "HF_HUB_OFFLINE": "1",
            "NAVSIM_EXP_ROOT": str(paths.metric_cache.parent),
            "NUPLAN_MAPS_ROOT": str(paths.map_root),
            "NUPLAN_MAP_VERSION": paths.map_version,
            "OPENSCENE_DATA_ROOT": str(paths.data_root),
            "PYTHONPATH": os.pathsep.join(python_paths),
            "PYTHONUNBUFFERED": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    return environment


def _run_command(command: Sequence[str], environment: dict[str, str], *, dry_run: bool) -> None:
    print("Command:")
    print(" ".join(command))
    if not dry_run:
        subprocess.run(command, cwd=REPOSITORY_ROOT, env=environment, check=True)


def _require_doctor(paths: RuntimePaths, *, require_metric_cache: bool) -> None:
    report = collect_doctor_report(paths, require_metric_cache=require_metric_cache)
    if not report["ok"]:
        raise RuntimeError("SUV NAVSIM v1 doctor failed:\n- " + "\n- ".join(report["errors"]))


def cache_metrics(args: argparse.Namespace, paths: RuntimePaths) -> None:
    _require_doctor(paths, require_metric_cache=False)
    gpu_ids = _gpu_ids(args.gpus)
    paths.metric_cache.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "navsim.planning.script.run_metric_caching",
        "train_test_split=navtest",
        f"cache.cache_path={paths.metric_cache}",
        "worker=single_machine_thread_pool",
        f"worker.max_workers={args.workers}",
        "worker.use_process_pool=false",
    ]
    if args.max_scenes is not None:
        command.append(f"train_test_split.scene_filter.max_scenes={args.max_scenes}")
    _run_command(command, _base_environment(paths, gpu_ids), dry_run=args.dry_run)


def precompute_text(args: argparse.Namespace, paths: RuntimePaths) -> None:
    _require_doctor(paths, require_metric_cache=False)
    gpu_ids = _gpu_ids(args.gpus)
    paths.text_cache.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={len(gpu_ids)}",
        str(REPOSITORY_ROOT / "experiments" / "navsimv1" / "precompute_text_embeds.py"),
        "--train-test-split",
        "navtest",
        "--navsim-log-path",
        str(paths.data_root / "navsim_logs" / "test"),
        "--sensor-blobs-path",
        str(paths.data_root / "sensor_blobs" / "test"),
        "--text-embedding-cache-dir",
        str(paths.text_cache),
        "--model-id",
        "Wan2.2-TI2V-5B",
        "--tokenizer-model-id",
        "Wan2.2-TI2V-5B",
        "--num-history-frames",
        "4",
        "--num-future-frames",
        "8",
        "--batch-size",
        str(args.batch_size),
        "--overwrite",
        str(args.overwrite).lower(),
        "--prompt-mode",
        str(args.prompt_mode),
    ]
    if args.max_scenes is not None:
        command.extend(["--max-tokens", str(args.max_scenes)])
    _run_command(command, _base_environment(paths, gpu_ids), dry_run=args.dry_run)


def evaluate(args: argparse.Namespace, paths: RuntimePaths) -> None:
    _require_doctor(paths, require_metric_cache=True)
    if not paths.text_cache.is_dir() or not any(paths.text_cache.rglob("*.pt")):
        raise RuntimeError(f"Text embedding cache is missing or empty: {paths.text_cache}")
    gpu_ids = _gpu_ids(args.gpus)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    is_itae = args.itae_adapter is not None
    itae_adapter = None
    tokenizer_config = None
    tokenizer_checkpoint = None
    if is_itae:
        itae_adapter = _absolute(args.itae_adapter)
        tokenizer_config = _absolute(
            args.action_tokenizer_config
            or REPOSITORY_ROOT
            / "output/navsim_trainval_v4_scratch_4gpu/resolved_config.json"
        )
        tokenizer_checkpoint = _absolute(
            args.action_tokenizer_checkpoint
            or REPOSITORY_ROOT / "output/navsim_trainval_v4_scratch_4gpu/best.pt"
        )
        missing = [
            str(path)
            for path in (itae_adapter, tokenizer_config, tokenizer_checkpoint)
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(f"SUV-ITAE evaluation inputs are missing: {missing}")
    agent_target = (
        "experiments.suv_itae.navsimv1_agent.SUVITAENavsimV1Agent"
        if is_itae
        else "experiments.navsimv1.pdm_agent.SUVNavsimV1Agent"
    )
    model_config = (
        REPOSITORY_ROOT / "experiments/suv_itae/config/model/suv_itae_navsim.yaml"
        if is_itae
        else REPOSITORY_ROOT / "experiments/navsimv1/config/model/suv_navsim.yaml"
    )
    overrides = [
        "train_test_split=navtest",
        "train_test_split.scene_filter.num_history_frames=4",
        "train_test_split.scene_filter.num_future_frames=8",
        f"metric_cache_path={paths.metric_cache}",
        f"navsim_log_path={paths.data_root / 'navsim_logs' / 'test'}",
        f"sensor_blobs_path={paths.data_root / 'sensor_blobs' / 'test'}",
        f"output_dir={paths.output_dir}",
        "experiment_name=suv_navsimv1_navtest",
        "worker=single_machine_thread_pool",
        "worker.max_workers=1",
        "worker.use_process_pool=false",
        f"agent._target_={agent_target}",
        f"++agent.checkpoint_path={paths.checkpoint_root / 'suv_navsim.pt'}",
        f"++agent.model_config_path={model_config}",
        f"++agent.text_embedding_cache_dir={paths.text_cache}",
        f"++agent.num_inference_steps={args.inference_steps}",
    ]
    if is_itae:
        assert itae_adapter is not None
        assert tokenizer_config is not None
        assert tokenizer_checkpoint is not None
        overrides.extend(
            [
                f"++agent.adapter_checkpoint_path={itae_adapter}",
                f"++agent.action_tokenizer_config_path={tokenizer_config}",
                f"++agent.action_tokenizer_checkpoint_path={tokenizer_checkpoint}",
            ]
        )
    else:
        overrides.extend(
            [
                "++agent.visual_conditioning=history_4",
                "++agent.slot_inference=true",
            ]
        )
    if args.max_scenes is not None:
        overrides.append(f"train_test_split.scene_filter.max_scenes={args.max_scenes}")
    command = [
        sys.executable,
        "-u",
        str(REPOSITORY_ROOT / "experiments/navsimv1/run_pdm_score_multigpu.py"),
        "--cuda-visible-devices",
        ",".join(gpu_ids),
        "--output-dir",
        str(paths.output_dir),
        "--overrides",
        *overrides,
    ]
    _run_command(command, _base_environment(paths, gpu_ids), dry_run=args.dry_run)


def _latest_result(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = sorted(path.glob("*.csv"), key=lambda item: item.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No evaluation CSV found under: {path}")
    return candidates[-1]


def summarize_results(csv_path: Path) -> dict[str, Any]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Evaluation CSV is empty: {csv_path}")
    data_rows = [row for row in rows if row.get("token") != "average"]
    valid_rows = [row for row in data_rows if str(row.get("valid", "")).lower() == "true"]
    average_rows = [row for row in rows if row.get("token") == "average"]
    if average_rows:
        source = average_rows[-1]
    elif valid_rows:
        source = {
            column: str(sum(float(row[column]) for row in valid_rows) / len(valid_rows))
            for column in PAPER_COLUMNS.values()
        }
    else:
        raise ValueError("Evaluation contains no valid samples")
    measured = {label: 100.0 * float(source[column]) for label, column in PAPER_COLUMNS.items()}
    comparison = {
        label: {
            "measured": measured[label],
            "published": PUBLISHED_NAVSIM_V1[label],
            "delta": measured[label] - PUBLISHED_NAVSIM_V1[label],
        }
        for label in PAPER_COLUMNS
    }
    return {
        "source_csv": str(csv_path.resolve()),
        "num_samples": len(data_rows),
        "num_valid": len(valid_rows),
        "num_failed": len(data_rows) - len(valid_rows),
        "published_protocol": {
            "benchmark": "NAVSIM v1 navtest PDMS",
            "front_cameras": 1,
            "solver_steps": 10,
            "trajectory_candidates": 1,
        },
        "comparison_percentage_points": comparison,
    }


def _comparison_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# SUV NAVSIM v1 reproduction comparison",
        "",
        f"Samples: {summary['num_samples']} total, {summary['num_valid']} valid, "
        f"{summary['num_failed']} failed.",
        "",
        "| Metric | Reproduced | Published | Delta (pp) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, values in summary["comparison_percentage_points"].items():
        lines.append(
            f"| {label} | {values['measured']:.2f} | {values['published']:.2f} | "
            f"{values['delta']:+.2f} |"
        )
    lines.extend(
        [
            "",
            "Published SUV protocol: one front camera, 10 solver steps, one trajectory.",
            "A partial `--max-scenes` run is a smoke test and is not directly comparable.",
            "",
        ]
    )
    return "\n".join(lines)


def summarize_command(args: argparse.Namespace, paths: RuntimePaths) -> dict[str, Any]:
    source = _latest_result(_absolute(Path(args.results)) if args.results else paths.output_dir)
    summary = summarize_results(source)
    json_output = (
        _absolute(Path(args.json_output))
        if args.json_output
        else source.parent / "summary.json"
    )
    markdown_output = (
        _absolute(Path(args.markdown_output))
        if args.markdown_output
        else source.parent / "comparison.md"
    )
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    markdown_output.write_text(_comparison_markdown(summary), encoding="utf-8")
    print(_comparison_markdown(summary))
    print(f"JSON: {json_output}")
    print(f"Markdown: {markdown_output}")
    return summary


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--map-root", type=Path)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--metric-cache", type=Path)
    parser.add_argument("--text-cache", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--map-version")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    common = argparse.ArgumentParser(add_help=False)
    _add_common_arguments(common)
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor = subparsers.add_parser("doctor", parents=[common])
    doctor.add_argument("--require-metric-cache", action="store_true")

    metric = subparsers.add_parser("cache-metrics", parents=[common])
    metric.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    metric.add_argument("--workers", type=int, default=16)
    metric.add_argument("--max-scenes", type=int)
    metric.add_argument("--dry-run", action="store_true")

    text = subparsers.add_parser("precompute", parents=[common])
    text.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    text.add_argument("--batch-size", type=int, default=16)
    text.add_argument("--max-scenes", type=int)
    text.add_argument("--overwrite", action="store_true")
    text.add_argument("--prompt-mode", choices=["static", "dynamic"], default="dynamic")
    text.add_argument("--dry-run", action="store_true")

    evaluation = subparsers.add_parser("evaluate", parents=[common])
    evaluation.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    evaluation.add_argument("--inference-steps", type=int, default=10)
    evaluation.add_argument("--max-scenes", type=int)
    evaluation.add_argument("--itae-adapter", type=Path)
    evaluation.add_argument("--action-tokenizer-config", type=Path)
    evaluation.add_argument("--action-tokenizer-checkpoint", type=Path)
    evaluation.add_argument("--dry-run", action="store_true")

    summary = subparsers.add_parser("summarize", parents=[common])
    summary.add_argument("--results", type=Path)
    summary.add_argument("--json-output", type=Path)
    summary.add_argument("--markdown-output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = _runtime_paths(args)
    if args.command == "doctor":
        report = collect_doctor_report(paths, require_metric_cache=args.require_metric_cache)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["ok"] else 2
    if args.command == "cache-metrics":
        cache_metrics(args, paths)
    elif args.command == "precompute":
        precompute_text(args, paths)
    elif args.command == "evaluate":
        evaluate(args, paths)
    elif args.command == "summarize":
        summarize_command(args, paths)
    else:  # pragma: no cover - argparse guarantees a known command
        raise ValueError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
