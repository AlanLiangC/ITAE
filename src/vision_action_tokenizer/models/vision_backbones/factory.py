"""Factories for configured planner vision backbones and transforms."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PIL import Image
from torch import Tensor

from ...data.dataset import VGGTOmegaResize
from .base import PlannerVisionBackbone
from .pe_spatial import PESpatialBackbone, PESpatialTransform
from .vggt_omega import SingleFrameVGGTOmegaBackbone

BackboneBuilder = Callable[[dict[str, Any]], PlannerVisionBackbone]
TransformBuilder = Callable[[dict[str, Any]], Callable[[Image.Image], Tensor]]
_BACKBONE_BUILDERS: dict[str, tuple[BackboneBuilder, TransformBuilder]] = {}


def register_planner_vision_backbone(
    name: str,
    backbone_builder: BackboneBuilder,
    transform_builder: TransformBuilder,
    *,
    replace: bool = False,
) -> None:
    """Register a new condition encoder without changing planner or dataset code."""
    if not name:
        raise ValueError("Planner vision backbone registry name cannot be empty")
    if name in _BACKBONE_BUILDERS and not replace:
        raise ValueError(f"Planner vision backbone is already registered: {name!r}")
    _BACKBONE_BUILDERS[name] = (backbone_builder, transform_builder)


def _build_pe(vision: dict[str, Any]) -> PlannerVisionBackbone:
    return PESpatialBackbone(
        model_name=str(vision["model_name"]),
        checkpoint_path=vision["checkpoint_path"],
        expected_sha256=vision.get("checkpoint_sha256"),
        source_path=vision.get("source_path"),
        image_size=int(vision.get("image_size", 512)),
        resize_mode=str(vision.get("resize_mode", "squash")),
        layer_idx=int(vision.get("layer_idx", -1)),
        strip_cls_token=bool(vision.get("strip_cls_token", True)),
        pool_grid=tuple(map(int, vision.get("pool_grid", [8, 8]))),
        freeze=bool(vision.get("freeze", True)),
    )


def _transform_pe(vision: dict[str, Any]) -> Callable[[Image.Image], Tensor]:
    return PESpatialTransform(
        image_size=int(vision.get("image_size", 512)),
        resize_mode=str(vision.get("resize_mode", "squash")),
    )


def _build_vggt(vision: dict[str, Any]) -> PlannerVisionBackbone:
    return SingleFrameVGGTOmegaBackbone(
        checkpoint_path=vision["checkpoint_path"],
        expected_sha256=vision.get("checkpoint_sha256"),
        image_resolution=int(vision.get("image_resolution", 512)),
        resize_mode=str(vision.get("resize_mode", "max_size")),
        patch_size=int(vision.get("patch_size", 16)),
    )


def _transform_vggt(vision: dict[str, Any]) -> Callable[[Image.Image], Tensor]:
    return VGGTOmegaResize(
        image_resolution=int(vision.get("image_resolution", 512)),
        mode=str(vision.get("resize_mode", "max_size")),
        patch_size=int(vision.get("patch_size", 16)),
    )


register_planner_vision_backbone("pe_spatial", _build_pe, _transform_pe)
register_planner_vision_backbone("vggt_omega", _build_vggt, _transform_vggt)


def build_planner_vision_transform(config: dict[str, Any]) -> Callable[[Image.Image], Tensor]:
    vision = config["vision_condition"]
    backbone_type = str(vision["type"])
    try:
        _, transform_builder = _BACKBONE_BUILDERS[backbone_type]
    except KeyError as error:
        raise ValueError(
            f"Unsupported planner vision backbone type: {backbone_type!r}; "
            f"registered={sorted(_BACKBONE_BUILDERS)}"
        ) from error
    return transform_builder(vision)


def build_planner_vision_backbone(config: dict[str, Any]) -> PlannerVisionBackbone:
    vision = config["vision_condition"]
    backbone_type = str(vision["type"])
    try:
        backbone_builder, _ = _BACKBONE_BUILDERS[backbone_type]
    except KeyError as error:
        raise ValueError(
            f"Unsupported planner vision backbone type: {backbone_type!r}; "
            f"registered={sorted(_BACKBONE_BUILDERS)}"
        ) from error
    return backbone_builder(vision)
