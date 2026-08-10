from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from vision_action_tokenizer.models.vision_backbones.pe_spatial import PESpatialTransform


def test_pe_transform_is_deterministic_and_normalized() -> None:
    array = np.full((120, 320, 3), 255, dtype=np.uint8)
    image = Image.fromarray(array)
    transform = PESpatialTransform(64, "squash")
    first = transform(image)
    second = transform(image)
    assert first.shape == (3, 64, 64)
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first, torch.ones_like(first))
