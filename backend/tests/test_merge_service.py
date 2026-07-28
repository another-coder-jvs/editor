"""Unit tests for merge_service."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tempfile
import numpy as np
import pytest
from PIL import Image
from schemas import LayerData, BoundingBox
from services.merge_service import merge_layers


def _save_rgba(path: str, color=(255, 0, 0, 200), size=(100, 100)):
    arr = np.full((*size[::-1], 4), color, dtype=np.uint8)
    Image.fromarray(arr, "RGBA").save(path)


def test_merge_single_layer():
    with tempfile.TemporaryDirectory() as tmp:
        png = os.path.join(tmp, "layer.png")
        _save_rgba(png, color=(255, 0, 0, 255), size=(50, 50))
        layer = LayerData(
            id="l1", name="test",
            mask_path=png, png_path=png,
            bbox=BoundingBox(x=10, y=10, width=50, height=50),
            z_index=0,
        )
        result = merge_layers([layer], 200, 200)
        assert result.size == (200, 200)
        assert result.mode == "RGBA"


def test_merge_hidden_layer_excluded():
    with tempfile.TemporaryDirectory() as tmp:
        png = os.path.join(tmp, "layer.png")
        _save_rgba(png, color=(0, 255, 0, 255), size=(50, 50))
        layer = LayerData(
            id="l1", name="test",
            mask_path=png, png_path=png,
            bbox=BoundingBox(x=0, y=0, width=50, height=50),
            z_index=0, visible=False,
        )
        result = merge_layers([layer], 100, 100)
        arr = np.array(result)
        assert np.all(arr[:, :, 3] == 0)  # fully transparent


def test_merge_opacity():
    with tempfile.TemporaryDirectory() as tmp:
        png = os.path.join(tmp, "layer.png")
        _save_rgba(png, color=(255, 0, 0, 255), size=(50, 50))
        layer = LayerData(
            id="l1", name="test",
            mask_path=png, png_path=png,
            bbox=BoundingBox(x=0, y=0, width=50, height=50),
            z_index=0, opacity=0.5,
        )
        result = merge_layers([layer], 100, 100)
        arr = np.array(result)
        # Alpha should be ~127 (50% of 255)
        assert arr[10, 10, 3] < 200
