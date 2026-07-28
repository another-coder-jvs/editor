"""Unit tests for editing_service non-AI transforms."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from PIL import Image
from services.editing_service import (
    _blur, _sharpen, _brightness, _contrast,
    _saturation, _erase, _cartoon, _sketch, _pixel_art, _recolor,
)


def _make_rgba(w=64, h=64) -> Image.Image:
    arr = np.random.randint(50, 200, (h, w, 4), dtype=np.uint8)
    arr[:, :, 3] = 200  # semi-transparent
    return Image.fromarray(arr, "RGBA")


def test_blur_preserves_size():
    img = _make_rgba()
    result = _blur(img, {"radius": 3})
    assert result.size == img.size
    assert result.mode == "RGBA"


def test_sharpen_preserves_size():
    img = _make_rgba()
    result = _sharpen(img, {"factor": 2.0})
    assert result.size == img.size


def test_brightness_preserves_size():
    img = _make_rgba()
    result = _brightness(img, {"value": 1.5})
    assert result.size == img.size


def test_contrast_preserves_size():
    img = _make_rgba()
    result = _contrast(img, {"value": 1.5})
    assert result.size == img.size


def test_saturation_preserves_size():
    img = _make_rgba()
    result = _saturation(img, {"value": 1.5})
    assert result.size == img.size


def test_erase_makes_transparent():
    img = _make_rgba()
    result = _erase(img, {})
    arr = np.array(result)
    assert np.all(arr[:, :, 3] == 0)


def test_cartoon_preserves_size():
    img = _make_rgba()
    result = _cartoon(img, {})
    assert result.size == img.size


def test_sketch_preserves_size():
    img = _make_rgba()
    result = _sketch(img, {})
    assert result.size == img.size


def test_pixel_art_preserves_size():
    img = _make_rgba()
    result = _pixel_art(img, {"size": 8})
    assert result.size == img.size


def test_recolor_changes_hue():
    img = _make_rgba()
    result = _recolor(img, {"color": "red"})
    assert result.size == img.size
    assert result.mode == "RGBA"
