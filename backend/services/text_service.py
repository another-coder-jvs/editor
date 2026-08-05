"""
Text detection and in-place editing service using EasyOCR + Pillow.
Detects text regions, samples their color/size, and re-renders with matching style.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

_ocr_reader = None

def _get_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        logger.info("[text_service] loading EasyOCR (en)...")
        _ocr_reader = easyocr.Reader(['en'], gpu=True, verbose=False)
        logger.info("[text_service] EasyOCR ready")
    return _ocr_reader


def detect_text(img: Image.Image) -> List[Dict]:
    """Return list of {bbox, text, color, font_size} for each text region."""
    arr = np.array(img.convert("RGB"))
    results = _get_reader().readtext(arr, detail=1, paragraph=False)
    regions = []
    for (box, text, conf) in results:
        if conf < 0.4 or not text.strip():
            continue
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        # Sample dominant text color from the bounding box
        crop = np.array(img.convert("RGB"))[y1:y2, x1:x2]
        color = _dominant_color(crop) if crop.size > 0 else (255, 255, 255)
        font_size = max(8, y2 - y1)
        regions.append({
            "bbox": [x1, y1, x2, y2],
            "text": text,
            "color": color,
            "font_size": font_size,
        })
    return regions


def edit_text_layer(img: Image.Image, params: Dict[str, Any]) -> Image.Image:
    """
    Replace text in the layer in-place.
    params keys:
      - new_text: str  (replace all detected text with this)
      - color: tuple|str  (optional override color)
      - font_size: int  (optional override)
      - target_text: str  (optional — only replace this specific text)
    """
    new_text = params.get("new_text", "")
    color_override = params.get("color")
    size_override = params.get("font_size")
    target_text = params.get("target_text", "").strip().lower()

    arr = np.array(img.convert("RGB"))
    results = _get_reader().readtext(arr, detail=1, paragraph=False)

    result = img.copy().convert("RGBA")
    draw = ImageDraw.Draw(result)

    for (box, text, conf) in results:
        if conf < 0.35:
            continue
        if target_text and target_text not in text.lower():
            continue

        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue

        # Sample background color just outside the text box
        bg_color = _sample_background(arr, x1, y1, x2, y2)
        text_color = color_override or _dominant_color(arr[y1:y2, x1:x2])
        font_size = size_override or max(8, h)

        # Paint over original text with background color
        draw.rectangle([x1, y1, x2, y2], fill=(*bg_color, 255))

        # Load font and fit text into box
        font = _load_font(font_size)
        render_text = new_text if new_text else text
        font, render_text = _fit_text(draw, render_text, font, w, font_size)

        # Center text in box
        bbox = draw.textbbox((0, 0), render_text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = x1 + (w - tw) // 2
        ty = y1 + (h - th) // 2

        if isinstance(text_color, np.ndarray):
            text_color = tuple(int(c) for c in text_color)

        draw.text((tx, ty), render_text, font=font, fill=(*text_color, 255))

    return result


def _dominant_color(crop: np.ndarray) -> tuple:
    """Return the most common non-background color in a crop."""
    if crop.size == 0:
        return (255, 255, 255)
    pixels = crop.reshape(-1, 3).astype(np.float32)
    # Use the pixel furthest from mid-gray as text color
    mid = np.array([128, 128, 128])
    dists = np.linalg.norm(pixels - mid, axis=1)
    idx = np.argmax(dists)
    return tuple(int(c) for c in pixels[idx])


def _sample_background(arr: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> tuple:
    """Sample color just outside the bounding box as background estimate."""
    h, w = arr.shape[:2]
    pad = 3
    samples = []
    for row in [max(0, y1 - pad), min(h - 1, y2 + pad)]:
        strip = arr[row, max(0, x1):min(w, x2)]
        if strip.size:
            samples.append(strip.mean(axis=0))
    if samples:
        bg = np.mean(samples, axis=0)
        return tuple(int(c) for c in bg)
    return (0, 0, 0)


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/content/editor/backend/assets/fonts/Roboto-Bold.ttf",
    ]
    for fp in font_paths:
        if Path(fp).exists():
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _fit_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int, font_size: int) -> tuple:
    """Shrink font until text fits within max_width."""
    while font_size > 6:
        bbox = draw.textbbox((0, 0), text, font=font)
        if (bbox[2] - bbox[0]) <= max_width:
            break
        font_size -= 1
        font = _load_font(font_size)
    return font, text
