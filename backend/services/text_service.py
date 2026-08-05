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
    arr = np.array(img.convert("RGB"))
    results = _get_reader().readtext(arr, detail=1, paragraph=False)
    regions = []
    for (box, text, conf) in results:
        if conf < 0.4 or not text.strip():
            continue
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        crop = arr[y1:y2, x1:x2]
        color = _text_color(crop) if crop.size > 0 else (255, 255, 255)
        font_size = max(8, y2 - y1)
        regions.append({"bbox": [x1, y1, x2, y2], "text": text, "color": list(color), "font_size": font_size})
    return regions


def edit_text_layer(img: Image.Image, params: Dict[str, Any]) -> Image.Image:
    import json as _json
    replacements = []
    if "replacements" in params:
        try:
            replacements = _json.loads(params["replacements"])
        except Exception:
            pass
    if not replacements:
        replacements = [{"target_text": params.get("target_text", "").strip().lower(),
                         "new_text": params.get("new_text", "")}]

    replace_map = {r["target_text"].strip().lower(): r["new_text"] for r in replacements if r.get("target_text")}
    replace_all = next((r["new_text"] for r in replacements if not r.get("target_text")), None)

    arr = np.array(img.convert("RGB"))
    alpha_arr = np.array(img.convert("RGBA"))[:, :, 3]
    results = _get_reader().readtext(arr, detail=1, paragraph=False)

    # Work on a copy; we'll restore background by copying original pixels
    result = img.copy().convert("RGBA")
    result_arr = np.array(result)

    for (box, text, conf) in results:
        if conf < 0.35:
            continue
        matched_new = replace_map.get(text.strip().lower()) or replace_all
        if matched_new is None:
            continue

        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue

        crop_rgb = arr[y1:y2, x1:x2]
        text_color = _text_color(crop_rgb)
        font_size = params.get("font_size") or max(8, h)

        # ── Erase original text: inpaint the bbox by copying surrounding rows ──
        pad = max(2, h // 4)
        above = arr[max(0, y1 - pad):y1, x1:x2]   # rows above
        below = arr[y2:min(arr.shape[0], y2 + pad), x1:x2]  # rows below

        if above.size and below.size:
            bg_fill = np.concatenate([above, below], axis=0).mean(axis=0).astype(np.uint8)
        elif above.size:
            bg_fill = above.mean(axis=0).astype(np.uint8)
        elif below.size:
            bg_fill = below.mean(axis=0).astype(np.uint8)
        else:
            bg_fill = arr[y1, x1]  # fallback: single pixel

        # Fill the bbox region with bg_fill color, preserving original alpha
        result_arr[y1:y2, x1:x2, :3] = bg_fill
        result_arr[y1:y2, x1:x2, 3] = alpha_arr[y1:y2, x1:x2]

        # ── Render new text onto a transparent scratch, then composite ──
        scratch = Image.fromarray(result_arr[y1:y2, x1:x2], "RGBA")
        draw = ImageDraw.Draw(scratch)

        font = _load_font(font_size)
        font, render_text = _fit_text(draw, matched_new, font, w, font_size)

        bbox_t = draw.textbbox((0, 0), render_text, font=font)
        tw, th = bbox_t[2] - bbox_t[0], bbox_t[3] - bbox_t[1]
        tx = (w - tw) // 2
        ty = (h - th) // 2

        draw.text((tx, ty), render_text, font=font,
                  fill=(int(text_color[0]), int(text_color[1]), int(text_color[2]), 255))

        result_arr[y1:y2, x1:x2] = np.array(scratch)

    return Image.fromarray(result_arr, "RGBA")


def _text_color(crop: np.ndarray) -> tuple:
    """Cluster pixels into dark/light; return the minority cluster (text) color."""
    if crop.size == 0:
        return (255, 255, 255)
    pixels = crop.reshape(-1, 3).astype(np.float32)
    brightness = pixels.mean(axis=1)
    median_b = np.median(brightness)
    # Text pixels are the minority — pick the cluster further from median
    dark = pixels[brightness < median_b]
    light = pixels[brightness >= median_b]
    if dark.size == 0:
        return tuple(int(c) for c in light.mean(axis=0))
    if light.size == 0:
        return tuple(int(c) for c in dark.mean(axis=0))
    # Whichever cluster is smaller is likely the text
    text_pixels = dark if len(dark) < len(light) else light
    return tuple(int(c) for c in text_pixels.mean(axis=0))


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


def _fit_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont,
              max_width: int, font_size: int) -> tuple:
    while font_size > 6:
        bbox = draw.textbbox((0, 0), text, font=font)
        if (bbox[2] - bbox[0]) <= max_width:
            break
        font_size -= 1
        font = _load_font(font_size)
    return font, text
