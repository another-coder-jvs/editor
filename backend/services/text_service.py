"""
Text detection and in-place editing.
Always operates on the FULL original image, not the layer crop.
Uses EasyOCR quads + cv2.inpaint for clean background, perspective-aware compositing.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

_ocr_reader = None

# All available bold fonts, ranked by preference
_FONT_CANDIDATES = [f for f in [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
] if Path(f).exists()]


def _get_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        logger.info("[text_service] loading EasyOCR...")
        _ocr_reader = easyocr.Reader(['en'], gpu=True, verbose=False)
        logger.info("[text_service] EasyOCR ready")
    return _ocr_reader


def detect_text(img: Image.Image) -> List[Dict]:
    """Detect text on the full image. Returns regions with quad, bbox, text, color, font_size."""
    arr = np.array(img.convert("RGB"))
    results = _get_reader().readtext(arr, detail=1, paragraph=False)
    regions = []
    for (quad, text, conf) in results:
        if conf < 0.35 or not text.strip():
            continue
        quad = [[int(p[0]), int(p[1])] for p in quad]
        xs = [p[0] for p in quad]
        ys = [p[1] for p in quad]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        crop = arr[y1:y2, x1:x2]
        color = _text_color(crop) if crop.size > 0 else [255, 255, 255]
        font_size = max(8, y2 - y1)
        regions.append({
            "quad": quad,
            "bbox": [x1, y1, x2, y2],
            "text": text,
            "color": list(color),
            "font_size": font_size,
            "conf": round(conf, 3),
        })
    return regions


def erase_text_regions(img: Image.Image, regions: list) -> Image.Image:
    """
    Erase all given text regions from img using cv2.inpaint (TELEA).
    Returns the inpainted image (RGBA preserved).
    """
    has_alpha = img.mode == "RGBA"
    arr = np.array(img.convert("RGB"))
    mask = np.zeros(arr.shape[:2], dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for r in regions:
        quad_pts = np.array([[int(p[0]), int(p[1])] for p in r["quad"]], dtype=np.int32)
        cv2.fillPoly(mask, [quad_pts], 255)

    mask = cv2.dilate(mask, kernel, iterations=2)
    inpainted = cv2.inpaint(arr, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    result = Image.fromarray(inpainted)
    if has_alpha:
        result = result.convert("RGBA")
        result.putalpha(img.getchannel("A"))
    return result


def render_text_patch(region: dict, new_text: str) -> Image.Image:
    """
    Render new_text onto a transparent RGBA patch matching the region's
    exact bbox size, color, and fitted font size.
    """
    bbox = region["bbox"]  # [x1, y1, x2, y2] in full-image coords
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    color = region.get("color", [255, 255, 255])

    patch = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(patch)

    # Fit font: start at 85% of bbox height, shrink until text fits width
    font_size = max(8, int(h * 0.85))
    font = _load_font(font_size)
    font, new_text = _fit_text(draw, new_text, font, w, font_size)

    tb = draw.textbbox((0, 0), new_text, font=font)
    tx = (w - (tb[2] - tb[0])) // 2
    ty = (h - (tb[3] - tb[1])) // 2
    draw.text((tx, ty), new_text, font=font,
              fill=(int(color[0]), int(color[1]), int(color[2]), 255))
    return patch


def edit_text_in_image(orig_img: Image.Image, params: Dict[str, Any]) -> Image.Image:
    """
    Edit text directly on the full original image.
    Returns the full edited image (same size as input).
    params: { replacements: JSON string of [{target_text, new_text}, ...] }
    """
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

    replace_map = {r["target_text"].strip().lower(): r["new_text"]
                   for r in replacements if r.get("target_text") and r.get("new_text")}
    if not replace_map:
        return orig_img

    arr = np.array(orig_img.convert("RGB"))
    results = _get_reader().readtext(arr, detail=1, paragraph=False)

    result_arr = arr.copy()

    for (quad, text, conf) in results:
        if conf < 0.30:
            continue
        new_text = replace_map.get(text.strip().lower())
        if new_text is None:
            continue

        quad_pts = np.array([[int(p[0]), int(p[1])] for p in quad], dtype=np.int32)
        xs = quad_pts[:, 0]
        ys = quad_pts[:, 1]
        x1, y1 = max(0, xs.min()), max(0, ys.min())
        x2, y2 = min(arr.shape[1], xs.max()), min(arr.shape[0], ys.max())
        w, h = x2 - x1, y2 - y1
        if w < 2 or h < 2:
            continue

        crop_rgb = arr[y1:y2, x1:x2]
        text_color = _text_color(crop_rgb)
        font_size = max(8, h)

        # ── 1. Erase original text with cv2.inpaint ──
        mask = np.zeros(arr.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [quad_pts], 255)
        # Dilate mask slightly to cover anti-aliased edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_d = cv2.dilate(mask, kernel, iterations=1)
        inpainted = cv2.inpaint(result_arr, mask_d, inpaintRadius=4, flags=cv2.INPAINT_TELEA)
        result_arr = inpainted

        # ── 2. Render new text onto a transparent patch ──
        patch_h, patch_w = h, w
        patch = Image.new("RGBA", (patch_w, patch_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(patch)
        font = _load_font(font_size)
        font, render_text = _fit_text(draw, new_text, font, patch_w, font_size)
        tb = draw.textbbox((0, 0), render_text, font=font)
        tx = (patch_w - (tb[2] - tb[0])) // 2
        ty = (patch_h - (tb[3] - tb[1])) // 2
        draw.text((tx, ty), render_text, font=font,
                  fill=(int(text_color[0]), int(text_color[1]), int(text_color[2]), 255))

        # ── 3. Perspective-warp patch onto result using quad ──
        # Source corners: top-left, top-right, bottom-right, bottom-left of patch
        src = np.float32([[0, 0], [patch_w, 0], [patch_w, patch_h], [0, patch_h]])
        dst = np.float32(quad_pts.astype(np.float32))
        M = cv2.getPerspectiveTransform(src, dst)
        warped_rgba = cv2.warpPerspective(
            np.array(patch), M, (arr.shape[1], arr.shape[0]),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
        )
        # Alpha-composite warped text onto result
        alpha = warped_rgba[:, :, 3:4].astype(np.float32) / 255.0
        text_rgb = warped_rgba[:, :, :3].astype(np.float32)
        result_arr = (result_arr.astype(np.float32) * (1 - alpha) + text_rgb * alpha).astype(np.uint8)

    return Image.fromarray(result_arr, "RGB")


def _text_color(crop: np.ndarray) -> tuple:
    if crop.size == 0:
        return (255, 255, 255)
    pixels = crop.reshape(-1, 3).astype(np.float32)
    brightness = pixels.mean(axis=1)
    median_b = float(np.median(brightness))
    dark = pixels[brightness < median_b]
    light = pixels[brightness >= median_b]
    if dark.size == 0:
        return tuple(int(c) for c in light.mean(axis=0))
    if light.size == 0:
        return tuple(int(c) for c in dark.mean(axis=0))
    # Minority cluster = text
    text_pixels = dark if len(dark) < len(light) else light
    return tuple(int(c) for c in text_pixels.mean(axis=0))


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for fp in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(fp, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _fit_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont,
              max_width: int, font_size: int) -> tuple:
    while font_size > 6:
        bb = draw.textbbox((0, 0), text, font=font)
        if (bb[2] - bb[0]) <= max_width:
            break
        font_size -= 1
        font = _load_font(font_size)
    return font, text
