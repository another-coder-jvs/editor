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


_lama = None

def _get_lama():
    global _lama
    if _lama is None:
        try:
            from simple_lama_inpainting import SimpleLama
            _lama = SimpleLama()
            logger.info("[text_service] LaMa inpainting loaded")
        except Exception as e:
            logger.warning(f"[text_service] LaMa unavailable ({e}), will use cv2.inpaint fallback")
            _lama = False
    return _lama if _lama is not False else None


def _build_glyph_mask(arr: np.ndarray, quad_pts: np.ndarray) -> np.ndarray:
    """
    Build a tight per-pixel glyph mask inside the quad by segmenting
    ink pixels from background using K-means (k=2) on the crop.
    Only pixels that belong to the text ink cluster are masked.
    Falls back to full quad fill if segmentation fails.
    """
    H, W = arr.shape[:2]
    # Bounding rect of quad
    xs, ys = quad_pts[:, 0], quad_pts[:, 1]
    x1, y1 = max(0, xs.min()), max(0, ys.min())
    x2, y2 = min(W, xs.max()), min(H, ys.max())
    if x2 - x1 < 2 or y2 - y1 < 2:
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [quad_pts], 255)
        return mask

    crop = arr[y1:y2, x1:x2].astype(np.float32)
    pixels = crop.reshape(-1, 3)

    try:
        # K-means into 2 clusters: ink vs background
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(
            pixels, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS
        )
        labels = labels.flatten().reshape(crop.shape[:2])

        # The ink cluster is the darker one (lower mean brightness)
        brightness = centers.mean(axis=1)
        ink_label = int(np.argmin(brightness))

        # If background is much darker than text (light text on dark bg), flip
        # Determine by minority: text usually covers less area than background
        count0 = np.sum(labels == 0)
        count1 = np.sum(labels == 1)
        # Minority cluster = ink
        ink_label = 0 if count0 < count1 else 1

        ink_mask_crop = (labels == ink_label).astype(np.uint8) * 255

        # Morphological cleanup: remove isolated noise, keep connected glyphs
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        ink_mask_crop = cv2.morphologyEx(ink_mask_crop, cv2.MORPH_OPEN, k)
        # Expand by 3px to cover anti-aliased edges and blurry halos
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        ink_mask_crop = cv2.dilate(ink_mask_crop, k2, iterations=2)

        # Place crop mask back into full-image mask, clipped to quad polygon
        full_mask = np.zeros((H, W), dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = ink_mask_crop

        # Clip to quad polygon so we don't touch pixels outside the text region
        quad_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(quad_mask, [quad_pts], 255)
        full_mask = cv2.bitwise_and(full_mask, quad_mask)

        # If segmentation produced almost nothing, fall back to quad fill
        if full_mask.sum() < 50:
            return quad_mask

        return full_mask

    except Exception:
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [quad_pts], 255)
        return mask


def erase_text_regions(img: Image.Image, regions: list) -> Image.Image:
    """
    Erase text regions using tight per-glyph masks (not rectangular bbox).
    Uses LaMa for reconstruction, cv2.inpaint TELEA as fallback.
    Only ink pixels are masked — background texture is never touched.
    """
    has_alpha = img.mode == "RGBA"
    alpha = img.getchannel("A") if has_alpha else None
    rgb = img.convert("RGB")
    arr = np.array(rgb)

    # Build combined tight glyph mask across all regions
    combined_mask = np.zeros(arr.shape[:2], dtype=np.uint8)
    for r in regions:
        quad_pts = np.array([[int(p[0]), int(p[1])] for p in r["quad"]], dtype=np.int32)
        glyph_mask = _build_glyph_mask(arr, quad_pts)
        combined_mask = np.maximum(combined_mask, glyph_mask)

    lama = _get_lama()
    if lama is not None:
        try:
            mask_pil = Image.fromarray(combined_mask)
            result_rgb = lama(rgb, mask_pil)
            result = result_rgb.convert("RGBA") if has_alpha else result_rgb
            if has_alpha and alpha is not None:
                result.putalpha(alpha)
            return result
        except Exception as e:
            logger.warning(f"[text_service] LaMa failed ({e}), falling back to cv2.inpaint")

    # Fallback: cv2.inpaint TELEA on tight glyph mask — two passes for cleaner result
    inpainted = cv2.inpaint(arr, combined_mask, inpaintRadius=8, flags=cv2.INPAINT_TELEA)
    inpainted = cv2.inpaint(inpainted, combined_mask, inpaintRadius=4, flags=cv2.INPAINT_NS)
    result = Image.fromarray(inpainted)
    if has_alpha:
        result = result.convert("RGBA")
        if alpha is not None:
            result.putalpha(alpha)
    return result


def render_text_patch(region: dict, new_text: str, overrides: dict | None = None) -> Image.Image:
    """
    Render new_text onto a transparent RGBA patch at the exact bbox size.
    overrides: { color, font_size, shadow_color, shadow_offset, rotation }
    """
    ov = overrides or {}
    bbox = region["bbox"]
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    raw_color = ov.get("color", region.get("color", [255, 255, 255])) or [255, 255, 255]
    color = [int(c) if c is not None else 255 for c in raw_color]
    font_size_override = ov.get("font_size")
    raw_shadow = ov.get("shadow_color")
    shadow_color = [int(c) if c is not None else 0 for c in raw_shadow] if raw_shadow else None
    shadow_offset = ov.get("shadow_offset", [2, 2])
    rotation = ov.get("rotation", 0)

    # Render on a larger canvas if rotated so text isn't clipped
    pad = int(max(w, h) * 0.5) if rotation else 0
    cw, ch = w + pad * 2, h + pad * 2

    patch = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    draw = ImageDraw.Draw(patch)

    font_size = font_size_override if font_size_override else max(8, int(h * 0.90))
    font = _load_font(font_size)
    font, new_text = _fit_text(draw, new_text, font, cw, font_size)

    tb = draw.textbbox((0, 0), new_text, font=font)
    text_w = tb[2] - tb[0]
    text_h = tb[3] - tb[1]
    tx = max(0, (cw - text_w) // 2) - tb[0]
    ty = max(0, (ch - text_h) // 2) - tb[1]

    # Shadow
    if shadow_color:
        sx, sy = int(shadow_offset[0]), int(shadow_offset[1])
        draw.text((tx + sx, ty + sy), new_text, font=font,
                  fill=(int(shadow_color[0]), int(shadow_color[1]), int(shadow_color[2]), 180))

    draw.text((tx, ty), new_text, font=font,
              fill=(int(color[0]), int(color[1]), int(color[2]), 255))

    if rotation:
        patch = patch.rotate(-rotation, expand=True, resample=Image.BICUBIC)
        # Crop back to original bbox size centered
        pw, ph = patch.size
        left = (pw - w) // 2
        top  = (ph - h) // 2
        patch = patch.crop((left, top, left + w, top + h))

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

        # ── 1. Erase original text with tight glyph mask + two-pass inpaint ──
        glyph_mask = _build_glyph_mask(arr, quad_pts)
        inpainted = cv2.inpaint(result_arr, glyph_mask, inpaintRadius=8, flags=cv2.INPAINT_TELEA)
        inpainted = cv2.inpaint(inpainted, glyph_mask, inpaintRadius=4, flags=cv2.INPAINT_NS)
        result_arr = inpainted

        # ── 2. Render new text onto a transparent patch ──
        patch_h, patch_w = h, w
        patch = Image.new("RGBA", (patch_w, patch_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(patch)
        # Start at full bbox height and shrink only if text overflows width
        font_size = max(8, int(h * 0.90))
        font = _load_font(font_size)
        font, render_text = _fit_text(draw, new_text, font, patch_w, font_size)
        tb = draw.textbbox((0, 0), render_text, font=font)
        text_w = tb[2] - tb[0]
        text_h = tb[3] - tb[1]
        # Center both horizontally and vertically, offset by textbbox top offset
        tx = max(0, (patch_w - text_w) // 2) - tb[0]
        ty = max(0, (patch_h - text_h) // 2) - tb[1]
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
