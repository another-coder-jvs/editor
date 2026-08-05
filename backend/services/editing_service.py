"""
Editing service – routes parsed prompts to the correct edit handler.
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from services.model_manager import model_manager, DEVICE
from services.prompt_service import parse_edit_prompt

logger = logging.getLogger(__name__)

TEMP_DIR = Path(__file__).resolve().parents[2] / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)


def edit_layer(
    session_id: str, layer_id: str, layer_name: str,
    layer_png_path: str, original_image_path: str, mask_path: str,
    prompt: str, strength: float = 0.75, guidance_scale: float = 7.5, steps: int = 20,
) -> str:
    logger.info(f"[editing] session={session_id} layer='{layer_name}' prompt='{prompt}'")

    parsed       = parse_edit_prompt(prompt, layer_name)
    edit_type    = parsed.get("edit_type", "other")
    edit_params  = parsed.get("edit_params", {})
    inpaint_prompt = parsed.get("inpaint_prompt", prompt)
    logger.info(f"[editing] edit_type={edit_type} params={edit_params}")

    # Check available RAM
    import psutil
    _available_gb = psutil.virtual_memory().available / 1024**3
    _low_ram = _available_gb < 5.5

    GENERATIVE_TYPES = {"replace", "generative_fill", "anime", "oil_painting", "other", "style_transfer"}

    if edit_type == "style_transfer" and "color" in edit_params:
        edit_type = "recolor"
        logger.info("[editing] style_transfer+color → recolor")
    elif _low_ram and edit_type in GENERATIVE_TYPES:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=422,
            detail=f"Not enough RAM for generative edit ({_available_gb:.1f}GB free, need 6GB+). "
                   f"Supported on this machine: recolor, blur, sharpen, brightness, contrast, cartoon, sketch, pixel_art."
        )

    layer_img = Image.open(layer_png_path).convert("RGBA")
    logger.debug(f"[editing] layer image size: {layer_img.size}")

    out_id   = uuid.uuid4().hex[:8]
    out_path = TEMP_DIR / session_id / f"{layer_id}_edited_{out_id}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    AI_TYPES = {"replace", "generative_fill", "anime", "oil_painting", "other", "style_transfer"}

    handlers = {
        "recolor":           _recolor,
        "blur":              _blur,
        "sharpen":           _sharpen,
        "brightness":        _brightness,
        "contrast":          _contrast,
        "saturation":        _saturation,
        "background_remove": _background_remove,
        "erase":             _erase,
        "cartoon":           _cartoon,
        "sketch":            _sketch,
        "pixel_art":         _pixel_art,
        "upscale":           _upscale,
        "replace":           _inpaint,
        "generative_fill":   _inpaint,
        "anime":             _inpaint,
        "oil_painting":      _inpaint,
        "other":             _inpaint,
        "style_transfer":    _style_transfer,
    }

    # On CPU cap steps to avoid OOM / multi-hour runs
    if DEVICE == "cpu":
        steps = min(steps, 8)

    handler = handlers.get(edit_type, _inpaint)
    logger.info(f"[editing] dispatching to handler: {handler.__name__}")
    if edit_type in AI_TYPES:
        result = handler(layer_img, original_image_path, mask_path, inpaint_prompt, strength, guidance_scale, steps, edit_params)
        # Free heavy model RAM immediately after use
        import gc
        if edit_type == "style_transfer":
            model_manager.unload_img2img_pipe()
        elif edit_type in {"replace", "generative_fill", "anime", "oil_painting", "other"}:
            model_manager.unload_inpaint_pipe()
        gc.collect()
    else:
        result = handler(layer_img, edit_params)

    result.save(str(out_path))
    logger.info(f"[editing] saved edited layer → {out_path}")
    return str(out_path)


# ── Non-AI edits ──────────────────────────────────────────────────────────────

def _recolor(img: Image.Image, params: Dict[str, Any]) -> Image.Image:
    color_name = params.get("color", "blue")
    logger.debug(f"[editing/recolor] color={color_name}")
    color_map = {
        "red":(0,100,100),"blue":(220,100,100),"green":(120,100,100),
        "black":(0,0,0),"white":(0,0,100),"yellow":(60,100,100),
        "orange":(30,100,100),"purple":(270,100,100),"pink":(330,80,100),
        "brown":(20,60,40),"gray":(0,0,50),"grey":(0,0,50),
    }
    h_t, s_t, v_t = color_map.get(color_name, (220,100,100))
    arr = np.array(img.convert("RGB"), dtype=np.float32)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    alpha = np.array(img)[:,:,3]
    obj_mask = alpha > 30
    hsv[:,:,0][obj_mask] = h_t / 2
    if s_t > 0: hsv[:,:,1][obj_mask] = s_t / 100 * 255
    if v_t < 100: hsv[:,:,2][obj_mask] = v_t / 100 * 255
    recolored = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.uint8)
    result = Image.fromarray(recolored).convert("RGBA")
    result.putalpha(img.getchannel("A"))
    return result


def _blur(img: Image.Image, params: Dict[str, Any]) -> Image.Image:
    radius = float(params.get("radius", 5))
    logger.debug(f"[editing/blur] radius={radius}")
    rgb = img.convert("RGB").filter(ImageFilter.GaussianBlur(radius=radius))
    result = rgb.convert("RGBA"); result.putalpha(img.getchannel("A"))
    return result


def _sharpen(img: Image.Image, params: Dict[str, Any]) -> Image.Image:
    factor = float(params.get("factor", 2.0))
    logger.debug(f"[editing/sharpen] factor={factor}")
    rgb = ImageEnhance.Sharpness(img.convert("RGB")).enhance(factor)
    result = rgb.convert("RGBA"); result.putalpha(img.getchannel("A"))
    return result


def _brightness(img: Image.Image, params: Dict[str, Any]) -> Image.Image:
    factor = float(params.get("value", 1.3))
    logger.debug(f"[editing/brightness] factor={factor}")
    rgb = ImageEnhance.Brightness(img.convert("RGB")).enhance(factor)
    result = rgb.convert("RGBA"); result.putalpha(img.getchannel("A"))
    return result


def _contrast(img: Image.Image, params: Dict[str, Any]) -> Image.Image:
    factor = float(params.get("value", 1.5))
    logger.debug(f"[editing/contrast] factor={factor}")
    rgb = ImageEnhance.Contrast(img.convert("RGB")).enhance(factor)
    result = rgb.convert("RGBA"); result.putalpha(img.getchannel("A"))
    return result


def _saturation(img: Image.Image, params: Dict[str, Any]) -> Image.Image:
    factor = float(params.get("value", 1.5))
    logger.debug(f"[editing/saturation] factor={factor}")
    rgb = ImageEnhance.Color(img.convert("RGB")).enhance(factor)
    result = rgb.convert("RGBA"); result.putalpha(img.getchannel("A"))
    return result


def _background_remove(img: Image.Image, params: Dict[str, Any]) -> Image.Image:
    logger.info("[editing/background_remove] running rembg…")
    from rembg import remove
    session = model_manager.get_rembg_session()
    result = remove(img, session=session)
    logger.info("[editing/background_remove] done")
    return result


def _erase(img: Image.Image, params: Dict[str, Any]) -> Image.Image:
    logger.debug("[editing/erase] making layer transparent")
    result = img.copy(); result.putalpha(0)
    return result


def _cartoon(img: Image.Image, params: Dict[str, Any]) -> Image.Image:
    logger.debug("[editing/cartoon] applying cartoon filter")
    arr  = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(arr, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    result = Image.fromarray(cartoon).convert("RGBA"); result.putalpha(img.getchannel("A"))
    return result


def _sketch(img: Image.Image, params: Dict[str, Any]) -> Image.Image:
    logger.debug("[editing/sketch] applying sketch filter")
    arr  = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    inv  = cv2.bitwise_not(gray)
    blurred = cv2.GaussianBlur(inv, (21, 21), 0)
    sketch  = cv2.divide(gray, cv2.bitwise_not(blurred), scale=256.0)
    sketch_rgb = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
    result = Image.fromarray(sketch_rgb).convert("RGBA"); result.putalpha(img.getchannel("A"))
    return result


def _pixel_art(img: Image.Image, params: Dict[str, Any]) -> Image.Image:
    size = params.get("size", 16)
    logger.debug(f"[editing/pixel_art] block_size={size}")
    w, h = img.size
    small = img.resize((w // size, h // size), Image.NEAREST)
    return small.resize((w, h), Image.NEAREST)


def _upscale(img: Image.Image, params: Dict[str, Any]) -> Image.Image:
    scale = int(params.get("scale", 2))
    logger.info(f"[editing/upscale] scale={scale}x with Real-ESRGAN")
    upsampler = model_manager.get_realesrgan(scale)
    arr = np.array(img.convert("RGB"))
    out_arr, _ = upsampler.enhance(arr, outscale=scale)
    result = Image.fromarray(out_arr).convert("RGBA")
    alpha  = img.getchannel("A").resize(result.size, Image.LANCZOS)
    result.putalpha(alpha)
    logger.info(f"[editing/upscale] done → {result.size}")
    return result


# ── AI inpainting ─────────────────────────────────────────────────────────────

def _inpaint(
    layer_img: Image.Image, original_image_path: str, mask_path: str,
    prompt: str, strength: float, guidance_scale: float, steps: int,
    params: Dict[str, Any],
) -> Image.Image:
    logger.info(f"[editing/inpaint] prompt='{prompt}' strength={strength} steps={steps}")
    logger.info("[editing/inpaint] loading inpainting pipeline…")
    pipe = model_manager.get_inpaint_pipe()

    orig = Image.open(original_image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    w, h = orig.size
    # On CPU, downscale to 512px max to avoid OOM
    if DEVICE == "cpu":
        max_side = 512
        scale_factor = min(max_side / w, max_side / h, 1.0)
        w, h = int(w * scale_factor), int(h * scale_factor)
    w8, h8 = (w // 8) * 8, (h // 8) * 8
    logger.debug(f"[editing/inpaint] resizing to {w8}x{h8} (multiple of 8)")

    orig_r = orig.resize((w8, h8), Image.LANCZOS)
    mask_r = mask.resize((w8, h8), Image.NEAREST)

    logger.info("[editing/inpaint] running diffusion pipeline…")
    result = pipe(
        prompt=prompt, image=orig_r, mask_image=mask_r,
        strength=strength, guidance_scale=guidance_scale,
        num_inference_steps=steps,
    ).images[0]
    logger.info("[editing/inpaint] diffusion complete")

    result = result.resize((w, h), Image.LANCZOS)
    orig_rgba   = orig.convert("RGBA")
    result_rgba = result.convert("RGBA")
    mask_full   = mask.resize((w, h), Image.NEAREST)
    composited  = Image.composite(result_rgba, orig_rgba, mask_full)
    logger.info("[editing/inpaint] composited result ready")
    return composited

def _style_transfer(
    layer_img: Image.Image,
    original_image_path: str,
    mask_path: str,          # unused, kept for unified interface
    prompt: str,
    strength: float,
    guidance_scale: float,
    steps: int,
    params: Dict[str, Any],
) -> Image.Image:
    logger.info(
        f"[editing/style_transfer] prompt='{prompt}' "
        f"strength={strength} guidance={guidance_scale} steps={steps}"
    )

    logger.info("[editing/style_transfer] loading img2img pipeline...")
    pipe = model_manager.get_img2img_pipe()

    image = Image.open(original_image_path).convert("RGB")

    w, h = image.size
    # On CPU, downscale to 512px max to avoid OOM
    if DEVICE == "cpu":
        max_side = 512
        scale_factor = min(max_side / w, max_side / h, 1.0)
        w, h = int(w * scale_factor), int(h * scale_factor)

    w8, h8 = (w // 8) * 8, (h // 8) * 8
    image = image.resize((w8, h8), Image.LANCZOS)

    logger.info("[editing/style_transfer] running img2img pipeline...")

    result = pipe(
        prompt=prompt,
        image=image,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
    ).images[0]

    result = result.resize((w, h), Image.LANCZOS)

    logger.info("[editing/style_transfer] completed")

    # return result.convert("RGBA")
    alpha = layer_img.getchannel("A").resize((w, h), Image.LANCZOS)
    result = result.convert("RGBA")
    result.putalpha(alpha)
    return result