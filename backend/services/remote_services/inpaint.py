"""
Remote inpainting service — replaces local FLUX.1 Fill and SDXL Inpainting.
Supports: fal.ai (FLUX Pro Fill, SDXL), Replicate (SDXL), Stability AI
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
from PIL import Image

from services.remote_config import remote_cfg
from services.remote_services._http import (
    fal_call,
    fal_image_result,
    replicate_call,
    replicate_image_result,
    image_to_data_uri,
)

logger = logging.getLogger(__name__)


def inpaint_remote(
    layer_img: Image.Image,
    prompt: str,
    strength: float = 0.75,
    guidance_scale: float = 7.5,
    steps: int = 20,
) -> Image.Image:
    """
    Run generative inpainting on a layer image using the configured remote provider.

    The layer image is RGBA — the alpha channel defines the inpaint region
    (non-transparent pixels = repaint these).

    Returns an RGBA PIL image with the original alpha channel preserved.
    """
    provider = remote_cfg.INPAINT_PROVIDER
    logger.info(f"[remote/inpaint] provider={provider} prompt='{prompt}' strength={strength}")

    orig_alpha = layer_img.getchannel("A")
    rgb = layer_img.convert("RGB")

    # Build mask: white = inpaint (where object pixels are)
    alpha_arr = np.array(orig_alpha)
    mask_arr = np.where(alpha_arr > 10, 255, 0).astype(np.uint8)
    mask_img = Image.fromarray(mask_arr)

    if provider == "fal":
        result_rgb = _inpaint_fal(rgb, mask_img, prompt, strength, guidance_scale, steps)
    elif provider == "replicate":
        result_rgb = _inpaint_replicate(rgb, mask_img, prompt, strength, guidance_scale, steps)
    elif provider == "stability":
        result_rgb = _inpaint_stability(rgb, mask_img, prompt, strength)
    else:
        raise ValueError(f"Unknown inpaint provider: {provider!r}. Choose 'fal', 'replicate', or 'stability'.")

    # Restore original RGBA alpha
    result_rgba = result_rgb.convert("RGBA")
    result_rgba = result_rgba.resize(layer_img.size, Image.LANCZOS)
    result_rgba.putalpha(orig_alpha)
    return result_rgba


def _inpaint_fal(
    rgb: Image.Image,
    mask: Image.Image,
    prompt: str,
    strength: float,
    guidance_scale: float,
    steps: int,
) -> Image.Image:
    payload = {
        "image_url":   image_to_data_uri(rgb),
        "mask_url":    image_to_data_uri(mask),
        "prompt":      prompt,
        "strength":    strength,
        "guidance_scale": guidance_scale,
        "num_inference_steps": steps,
    }
    result = fal_call(remote_cfg.ENDPOINTS["flux_fill_fal"], payload)
    return fal_image_result(result).convert("RGB")


def _inpaint_replicate(
    rgb: Image.Image,
    mask: Image.Image,
    prompt: str,
    strength: float,
    guidance_scale: float,
    steps: int,
) -> Image.Image:
    payload = {
        "input": {
            "image":           image_to_data_uri(rgb),
            "mask":            image_to_data_uri(mask),
            "prompt":          prompt,
            "prompt_strength": strength,
            "guidance_scale":  guidance_scale,
            "num_inference_steps": steps,
        }
    }
    result = replicate_call(remote_cfg.ENDPOINTS["sdxl_inpaint_replicate"], payload)
    return replicate_image_result(result).convert("RGB")


def _inpaint_stability(
    rgb: Image.Image,
    mask: Image.Image,
    prompt: str,
    strength: float,
) -> Image.Image:
    import io, requests
    buf_img = io.BytesIO()
    rgb.save(buf_img, format="PNG")
    buf_img.seek(0)

    buf_mask = io.BytesIO()
    mask.save(buf_mask, format="PNG")
    buf_mask.seek(0)

    resp = requests.post(
        remote_cfg.ENDPOINTS["inpaint_stability"],
        headers={"Authorization": f"Bearer {remote_cfg.STABILITY_API_KEY}"},
        files={
            "image": ("image.png", buf_img, "image/png"),
            "mask":  ("mask.png",  buf_mask, "image/png"),
        },
        data={
            "prompt":   prompt,
            "strength": str(strength),
            "output_format": "png",
        },
        timeout=remote_cfg.HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def img2img_remote(
    layer_img: Image.Image,
    prompt: str,
    strength: float = 0.75,
    guidance_scale: float = 7.5,
    steps: int = 20,
) -> Image.Image:
    """
    Remote img2img (style transfer) — uses SDXL img2img on fal.ai.
    Returns RGBA with original alpha preserved.
    """
    provider = remote_cfg.INPAINT_PROVIDER
    logger.info(f"[remote/img2img] provider={provider} prompt='{prompt}'")

    orig_alpha = layer_img.getchannel("A")
    rgb = layer_img.convert("RGB")

    if provider == "fal":
        payload = {
            "image_url":  image_to_data_uri(rgb),
            "prompt":     prompt,
            "strength":   strength,
            "guidance_scale": guidance_scale,
            "num_inference_steps": steps,
        }
        result = fal_call(remote_cfg.ENDPOINTS["sdxl_img2img_fal"], payload)
        result_rgb = fal_image_result(result).convert("RGB")
    elif provider == "replicate":
        payload = {
            "input": {
                "image":           image_to_data_uri(rgb),
                "prompt":          prompt,
                "prompt_strength": strength,
                "guidance_scale":  guidance_scale,
                "num_inference_steps": steps,
            }
        }
        result = replicate_call(remote_cfg.ENDPOINTS["sdxl_inpaint_replicate"], payload)
        result_rgb = replicate_image_result(result).convert("RGB")
    else:
        # Stability doesn't have a cheap img2img endpoint; fall back to inpainting
        logger.info("[remote/img2img] stability doesn't support img2img, using full-image inpaint")
        full_mask = Image.new("L", rgb.size, 255)
        result_rgb = _inpaint_stability(rgb, full_mask, prompt, strength)

    result_rgba = result_rgb.convert("RGBA").resize(layer_img.size, Image.LANCZOS)
    result_rgba.putalpha(orig_alpha)
    return result_rgba
