"""
Remote upscaling service — replaces local Real-ESRGAN.
Supports: fal.ai (Real-ESRGAN), Replicate (Real-ESRGAN)
"""
from __future__ import annotations

import logging

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


def upscale_remote(img: Image.Image, scale: int = 2) -> Image.Image:
    """
    Upscale a PIL image using the configured remote provider.
    Returns an RGB PIL image at `scale`× the original resolution.
    Alpha channel is NOT included (same as local Real-ESRGAN output).
    """
    provider = remote_cfg.UPSCALE_PROVIDER
    logger.info(f"[remote/upscale] provider={provider} scale={scale}x size={img.size}")

    if provider == "fal":
        return _upscale_fal(img, scale)
    elif provider == "replicate":
        return _upscale_replicate(img, scale)
    else:
        raise ValueError(f"Unknown upscale provider: {provider!r}. Choose 'fal' or 'replicate'.")


def _upscale_fal(img: Image.Image, scale: int) -> Image.Image:
    payload = {
        "image_url": image_to_data_uri(img.convert("RGB")),
        "scale":     scale,
        "face_enhance": False,
    }
    result = fal_call(remote_cfg.ENDPOINTS["esrgan_fal"], payload)
    return fal_image_result(result).convert("RGB")


def _upscale_replicate(img: Image.Image, scale: int) -> Image.Image:
    payload = {
        "input": {
            "image": image_to_data_uri(img.convert("RGB")),
            "scale": scale,
            "face_enhance": False,
        }
    }
    result = replicate_call(remote_cfg.ENDPOINTS["esrgan_replicate"], payload)
    return replicate_image_result(result).convert("RGB")
