"""
Remote background removal service — replaces local rembg u2net.
Supports: fal.ai (rembg / Bria RMBG), remove.bg, Stability AI
"""
from __future__ import annotations

import io
import logging

import requests
from PIL import Image

from services.remote_config import remote_cfg
from services.remote_services._http import (
    fal_call,
    fal_image_result,
    image_to_data_uri,
)

logger = logging.getLogger(__name__)


def remove_background_remote(img: Image.Image) -> Image.Image:
    """
    Remove background from a PIL image using the configured remote provider.
    Returns an RGBA PIL image with background removed (transparent).
    """
    provider = remote_cfg.REMBG_PROVIDER
    logger.info(f"[remote/rembg] provider={provider}")

    if provider == "fal":
        return _rembg_fal(img)
    elif provider == "removebg":
        return _rembg_removebg(img)
    elif provider == "stability":
        return _rembg_stability(img)
    else:
        raise ValueError(f"Unknown rembg provider: {provider!r}. Choose 'fal', 'removebg', or 'stability'.")


def _rembg_fal(img: Image.Image) -> Image.Image:
    payload = {"image_url": image_to_data_uri(img.convert("RGB"))}
    result = fal_call(remote_cfg.ENDPOINTS["rembg_fal"], payload)
    return fal_image_result(result).convert("RGBA")


def _rembg_removebg(img: Image.Image) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    buf.seek(0)

    resp = requests.post(
        remote_cfg.ENDPOINTS["removebg"],
        headers={"X-Api-Key": remote_cfg.REMOVEBG_API_KEY},
        files={"image_file": ("image.png", buf, "image/png")},
        data={"size": "auto"},
        timeout=remote_cfg.HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGBA")


def _rembg_stability(img: Image.Image) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    buf.seek(0)

    resp = requests.post(
        remote_cfg.ENDPOINTS["rembg_stability"],
        headers={"Authorization": f"Bearer {remote_cfg.STABILITY_API_KEY}"},
        files={"image": ("image.png", buf, "image/png")},
        data={"output_format": "png"},
        timeout=remote_cfg.HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGBA")
