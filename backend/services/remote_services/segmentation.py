"""
Remote segmentation service — replaces local SAM2.
Returns a binary mask (numpy uint8 array, 0/255) for each detected object.
Supports: fal.ai SAM2, Replicate SAM2
"""
from __future__ import annotations

import logging
from io import BytesIO
from typing import Any, Dict, List, Optional

import numpy as np
import requests
from PIL import Image

from services.remote_config import remote_cfg
from services.remote_services._http import (
    fal_call,
    fal_image_result,
    replicate_call,
    path_to_data_uri,
    image_to_data_uri,
)

logger = logging.getLogger(__name__)


def segment_object_remote(
    image_path: str,
    bbox: Dict[str, float],
    image_size: tuple[int, int],
) -> np.ndarray:
    """
    Segment a single object given its bounding box.

    Args:
        image_path: path to the full image
        bbox: {"x": x1, "y": y1, "width": w, "height": h}
        image_size: (width, height) of the image

    Returns:
        Binary mask as numpy uint8 array (0 or 255), same size as image.
    """
    provider = remote_cfg.SEGMENTATION_PROVIDER
    logger.info(f"[remote/segmentation] provider={provider} bbox={bbox}")

    if provider == "fal":
        return _segment_fal(image_path, bbox, image_size)
    elif provider == "replicate":
        return _segment_replicate(image_path, bbox, image_size)
    else:
        raise ValueError(f"Unknown segmentation provider: {provider!r}. Choose 'fal' or 'replicate'.")


def _bbox_to_points(bbox: Dict[str, float], img_w: int, img_h: int) -> list:
    """Convert a bbox dict to normalized [x_min, y_min, x_max, y_max] coords."""
    x1 = bbox["x"] / img_w
    y1 = bbox["y"] / img_h
    x2 = (bbox["x"] + bbox["width"]) / img_w
    y2 = (bbox["y"] + bbox["height"]) / img_h
    return [max(0.0, x1), max(0.0, y1), min(1.0, x2), min(1.0, y2)]


def _segment_fal(
    image_path: str,
    bbox: Dict[str, float],
    image_size: tuple[int, int],
) -> np.ndarray:
    img_w, img_h = image_size
    # fal SAM2 takes pixel-space prompts
    x1 = bbox["x"]
    y1 = bbox["y"]
    x2 = bbox["x"] + bbox["width"]
    y2 = bbox["y"] + bbox["height"]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    payload = {
        "image_url": path_to_data_uri(image_path),
        "prompts": [
            {
                "type": "box",
                "x_min": x1,
                "y_min": y1,
                "x_max": x2,
                "y_max": y2,
            },
            {
                "type": "point",
                "x": cx,
                "y": cy,
                "label": 1,  # foreground
            },
        ],
    }

    result = fal_call(remote_cfg.ENDPOINTS["sam2_fal"], payload)

    # fal SAM2 returns a mask image URL
    mask_data = result.get("mask") or result.get("masks", [{}])[0]
    if isinstance(mask_data, dict):
        url = mask_data.get("url", "")
    else:
        url = str(mask_data)

    if not url:
        logger.warning("[remote/segmentation] fal returned no mask URL, using bbox mask")
        return _bbox_fallback_mask(img_w, img_h, bbox)

    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    mask_img = Image.open(BytesIO(resp.content)).convert("L").resize((img_w, img_h), Image.NEAREST)
    mask_arr = np.array(mask_img)
    return (mask_arr > 128).astype(np.uint8) * 255


def _segment_replicate(
    image_path: str,
    bbox: Dict[str, float],
    image_size: tuple[int, int],
) -> np.ndarray:
    img_w, img_h = image_size
    x1, y1 = bbox["x"], bbox["y"]
    x2, y2 = x1 + bbox["width"], y1 + bbox["height"]

    payload = {
        "input": {
            "image": path_to_data_uri(image_path),
            "input_boxes": [[x1, y1, x2, y2]],
        }
    }

    result = replicate_call(remote_cfg.ENDPOINTS["sam2_replicate"], payload)
    output = result.get("output", [])

    if not output:
        logger.warning("[remote/segmentation] replicate returned no masks, using bbox mask")
        return _bbox_fallback_mask(img_w, img_h, bbox)

    # output is a list of mask image URLs
    mask_url = output[0] if isinstance(output[0], str) else output[0].get("url", "")
    resp = requests.get(mask_url, timeout=60)
    resp.raise_for_status()
    mask_img = Image.open(BytesIO(resp.content)).convert("L").resize((img_w, img_h), Image.NEAREST)
    mask_arr = np.array(mask_img)
    return (mask_arr > 128).astype(np.uint8) * 255


def _bbox_fallback_mask(img_w: int, img_h: int, bbox: Dict[str, float]) -> np.ndarray:
    """Last-resort fallback: fill the bounding box as a rectangular mask."""
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    x1 = max(0, int(bbox["x"]))
    y1 = max(0, int(bbox["y"]))
    x2 = min(img_w, int(bbox["x"] + bbox["width"]))
    y2 = min(img_h, int(bbox["y"] + bbox["height"]))
    mask[y1:y2, x1:x2] = 255
    return mask
