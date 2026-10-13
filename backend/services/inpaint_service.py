"""Shared background inpainting helpers."""
from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image

from services.text_service import _get_lama

logger = logging.getLogger(__name__)


def cv2_inpaint(img: Image.Image, mask_arr: np.ndarray) -> Image.Image:
    arr = np.array(img)
    inpainted = cv2.inpaint(arr, mask_arr, inpaintRadius=12, flags=cv2.INPAINT_TELEA)
    inpainted = cv2.inpaint(inpainted, mask_arr, inpaintRadius=6, flags=cv2.INPAINT_NS)
    return Image.fromarray(inpainted)


def prepare_inpaint_mask(mask: np.ndarray, dilate: int = 15) -> np.ndarray:
    """Convert a soft mask to a binary inpaint region, dilated to cover feathered edges."""
    binary = (mask > 32).astype(np.uint8) * 255
    if dilate > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
        binary = cv2.dilate(binary, kernel, iterations=1)
    return binary


def inpaint_background(image: Image.Image, mask: np.ndarray, dilate: int = 15) -> Image.Image:
    """Fill masked regions with reconstructed background pixels."""
    mask_arr = prepare_inpaint_mask(mask, dilate=dilate)
    if not np.any(mask_arr):
        return image.copy()

    mask_pil = Image.fromarray(mask_arr)
    lama = _get_lama()
    if lama is not None:
        try:
            result = lama(image, mask_pil)
            logger.info("[inpaint] LaMa reconstruction done")
            return result
        except Exception as e:
            logger.warning(f"[inpaint] LaMa failed ({e}), falling back to cv2.inpaint")

    return cv2_inpaint(image, mask_arr)
