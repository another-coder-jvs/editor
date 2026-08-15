"""Router: POST /inpaint-bg — reconstruct background behind a layer using its mask."""
from __future__ import annotations
import uuid
import logging
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from PIL import Image
from utils import config
from services.text_service import _get_lama

router = APIRouter()
logger = logging.getLogger(__name__)


class InpaintBgRequest(BaseModel):
    session_id: str
    layer_id: str
    image_path: str   # original full image
    mask_path: str    # layer mask (white = object area to reconstruct)


@router.post("")
def inpaint_bg(req: InpaintBgRequest):
    def _resolve(p: str):
        raw = p.lstrip("/")
        if raw.startswith("temp/"):
            raw = raw[len("temp/"):]
        return config.TEMP_DIR / raw

    img_path  = _resolve(req.image_path)
    mask_path = _resolve(req.mask_path)

    if not img_path.exists():
        raise HTTPException(404, f"Image not found: {req.image_path}")
    if not mask_path.exists():
        raise HTTPException(404, f"Mask not found: {req.mask_path}")

    orig = Image.open(img_path).convert("RGB")
    mask_raw = Image.open(mask_path).convert("L")

    # Resize mask to match original image if needed
    if mask_raw.size != orig.size:
        mask_raw = mask_raw.resize(orig.size, Image.NEAREST)

    # Dilate mask slightly to cover edges/anti-aliasing
    import cv2
    mask_arr = np.array(mask_raw)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask_arr = cv2.dilate(mask_arr, kernel, iterations=1)
    mask_pil = Image.fromarray(mask_arr)

    lama = _get_lama()
    if lama is not None:
        try:
            result = lama(orig, mask_pil)
            logger.info("[inpaint-bg] LaMa reconstruction done")
        except Exception as e:
            logger.warning(f"[inpaint-bg] LaMa failed ({e}), falling back to cv2.inpaint")
            result = _cv2_inpaint(orig, mask_arr)
    else:
        result = _cv2_inpaint(orig, mask_arr)

    out_name = f"{req.layer_id}_bg_reconstructed_{uuid.uuid4().hex[:8]}.png"
    out_path = config.TEMP_DIR / req.session_id / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(str(out_path))

    return {"path": f"/temp/{req.session_id}/{out_name}"}


def _cv2_inpaint(img: Image.Image, mask_arr: np.ndarray) -> Image.Image:
    import cv2
    arr = np.array(img)
    inpainted = cv2.inpaint(arr, mask_arr, inpaintRadius=12, flags=cv2.INPAINT_TELEA)
    inpainted = cv2.inpaint(inpainted, mask_arr, inpaintRadius=6, flags=cv2.INPAINT_NS)
    return Image.fromarray(inpainted)
