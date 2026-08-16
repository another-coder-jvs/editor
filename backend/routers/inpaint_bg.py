"""Router: POST /inpaint-bg — reconstruct background behind a layer using its mask."""
from __future__ import annotations
import uuid
import logging
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from PIL import Image
from utils import config
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

    from services.inpaint_service import inpaint_background
    result = inpaint_background(orig, np.array(mask_raw))

    out_name = f"{req.layer_id}_bg_reconstructed_{uuid.uuid4().hex[:8]}.png"
    out_path = config.TEMP_DIR / req.session_id / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(str(out_path))

    return {"path": f"/temp/{req.session_id}/{out_name}"}
