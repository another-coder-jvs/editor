"""Router: POST /inpaint-bg — reconstruct background behind a layer using its mask.

Uses FLUX.1 Fill for generative reconstruction (preferred),
with LaMa / cv2 fallback if FLUX is unavailable.
"""
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
    prompt: str = "clean background, natural scene continuation, empty area"  # FLUX guidance


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

    mask_arr = np.array(mask_raw)

    # Try FLUX.1 Fill first (generative — never uses original cutout pixels)
    result = None
    try:
        from services.flux_inpaint_service import reconstruct_background_with_flux
        logger.info("[inpaint-bg] attempting FLUX.1 Fill reconstruction...")
        result = reconstruct_background_with_flux(
            original_image=orig,
            combined_mask=mask_arr,
            prompt=req.prompt,
            num_inference_steps=20,
            guidance_scale=3.5,
        )
        logger.info("[inpaint-bg] FLUX.1 Fill reconstruction complete")
    except Exception as e:
        logger.warning(f"[inpaint-bg] FLUX.1 Fill failed ({e}), trying LaMa fallback")

    # Fallback to LaMa inpainting
    if result is None:
        try:
            from services.inpaint_service import inpaint_background
            result = inpaint_background(orig, mask_arr)
            logger.info("[inpaint-bg] LaMa reconstruction complete")
        except Exception as e2:
            logger.warning(f"[inpaint-bg] LaMa failed ({e2}), using cv2 fallback")
            from services.inpaint_service import cv2_inpaint
            result = cv2_inpaint(orig, mask_arr)

    # Build an RGBA patch: reconstructed pixels only inside the layer's mask,
    # transparent elsewhere.  Writing the whole opaque frame would hide every
    # layer below this one, so the alpha channel is the mask.
    if result.size != orig.size:
        result = result.resize(orig.size, Image.LANCZOS)
    recon_rgb = np.array(result.convert("RGB"))
    rgba = np.dstack([recon_rgb, mask_arr]).astype(np.uint8)

    # Crop to the mask's content box so the patch matches the layer's tight
    # bbox (segmentation stores layer PNGs cropped to their content).
    cols = np.flatnonzero(rgba[:, :, 3].max(axis=0) > 0)
    rows = np.flatnonzero(rgba[:, :, 3].max(axis=1) > 0)
    if cols.size == 0 or rows.size == 0:
        raise HTTPException(422, "Mask is empty")
    x1, x2 = int(cols[0]), int(cols[-1]) + 1
    y1, y2 = int(rows[0]), int(rows[-1]) + 1
    patch = rgba[y1:y2, x1:x2]
    bbox = {
        "x": float(x1), "y": float(y1),
        "width": float(x2 - x1), "height": float(y2 - y1),
    }

    out_name = f"{req.layer_id}_bg_reconstructed_{uuid.uuid4().hex[:8]}.png"
    out_path = config.TEMP_DIR / req.session_id / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(patch).save(str(out_path))

    return {
        "path": f"/temp/{req.session_id}/{out_name}",
        "layer_id": req.layer_id,
        "bbox": bbox,
    }
