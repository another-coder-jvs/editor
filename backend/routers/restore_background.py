"""
Router: POST /restore-background

On-demand background restoration for an existing session layer.

Use this when the automatic restoration during segmentation needs to be
re-run with stronger settings (more inference steps, forced generative mode)
or when the user manually triggers a "Restore Background" action in the UI.

Request body
────────────
{
  "session_id":    "abc123",          // session identifier
  "image_path":    "/temp/abc123/original.jpg",  // original uploaded image
  "mask_paths":    [                  // paths to per-object mask PNGs
    "/temp/abc123/abc_0_shoe_mask.png",
    "/temp/abc123/abc_1_logo_mask.png"
  ],
  "object_labels": ["shoe", "logo"], // names of the removed objects (optional)
  "force_generative": false,          // set true to skip complexity check
  "steps":         30                 // diffusion steps (default 30)
}

Response body
─────────────
{
  "status": "ok",
  "png_path": "/temp/abc123/abc_bg_restored_XXXX_background_layer.png",
  "session_id": "abc123"
}
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from PIL import Image
from pydantic import BaseModel

from utils import config

router = APIRouter()
logger = logging.getLogger(__name__)
TEMP_DIR = config.TEMP_DIR


# ── Request / Response schemas ─────────────────────────────────────────────

class RestoreBackgroundRequest(BaseModel):
    session_id: str
    image_path: str
    mask_paths: List[str]
    object_labels: Optional[List[str]] = None
    force_generative: bool = False
    steps: int = 30


class RestoreBackgroundResponse(BaseModel):
    status: str
    png_path: str
    session_id: str


# ── Route ─────────────────────────────────────────────────────────────────

@router.post("", response_model=RestoreBackgroundResponse)
async def restore_background_endpoint(req: RestoreBackgroundRequest):
    """
    Re-run background restoration on an existing session.

    Loads the original image and the per-object mask files, then runs the
    full hybrid pipeline (Stage A → cv2, Stage B → SDXL/FLUX generative,
    Stage C → seam blend) and saves the result as a new background PNG.

    The existing object layers are not modified.
    """
    logger.info(
        f"[restore-bg] session={req.session_id} "
        f"masks={len(req.mask_paths)} "
        f"force_gen={req.force_generative} steps={req.steps}"
    )

    # ── 1. Resolve original image ──────────────────────────────────────────
    p = Path(req.image_path)
    if not p.is_absolute():
        rel = req.image_path.lstrip("/")
        if rel.startswith("temp/"):
            rel = rel[len("temp/"):]
        p = (TEMP_DIR / rel).resolve()

    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {req.image_path}")

    try:
        image_pil = Image.open(p).convert("RGB")
        image_np  = np.array(image_pil)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Cannot open image: {e}")

    img_h, img_w = image_np.shape[:2]

    # ── 2. Run restoration ─────────────────────────────────────────────────
    try:
        from services.restoration_service import restore_background_from_mask_files
        restored_np = restore_background_from_mask_files(
            image_np=image_np,
            mask_paths=req.mask_paths,
            object_labels=req.object_labels,
            force_generative=req.force_generative,
            steps=req.steps,
        )
    except Exception as e:
        logger.exception(f"[restore-bg] restoration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Restoration failed: {e}")

    # ── 3. Save new background PNG ─────────────────────────────────────────
    session_dir = TEMP_DIR / req.session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    layer_id = f"{req.session_id}_bg_restored_{uuid.uuid4().hex[:6]}"
    png_name = f"{layer_id}_background_layer.png"
    png_file = session_dir / png_name

    rgba = cv2.cvtColor(restored_np, cv2.COLOR_RGB2RGBA)
    rgba[:, :, 3] = 255  # fully opaque
    Image.fromarray(rgba).save(str(png_file), optimize=False)

    png_url = f"/temp/{req.session_id}/{png_name}"
    logger.info(f"[restore-bg] saved restored background → {png_url}")

    return RestoreBackgroundResponse(
        status="ok",
        png_path=png_url,
        session_id=req.session_id,
    )
