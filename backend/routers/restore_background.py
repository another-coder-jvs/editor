"""
Router: POST /restore-background

On-demand background restoration for an existing session.

Request body
────────────
{
  "session_id":    "abc123",
  "image_path":    "/temp/abc123/original.jpg",
  "mask_paths":    ["/temp/abc123/abc_0_shoe_mask.png", ...],
  "object_labels": ["shoe", "logo"],   // optional, one per mask
  "use_sdxl":      true,               // run SDXL texture refinement (default true)
  "force_generative": false            // kept for backwards compat, alias for use_sdxl
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
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from PIL import Image
from pydantic import BaseModel

from utils import config

router = APIRouter()
logger = logging.getLogger(__name__)
TEMP_DIR = config.TEMP_DIR


class RestoreBackgroundRequest(BaseModel):
    session_id: str
    image_path: str
    mask_paths: List[str]
    object_labels: Optional[List[str]] = None
    use_sdxl: bool = True
    force_generative: bool = False   # alias for use_sdxl (backwards compat)


class RestoreBackgroundResponse(BaseModel):
    status: str
    png_path: str
    session_id: str


@router.post("", response_model=RestoreBackgroundResponse)
async def restore_background_endpoint(req: RestoreBackgroundRequest):
    """
    Re-run background restoration on an existing session.

    Loads the original image and per-object mask files, then runs the full
    pipeline (LaMa Stage A → SDXL texture refinement → DINO validation →
    feathered blend) per object separately.

    The existing object layers are not modified.
    """
    use_sdxl = req.use_sdxl or req.force_generative
    logger.info(
        f"[restore-bg] session={req.session_id} "
        f"masks={len(req.mask_paths)} use_sdxl={use_sdxl}"
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

    # ── 2. Load mask files and build per_object_data ───────────────────────
    per_object_data: List[Dict[str, Any]] = []
    for i, mp in enumerate(req.mask_paths):
        mp_path = Path(mp)
        if not mp_path.is_absolute():
            rel = str(mp).lstrip("/")
            if rel.startswith("temp/"):
                rel = rel[len("temp/"):]
            mp_path = (TEMP_DIR / rel).resolve()
        if not mp_path.exists():
            logger.warning(f"[restore-bg] mask not found: {mp_path} — skipping")
            continue
        try:
            m = np.array(Image.open(mp_path).convert("L"))
            if m.shape != (img_h, img_w):
                m = cv2.resize(m, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            label = (
                req.object_labels[i]
                if req.object_labels and i < len(req.object_labels)
                else f"object_{i}"
            )
            per_object_data.append({"mask": m, "label": label})
        except Exception as e:
            logger.warning(f"[restore-bg] failed to load mask {mp_path}: {e}")

    if not per_object_data:
        raise HTTPException(status_code=422, detail="No valid mask files found")

    # ── 3. Run restoration ─────────────────────────────────────────────────
    try:
        from services.restoration_service import restore_background
        restored_np = restore_background(
            image_np=image_np,
            per_object_data=per_object_data,
            use_sdxl=use_sdxl,
        )
    except Exception as e:
        logger.exception(f"[restore-bg] restoration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Restoration failed: {e}")

    # ── 4. Save new background PNG ─────────────────────────────────────────
    session_dir = TEMP_DIR / req.session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    layer_id = f"{req.session_id}_bg_restored_{uuid.uuid4().hex[:6]}"
    png_name = f"{layer_id}_background_layer.png"
    png_file = session_dir / png_name

    rgba = cv2.cvtColor(restored_np, cv2.COLOR_RGB2RGBA)
    rgba[:, :, 3] = 255
    Image.fromarray(rgba).save(str(png_file), optimize=False)

    png_url = f"/temp/{req.session_id}/{png_name}"
    logger.info(f"[restore-bg] saved → {png_url}")

    return RestoreBackgroundResponse(
        status="ok",
        png_path=png_url,
        session_id=req.session_id,
    )
