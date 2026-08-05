"""Router: POST /edit"""
from __future__ import annotations

import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException
from schemas import EditRequest, EditResponse
from services.editing_service import edit_layer
from services.progress_store import set_progress

from utils import config
router = APIRouter()
logger = logging.getLogger(__name__)
TEMP_DIR = config.TEMP_DIR


@router.post("", response_model=EditResponse)
async def edit(req: EditRequest):
    logger.info(f"[edit] session={req.session_id} layer={req.layer_id} prompt='{req.prompt}'")
    logger.info(f"FILE : : {BASE_DIR} ")
    session_dir = TEMP_DIR / req.session_id

    layer_png = next(session_dir.glob(f"{req.layer_id}_layer*.png"), None)
    mask_png  = next(session_dir.glob(f"{req.layer_id}_mask*.png"),  None)
    logger.debug(f"[edit] layer_png={layer_png} mask_png={mask_png}")

    if not layer_png or not layer_png.exists():
        logger.error(f"[edit] layer PNG not found for {req.layer_id} in {session_dir}")
        raise HTTPException(status_code=404, detail="Layer PNG not found")
    if not mask_png or not mask_png.exists():
        logger.error(f"[edit] mask not found for {req.layer_id}")
        raise HTTPException(status_code=404, detail="Mask not found")
    # Resolve original image path: if req.image_path is absolute use it, otherwise join with BASE_DIR
    image_path = Path(req.image_path)
    if not image_path.is_absolute():
        image_path = TEMP_DIR / req.image_path
    logger.info(f"[edit] resolved original image path: {image_path}")
    if not image_path.exists():
        logger.error(f"[edit] original image not found: {image_path}")
        raise HTTPException(status_code=404, detail="Original image not found")

    set_progress(req.session_id, "edit", 0.1, f"Editing layer {req.layer_id}…")
    try:
        edited_path = edit_layer(
            session_id=req.session_id,
            layer_id=req.layer_id,
            layer_name=req.layer_id.split("_")[2] if "_" in req.layer_id else req.layer_id,
            layer_png_path=str(layer_png),
            original_image_path=str(image_path),
            mask_path=str(mask_png),
            prompt=req.prompt,
            strength=req.strength,
            guidance_scale=req.guidance_scale,
            steps=req.steps,
        )
    except Exception as e:
        logger.exception(f"[edit] edit_layer failed: {e}")
        raise
    logger.info(f"[edit] done → {edited_path}")
    set_progress(req.session_id, "edit", 1.0, "Edit complete", done=True)
    edited_path = "/temp/" + str(Path(edited_path).relative_to(TEMP_DIR)).replace("\\", "/")
    return EditResponse(layer_id=req.layer_id, edited_png_path=edited_path, session_id=req.session_id)
