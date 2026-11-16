"""Router: POST /edit"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException
from PIL import Image
from schemas import EditRequest, EditResponse
from services.editing_service import edit_layer
from services.progress_store import set_progress

from utils import config
router = APIRouter()
logger = logging.getLogger(__name__)
TEMP_DIR = config.TEMP_DIR
# BASE_DIR = config.BASE_DIR


@router.post("", response_model=EditResponse)
async def edit(req: EditRequest):
    logger.info(f"[edit] session={req.session_id} layer={req.layer_id} prompt='{req.prompt}'")
    # logger.info(f"FILE : : {BASE_DIR} ")
    session_dir = TEMP_DIR / req.session_id

    layer_png = next(session_dir.glob(f"{req.layer_id}_*layer*.png"), None)
    mask_png  = next(session_dir.glob(f"{req.layer_id}_*mask*.png"),  None)
    logger.debug(f"[edit] layer_png={layer_png} mask_png={mask_png}")

    if not layer_png or not layer_png.exists():
        logger.error(f"[edit] layer PNG not found for {req.layer_id} in {session_dir}")
        raise HTTPException(status_code=404, detail="Layer PNG not found")
    if not mask_png or not mask_png.exists():
        logger.error(f"[edit] mask not found for {req.layer_id}")
        raise HTTPException(status_code=404, detail="Mask not found")
    # Resolve original image path: if req.image_path is absolute use it, otherwise join with BASE_DIR
    _raw = req.image_path.lstrip("/")
    if _raw.startswith("temp/"):
        _raw = _raw[len("temp/"):]
    image_path = TEMP_DIR / _raw
    if not image_path.exists():
        # fallback: try as absolute
        image_path = Path(req.image_path)
    logger.info(f"[edit] resolved original image path: {image_path}")
    if not image_path.exists():
        logger.error(f"[edit] original image not found: {image_path}")
        raise HTTPException(status_code=404, detail="Original image not found")

    # Layer bbox from session meta — used to crop results that come back in
    # full-canvas coordinates (e.g. text_edit) down to the layer's tight box.
    meta_file = session_dir / "session_meta.json"
    meta = None
    layer_bbox = None
    if meta_file.exists():
        try:
            meta = json.loads(meta_file.read_text())
            for layer in meta.get("layers", []):
                if layer.get("id") == req.layer_id:
                    layer_bbox = layer.get("bbox")
                    break
        except Exception as e:
            logger.warning(f"[edit] failed to read session_meta: {e}")

    set_progress(req.session_id, "edit", 0.1, f"Editing layer {req.layer_id}…")
    try:
        edited_path = edit_layer(
            session_id=req.session_id,
            layer_id=req.layer_id,
            layer_name=req.layer_id.split("_")[2] if "_" in req.layer_id else req.layer_id,
            edit_type_override=req.edit_type,
            edit_params_override=req.edit_params,
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
    # Layer PNGs are cropped to their content bbox. If a handler returned a
    # full-canvas image, crop it back so it still lines up with the layer.
    if layer_bbox:
        try:
            edited = Image.open(edited_path)
            base = Image.open(layer_png)
            if edited.size != base.size:
                bx = max(0, int(round(layer_bbox["x"])))
                by = max(0, int(round(layer_bbox["y"])))
                bw = int(round(layer_bbox["width"]))
                bh = int(round(layer_bbox["height"]))
                edited.crop(
                    (bx, by, min(edited.width, bx + bw), min(edited.height, by + bh))
                ).save(edited_path)
                logger.info(f"[edit] cropped full-canvas result to layer bbox ({bw}x{bh})")
        except Exception as e:
            logger.warning(f"[edit] could not crop oversized edit result: {e}")

    logger.info(f"[edit] done → {edited_path}")
    set_progress(req.session_id, "edit", 1.0, "Edit complete", done=True)
    try:
        rel = Path(edited_path).relative_to(TEMP_DIR)
    except ValueError:
        rel = Path(*Path(edited_path).parts[-2:])
    edited_path = "/temp/" + str(rel).replace("\\", "/")

    # Persist updated png_path to session_meta.json so refresh restores the edit
    if meta is not None:
        try:
            for layer in meta.get("layers", []):
                if layer.get("id") == req.layer_id:
                    layer["png_path"] = edited_path
                    break
            meta_file.write_text(json.dumps(meta))
        except Exception as e:
            logger.warning(f"[edit] failed to update session_meta: {e}")

    return EditResponse(layer_id=req.layer_id, edited_png_path=edited_path, session_id=req.session_id)
