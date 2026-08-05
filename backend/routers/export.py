"""Router: POST /export"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path

import numpy as np
from PIL import Image
from fastapi import APIRouter
from fastapi.responses import FileResponse
from schemas import ExportRequest
from services.merge_service import merge_layers
from services.model_manager import model_manager
from services.progress_store import set_progress

router = APIRouter()
logger = logging.getLogger(__name__)

from utils import config
OUTPUTS_DIR = config.OUTPUT_DIR
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


@router.post("")
async def export_image(req: ExportRequest):
    logger.info(f"[export] session={req.session_id} format={req.format} upscale={req.upscale}x{req.upscale_factor}")
    set_progress(req.session_id, "export", 0.1, "Exporting…")
    try:
        merged = merge_layers(req.layers, req.canvas_width, req.canvas_height)
        logger.info(f"[export] merged canvas {req.canvas_width}x{req.canvas_height}")

        if req.upscale:
            logger.info(f"[export] upscaling {req.upscale_factor}x with Real-ESRGAN")
            upsampler = model_manager.get_realesrgan(req.upscale_factor)
            arr = np.array(merged.convert("RGB"))
            out_arr, _ = upsampler.enhance(arr, outscale=req.upscale_factor)
            merged = Image.fromarray(out_arr)

        fmt = req.format.lower()
        out_name = f"export_{uuid.uuid4().hex[:8]}.{fmt}"
        out_path = OUTPUTS_DIR / out_name

        if fmt in ("jpg", "jpeg"):
            bg = Image.new("RGB", merged.size, (255, 255, 255))
            bg.paste(merged, mask=merged.getchannel("A"))
            bg.save(str(out_path), quality=95)
        elif fmt == "webp":
            merged.save(str(out_path), format="WEBP", quality=90)
        else:
            merged.save(str(out_path), format="PNG")

        logger.info(f"[export] saved → {out_path}")
    except Exception as e:
        logger.exception(f"[export] failed: {e}")
        raise

    set_progress(req.session_id, "export", 1.0, "Export complete", done=True)
    return FileResponse(str(out_path), filename=out_name)
