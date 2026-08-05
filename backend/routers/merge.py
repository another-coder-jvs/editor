"""Router: POST /merge"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter
from schemas import MergeRequest, MergeResponse
from services.merge_service import merge_layers
from services.progress_store import set_progress

router = APIRouter()
logger = logging.getLogger(__name__)

from utils import config
OUTPUTS_DIR = config.OUTPUT_DIR
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


@router.post("", response_model=MergeResponse)
async def merge(req: MergeRequest):
    logger.info(f"[merge] session={req.session_id} layers={len(req.layers)} size={req.canvas_width}x{req.canvas_height} fmt={req.output_format}")
    set_progress(req.session_id, "merge", 0.1, "Merging layers…")
    try:
        merged = merge_layers(req.layers, req.canvas_width, req.canvas_height)
    except Exception as e:
        logger.exception(f"[merge] failed: {e}")
        raise

    fmt = req.output_format.lower()
    out_name = f"{req.session_id}_merged_{uuid.uuid4().hex[:6]}.{fmt}"
    out_path = OUTPUTS_DIR / out_name

    if fmt in ("jpeg", "jpg"):
        merged.convert("RGB").save(str(out_path), quality=95)
    elif fmt == "webp":
        merged.save(str(out_path), format="WEBP", quality=90)
    else:
        merged.save(str(out_path), format="PNG")

    logger.info(f"[merge] saved → {out_path}")
    set_progress(req.session_id, "merge", 1.0, "Merge complete", done=True)
    return MergeResponse(output_path=str(out_path), session_id=req.session_id)
