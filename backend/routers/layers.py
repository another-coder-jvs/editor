"""Router: POST /layers – rebuild layers from an existing session."""
from __future__ import annotations

import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException
from schemas import SegmentRequest, SegmentResponse
from services.segmentation_service import segment_objects
from services.progress_store import set_progress

router = APIRouter()
logger = logging.getLogger(__name__)

from utils import config
TEMP_DIR = config.TEMP_DIR


@router.post("", response_model=SegmentResponse)
async def build_layers(req: SegmentRequest):
    logger.info(f"[layers] session={req.session_id} image={req.image_path} objects={len(req.objects)}")
    image_path = Path(req.image_path)
    if not image_path.exists():
        logger.error(f"[layers] image not found: {image_path}")
        raise HTTPException(status_code=404, detail="Image not found")

    set_progress(req.session_id, "layers", 0.1, "Building layers…")
    try:
        layers = segment_objects(req.session_id, str(image_path), req.objects)
    except Exception as e:
        logger.exception(f"[layers] failed: {e}")
        raise
    logger.info(f"[layers] built {len(layers)} layers")
    set_progress(req.session_id, "layers", 1.0, "Layers ready", done=True)
    return SegmentResponse(session_id=req.session_id, layers=layers)
