"""Router: POST /segment"""
from __future__ import annotations

import logging
from fastapi import APIRouter
from schemas import SegmentRequest, SegmentResponse
from services.segmentation_service import segment_objects
from services.progress_store import set_progress

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("", response_model=SegmentResponse)
async def segment(req: SegmentRequest):
    logger.info(f"[segment] session={req.session_id} objects={len(req.objects)} image={req.image_path}")
    set_progress(req.session_id, "segment", 0.1, "Running SAM2 segmentation…")
    try:
        layers = segment_objects(req.session_id, req.image_path, req.objects)
    except Exception as e:
        logger.exception(f"[segment] failed: {e}")
        raise
    logger.info(f"[segment] created {len(layers)} layers")
    set_progress(req.session_id, "segment", 1.0, "Segmentation complete", done=True)
    return SegmentResponse(session_id=req.session_id, layers=layers)
