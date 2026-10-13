"""Router: POST /detect/re — re-detect objects with a custom prompt on an existing session image."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.detection_service import detect_objects
from services.progress_store import set_progress
from utils import config

router = APIRouter()
logger = logging.getLogger(__name__)
TEMP_DIR = config.TEMP_DIR


class RedetectRequest(BaseModel):
    session_id: str
    image_path: str
    prompt: Optional[str] = None  # custom prompt; empty/None = use fallback


class RedetectResponse(BaseModel):
    session_id: str
    image_path: str
    objects: List[Dict[str, Any]]


@router.post("/re", response_model=RedetectResponse)
async def redetect(req: RedetectRequest):
    logger.info(f"[redetect] session={req.session_id} prompt='{req.prompt}'")

    # Resolve image path
    _raw = req.image_path.lstrip("/")
    if _raw.startswith("temp/"):
        _raw = _raw[len("temp/"):]
    image_path = TEMP_DIR / _raw
    if not image_path.exists():
        image_path = Path(req.image_path)
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {req.image_path}")

    set_progress(req.session_id, "detect", 0.1, "Running object detection…")
    try:
        objects = detect_objects(str(image_path), req.prompt or None)
    except Exception as e:
        logger.exception(f"[redetect] detection failed: {e}")
        raise

    logger.info(f"[redetect] found {len(objects)} objects: {[o['label'] for o in objects]}")
    set_progress(req.session_id, "detect", 1.0, "Detection complete", done=True)

    return RedetectResponse(
        session_id=req.session_id,
        image_path=str(image_path),
        objects=objects,
    )
