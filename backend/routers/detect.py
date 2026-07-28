"""Router: POST /detect"""
from __future__ import annotations

import logging
import uuid
import shutil
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form
from typing import Optional

from schemas import DetectResponse
from services.detection_service import detect_objects
from services.progress_store import set_progress

router = APIRouter()

TEMP_DIR = Path(__file__).resolve().parents[2] / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)


logger = logging.getLogger(__name__)


@router.post("", response_model=DetectResponse)
async def detect(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
):
    session_id = uuid.uuid4().hex
    session_dir = TEMP_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[detect] session={session_id} file={file.filename} content_type={file.content_type}")

    filename = file.filename or "upload.png"
    image_path = session_dir / filename
    with open(image_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    logger.info(f"[detect] saved image to {image_path}")

    set_progress(session_id, "detect", 0.1, "Running object detection…")
    try:
        objects = detect_objects(str(image_path), prompt)
    except Exception as e:
        logger.exception(f"[detect] detection failed: {e}")
        raise
    logger.info(f"[detect] found {len(objects)} objects: {[o['label'] for o in objects]}")
    set_progress(session_id, "detect", 1.0, "Detection complete", done=True)

    return DetectResponse(objects=objects, session_id=session_id)
