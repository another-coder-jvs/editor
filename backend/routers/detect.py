"""Router: POST /detect"""
from __future__ import annotations

import logging
import uuid
import shutil
import numpy as np
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form
from typing import Optional

from schemas import DetectResponse
from services.detection_service import detect_objects
from services.background_detector import analyze_background, BackgroundType
from services.progress_store import set_progress

router = APIRouter()
from utils import config
TEMP_DIR = config.TEMP_DIR
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

    set_progress(session_id, "detect", 0.1, "Analyzing background...")
    try:
        from PIL import Image
        image_pil = Image.open(image_path).convert("RGB")
        image_np = np.array(image_pil)
        bg_analysis = analyze_background(image_np)
        logger.info(
            f"[detect] background analysis: type={bg_analysis.bg_type.value}, "
            f"dominant_color={bg_analysis.dominant_color}, "
            f"confidence={bg_analysis.confidence:.2f}"
        )
    except Exception as e:
        logger.warning(f"[detect] background analysis failed: {e}")
        bg_analysis = None

    set_progress(session_id, "detect", 0.3, "Running object detection…")
    try:
        objects = detect_objects(str(image_path), prompt)
    except Exception as e:
        logger.exception(f"[detect] detection failed: {e}")
        raise
    logger.info(f"[detect] found {len(objects)} objects: {[o['label'] for o in objects]}")
    set_progress(session_id, "detect", 1.0, "Detection complete", done=True)

    response_data = DetectResponse(objects=objects, session_id=session_id, image_path=str(image_path))
    
    # Add background analysis to response if available
    if bg_analysis:
        response_data_dict = response_data.model_dump()
        response_data_dict["background_analysis"] = {
            "type": bg_analysis.bg_type.value,
            "dominant_color": list(bg_analysis.dominant_color),
            "color_variance": bg_analysis.color_variance,
            "gradient_direction": bg_analysis.gradient_direction,
            "confidence": bg_analysis.confidence,
        }
        return response_data_dict
    
    return response_data
