"""Router: POST /identify — use Ollama vision to identify objects in an image."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.identify_service import identify_objects, check_ollama
from utils import config

router = APIRouter()
logger = logging.getLogger(__name__)
TEMP_DIR = config.TEMP_DIR


class IdentifyRequest(BaseModel):
    image_path: str


class IdentifyResponse(BaseModel):
    objects: str  # dot-separated object list


@router.post("", response_model=IdentifyResponse)
async def identify(req: IdentifyRequest):
    logger.info(f"[identify] request for: {req.image_path}")

    # Resolve image path
    _raw = req.image_path.lstrip("/")
    if _raw.startswith("temp/"):
        _raw = _raw[len("temp/"):]
    image_path = TEMP_DIR / _raw
    if not image_path.exists():
        image_path = Path(req.image_path)
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {req.image_path}")

    if not check_ollama():
        raise HTTPException(
            status_code=503,
            detail="Ollama is not running. Start it with: ollama serve",
        )

    try:
        result = identify_objects(str(image_path))
        logger.info(f"[identify] result: {result}")
        return IdentifyResponse(objects=result)
    except Exception as e:
        logger.exception(f"[identify] failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
