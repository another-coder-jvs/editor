"""Router: POST /upload — save an image without running detection."""
from __future__ import annotations

import logging
import uuid
import shutil
from pathlib import Path

from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel

from utils import config

router = APIRouter()
logger = logging.getLogger(__name__)
TEMP_DIR = config.TEMP_DIR
TEMP_DIR.mkdir(parents=True, exist_ok=True)


class UploadResponse(BaseModel):
    session_id: str
    image_path: str


@router.post("", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    session_id = uuid.uuid4().hex
    session_dir = TEMP_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[upload] session={session_id} file={file.filename}")

    filename = file.filename or "upload.png"
    image_path = session_dir / filename
    with open(image_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    logger.info(f"[upload] saved image to {image_path}")

    return UploadResponse(session_id=session_id, image_path=str(image_path))
