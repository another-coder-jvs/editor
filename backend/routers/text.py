"""Router: /text/detect"""
from __future__ import annotations
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from PIL import Image
from services.text_service import detect_text
from utils import config

router = APIRouter()
logger = logging.getLogger(__name__)


class DetectTextRequest(BaseModel):
    session_id: str
    layer_id: str
    image_path: str = ""   # full original image path (preferred)


@router.post("/detect")
def detect_text_in_layer(req: DetectTextRequest):
    # Prefer full original image for accurate detection
    img = None
    if req.image_path:
        raw = req.image_path.lstrip("/")
        if raw.startswith("temp/"):
            raw = raw[len("temp/"):]
        p = config.TEMP_DIR / raw
        if p.exists():
            img = Image.open(p)

    if img is None:
        session_dir = config.TEMP_DIR / req.session_id
        layer_png = next(session_dir.glob(f"{req.layer_id}_edited*.png"), None) or \
                    next(session_dir.glob(f"{req.layer_id}_layer*.png"), None)
        if not layer_png:
            raise HTTPException(404, "Layer not found")
        img = Image.open(layer_png)

    regions = detect_text(img)
    logger.info(f"[text/detect] found {len(regions)} regions")
    return {"regions": regions}
