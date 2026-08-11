"""Router: /text/detect, /text/erase-bg, /text/render"""
from __future__ import annotations
import logging
import uuid
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Any
from PIL import Image
from services.text_service import detect_text, erase_text_regions, render_text_patch
from utils import config

router = APIRouter()
logger = logging.getLogger(__name__)


class DetectTextRequest(BaseModel):
    session_id: str
    layer_id: str
    image_path: str = ""


class EraseBgRequest(BaseModel):
    session_id: str
    layer_id: str
    image_path: str = ""   # original layer PNG path
    regions: List[Any]     # list of region dicts with "quad"


class RenderTextRequest(BaseModel):
    session_id: str
    region: Any            # region dict: bbox, color, font_size, quad
    new_text: str
    overrides: Any = None  # { color, font_size, shadow_color, shadow_offset, rotation }


@router.post("/detect")
def detect_text_in_layer(req: DetectTextRequest):
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


@router.post("/erase-bg")
def erase_bg(req: EraseBgRequest):
    """Erase text regions from the layer PNG using cv2.inpaint, return path to result."""
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
        layer_png = next(session_dir.glob(f"{req.layer_id}_layer*.png"), None)
        if not layer_png:
            raise HTTPException(404, "Layer not found")
        img = Image.open(layer_png)

    result = erase_text_regions(img, req.regions)

    out_name = f"{req.layer_id}_bg_{uuid.uuid4().hex[:8]}.png"
    out_path = config.TEMP_DIR / req.session_id / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(str(out_path))

    return {"path": f"/temp/{req.session_id}/{out_name}"}


@router.post("/render")
def render_text(req: RenderTextRequest):
    """Render new_text onto a transparent PNG matching the region's style."""
    result = render_text_patch(req.region, req.new_text, req.overrides)

    out_name = f"textpatch_{uuid.uuid4().hex[:8]}.png"
    out_path = config.TEMP_DIR / req.session_id / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(str(out_path))

    return {"path": f"/temp/{req.session_id}/{out_name}"}

