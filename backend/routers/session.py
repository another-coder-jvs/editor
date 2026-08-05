"""Router: GET /session/latest – return the most recently modified valid session."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from utils import config
router = APIRouter()
logger = logging.getLogger(__name__)

TEMP_DIR = config.TEMP_DIR or  Path(__file__).resolve().parents[2] / "temp" 


@router.get("/latest")
def get_latest_session():
    logger.info(f"Checking Dir : {TEMP_DIR}")
    if not TEMP_DIR.exists():
        return JSONResponse({"session": None})

    best_dir = None
    best_mtime = 0.0
    logger.info(f"Checking list in : {TEMP_DIR}")
    for d in TEMP_DIR.iterdir():
        if not d.is_dir():
            continue
        meta_file = d / "session_meta.json"
        if not meta_file.exists():
            continue
        mtime = meta_file.stat().st_mtime
        if mtime > best_mtime:
            best_mtime = mtime
            best_dir = d
    logger.info(f"best_dir : {best_dir}")
    if best_dir is None:
        return JSONResponse({"session": None})

    try:
        meta = json.loads((best_dir / "session_meta.json").read_text())
        logger.info(f"meta : {meta}")
    except Exception as e:
        logger.warning(f"[session/latest] failed to read meta: {e}")
        return JSONResponse({"session": None})

    # Validate all layer PNGs still exist
    for layer in meta.get("layers", []):
        png_rel = layer.get("png_path", "").lstrip("/temp")
        logger.info(f"Checking : {config.TEMP_DIR}/{png_rel}")
        if not (config.TEMP_DIR / png_rel).exists():
            logger.warning(f"[session/latest] missing layer file {png_rel}, skipping session")
            return JSONResponse({"session": None})

    # Validate original image exists
    img_path = meta.get("image_path", "").lstrip("/")
    img_abs = (config.TEMP_DIR / img_path).resolve()
    if not img_abs.exists():
        return JSONResponse({"session": None})

    logger.info(f"[session/latest] restoring session {meta['session_id']}")
    return JSONResponse({"session": meta})
