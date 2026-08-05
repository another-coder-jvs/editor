"""
Segmentation service – SAM2 masks + refinement + transparent PNG layers.
"""
from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

from services.model_manager import model_manager
from schemas import BoundingBox, LayerData
 
from utils import config
BASE_DIR = Path(__file__).resolve().parents[2]  
logger = logging.getLogger(__name__)

# TEMP_DIR = Path(__file__).resolve().parents[2] / "temp"
TEMP_DIR = config.TEMP_DIR

TEMP_DIR.mkdir(parents=True, exist_ok=True)


def refine_mask(mask: np.ndarray) -> np.ndarray:
    logger.debug("[segmentation] refining mask…")
    mask = mask.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    flood = mask.copy()
    h, w = flood.shape
    flood_fill_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_fill_mask, (0, 0), 255)
    mask = mask | cv2.bitwise_not(flood)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    logger.debug("[segmentation] mask refined")
    return mask


def mask_to_transparent_png(image: np.ndarray, mask: np.ndarray, bbox: Dict[str, float], out_path: Path) -> None:
    x, y, w, h = int(bbox["x"]), int(bbox["y"]), int(bbox["width"]), int(bbox["height"])
    img_h, img_w = image.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(img_w, x + w), min(img_h, y + h)
    crop_rgb  = image[y1:y2, x1:x2]
    crop_mask = mask[y1:y2, x1:x2]
    rgba = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2RGBA)
    rgba[:, :, 3] = crop_mask
    Image.fromarray(rgba).save(str(out_path))
    logger.debug(f"[segmentation] saved transparent PNG → {out_path}")


def segment_objects(session_id: str, image_path: str, objects: List[Dict[str, Any]]) -> List[LayerData]:
    logger.info(f"[segmentation] session={session_id} objects={len(objects)} image={image_path}")

    logger.info("[segmentation] loading SAM2 predictor…")
    predictor = model_manager.get_sam2()

    logger.info(f"[segmentation] opening image: {image_path}")
    # image_pil = Image.open(image_path).convert("RGB")
    # image_np  = np.array(image_pil)
    image_path = str(image_path).lstrip("/")

    # Build path relative to project root
    p = (TEMP_DIR / image_path).resolve()

    logger.info(f"BASE_DIR={BASE_DIR}")
    logger.info(f"RAW={image_path}")
    logger.info(f"FINAL={p} exists={p.exists()}")

    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")

    image_pil = Image.open(p).convert("RGB")
    image_np  = np.array(image_pil)
    logger.info(f"[segmentation] image shape: {image_np.shape}")

    logger.info("[segmentation] setting image in SAM2…")
    predictor.set_image(image_np)
    logger.info("[segmentation] image set, processing objects…")

    session_dir = TEMP_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    layers: List[LayerData] = []

    for idx, obj in enumerate(objects):
        bbox  = obj["bbox"]
        label = obj["label"]
        logger.info(f"[segmentation] [{idx+1}/{len(objects)}] '{label}' bbox={bbox}")

        box = np.array([bbox["x"], bbox["y"], bbox["x"] + bbox["width"], bbox["y"] + bbox["height"]])
        logger.debug(f"[segmentation] SAM2 box prompt: {box}")

        try:
            masks, scores, _ = predictor.predict(box=box, multimask_output=True)
            best_idx  = int(np.argmax(scores))
            raw_mask  = masks[best_idx]
            logger.info(f"[segmentation] '{label}' → best mask score={scores[best_idx]:.4f} (of {len(scores)})")
        except Exception as e:
            logger.warning(f"[segmentation] SAM2 failed for '{label}': {e} — using bbox fallback")
            raw_mask = _bbox_mask(image_np.shape[:2], bbox)
        refined = refine_mask(raw_mask)

        # generate ONCE
        layer_id = f"{session_id}_{idx}_{uuid.uuid4().hex[:6]}"

        # filenames
        mask_name = f"{layer_id}_mask.png"
        png_name  = f"{layer_id}_layer.png"

        # absolute filesystem paths (save here)
        mask_file = session_dir / mask_name
        png_file  = session_dir / png_name

        # save files
        Image.fromarray(refined).save(str(mask_file))
        logger.debug(f"[segmentation] mask saved → {mask_file}")

        mask_to_transparent_png(image_np, refined, bbox, png_file)

        # public URLs (return these)
        mask_url = f"/temp/{session_id}/{mask_name}"
        png_url  = f"/temp/{session_id}/{png_name}"

        layers.append(LayerData(
            id=layer_id,
            name=label,
            mask_path=mask_url,
            png_path=png_url,
            bbox=BoundingBox(**bbox),
            z_index=len(objects) - idx,
            visible=True,
            opacity=1.0,
        ))

        logger.info(f"[segmentation] layer '{label}' created: id={layer_id}")
        # refined   = refine_mask(raw_mask)
        # layer_id  = f"{session_id}_{idx}_{uuid.uuid4().hex[:6]}"
        # mask_path = session_dir / f"{layer_id}_mask.png"
        # png_path  = session_dir / f"{layer_id}_layer.png"

        # Image.fromarray(refined).save(str(mask_path))
        # logger.debug(f"[segmentation] mask saved → {mask_path}")

        # mask_to_transparent_png(image_np, refined, bbox, png_path)
        # refined  = refine_mask(raw_mask)
        # layer_id = f"{session_id}_{idx}_{uuid.uuid4().hex[:6]}"

        # # filenames
        # mask_name = f"{layer_id}_mask.png"
        # png_name  = f"{layer_id}_layer.png"

        # # absolute filesystem paths (for saving files)
        # mask_file = session_dir / mask_name
        # png_file  = session_dir / png_name

        # # public URLs (for frontend response)
        # mask_path = f"/temp/{session_id}/{mask_name}"
        # png_path  = f"/temp/{session_id}/{png_name}"
        # layers.append(LayerData(
        #     id=layer_id, name=label,
        #     mask_path=str(mask_path), png_path=str(png_path),
        #     bbox=BoundingBox(**bbox),
        #     z_index=len(objects) - idx,
        #     visible=True, opacity=1.0,
        # ))
        logger.info(f"[segmentation] layer '{label}' created: id={layer_id}")

    # Save session metadata for cache restore
    meta = {
        "session_id": session_id,
        "image_path": str(image_path),
        "layers": [l.model_dump() for l in layers],
    }
    (session_dir / "session_meta.json").write_text(json.dumps(meta))

    logger.info(f"[segmentation] done — {len(layers)} layers created for session {session_id}")
    return layers


def _bbox_mask(shape: Tuple[int, int], bbox: Dict[str, float]) -> np.ndarray:
    logger.debug(f"[segmentation] creating bbox fallback mask for {bbox}")
    mask = np.zeros(shape, dtype=bool)
    x1 = max(0, int(bbox["x"]))
    y1 = max(0, int(bbox["y"]))
    x2 = min(shape[1], int(bbox["x"] + bbox["width"]))
    y2 = min(shape[0], int(bbox["y"] + bbox["height"]))
    mask[y1:y2, x1:x2] = True
    return mask
