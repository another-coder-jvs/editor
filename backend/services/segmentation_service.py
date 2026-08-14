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
logger = logging.getLogger(__name__)

# TEMP_DIR = Path(__file__).resolve().parents[2] / "temp"
TEMP_DIR = config.TEMP_DIR

TEMP_DIR.mkdir(parents=True, exist_ok=True)


def refine_mask(mask: np.ndarray) -> np.ndarray:
    logger.debug("[segmentation] refining mask…")
    mask = mask.astype(np.uint8) * 255
    # Small kernel — preserves fine details like hair and fingers
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # Close small holes only
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    # Flood-fill interior holes
    flood = mask.copy()
    h, w = flood.shape
    flood_fill_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_fill_mask, (0, 0), 255)
    mask = mask | cv2.bitwise_not(flood)
    # Light open to remove isolated noise — skip aggressive erosion
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    # Use bilateral filter instead of Gaussian — preserves edges while smoothing
    mask_f = mask.astype(np.float32)
    mask_f = cv2.bilateralFilter(mask_f, d=5, sigmaColor=25, sigmaSpace=5)
    mask = mask_f.astype(np.uint8)
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
    Image.fromarray(rgba).save(str(out_path), optimize=False)
    logger.debug(f"[segmentation] saved transparent PNG → {out_path}")


def segment_objects(session_id: str, image_path: str, objects: List[Dict[str, Any]]) -> List[LayerData]:
    logger.info(f"[segmentation] session={session_id} objects={len(objects)} image={image_path}")

    logger.info("[segmentation] loading SAM2 predictor…")
    predictor = model_manager.get_sam2()

    logger.info(f"[segmentation] opening image: {image_path}")
    image_path = str(image_path).lstrip("/temp")

    # Build path relative to project root
    p = (TEMP_DIR / image_path).resolve()
 
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
    # Accumulate all refined masks to compute the unrecognized remainder
    img_h, img_w = image_np.shape[:2]
    combined_mask = np.zeros((img_h, img_w), dtype=np.uint8)

    # Sanitize label for use in filename
    def safe_label(lbl: str) -> str:
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in lbl.strip())

    for idx, obj in enumerate(objects):
        bbox  = obj["bbox"]
        label = obj["label"]
        logger.info(f"[segmentation] [{idx+1}/{len(objects)}] '{label}' bbox={bbox}")

        box = np.array([bbox["x"], bbox["y"], bbox["x"] + bbox["width"], bbox["y"] + bbox["height"]])

        # Pad the box slightly so SAM2 has context around the object
        pad = max(10, int(min(bbox["width"], bbox["height"]) * 0.05))
        padded_box = np.array([
            max(0, box[0] - pad), max(0, box[1] - pad),
            min(img_w, box[2] + pad), min(img_h, box[3] + pad),
        ])

        # Center point prompt — guides SAM2 toward the object interior
        cx = (padded_box[0] + padded_box[2]) / 2
        cy = (padded_box[1] + padded_box[3]) / 2
        point_coords = np.array([[cx, cy]])
        point_labels = np.array([1])  # 1 = foreground

        logger.debug(f"[segmentation] SAM2 box prompt: {padded_box}")

        try:
            masks, scores, _ = predictor.predict(
                box=padded_box,
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )
            best_idx  = int(np.argmax(scores))
            raw_mask  = masks[best_idx]
            logger.info(f"[segmentation] '{label}' → best mask score={scores[best_idx]:.4f} (of {len(scores)})")
        except Exception as e:
            logger.warning(f"[segmentation] SAM2 failed for '{label}': {e} — using bbox fallback")
            raw_mask = _bbox_mask(image_np.shape[:2], bbox)
        refined = refine_mask(raw_mask)

        # Accumulate into combined mask (any pixel > 128 is "recognized")
        combined_mask = np.maximum(combined_mask, (refined > 128).astype(np.uint8) * 255)

        layer_id = f"{session_id}_{idx}_{uuid.uuid4().hex[:6]}"
        label_slug = safe_label(label)

        mask_name = f"{layer_id}_{label_slug}_mask.png"
        png_name  = f"{layer_id}_{label_slug}_layer.png"

        mask_file = session_dir / mask_name
        png_file  = session_dir / png_name

        Image.fromarray(refined).save(str(mask_file))
        logger.debug(f"[segmentation] mask saved → {mask_file}")

        mask_to_transparent_png(image_np, refined, bbox, png_file)

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

    # --- Unrecognized remainder layer (0% data loss) ---
    remainder_mask = cv2.bitwise_not(combined_mask)  # pixels not covered by any object

    # Remove tiny disconnected blobs (shadow noise, compression artifacts)
    # Keep only connected components larger than 0.5% of total image area
    min_area = int(img_h * img_w * 0.005)
    num_labels, cc_labels, stats, _ = cv2.connectedComponentsWithStats(
        (remainder_mask > 128).astype(np.uint8), connectivity=8
    )
    clean_remainder = np.zeros_like(remainder_mask)
    for cc_idx in range(1, num_labels):  # skip background (0)
        if stats[cc_idx, cv2.CC_STAT_AREA] >= min_area:
            clean_remainder[cc_labels == cc_idx] = 255

    remainder_pixel_count = int(np.sum(clean_remainder > 0))
    logger.info(f"[segmentation] remainder pixels after noise removal: {remainder_pixel_count}")

    if remainder_pixel_count > 0:
        layer_id   = f"{session_id}_remainder_{uuid.uuid4().hex[:6]}"
        mask_name  = f"{layer_id}_unrecognized_mask.png"
        png_name   = f"{layer_id}_unrecognized_layer.png"
        mask_file  = session_dir / mask_name
        png_file   = session_dir / png_name

        Image.fromarray(clean_remainder).save(str(mask_file))

        rgba = cv2.cvtColor(image_np, cv2.COLOR_RGB2RGBA)
        rgba[:, :, 3] = clean_remainder
        Image.fromarray(rgba).save(str(png_file), optimize=False)

        full_bbox = {"x": 0.0, "y": 0.0, "width": float(img_w), "height": float(img_h)}
        layers.append(LayerData(
            id=layer_id,
            name="unrecognized",
            mask_path=f"/temp/{session_id}/{mask_name}",
            png_path=f"/temp/{session_id}/{png_name}",
            bbox=BoundingBox(**full_bbox),
            z_index=0,  # bottom layer
            visible=True,
            opacity=1.0,
        ))
        logger.info(f"[segmentation] unrecognized remainder layer created")

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
