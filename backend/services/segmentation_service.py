"""
Segmentation service – SAM2 masks + refinement + transparent PNG layers.
"""
from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from services.model_manager import model_manager
from services.background_detector import (
    BackgroundType,
    analyze_background,
    create_background_mask,
    analyze_and_create_masks,
)
from schemas import BoundingBox, LayerData
 
from utils import config
logger = logging.getLogger(__name__)

# TEMP_DIR = Path(__file__).resolve().parents[2] / "temp"
TEMP_DIR = config.TEMP_DIR

TEMP_DIR.mkdir(parents=True, exist_ok=True)


def refine_mask(mask: np.ndarray) -> np.ndarray:
    logger.debug("[segmentation] refining mask…")
    mask = mask.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # Close small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    # Flood-fill interior holes
    flood = mask.copy()
    h, w = flood.shape
    flood_fill_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_fill_mask, (0, 0), 255)
    mask = mask | cv2.bitwise_not(flood)
    # Remove isolated noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # --- Edge feathering via alpha matting ---
    # Erode to get definite foreground, dilate to get definite background boundary
    fg_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded  = cv2.erode(mask,  fg_kernel, iterations=2)   # certain foreground
    dilated = cv2.dilate(mask, fg_kernel, iterations=2)   # certain background boundary

    # Transition zone = pixels between eroded and dilated
    transition = (dilated > 128) & (eroded < 128)

    # In the transition zone, use distance-based smooth alpha
    dist_fg = cv2.distanceTransform((mask > 128).astype(np.uint8), cv2.DIST_L2, 5)
    dist_bg = cv2.distanceTransform((mask < 128).astype(np.uint8), cv2.DIST_L2, 5)
    smooth = dist_fg / (dist_fg + dist_bg + 1e-6)  # 0..1 gradient across edge

    result = mask.copy().astype(np.float32)
    result[transition] = smooth[transition] * 255.0

    # Final light blur only on the transition band to remove staircase artifacts
    blurred = cv2.GaussianBlur(result, (3, 3), 0)
    result[transition] = blurred[transition]

    logger.debug("[segmentation] mask refined")
    return np.clip(result, 0, 255).astype(np.uint8)


def mask_to_transparent_png(image: np.ndarray, mask: np.ndarray, bbox: Dict[str, float], out_path: Path) -> None:
    img_h, img_w = image.shape[:2]
    # Expand crop by feather radius so soft edge pixels aren't clipped
    FEATHER_PAD = 8
    x1 = max(0, int(bbox["x"]) - FEATHER_PAD)
    y1 = max(0, int(bbox["y"]) - FEATHER_PAD)
    x2 = min(img_w, int(bbox["x"]) + int(bbox["width"])  + FEATHER_PAD)
    y2 = min(img_h, int(bbox["y"]) + int(bbox["height"]) + FEATHER_PAD)
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

    # ── Background Analysis ────────────────────────────────────────────────────
    logger.info("[segmentation] analyzing background...")
    bg_analysis, bg_mask = analyze_and_create_masks(
        image_np,
        session_dir=str(TEMP_DIR / session_id),
        session_id=session_id,
    )
    logger.info(
        f"[segmentation] background type: {bg_analysis.bg_type.value}, "
        f"dominant_color: {bg_analysis.dominant_color}, "
        f"confidence: {bg_analysis.confidence:.2f}"
    )

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

        # Bbox must match the expanded crop used in mask_to_transparent_png (FEATHER_PAD=8)
        FP = 8
        expanded_bbox = {
            "x":      float(max(0,      int(bbox["x"]) - FP)),
            "y":      float(max(0,      int(bbox["y"]) - FP)),
            "width":  float(min(img_w,  int(bbox["x"]) + int(bbox["width"])  + FP) - max(0, int(bbox["x"]) - FP)),
            "height": float(min(img_h,  int(bbox["y"]) + int(bbox["height"]) + FP) - max(0, int(bbox["y"]) - FP)),
        }

        layers.append(LayerData(
            id=layer_id,
            name=label,
            mask_path=mask_url,
            png_path=png_url,
            bbox=BoundingBox(**expanded_bbox),
            z_index=len(objects) - idx,
            visible=True,
            opacity=1.0,
        ))

        logger.info(f"[segmentation] layer '{label}' created: id={layer_id}")

    # --- Background Layer ────────────────────────────────────────────────────
    # Use analyzed background mask or create from remainder
    if bg_mask is not None and np.sum(bg_mask > 0) > 0:
        # Use the analyzed background mask
        logger.info("[segmentation] using analyzed background mask")
        background_mask = bg_mask
    else:
        # Fallback: create from remainder of objects
        logger.info("[segmentation] creating background mask from remainder")
        background_mask = cv2.bitwise_not(combined_mask)  # pixels not covered by any object
    
    # Remove tiny disconnected blobs (shadow noise, compression artifacts)
    # Keep only connected components larger than 0.5% of total image area
    min_area = int(img_h * img_w * 0.005)
    num_labels, cc_labels, stats, _ = cv2.connectedComponentsWithStats(
        (background_mask > 128).astype(np.uint8), connectivity=8
    )
    clean_background = np.zeros_like(background_mask)
    for cc_idx in range(1, num_labels):  # skip background (0)
        if stats[cc_idx, cv2.CC_STAT_AREA] >= min_area:
            clean_background[cc_labels == cc_idx] = 255

    # Use the larger of analyzed mask vs remainder for better coverage
    if bg_mask is not None:
        clean_background = np.maximum(clean_background, bg_mask)

    background_pixel_count = int(np.sum(clean_background > 0))
    logger.info(f"[segmentation] background pixels: {background_pixel_count}")

    if background_pixel_count > 0:
        layer_id   = f"{session_id}_bg_{uuid.uuid4().hex[:6]}"
        mask_name  = f"{layer_id}_background_mask.png"
        png_name   = f"{layer_id}_background_layer.png"
        mask_file  = session_dir / mask_name
        png_file   = session_dir / png_name

        Image.fromarray(clean_background).save(str(mask_file))

        rgba = cv2.cvtColor(image_np, cv2.COLOR_RGB2RGBA)
        rgba[:, :, 3] = clean_background
        Image.fromarray(rgba).save(str(png_file), optimize=False)

        full_bbox = {"x": 0.0, "y": 0.0, "width": float(img_w), "height": float(img_h)}
        layers.append(LayerData(
            id=layer_id,
            name="background",
            mask_path=f"/temp/{session_id}/{mask_name}",
            png_path=f"/temp/{session_id}/{png_name}",
            bbox=BoundingBox(**full_bbox),
            z_index=0,  # bottom layer
            visible=True,
            opacity=1.0,
        ))
        logger.info(f"[segmentation] background layer created: type={bg_analysis.bg_type.value}")

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
