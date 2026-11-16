"""
Segmentation service – SAM2 masks + refinement + transparent PNG layers.

Key design invariants (fixed vs. previous version):
  1. PIXEL OWNERSHIP — objects are sorted small-to-large before masking.
     Each object claims pixels via a running ownership_map. A pixel already
     claimed by a smaller object CANNOT be claimed again by a larger one.
     This eliminates overlapping foreground content across layers.

  2. FULL-CANVAS RGBA PNGs — every foreground layer is saved as a PNG the
     same size as the original image (not a cropped bbox cutout).
     LayerData.bbox is always {x:0, y:0, width:img_w, height:img_h}.
     merge_service.py pastes at bbox.x + position.x = 0, which is correct.
     Transforms applied in the frontend/backend operate on the full canvas,
     so there are no off-center rotation artefacts.

  3. NO EAGER BACKGROUND RECONSTRUCTION — segmentation never runs inpainting
     by default; that would load SDXL/LaMa and stall every segmentation call.
     The background layer is built from the original image pixels with
     object-owned pixels zeroed in the alpha channel.  Reconstruction happens
     only on demand (POST /restore-background or POST /inpaint-bg) for the
     parts the user actually asks to reconstruct.  Pass
     reconstruct_background=True to opt into the eager path.

  4. LAYER TYPE METADATA — each LayerData now carries `layer_type`
     (object / background / unrecognized) and `is_background` boolean so
     the frontend can treat the background layer differently.
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
from schemas import BoundingBox, LayerData
from utils import config

logger = logging.getLogger(__name__)

TEMP_DIR = config.TEMP_DIR
TEMP_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Mask refinement
# ---------------------------------------------------------------------------

def refine_mask(mask: np.ndarray) -> np.ndarray:
    """Morphological clean-up + distance-based edge feathering."""
    logger.debug("[segmentation] refining mask…")
    mask = mask.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Close small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Flood-fill interior holes from corner
    flood = mask.copy()
    h, w = flood.shape
    flood_fill_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_fill_mask, (0, 0), 255)
    mask = mask | cv2.bitwise_not(flood)

    # Remove isolated noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # --- Edge feathering via distance transform ---
    fg_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded  = cv2.erode(mask,  fg_kernel, iterations=2)
    dilated = cv2.dilate(mask, fg_kernel, iterations=2)
    transition = (dilated > 128) & (eroded < 128)

    dist_fg = cv2.distanceTransform((mask > 128).astype(np.uint8), cv2.DIST_L2, 5)
    dist_bg = cv2.distanceTransform((mask < 128).astype(np.uint8), cv2.DIST_L2, 5)
    smooth = dist_fg / (dist_fg + dist_bg + 1e-6)

    result = mask.copy().astype(np.float32)
    result[transition] = smooth[transition] * 255.0
    blurred = cv2.GaussianBlur(result, (3, 3), 0)
    result[transition] = blurred[transition]

    logger.debug("[segmentation] mask refined")
    return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Full-canvas layer PNG writer
# ---------------------------------------------------------------------------

def mask_to_fullcanvas_png(
    image_np: np.ndarray,
    owned_mask: np.ndarray,
    out_path: Path,
) -> None:
    """
    Save a full-canvas RGBA PNG where alpha = owned_mask.

    Unlike the old mask_to_transparent_png() this does NOT crop to bbox.
    The layer PNG has exactly the same width/height as the source image, so
    its position is always (0, 0) relative to the canvas.

    Args:
        image_np:   Full original image as (H, W, 3) uint8 RGB.
        owned_mask: (H, W) uint8 mask — pixels the object owns (0-255 alpha).
        out_path:   Destination file path.
    """
    rgba = cv2.cvtColor(image_np, cv2.COLOR_RGB2RGBA)
    rgba[:, :, 3] = owned_mask
    Image.fromarray(rgba).save(str(out_path), optimize=False)
    logger.debug(f"[segmentation] saved full-canvas transparent PNG → {out_path}")


# ---------------------------------------------------------------------------
# Bbox fallback mask
# ---------------------------------------------------------------------------

def crop_layer_to_content(rgba: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Crop a full-canvas RGBA layer down to its non-transparent pixels.

    Returns (cropped_rgba, bbox) where bbox is the content box in canvas
    coordinates. Layer PNGs are stored tightly cropped so their clickable area
    matches the object instead of covering — and blocking — every layer below.

    If the layer has no visible pixels, the input is returned unchanged with a
    zero-sized bbox so the caller can skip it.
    """
    alpha = rgba[:, :, 3]
    cols = np.flatnonzero(alpha.max(axis=0) > 0)
    rows = np.flatnonzero(alpha.max(axis=1) > 0)
    if cols.size == 0 or rows.size == 0:
        return rgba, {"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0}
    x1, x2 = int(cols[0]), int(cols[-1]) + 1
    y1, y2 = int(rows[0]), int(rows[-1]) + 1
    return rgba[y1:y2, x1:x2], {
        "x": float(x1),
        "y": float(y1),
        "width": float(x2 - x1),
        "height": float(y2 - y1),
    }


def _bbox_mask(shape: Tuple[int, int], bbox: Dict[str, float]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    x1 = max(0, int(bbox["x"]))
    y1 = max(0, int(bbox["y"]))
    x2 = min(shape[1], int(bbox["x"] + bbox["width"]))
    y2 = min(shape[0], int(bbox["y"] + bbox["height"]))
    mask[y1:y2, x1:x2] = True
    return mask


# ---------------------------------------------------------------------------
# Remote segmentation path
# ---------------------------------------------------------------------------

def _segment_objects_remote(
    session_id: str,
    image_path: str,
    objects: List[Dict[str, Any]],
    reconstruct_background: bool = False,
) -> List[LayerData]:
    """
    Remote segmentation: calls cloud SAM2 API for each detected object,
    then applies pixel-ownership and full-canvas layer creation.
    """
    from services.remote_services.segmentation import segment_object_remote

    p = Path(image_path)
    if not p.is_absolute():
        rel = str(image_path).lstrip("/")
        if rel.startswith("temp/"):
            rel = rel[len("temp/"):]
        p = (TEMP_DIR / rel).resolve()

    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")

    image_pil = Image.open(p).convert("RGB")
    image_np  = np.array(image_pil)
    img_h, img_w = image_np.shape[:2]

    session_dir = TEMP_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    def safe_label(lbl: str) -> str:
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in lbl.strip())

    # Sort objects small → large by bbox area so smaller objects get priority.
    sorted_objects = sorted(
        list(enumerate(objects)),
        key=lambda t: t[1]["bbox"]["width"] * t[1]["bbox"]["height"],
    )

    # Ownership map: each pixel stores the index of its owning object, -1 = unclaimed.
    ownership_map = np.full((img_h, img_w), -1, dtype=np.int32)

    raw_masks: Dict[int, np.ndarray] = {}
    refined_masks: Dict[int, np.ndarray] = {}

    # --- Pass 1: obtain and refine all SAM2 masks ---
    for orig_idx, obj in sorted_objects:
        bbox  = obj["bbox"]
        label = obj["label"]
        logger.info(f"[segmentation/remote] '{label}' (orig_idx={orig_idx})")
        try:
            raw_mask = segment_object_remote(str(p), bbox, (img_w, img_h))
        except Exception as e:
            logger.warning(f"[segmentation/remote] API failed for '{label}': {e} — bbox fallback")
            raw_mask = _bbox_mask(image_np.shape[:2], bbox)
        raw_masks[orig_idx] = raw_mask
        refined_masks[orig_idx] = refine_mask(raw_mask)

    # --- Pass 2: assign pixel ownership small → large ---
    layers: List[LayerData] = []

    for orig_idx, obj in sorted_objects:
        label  = obj["label"]
        refined = refined_masks[orig_idx]

        # Claim pixels not yet owned by any earlier (smaller) object
        binary = (refined > 128).astype(bool)
        newly_claimed = binary & (ownership_map < 0)
        ownership_map[newly_claimed] = orig_idx

        # owned_mask: soft alpha only for newly-claimed pixels; zero everywhere else
        owned_mask = np.where(newly_claimed, refined, 0).astype(np.uint8)

        if not np.any(owned_mask > 0):
            logger.warning(f"[segmentation/remote] '{label}' has no unclaimed pixels — skipping")
            continue

        layer_id   = f"{session_id}_{orig_idx}_{uuid.uuid4().hex[:6]}"
        label_slug = safe_label(label)
        mask_name  = f"{layer_id}_{label_slug}_mask.png"
        png_name   = f"{layer_id}_{label_slug}_layer.png"
        mask_file  = session_dir / mask_name
        png_file   = session_dir / png_name

        Image.fromarray(owned_mask).save(str(mask_file))

        # Build the layer in memory, decontaminate edges, then crop it to the
        # object's own content box instead of keeping a full-canvas PNG.
        rgba = cv2.cvtColor(image_np, cv2.COLOR_RGB2RGBA)
        rgba[:, :, 3] = owned_mask
        try:
            from services.restoration_service import decontaminate_foreground_edges
            rgba = decontaminate_foreground_edges(rgba, band_px=3)
        except Exception as _de:
            logger.debug(f"[segmentation/remote] foreground decontamination skipped: {_de}")
        rgba, content_bbox = crop_layer_to_content(rgba)
        Image.fromarray(rgba).save(str(png_file), optimize=False)

        layers.append(LayerData(
            id=layer_id,
            name=label,
            mask_path=f"/temp/{session_id}/{mask_name}",
            png_path=f"/temp/{session_id}/{png_name}",
            bbox=BoundingBox(**content_bbox),
            z_index=len(objects) - orig_idx,
            visible=True,
            opacity=1.0,
            layer_type="object",
            is_background=False,
        ))
        logger.info(f"[segmentation/remote] layer '{label}' created: id={layer_id}")

    # --- Background layer ---
    bg_layer = _build_background_layer(
        image_np, ownership_map, session_id, session_dir, img_w, img_h, len(objects),
        refined_masks=list(refined_masks.values()),
        object_labels=[obj["label"] for _, obj in sorted_objects if obj["label"]],
        reconstruct=reconstruct_background,
    )
    if bg_layer:
        layers.append(bg_layer)

    return layers


# ---------------------------------------------------------------------------
# Background layer builder
# ---------------------------------------------------------------------------

def _build_background_layer(
    image_np: np.ndarray,
    ownership_map: np.ndarray,
    session_id: str,
    session_dir: Path,
    img_w: int,
    img_h: int,
    num_objects: int,
    refined_masks: Optional[List[np.ndarray]] = None,
    object_labels: Optional[List[str]] = None,
    reconstruct: bool = False,
) -> Optional[LayerData]:
    """
    Build the background layer.

    Default (reconstruct=False): no inpainting is run.  The layer is the
    original image with object-owned pixels made transparent (alpha=0), so
    object layers composite on top of the untouched original background.
    Reconstruction is left to the on-demand endpoints, which only run for
    the part the user explicitly asks for.

    Opt-in (reconstruct=True): call restoration_service.restore_background()
    to reconstruct the pixels behind every removed object (Stage A → LaMa/cv2,
    Stage B → SDXL generative, Stage C → seam blend) and save a fully opaque
    full-canvas RGBA PNG.
    """
    # Build the union mask of all object-owned pixels
    union_mask = (ownership_map >= 0).astype(np.uint8) * 255

    object_pixel_count = int(np.sum(union_mask > 0))
    logger.info(f"[segmentation] object pixels: {object_pixel_count} (reconstruct={reconstruct})")

    if object_pixel_count == 0:
        logger.info("[segmentation] no objects — using original image as background")
        restored_np = image_np.copy()
        background_alpha = None  # fully opaque
    elif not reconstruct:
        logger.info(
            "[segmentation] background reconstruction skipped — transparent gaps kept "
            "until the user requests reconstruction"
        )
        if object_pixel_count >= img_w * img_h:
            logger.info("[segmentation] every pixel is object-owned — no background layer")
            return None
        restored_np = image_np.copy()
        background_alpha = np.where(ownership_map >= 0, 0, 255).astype(np.uint8)
    else:
        from services.restoration_service import restore_background

        _labels = object_labels or []
        _masks  = refined_masks or []

        # Build per_object_data list: pair each mask with its label
        per_object_data = [
            {"mask": m, "label": _labels[i] if i < len(_labels) else f"object_{i}"}
            for i, m in enumerate(_masks)
        ]
        # If no individual masks supplied, fall back to the combined union mask
        if not per_object_data and np.any(union_mask > 0):
            per_object_data = [{"mask": union_mask, "label": _labels[0] if _labels else "object"}]

        try:
            logger.info(
                f"[segmentation] running background restoration for "
                f"{len(per_object_data)} object(s): {[o['label'] for o in per_object_data]}"
            )
            restored_np = restore_background(
                image_np=image_np,
                per_object_data=per_object_data,
                use_sdxl=True,
            )
            logger.info("[segmentation] background restoration complete")
        except Exception as e:
            logger.warning(
                f"[segmentation] background restoration failed ({e}) "
                f"— falling back to original image for background layer"
            )
            restored_np = image_np.copy()
        background_alpha = None  # fully opaque

    layer_id  = f"{session_id}_bg_{uuid.uuid4().hex[:6]}"
    mask_name = f"{layer_id}_background_mask.png"
    png_name  = f"{layer_id}_background_layer.png"
    mask_file = session_dir / mask_name
    png_file  = session_dir / png_name

    # Mask PNG: white everywhere (for reference / editing purposes)
    full_mask = np.full((img_h, img_w), 255, dtype=np.uint8)
    Image.fromarray(full_mask).save(str(mask_file))

    # Layer PNG: original (or restored) pixels; object gaps are transparent
    # unless restoration filled them.
    rgba = cv2.cvtColor(restored_np, cv2.COLOR_RGB2RGBA)
    if background_alpha is None:
        rgba[:, :, 3] = 255  # fully opaque — objects composite on top
    else:
        rgba[:, :, 3] = background_alpha
    Image.fromarray(rgba).save(str(png_file), optimize=False)

    full_bbox = {"x": 0.0, "y": 0.0, "width": float(img_w), "height": float(img_h)}
    return LayerData(
        id=layer_id,
        name="background",
        mask_path=f"/temp/{session_id}/{mask_name}",
        png_path=f"/temp/{session_id}/{png_name}",
        bbox=BoundingBox(**full_bbox),
        z_index=0,  # bottom layer
        visible=True,
        opacity=1.0,
        layer_type="background",
        is_background=True,
    )


# ---------------------------------------------------------------------------
# Main local segmentation entry point
# ---------------------------------------------------------------------------

def segment_objects(
    session_id: str,
    image_path: str,
    objects: List[Dict[str, Any]],
    reconstruct_background: bool = False,
) -> List[LayerData]:
    logger.info(f"[segmentation] session={session_id} objects={len(objects)} image={image_path}")

    # ── Remote mode ────────────────────────────────────────────────────────
    from services.remote_config import remote_cfg
    if remote_cfg.COMPUTE_MODE == "remote":
        return _segment_objects_remote(
            session_id, image_path, objects, reconstruct_background
        )

    # ── Local mode ─────────────────────────────────────────────────────────
    logger.info("[segmentation] loading SAM2 predictor…")
    predictor = model_manager.get_sam2()

    # Resolve image path
    p = Path(image_path)
    if not p.is_absolute():
        rel = str(image_path).lstrip("/")
        if rel.startswith("temp/"):
            rel = rel[len("temp/"):]
        p = (TEMP_DIR / rel).resolve()

    logger.info(f"[segmentation] resolved path: {p} exists={p.exists()}")
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")

    image_pil = Image.open(p).convert("RGB")
    image_np  = np.array(image_pil)
    img_h, img_w = image_np.shape[:2]
    logger.info(f"[segmentation] image shape: {image_np.shape}")

    logger.info("[segmentation] setting image in SAM2…")
    predictor.set_image(image_np)
    logger.info("[segmentation] image set, processing objects…")

    session_dir = TEMP_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    def safe_label(lbl: str) -> str:
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in lbl.strip())

    # ── Sort objects small → large by bbox area ────────────────────────────
    # Smaller objects claim their pixels first, so larger containers cannot
    # duplicate pixels already owned by a smaller object.
    sorted_objects = sorted(
        list(enumerate(objects)),
        key=lambda t: t[1]["bbox"]["width"] * t[1]["bbox"]["height"],
    )
    logger.info(
        f"[segmentation] processing order (small→large): "
        + ", ".join(f"'{o[1]['label']}' ({o[1]['bbox']['width']*o[1]['bbox']['height']:.0f}px²)" for o in sorted_objects)
    )

    # ── Ownership map: -1 = unclaimed, >=0 = orig_idx of owning object ─────
    ownership_map = np.full((img_h, img_w), -1, dtype=np.int32)

    refined_masks: Dict[int, np.ndarray] = {}

    # ── Pass 1: SAM2 inference for all objects ─────────────────────────────
    for orig_idx, obj in sorted_objects:
        bbox  = obj["bbox"]
        label = obj["label"]
        logger.info(f"[segmentation] SAM2 [{orig_idx+1}/{len(objects)}] '{label}' bbox={bbox}")

        box = np.array([
            bbox["x"], bbox["y"],
            bbox["x"] + bbox["width"], bbox["y"] + bbox["height"],
        ])
        pad = max(10, int(min(bbox["width"], bbox["height"]) * 0.05))
        padded_box = np.array([
            max(0, box[0] - pad), max(0, box[1] - pad),
            min(img_w, box[2] + pad), min(img_h, box[3] + pad),
        ])
        cx = (padded_box[0] + padded_box[2]) / 2
        cy = (padded_box[1] + padded_box[3]) / 2
        point_coords = np.array([[cx, cy]])
        point_labels = np.array([1])

        try:
            masks, scores, _ = predictor.predict(
                box=padded_box,
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )
            best_idx = int(np.argmax(scores))
            raw_mask = masks[best_idx]
            logger.info(f"[segmentation] '{label}' → best score={scores[best_idx]:.4f}")
        except Exception as e:
            logger.warning(f"[segmentation] SAM2 failed for '{label}': {e} — using bbox fallback")
            raw_mask = _bbox_mask(image_np.shape[:2], bbox)

        refined_masks[orig_idx] = refine_mask(raw_mask)

    # ── Pass 2: assign pixel ownership and write full-canvas layers ────────
    layers: List[LayerData] = []

    for orig_idx, obj in sorted_objects:
        label   = obj["label"]
        refined = refined_masks[orig_idx]

        # Claim pixels not yet owned by any earlier (smaller) object
        binary       = (refined > 128).astype(bool)
        newly_claimed = binary & (ownership_map < 0)
        ownership_map[newly_claimed] = orig_idx

        # owned_mask: preserve soft-alpha feathering only for newly-claimed pixels
        owned_mask = np.where(newly_claimed, refined, 0).astype(np.uint8)

        if not np.any(owned_mask > 0):
            logger.warning(
                f"[segmentation] '{label}' has no unclaimed pixels after ownership resolution "
                f"— all its pixels were already owned by a smaller object. Skipping layer."
            )
            continue

        layer_id   = f"{session_id}_{orig_idx}_{uuid.uuid4().hex[:6]}"
        label_slug = safe_label(label)
        mask_name  = f"{layer_id}_{label_slug}_mask.png"
        png_name   = f"{layer_id}_{label_slug}_layer.png"
        mask_file  = session_dir / mask_name
        png_file   = session_dir / png_name

        Image.fromarray(owned_mask).save(str(mask_file))

        # Build the layer in memory, decontaminate edges, then crop it to the
        # object's own content box instead of keeping a full-canvas PNG.
        rgba = cv2.cvtColor(image_np, cv2.COLOR_RGB2RGBA)
        rgba[:, :, 3] = owned_mask
        try:
            from services.restoration_service import decontaminate_foreground_edges
            rgba = decontaminate_foreground_edges(rgba, band_px=3)
        except Exception as _de:
            logger.debug(f"[segmentation] foreground decontamination skipped: {_de}")
        rgba, content_bbox = crop_layer_to_content(rgba)
        Image.fromarray(rgba).save(str(png_file), optimize=False)

        # Determine a sensible layer_type from the label
        lbl_lower = label.lower()
        if any(kw in lbl_lower for kw in ("logo", "icon", "brand", "emblem", "badge")):
            ltype = "logo"
        elif any(kw in lbl_lower for kw in ("text", "sign", "word", "letter", "title", "heading", "label")):
            ltype = "text"
        else:
            ltype = "object"

        layers.append(LayerData(
            id=layer_id,
            name=label,
            mask_path=f"/temp/{session_id}/{mask_name}",
            png_path=f"/temp/{session_id}/{png_name}",
            bbox=BoundingBox(**content_bbox),
            z_index=len(objects) - orig_idx,
            visible=True,
            opacity=1.0,
            layer_type=ltype,
            is_background=False,
        ))
        logger.info(f"[segmentation] layer '{label}' (type={ltype}) created: id={layer_id}")

    # ── Background layer ───────────────────────────────────────────────────
    # Collect per-object masks and labels for the restoration service
    _refined_mask_list = [refined_masks[orig_idx] for orig_idx, _ in sorted_objects if orig_idx in refined_masks]
    _object_label_list = [obj["label"] for _, obj in sorted_objects]

    bg_layer = _build_background_layer(
        image_np, ownership_map, session_id, session_dir, img_w, img_h, len(objects),
        refined_masks=_refined_mask_list,
        object_labels=_object_label_list,
        reconstruct=reconstruct_background,
    )
    if bg_layer:
        layers.append(bg_layer)

    # ── Unrecognized layer: pixels not owned by any detected object ─────────
    # (after background exclusion this handles genuinely unresolved content)
    # NOTE: with the current design the background layer already captures
    # unowned pixels, so a separate unrecognized layer is only needed if
    # there are isolated islands far from the background region.  We skip
    # this for now to avoid an extra layer that is always identical to the
    # background in most cases.

    # ── Persist session metadata ───────────────────────────────────────────
    meta = {
        "session_id": session_id,
        "image_path": str(image_path),
        "layers": [l.model_dump() for l in layers],
    }
    (session_dir / "session_meta.json").write_text(json.dumps(meta))

    logger.info(f"[segmentation] done — {len(layers)} layers for session {session_id}")
    return layers
