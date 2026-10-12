"""
Detection service – uses Grounding DINO to find all objects in an image.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

from services.model_manager import model_manager, DEVICE
from services.identify_service import identify_objects, check_ollama

logger = logging.getLogger(__name__)

# Fallback when Ollama is unavailable — comprehensive object list for Grounding DINO.
# NOTE: "sign" and "poster" are intentionally EXCLUDED.  Those labels cause DINO to
# detect the entire uploaded advertisement/poster as a single foreground object, which
# then appears as a duplicate full-image layer in the output.
FALLBACK_PROMPT = (
    "person . car . truck . bus . motorcycle . bicycle . boat . airplane . train . "
    "dog . cat . bird . horse . cow . sheep . elephant . bear . "
    "tree . flower . plant . "
    "building . house . bridge . tower . fence . "
    "pillow . chair . table . sofa . bed . desk . cabinet . shelf . lamp . "
    "bottle . cup . bowl . plate . fork . knife . spoon . "
    "book . laptop . phone . keyboard . monitor . television . camera . "
    "bag . backpack . suitcase . umbrella . hat . shoe . glasses . "
    "door . window . stairs . "
    "fire hydrant . traffic light . bench . trash can . "
    "clock . mirror . painting . vase . "
    "ball . helmet . food . mobile phone . stone . logo . graphic"
)


def _iou(a: List[float], b: List[float]) -> float:
    """IoU between two [x1,y1,x2,y2] boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _nms(objects: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
    """Remove duplicate detections — keep highest-score box when IoU > threshold."""
    objects = sorted(objects, key=lambda o: o["score"], reverse=True)
    kept = []
    for obj in objects:
        b = obj["bbox"]
        box = [b["x"], b["y"], b["x"] + b["width"], b["y"] + b["height"]]
        if all(_iou(box, [k["bbox"]["x"], k["bbox"]["y"],
                          k["bbox"]["x"] + k["bbox"]["width"],
                          k["bbox"]["y"] + k["bbox"]["height"]]) < iou_threshold
               for k in kept):
            kept.append(obj)
    return kept





def _box_area(bbox: Dict[str, Any], img_w: int, img_h: int) -> float:
    """Clipped area of bbox in pixels (never exceeds image area)."""
    x1 = max(0.0, min(float(img_w), float(bbox.get("x", 0))))
    y1 = max(0.0, min(float(img_h), float(bbox.get("y", 0))))
    x2 = max(0.0, min(float(img_w), float(bbox.get("x", 0)) + float(bbox.get("width", 0))))
    y2 = max(0.0, min(float(img_h), float(bbox.get("y", 0)) + float(bbox.get("height", 0))))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _is_degenerate_box(bbox: Dict[str, Any], img_w: int, img_h: int) -> bool:
    """True if the box covers >= 85% of the full image canvas.

    Lowered from 0.98 → 0.85 because typical poster/full-advertisement detections
    from Grounding DINO land in the 86-97% range and the old threshold missed them.
    """
    if not bbox:
        return True
    area = _box_area(bbox, img_w, img_h)
    image_area = float(img_w) * float(img_h)
    if image_area <= 0:
        return True
    return (area / image_area) >= 0.85


def _containment_ratio(inner: Dict[str, Any], outer: Dict[str, Any], img_w: int, img_h: int) -> float:
    """Return the fraction of *inner*'s area that lies inside *outer*.

    A value >= 0.85 means inner is almost entirely contained by outer.
    """
    ix1 = max(0.0, float(inner.get("x", 0)))
    iy1 = max(0.0, float(inner.get("y", 0)))
    ix2 = min(float(img_w), ix1 + float(inner.get("width", 0)))
    iy2 = min(float(img_h), iy1 + float(inner.get("height", 0)))

    ox1 = max(0.0, float(outer.get("x", 0)))
    oy1 = max(0.0, float(outer.get("y", 0)))
    ox2 = min(float(img_w), ox1 + float(outer.get("width", 0)))
    oy2 = min(float(img_h), oy1 + float(outer.get("height", 0)))

    inter_x1, inter_y1 = max(ix1, ox1), max(iy1, oy1)
    inter_x2, inter_y2 = min(ix2, ox2), min(iy2, oy2)
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)

    inner_area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inner_area <= 0:
        return 0.0
    return inter_area / inner_area


def _drop_redundant_objects(objects: List[Dict[str, Any]], img_w: int, img_h: int) -> List[Dict[str, Any]]:
    """Remove:
    1. Boxes covering >= 85% of the image canvas (full-image / poster duplicates).
    2. Boxes that are a superset of smaller boxes: if box A contains >= 85% of box B,
       AND box A is also larger than box B by >= 20%, then A is a duplicate container
       and should be dropped (B is the real distinct object).
    """
    if not objects:
        return objects

    # --- Pass 1: drop near-full-canvas boxes ---
    filtered: List[Dict[str, Any]] = []
    dropped_degenerate = 0
    for obj in objects:
        if _is_degenerate_box(obj.get("bbox"), img_w, img_h):
            logger.warning(
                f"[detection] dropping near-full-canvas detection '{obj.get('label')}' "
                f"bbox={obj.get('bbox')} (covers >= 85% of image)"
            )
            dropped_degenerate += 1
            continue
        filtered.append(obj)

    if dropped_degenerate:
        logger.info(f"[detection] dropped {dropped_degenerate} near-full-canvas detection(s)")

    if not filtered:
        logger.info("[detection] all detections were degenerate — returning empty object list")
        return []

    # --- Pass 2: drop superset-container boxes ---
    # Sort largest → smallest so we can check whether a big box contains smaller ones.
    by_area = sorted(filtered, key=lambda o: _box_area(o.get("bbox", {}), img_w, img_h), reverse=True)
    image_area = float(img_w) * float(img_h)
    to_drop: set = set()

    for i, big in enumerate(by_area):
        if i in to_drop:
            continue
        big_area = _box_area(big.get("bbox", {}), img_w, img_h)
        for j, small in enumerate(by_area):
            if j <= i or j in to_drop:
                continue
            small_area = _box_area(small.get("bbox", {}), img_w, img_h)
            if small_area <= 0:
                continue
            # big contains >= 85% of small, AND big is at least 20% larger
            if (
                _containment_ratio(small.get("bbox", {}), big.get("bbox", {}), img_w, img_h) >= 0.85
                and big_area >= small_area * 1.20
            ):
                # Additionally: only suppress the large container if it covers a
                # significant fraction of the canvas (>= 40%) to avoid dropping
                # genuinely separate large objects that just happen to share area.
                if big_area / image_area >= 0.40:
                    logger.warning(
                        f"[detection] dropping container detection '{big.get('label')}' "
                        f"(contains {_containment_ratio(small.get('bbox',{}), big.get('bbox',{}), img_w, img_h):.0%} "
                        f"of '{small.get('label')}' and covers "
                        f"{big_area/image_area:.0%} of canvas)"
                    )
                    to_drop.add(i)
                    break  # no need to check further small boxes for this big

    result = [obj for idx, obj in enumerate(by_area) if idx not in to_drop]
    if to_drop:
        logger.info(f"[detection] dropped {len(to_drop)} superset-container detection(s)")

    return result


def _clean_ollama_prompt(raw: str) -> str:
    """Clean Ollama output into a valid Grounding DINO dot-separated prompt."""
    if not raw or not raw.strip():
        return ""
    # Take the last non-empty line
    lines = raw.strip().splitlines()
    for line in reversed(lines):
        line = line.strip()
        if line:
            raw = line
            break
    # Remove bullet points, numbers, dashes at start
    raw = re.sub(r'^[\d\-\*\)\(]+\s*', '', raw)
    # Split into individual items
    items = re.split(r'[,;\n]+', raw)
    # Clean each item: keep only single lowercase words
    cleaned = []
    seen = set()
    for item in items:
        item = re.sub(r'[^a-zA-Z\s]', '', item).strip().lower()
        if not item or len(item) < 2:
            continue
        # Take only the first word if multiple words slipped through
        word = item.split()[0] if item.split() else item
        if word not in seen:
            seen.add(word)
            cleaned.append(word)
    if not cleaned:
        return ""
    return " . ".join(cleaned) + " ."


def detect_objects(
    image_path: str,
    prompt: Optional[str] = None,
    box_threshold: float = 0.25,
    text_threshold: float = 0.25,
) -> List[Dict[str, Any]]:
    logger.info(f"[detection] loading image: {image_path}")

    # ── Remote mode: delegate entirely to the remote detection service ────────
    from services.remote_config import remote_cfg
    if remote_cfg.COMPUTE_MODE == "remote":
        logger.info("[detection] COMPUTE_MODE=remote — using remote detection API")
        from services.remote_services.detection import detect_objects_remote
        return detect_objects_remote(image_path, prompt, box_threshold, text_threshold)

    # ── Local mode (unchanged below) ─────────────────────────────────────────
    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size
    logger.info(f"[detection] image size: {img_w}x{img_h}")

    # --- Step 1: Determine the prompt for Grounding DINO ---
    if prompt:
        text = prompt
        logger.info(f"[detection] using user prompt: '{text[:80]}'")
    else:
        ollama_available = False
        try:
            ollama_available = check_ollama()
        except Exception:
            pass

        if ollama_available:
            try:
                logger.info("[detection] using Ollama vision to identify objects...")
                raw_text = identify_objects(image_path)
                logger.info(f"[detection] Ollama raw output: '{raw_text}'")
                text = _clean_ollama_prompt(raw_text)
                if not text:
                    logger.warning("[detection] Ollama returned empty prompt, using fallback")
                    text = FALLBACK_PROMPT
                logger.info(f"[detection] final prompt for Grounding DINO: '{text}'")
            except Exception as e:
                logger.warning(f"[detection] Ollama failed ({e}), using fallback prompt")
                text = FALLBACK_PROMPT
        else:
            logger.info("[detection] Ollama not available, using fallback prompt")
            text = FALLBACK_PROMPT

    # --- Step 2: Run Grounding DINO ---
    logger.info("[detection] loading Grounding DINO model…")
    model, processor = model_manager.get_grounding_dino()
    logger.info("[detection] model ready, running inference…")

    inputs = processor(images=image, text=text, return_tensors="pt").to(DEVICE)
    logger.info(f"[detection] prompt='{text}' input_ids shape={inputs.input_ids.shape}")

    with torch.no_grad():
        outputs = model(**inputs)
    logger.info("[detection] inference complete, post-processing…")

    # Try newer API first (with thresholds), fall back to older API
    try:
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[(img_h, img_w)],
        )[0]
    except TypeError:
        results = processor.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=inputs.input_ids,
            target_sizes=[(img_h, img_w)],
        )[0]
        keep = results["scores"] > box_threshold
        results = {
            "boxes": results["boxes"][keep],
            "scores": results["scores"][keep],
            "labels": [results["labels"][i] for i, k in enumerate(keep.tolist()) if k],
        }
    logger.info(f"[detection] raw detections: {len(results['scores'])}")

    objects: List[Dict[str, Any]] = []

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        x1, y1, x2, y2 = box.tolist()
        objects.append({
            "label": label.strip(),
            "score": round(float(score), 4),
            "bbox": {
                "x": round(x1, 2), "y": round(y1, 2),
                "width": round(x2 - x1, 2), "height": round(y2 - y1, 2),
            },
        })

    objects = _nms(objects, iou_threshold=0.5)

    # Drop detections that are essentially full-image duplicates
    objects = _drop_redundant_objects(objects, img_w, img_h)

    seen: Dict[str, int] = {}
    for obj in objects:
        lbl = obj["label"]
        seen[lbl] = seen.get(lbl, 0) + 1
        obj["label"] = lbl if seen[lbl] == 1 else f"{lbl} {seen[lbl]}"

    logger.info(f"[detection] final objects ({len(objects)}): {[o['label'] for o in objects]}")
    return objects or []
