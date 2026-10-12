"""
Remote detection service — replaces local Grounding DINO.
Supports: Replicate (grounding-dino), fal.ai (grounding-dino)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image as _PILImage

from services.remote_config import remote_cfg
from services.remote_services._http import (
    replicate_call,
    fal_call,
    path_to_data_uri,
)

logger = logging.getLogger(__name__)


# Fallback broad object list (same as local detection_service.py)
# NOTE: "sign" and "poster" intentionally excluded — they cause the full advertisement
# to be returned as a foreground object.
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
    "clock . mirror . painting . vase . ball . helmet . food . mobile phone . stone . logo . graphic"
)


def _iou(a: List[float], b: List[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _box_area_r(bbox: Dict[str, Any], img_w: int, img_h: int) -> float:
    x1 = max(0.0, min(float(img_w), float(bbox.get("x", 0))))
    y1 = max(0.0, min(float(img_h), float(bbox.get("y", 0))))
    x2 = max(0.0, min(float(img_w), float(bbox.get("x", 0)) + float(bbox.get("width", 0))))
    y2 = max(0.0, min(float(img_h), float(bbox.get("y", 0)) + float(bbox.get("height", 0))))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _containment_r(inner: Dict[str, Any], outer: Dict[str, Any], img_w: int, img_h: int) -> float:
    ix1 = max(0.0, float(inner.get("x", 0)))
    iy1 = max(0.0, float(inner.get("y", 0)))
    ix2 = min(float(img_w), ix1 + float(inner.get("width", 0)))
    iy2 = min(float(img_h), iy1 + float(inner.get("height", 0)))
    ox1 = max(0.0, float(outer.get("x", 0)))
    oy1 = max(0.0, float(outer.get("y", 0)))
    ox2 = min(float(img_w), ox1 + float(outer.get("width", 0)))
    oy2 = min(float(img_h), oy1 + float(outer.get("height", 0)))
    inter = max(0.0, min(ix2, ox2) - max(ix1, ox1)) * max(0.0, min(iy2, oy2) - max(iy1, oy1))
    inner_a = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return (inter / inner_a) if inner_a > 0 else 0.0


def _is_degenerate_box(bbox: Dict[str, Any], img_w: int, img_h: int) -> bool:
    """Remote equivalent of the local degenerate-box guard. Threshold lowered to 0.85."""
    if not bbox:
        return True
    area = _box_area_r(bbox, img_w, img_h)
    image_area = float(img_w) * float(img_h)
    if image_area <= 0:
        return True
    return (area / image_area) >= 0.85


def _nms(objects: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
    objects = sorted(objects, key=lambda o: o["score"], reverse=True)
    kept = []
    for obj in objects:
        b = obj["bbox"]
        box = [b["x"], b["y"], b["x"] + b["width"], b["y"] + b["height"]]
        if all(
            _iou(box, [k["bbox"]["x"], k["bbox"]["y"],
                       k["bbox"]["x"] + k["bbox"]["width"],
                       k["bbox"]["y"] + k["bbox"]["height"]]) < iou_threshold
            for k in kept
        ):
            kept.append(obj)
    return kept


def _drop_redundant_objects(objects: List[Dict[str, Any]], img_w: int, img_h: int) -> List[Dict[str, Any]]:
    """Remote-side equivalent of the local degenerate-box + containment filter."""
    if not objects:
        return objects

    filtered: List[Dict[str, Any]] = []
    dropped_degenerate = 0
    for obj in objects:
        if _is_degenerate_box(obj.get("bbox"), img_w, img_h):
            logger.warning(
                f"[remote/detection] dropping near-full-canvas detection '{obj.get('label')}' "
                f"(covers >= 85% of image)"
            )
            dropped_degenerate += 1
            continue
        filtered.append(obj)

    if dropped_degenerate:
        logger.info(f"[remote/detection] dropped {dropped_degenerate} near-full-canvas detection(s)")

    if not filtered:
        logger.info("[remote/detection] all detections were degenerate — returning empty object list")
        return []

    # Drop superset-container boxes
    image_area = float(img_w) * float(img_h)
    by_area = sorted(filtered, key=lambda o: _box_area_r(o.get("bbox", {}), img_w, img_h), reverse=True)
    to_drop: set = set()
    for i, big in enumerate(by_area):
        if i in to_drop:
            continue
        big_area = _box_area_r(big.get("bbox", {}), img_w, img_h)
        for j, small in enumerate(by_area):
            if j <= i or j in to_drop:
                continue
            small_area = _box_area_r(small.get("bbox", {}), img_w, img_h)
            if small_area <= 0:
                continue
            if (
                _containment_r(small.get("bbox", {}), big.get("bbox", {}), img_w, img_h) >= 0.85
                and big_area >= small_area * 1.20
                and big_area / image_area >= 0.40
            ):
                logger.warning(f"[remote/detection] dropping container '{big.get('label')}' (superset of '{small.get('label')}')")
                to_drop.add(i)
                break

    result = [obj for idx, obj in enumerate(by_area) if idx not in to_drop]
    if to_drop:
        logger.info(f"[remote/detection] dropped {len(to_drop)} superset-container detection(s)")
    return result




def detect_objects_remote(
    image_path: str,
    prompt: Optional[str] = None,
    box_threshold: float = 0.25,
    text_threshold: float = 0.25,
) -> List[Dict[str, Any]]:
    """
    Detect objects using the configured remote provider.
    Returns the same list-of-dicts format as the local detection_service.
    """
    text = prompt or FALLBACK_PROMPT
    provider = remote_cfg.DETECTION_PROVIDER
    logger.info(f"[remote/detection] provider={provider} prompt='{text[:60]}…'")

    try:
        image = Image.open(image_path).convert("RGB")
        img_w, img_h = image.size
    except Exception as e:
        logger.warning(f"[remote/detection] cannot read image for degenerate check ({e})")
        img_w = img_h = 0

    if provider == "replicate":
        objects = _detect_replicate(image_path, text, box_threshold)
    elif provider == "fal":
        objects = _detect_fal(image_path, text, box_threshold)
    else:
        raise ValueError(f"Unknown detection provider: {provider!r}. Choose 'replicate' or 'fal'.")

    if img_w and img_h:
        objects = _drop_redundant_objects(objects, img_w, img_h)

    return objects


def _detect_replicate(image_path: str, text: str, threshold: float) -> List[Dict[str, Any]]:
    data_uri = path_to_data_uri(image_path)
    payload = {
        "input": {
            "image": data_uri,
            "query": text,
            "box_threshold": threshold,
            "text_threshold": threshold,
        }
    }
    result = replicate_call(remote_cfg.ENDPOINTS["grounding_dino_replicate"], payload)
    raw = result.get("output", [])

    objects = []
    for det in raw:
        # Replicate grounding-dino returns: {label, score, box: [x1,y1,x2,y2]}
        box = det.get("box", det.get("bbox", [0, 0, 0, 0]))
        if len(box) == 4:
            x1, y1, x2, y2 = box
        else:
            continue
        objects.append({
            "label": det.get("label", "object").strip(),
            "score": round(float(det.get("score", 0.5)), 4),
            "bbox": {
                "x": round(x1, 2), "y": round(y1, 2),
                "width": round(x2 - x1, 2), "height": round(y2 - y1, 2),
            },
        })

    objects = _nms(objects)
    # Deduplicate labels
    seen: Dict[str, int] = {}
    for obj in objects:
        lbl = obj["label"]
        seen[lbl] = seen.get(lbl, 0) + 1
        obj["label"] = lbl if seen[lbl] == 1 else f"{lbl} {seen[lbl]}"

    logger.info(f"[remote/detection] replicate returned {len(objects)} objects")
    return objects


def _detect_fal(image_path: str, text: str, threshold: float) -> List[Dict[str, Any]]:
    data_uri = path_to_data_uri(image_path)
    payload = {
        "image_url": data_uri,
        "query": text,
        "box_threshold": threshold,
        "text_threshold": threshold,
    }
    result = fal_call(remote_cfg.ENDPOINTS.get("grounding_dino_fal",
                      "https://queue.fal.run/fal-ai/grounding-dino"), payload)
    raw = result.get("objects") or result.get("detections") or result.get("output", [])

    objects = []
    for det in raw:
        box = det.get("box") or det.get("bbox") or []
        if len(box) == 4:
            x1, y1, x2, y2 = box
        else:
            continue
        objects.append({
            "label": det.get("label", "object").strip(),
            "score": round(float(det.get("score", 0.5)), 4),
            "bbox": {
                "x": round(x1, 2), "y": round(y1, 2),
                "width": round(x2 - x1, 2), "height": round(y2 - y1, 2),
            },
        })

    objects = _nms(objects)
    seen: Dict[str, int] = {}
    for obj in objects:
        lbl = obj["label"]
        seen[lbl] = seen.get(lbl, 0) + 1
        obj["label"] = lbl if seen[lbl] == 1 else f"{lbl} {seen[lbl]}"

    logger.info(f"[remote/detection] fal returned {len(objects)} objects")
    return objects
