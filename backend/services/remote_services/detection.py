"""
Remote detection service — replaces local Grounding DINO.
Supports: Replicate (grounding-dino), fal.ai (grounding-dino)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from services.remote_config import remote_cfg
from services.remote_services._http import (
    replicate_call,
    fal_call,
    path_to_data_uri,
)

logger = logging.getLogger(__name__)


# Fallback broad object list (same as local detection_service.py)
FALLBACK_PROMPT = (
    "person . car . truck . bus . motorcycle . bicycle . boat . airplane . train . "
    "dog . cat . bird . horse . cow . sheep . elephant . bear . "
    "tree . flower . plant . "
    "building . house . bridge . tower . fence . "
    "pillow . chair . table . sofa . bed . desk . cabinet . shelf . lamp . "
    "bottle . cup . bowl . plate . fork . knife . spoon . "
    "book . laptop . phone . keyboard . monitor . television . camera . "
    "bag . backpack . suitcase . umbrella . hat . shoe . glasses . "
    "door . window . stairs . sign . poster . "
    "fire hydrant . traffic light . bench . trash can . "
    "clock . mirror . painting . vase . ball . helmet . food . mobile phone . stone"
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

    if provider == "replicate":
        return _detect_replicate(image_path, text, box_threshold)
    elif provider == "fal":
        return _detect_fal(image_path, text, box_threshold)
    else:
        raise ValueError(f"Unknown detection provider: {provider!r}. Choose 'replicate' or 'fal'.")


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
