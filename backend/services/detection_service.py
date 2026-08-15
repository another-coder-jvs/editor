"""
Detection service – uses Grounding DINO to find all objects in an image.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

from services.model_manager import model_manager, DEVICE

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = (
    "person . car . truck . bus . motorcycle . bicycle . boat . airplane . train . "
    "dog . cat . bird . horse . cow . sheep . elephant . bear . "
    "tree . flower . plant . "
    "building . house . bridge . tower . fence . "
    "pillow . chair . table . sofa . bed . desk . cabinet . shelf . lamp . "
    "bottle . cup . bowl . plate . fork . knife . spoon . "
    "book . laptop . phone . keyboard . monitor . television . camera . "
    "bag . backpack . suitcase . umbrella . hat . shoe . glasses . "
    "door . window . stairs . "
    "sign . poster . "
    "fire hydrant . traffic light . bench . trash can . "
    "clock . mirror . painting . vase . wall . "
    "ball . helmet . food . mobile phone . stone"
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
    # Sort by score descending
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


def detect_objects(
    image_path: str,
    prompt: Optional[str] = None,
    box_threshold: float = 0.25,
    text_threshold: float = 0.25,
) -> List[Dict[str, Any]]:
    logger.info(f"[detection] loading image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    logger.info(f"[detection] image size: {w}x{h}")

    text = prompt or DEFAULT_PROMPT
    logger.info(f"[detection] using prompt: '{text[:80]}…'")

    logger.info("[detection] loading Grounding DINO model…")
    model, processor = model_manager.get_grounding_dino()
    logger.info("[detection] model ready, running inference…")

    inputs = processor(images=image, text=text, return_tensors="pt").to(DEVICE)
    logger.debug(f"[detection] input_ids shape: {inputs.input_ids.shape}")

    with torch.no_grad():
        outputs = model(**inputs)
    logger.info("[detection] inference complete, post-processing…")

    # results = processor.post_process_grounded_object_detection(
    #     outputs,
    #     inputs.input_ids,
    #     box_threshold=box_threshold,
    #     text_threshold=text_threshold,
    #     target_sizes=[(h, w)],
    # )[0]

    results = processor.post_process_grounded_object_detection(
        outputs=outputs,
        input_ids=inputs.input_ids,
        target_sizes=[(h, w)],
    )[0]

    # Manual filtering for older transformers versions
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

    # Deduplicate overlapping boxes across all labels (IoU-based NMS)
    objects = _nms(objects, iou_threshold=0.5)

    # Assign display names after dedup (same label appearing multiple times = distinct objects)
    seen: Dict[str, int] = {}
    for obj in objects:
        lbl = obj["label"]
        seen[lbl] = seen.get(lbl, 0) + 1
        obj["label"] = lbl if seen[lbl] == 1 else f"{lbl} {seen[lbl]}"
        logger.debug(f"[detection]   {obj['label']} score={obj['score']} bbox={obj['bbox']}")

    logger.info(f"[detection] final objects ({len(objects)}): {[o['label'] for o in objects]}")
    return objects
