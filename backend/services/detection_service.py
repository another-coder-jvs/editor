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
    "person . car . road . sky . mountain . tree . building . animal . "
    "water . grass . floor . wall . window . door . furniture . food . "
    "vehicle . sign . light . shadow . background"
)


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
    seen_labels: Dict[str, int] = {}

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        x1, y1, x2, y2 = box.tolist()
        label_str = label.strip()
        count = seen_labels.get(label_str, 0)
        seen_labels[label_str] = count + 1
        display_name = label_str if count == 0 else f"{label_str} {count + 1}"

        obj = {
            "label": display_name,
            "score": round(float(score), 4),
            "bbox": {
                "x": round(x1, 2),
                "y": round(y1, 2),
                "width": round(x2 - x1, 2),
                "height": round(y2 - y1, 2),
            },
        }
        objects.append(obj)
        logger.debug(f"[detection]   {display_name} score={obj['score']} bbox={obj['bbox']}")

    logger.info(f"[detection] final objects ({len(objects)}): {[o['label'] for o in objects]}")
    return objects
