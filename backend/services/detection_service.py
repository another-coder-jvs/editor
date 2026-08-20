"""
Detection service – BLIP captions the image to find object names,
then Grounding DINO localises them with bounding boxes.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

from services.model_manager import model_manager, DEVICE

logger = logging.getLogger(__name__)

# ── BLIP pre-detector ─────────────────────────────────────────────────────
_blip_processor = None
_blip_model = None

BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"


def _get_blip():
    global _blip_processor, _blip_model
    if _blip_processor is None:
        import os
        os.environ["MPLBACKEND"] = "Agg"
        from transformers import BlipProcessor, BlipForConditionalGeneration

        logger.info("[detection] loading BLIP model…")
        _blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
        _blip_model = BlipForConditionalGeneration.from_pretrained(
            BLIP_MODEL_NAME,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        ).to(DEVICE)
        _blip_model.eval()
        logger.info("[detection] BLIP model ready")
    return _blip_processor, _blip_model


# Words that aren't useful as Grounding DINO object names
_STOP_WORDS = {
    "a", "an", "the", "and", "with", "on", "in", "at", "of", "to", "is",
    "are", "this", "that", "there", "image", "photo", "picture", "showing",
    "shows", "looking", "standing", "sitting", "wearing", "next", "near",
    "front", "back", "side", "background", "against", "featuring",
    "large", "small", "white", "black", "red", "blue", "green", "yellow",
    "brown", "gray", "grey", "orange", "pink", "purple", "color",
}


def _blip_detect(image_path: str) -> List[str]:
    """Caption the image with BLIP, extract object names from the caption."""
    processor, model = _get_blip()

    image = Image.open(image_path).convert("RGB")
    image.thumbnail((1024, 1024))

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=80, num_beams=3)

    caption = processor.decode(output[0], skip_special_tokens=True)
    logger.info(f"[detection] BLIP caption: '{caption}'")

    # Extract words from caption
    words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_-]*\b", caption.lower())

    objects: List[str] = []
    for word in words:
        if word in _STOP_WORDS:
            continue
        if word not in objects:
            objects.append(word)

    logger.info(f"[detection] extracted {len(objects)} object names: {objects}")
    return objects


# ── Helpers ─────────────────────────────────────────────────────────────────

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
                          k["bbox"]["x"] + k["bbox"]["width"], k["bbox"]["y"] + k["bbox"]["height"]]) < iou_threshold
               for k in kept):
            kept.append(obj)
    return kept


# ── Main detection pipeline ────────────────────────────────────────────────

def detect_objects(
    image_path: str,
    prompt: Optional[str] = None,
    box_threshold: float = 0.15,
    text_threshold: float = 0.15,
) -> List[Dict[str, Any]]:
    logger.info(f"[detection] loading image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    logger.info(f"[detection] image size: {w}x{h}")

    # ── Step 1: BLIP captions the image → extract object names → build DINO prompt ──
    if prompt:
        # User supplied a custom prompt — use it directly
        dino_prompt = prompt
        logger.info(f"[detection] using custom prompt: '{dino_prompt[:80]}'")
    else:
        blip_objects = _blip_detect(image_path)
        if not blip_objects:
            logger.info("[detection] BLIP found nothing — falling back to broad prompt")
            dino_prompt = (
                "person . car . truck . bus . motorcycle . bicycle . boat . airplane . train . "
                "dog . cat . bird . horse . cow . sheep . elephant . bear . "
                "tree . flower . plant . building . house . bridge . tower . fence . "
                "pillow . chair . table . sofa . bed . desk . cabinet . shelf . lamp . "
                "bottle . cup . bowl . plate . fork . knife . spoon . "
                "book . laptop . phone . keyboard . monitor . television . camera . "
                "bag . backpack . suitcase . umbrella . hat . shoe . sneaker . glasses . "
                "door . window . stairs . sign . poster . banner . "
                "fire hydrant . traffic light . bench . trash can . "
                "clock . mirror . painting . vase . ball . helmet . food . mobile phone . stone . "
                "logo . icon . globe . earth . location . pin . map marker . "
                "discount . sale . offer"
            )
        else:
            dino_prompt = " . ".join(blip_objects)
            logger.info(f"[detection] built DINO prompt from BLIP: '{dino_prompt}'")

    # ── Step 2: Grounding DINO localises each object ──
    logger.info("[detection] loading Grounding DINO model…")
    model, processor = model_manager.get_grounding_dino()
    logger.info("[detection] model ready, running inference…")

    inputs = processor(images=image, text=dino_prompt, return_tensors="pt").to(DEVICE)
    logger.debug(f"[detection] input_ids shape: {inputs.input_ids.shape}")

    with torch.no_grad():
        outputs = model(**inputs)
    logger.info("[detection] inference complete, post-processing…")

    results = processor.post_process_grounded_object_detection(
        outputs=outputs,
        input_ids=inputs.input_ids,
        target_sizes=[(h, w)],
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
