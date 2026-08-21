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

# Fallback when Ollama is unavailable — comprehensive object list for Grounding DINO
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
    "sign . poster . "
    "fire hydrant . traffic light . bench . trash can . "
    "clock . mirror . painting . vase . "
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


# Synonyms map — merge common duplicates into one canonical name
SYNONYM_MAP = {
    "sneaker": "shoe", "footwear": "shoe", "sneakers": "shoe", "boots": "shoe",
    "heel": "shoe", "sandal": "shoe",
    "vehicle": "car", "automobile": "car", "sedan": "car", "suv": "car",
    "canine": "dog", "puppy": "dog",
    "feline": "cat", "kitten": "cat",
    "sofa": "chair", "couch": "chair", "seat": "chair",
    "glasses": "glasses", "spectacles": "glasses", "eyeglasses": "glasses",
    "television": "tv", "monitor": "tv",
    "mobile": "phone", "cellphone": "phone", "smartphone": "phone",
    "laptop": "laptop", "notebook": "laptop", "computer": "laptop",
    "backpack": "bag", "handbag": "bag", "purse": "bag",
    "bottle": "bottle", "flask": "bottle",
    "cup": "cup", "mug": "cup", "glass": "cup",
    "plate": "plate", "dish": "plate",
    "bicycle": "bicycle", "bike": "bicycle",
    "motorcycle": "motorcycle", "motorbike": "motorcycle",
    "building": "building", "house": "building", "structure": "building",
    "tree": "tree", "plant": "plant",
    "flower": "flower", "blossom": "flower",
    "pillow": "pillow", "cushion": "pillow", "pillowcase": "pillow",
    "poster": "poster", "flyer": "poster", "billboard": "poster",
    "sign": "sign", "signboard": "sign",
    "window": "window", "door": "door",
    "hat": "hat", "cap": "hat", "helmet": "hat",
    "book": "book",
    "clock": "clock", "watch": "clock",
    "mirror": "mirror",
    "vase": "vase",
    "lamp": "lamp", "light": "lamp", "lantern": "lamp",
    "floor": "floor", "carpet": "floor", "rug": "floor",
    "wall": "wall", "wallpaper": "wall",
    "table": "table", "desk": "table",
}

# Words to always skip (text-related, abstract, hallucinated)
SKIP_WORDS = {
    "letter", "letters", "number", "numbers", "alphabet", "text",
    "word", "words", "font", "type", "typography", "logo",
    "watermark", "caption", "title", "heading", "label",
    "background", "foreground", "shadow", "reflection",
    "image", "photo", "picture", "scene", "view",
    "design", "pattern", "texture", "color", "colour",
    "stand", "hook", "hanger", "clip", "magnet", "tape",
    "adhesive", "dispenser", "frame",
}


def _clean_ollama_prompt(raw: str) -> str:
    """Clean Ollama output into a valid Grounding DINO dot-separated prompt."""
    if not raw or not raw.strip():
        return ""
    # Take the last non-empty line (usually the actual list)
    lines = raw.strip().splitlines()
    for line in reversed(lines):
        line = line.strip()
        if line:
            raw = line
            break
    # Remove bullet points, numbers, dashes
    raw = re.sub(r'^[\d\-\*\)\(]+\s*', '', raw)
    # Split into individual items
    items = re.split(r'[,;\n]+', raw)
    # Clean each item
    cleaned = []
    seen = set()
    for item in items:
        item = re.sub(r'[^a-zA-Z0-9\s\-]', '', item).strip().lower()
        if not item or len(item) < 2:
            continue
        # Skip multi-word items (likely hallucinated)
        if ' ' in item:
            continue
        # Skip known bad words
        if item in SKIP_WORDS:
            continue
        # Map synonyms
        canonical = SYNONYM_MAP.get(item, item)
        if canonical not in seen:
            seen.add(canonical)
            cleaned.append(canonical)
    if not cleaned:
        return ""
    result = " . ".join(cleaned) + " ."
    return result


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
            target_sizes=[(h, w)],
        )[0]
    except TypeError:
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

    objects = _nms(objects, iou_threshold=0.5)

    seen: Dict[str, int] = {}
    for obj in objects:
        lbl = obj["label"]
        seen[lbl] = seen.get(lbl, 0) + 1
        obj["label"] = lbl if seen[lbl] == 1 else f"{lbl} {seen[lbl]}"

    logger.info(f"[detection] final objects ({len(objects)}): {[o['label'] for o in objects]}")
    return objects or []
