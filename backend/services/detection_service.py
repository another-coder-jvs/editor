"""
Detection service – BLIP captions + CLIP zero-shot classification
find object names, then Grounding DINO localises them with bounding boxes.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Set

import torch
from PIL import Image

from services.model_manager import model_manager, DEVICE

logger = logging.getLogger(__name__)

os.environ["MPLBACKEND"] = "Agg"  # avoid Colab matplotlib backend error


# ── BLIP pre-detector ─────────────────────────────────────────────────────
_blip_processor = None
_blip_model = None

BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"


def _get_blip():
    global _blip_processor, _blip_model
    if _blip_processor is None:
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
_STOP_WORDS: Set[str] = {
    # Articles, prepositions, conjunctions
    "a", "an", "the", "and", "with", "on", "in", "at", "of", "to", "is",
    "are", "this", "that", "there", "or", "but", "not", "for", "from",
    # Photography/image terms
    "image", "photo", "picture", "showing", "shows", "looking", "featuring",
    # Positional terms
    "standing", "sitting", "wearing", "next", "near", "front", "back",
    "side", "top", "bottom", "left", "right", "center", "middle",
    # Descriptive adjectives (not objects)
    "large", "small", "big", "tiny", "tall", "short",
    "white", "black", "red", "blue", "green", "yellow", "brown", "gray",
    "grey", "orange", "pink", "purple", "color", "colored", "dark", "light",
    "bright", "vibrant", "solid", "plain", "bright", "neon",
    # Poster/design terms (not objects)
    "background", "against", "main", "original", "digital", "modern",
    "clean", "minimal", "simple", "abstract", "cool", "nice", "beautiful",
    # Generic/meaningless terms
    "call", "global", "address", "white", "black", "the", "logo",
    "text", "font", "design", "style", "layout", "template", "poster",
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

    words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_-]*\b", caption.lower())

    objects: List[str] = []
    for word in words:
        if word in _STOP_WORDS:
            continue
        if word not in objects:
            objects.append(word)

    logger.info(f"[detection] BLIP extracted {len(objects)} object names: {objects}")
    return objects


# ── CLIP zero-shot classifier ────────────────────────────────────────────
_clip_processor = None
_clip_model = None

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# Candidate labels CLIP evaluates against the image — ONLY real objects
_CLIP_CANDIDATES = [
    # People & animals
    "person", "man", "woman", "child", "baby",
    "dog", "cat", "bird", "horse", "fish", "elephant", "bear", "lion",
    "tiger", "rabbit", "monkey", "deer", "snake", "turtle", "frog",
    # Clothing & accessories
    "shoe", "sneaker", "boot", "sandal", "hat", "cap", "glasses",
    "sunglasses", "watch", "ring", "necklace", "bracelet", "bag",
    "backpack", "handbag", "purse", "wallet", "belt", "scarf", "gloves",
    "jacket", "coat", "shirt", "pants", "dress", "skirt", "suit", "tie",
    # Electronics
    "phone", "smartphone", "laptop", "computer", "tablet", "monitor",
    "television", "camera", "headphones", "speaker", "keyboard", "mouse",
    "remote", "earbuds",
    # Food & drink
    "food", "fruit", "apple", "banana", "orange", "grape", "strawberry",
    "pizza", "burger", "sandwich", "cake", "cookie", "bread",
    "coffee", "cup", "mug", "bottle", "glass", "wine glass",
    # Objects
    "book", "magazine", "newspaper", "pen", "pencil", "scissors",
    "clock", "lamp", "chair", "table", "desk", "sofa", "bed",
    "pillow", "blanket", "towel", "mirror", "vase", "flower",
    "plant", "tree", "umbrella", "key", "coin", "ball", "toy",
    "box", "basket", "bowl", "plate", "fork", "knife", "spoon",
    # Vehicles
    "car", "truck", "bus", "motorcycle", "bicycle", "boat", "airplane",
    "train", "helicopter", "scooter", "skateboard",
    # Buildings & structures
    "house", "building", "bridge", "tower", "fence", "door", "window",
    "stairs", "roof", "wall",
    # Nature
    "mountain", "hill", "river", "lake", "ocean", "beach", "forest",
    "cloud", "sun", "moon", "star", "sky",
    # Design elements (actual visual objects)
    "logo", "icon", "badge", "sticker", "frame", "border",
    "globe", "earth", "pin", "location", "heart", "arrow",
    "circle", "triangle", "square",
    # Specific icons that appear in posters
    "phone icon", "map pin", "envelope", "mail",
    "camera", "microphone", "speaker",
    "fire", "lightning", "sparkle",
    "star", "crown", "shield",
]


def _get_clip():
    global _clip_processor, _clip_model
    if _clip_processor is None:
        from transformers import CLIPProcessor, CLIPModel

        logger.info("[detection] loading CLIP model…")
        _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        _clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
        _clip_model.eval()
        logger.info("[detection] CLIP model ready")
    return _clip_processor, _clip_model


def _clip_classify(image_path: str, top_k: int = 20, threshold: float = 0.22) -> List[str]:
    """Run CLIP zero-shot classification. Return labels above threshold, sorted by score."""
    processor, model = _get_clip()

    image = Image.open(image_path).convert("RGB")
    image.thumbnail((1024, 1024))

    inputs = processor(
        text=_CLIP_CANDIDATES,
        images=image,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits_per_image[0]
    probs = logits.softmax(dim=-1).cpu().tolist()

    scored = sorted(zip(_CLIP_CANDIDATES, probs), key=lambda x: x[1], reverse=True)

    objects: List[str] = []
    for label, score in scored[:top_k]:
        if score >= threshold:
            objects.append(label)

    logger.info(f"[detection] CLIP found {len(objects)} objects: {objects}")
    return objects


# ── Merge BLIP + CLIP results ─────────────────────────────────────────────

def _merge_object_names(blip_objects: List[str], clip_objects: List[str]) -> List[str]:
    """Merge BLIP caption words + CLIP classified labels into one deduplicated list."""
    merged: List[str] = []

    for name in clip_objects:
        if name not in merged:
            merged.append(name)

    for name in blip_objects:
        if name not in merged:
            merged.append(name)

    logger.info(f"[detection] merged {len(merged)} object names: {merged}")
    return merged


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


def _is_oversized(box: Dict[str, float], img_w: int, img_h: int, threshold: float = 0.85) -> bool:
    """Check if a bounding box covers more than threshold fraction of the image."""
    box_area = box["width"] * box["height"]
    img_area = img_w * img_h
    ratio = box_area / img_area if img_area > 0 else 0
    return ratio > threshold


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

    # ── Step 1: Get object names from BLIP + CLIP ──
    if prompt:
        dino_prompt = prompt
        logger.info(f"[detection] using custom prompt: '{dino_prompt[:80]}'")
    else:
        blip_objects = _blip_detect(image_path)
        clip_objects = _clip_classify(image_path)
        merged = _merge_object_names(blip_objects, clip_objects)

        if not merged:
            logger.info("[detection] both BLIP and CLIP found nothing — falling back to broad prompt")
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
            dino_prompt = " . ".join(merged)
            logger.info(f"[detection] built DINO prompt from BLIP+CLIP: '{dino_prompt}'")

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
        obj_box = {
            "x": round(x1, 2), "y": round(y1, 2),
            "width": round(x2 - x1, 2), "height": round(y2 - y1, 2),
        }

        # Skip oversized detections — these are DINO hallucinating the whole image as one object
        if _is_oversized(obj_box, w, h):
            logger.warning(f"[detection] skipping oversized detection: '{label}' covers {obj_box['width']:.0f}x{obj_box['height']:.0f} of {w}x{h}")
            continue

        # Clean label: if DINO returned a multi-word label, take only the first meaningful word
        clean_label = label.strip()
        # If label contains spaces, it's likely DINO merged multiple terms — keep first word only
        if " " in clean_label:
            words = clean_label.split()
            # Pick the first word that isn't a stop word
            for w_word in words:
                if w_word.lower() not in _STOP_WORDS:
                    clean_label = w_word
                    break
            else:
                clean_label = words[0]

        objects.append({
            "label": clean_label,
            "score": round(float(score), 4),
            "bbox": obj_box,
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
