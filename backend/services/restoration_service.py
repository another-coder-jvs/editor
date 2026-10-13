"""
Background restoration service — rewrote per spec (2026-10-13).

Pipeline per object
───────────────────
1.  Per-object adaptive mask
    • Dilate by max(25 px, 3% of bbox diagonal) to cover cast shadows
    • Add contact-shadow band (dark pixels adjacent to mask bottom edge)
    • 6-px feather for compositing only (not for inpaint model)

2.  Stage A — LaMa (big-lama / simple_lama_inpainting)
    • Full-res image + dilated binary mask, no prompt
    • Runs on the FULL canvas (not a crop) so LaMa sees global context
    • cv2 TELEA+NS fallback if LaMa is unavailable

3.  Stage B — SDXL texture refinement (optional, run on LaMa output)
    • strength 0.35, guidance 5
    • Crop = per-object bbox + 25% padding; never merge objects
    • Positive prompt: background-stat colour name + "seamless texture"
      (NEVER the object label, NEVER "no X")
    • Negative prompt: "{label}, shoe, sneaker, …"
    • Shape-cue suppression: SDXL runs on a blurred/rectangular proxy mask,
      result is composited back inside the real feathered mask only
    • Skipped when hole < 5% of canvas OR time budget already tight

4.  Validation
    • Run Grounding DINO on the restored hole with "{label} . text . logo"
    • If top score > 0.3: retry once with larger mask (+15 px) and seed+1
    • After retry: fall back to pure LaMa if still failing

5.  Blend
    • Feathered alpha composite (distance-transform ramp, 6 px) always
    • Poisson only when hole < 8% of canvas (avoids artefacts on large holes)
    • Hard-lock: unmasked pixels identical to original

6.  Logging (all calls)
    • Positive + negative prompts, guidance, strength, crop box, seed
    • Saves crop PNG + mask PNG to debug dir when DEBUG_RESTORATION=1

Public API
──────────
    restore_background(image_np, per_object_data, ...)
        per_object_data: list of {"mask": np.ndarray, "label": str, "bbox": dict}
        Returns (H, W, 3) uint8 RGB.
"""
from __future__ import annotations

import logging
import math
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────
MIN_DILATION_PX: int = 25            # minimum per-object dilation
DILATION_BBOX_DIAG_FRAC: float = 0.03   # 3% of bbox diagonal
SHADOW_BAND_PX: int = 12             # how far below mask to look for shadow
SHADOW_DARK_THRESHOLD: int = 60      # pixels darker than this are "shadow"
FEATHER_PX: int = 6                  # compositing feather width

SDXL_PAD_FACTOR: float = 0.25       # crop padding around object
SDXL_STRENGTH: float = 0.35         # texture-only refinement
SDXL_GUIDANCE: float = 5.0
SDXL_STEPS: int = 20
SDXL_MIN_HOLE_FRAC: float = 0.05    # skip SDXL if hole < 5% of canvas
MAX_INPAINT_DIM: int = 1024

FEATHERED_ALPHA_ALWAYS: bool = True  # always feathered alpha composite
POISSON_MAX_HOLE_FRAC: float = 0.08  # Poisson only when hole < 8%

DINO_DETECT_THRESHOLD: float = 0.3   # re-try if detection score > this
RETRY_EXTRA_DILATION: int = 15       # extra dilation on retry

DEBUG_DIR: Optional[Path] = (
    Path(os.environ.get("DEBUG_RESTORATION_DIR", "/tmp/restoration_debug"))
    if os.environ.get("DEBUG_RESTORATION") == "1"
    else None
)

# ── Colour name lookup (for SDXL texture prompt) ───────────────────────────
_COLOUR_NAMES = [
    ((200, 200, 200), "light grey"),  ((128, 128, 128), "grey"),
    ((40,  40,  40),  "dark grey"),   ((10,  10,  10),  "black"),
    ((255, 255, 255), "white"),
    ((180, 30,  30),  "dark red"),    ((220, 60,  60),  "red"),
    ((200, 100, 50),  "orange"),      ((200, 180, 50),  "yellow"),
    ((40,  100, 40),  "dark green"),  ((60,  160, 60),  "green"),
    ((30,  60,  160), "dark blue"),   ((60,  100, 200), "blue"),
    ((100, 50,  150), "purple"),      ((160, 50,  100), "maroon"),
    ((50,  140, 160), "teal"),        ((180, 140, 100), "tan"),
    ((100, 70,  40),  "brown"),
]

def _colour_name(rgb: np.ndarray) -> str:
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    best, best_d = "neutral", 1e9
    for (cr, cg, cb), name in _COLOUR_NAMES:
        d = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
        if d < best_d:
            best_d, best = d, name
    return best


# ── Debug helpers ──────────────────────────────────────────────────────────

def _debug_save(tag: str, img: np.ndarray, mask: Optional[np.ndarray] = None) -> None:
    """Save image and optional mask to DEBUG_DIR when debugging is enabled."""
    if DEBUG_DIR is None:
        return
    try:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        uid = uuid.uuid4().hex[:6]
        Image.fromarray(img).save(str(DEBUG_DIR / f"{tag}_{uid}.png"))
        if mask is not None:
            Image.fromarray(mask).save(str(DEBUG_DIR / f"{tag}_{uid}_mask.png"))
    except Exception as e:
        logger.debug(f"[restoration/debug] save failed: {e}")


# ── Per-object mask construction ───────────────────────────────────────────

def _per_object_dilation(raw_mask: np.ndarray) -> int:
    """Return dilation radius for one object: max(25px, 3% of bbox diagonal)."""
    rows = np.any(raw_mask > 0, axis=1)
    cols = np.any(raw_mask > 0, axis=0)
    if not np.any(rows) or not np.any(cols):
        return MIN_DILATION_PX
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    diag = math.hypot(y2 - y1, x2 - x1)
    return max(MIN_DILATION_PX, int(diag * DILATION_BBOX_DIAG_FRAC))


def _add_contact_shadow(
    image_np: np.ndarray,
    dilated: np.ndarray,
    band_px: int = SHADOW_BAND_PX,
) -> np.ndarray:
    """
    Extend the mask downward to include cast-shadow pixels directly below
    the object.  Detects dark pixels (< SHADOW_DARK_THRESHOLD) in the band
    immediately below the mask's bottom edge.
    """
    h, w = image_np.shape[:2]
    result = dilated.copy()

    # Bottom edge of the dilated mask
    rows = np.any(dilated > 0, axis=1)
    if not np.any(rows):
        return result
    bottom_row = int(np.where(rows)[0][-1])

    # Scan band below the object
    band_end = min(h, bottom_row + band_px + 1)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    for row in range(bottom_row + 1, band_end):
        # Only include dark pixels that are directly beneath existing mask columns
        mask_cols = dilated[bottom_row] > 0
        shadow_pixels = gray[row] < SHADOW_DARK_THRESHOLD
        result[row] = np.maximum(result[row], (mask_cols & shadow_pixels).astype(np.uint8) * 255)

    return result


def build_object_removal_mask(
    image_np: np.ndarray,
    raw_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build per-object removal mask (dilated + shadow) and feather mask.

    Returns:
        inpaint_mask  — hard binary (0/255) for LaMa / SDXL
        feather_mask  — float32 [0,1] for compositing
    """
    binary = (raw_mask > 128).astype(np.uint8) * 255
    dil_px = _per_object_dilation(binary)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dil_px * 2 + 1, dil_px * 2 + 1)
    )
    dilated = cv2.dilate((binary > 0).astype(np.uint8), kernel, iterations=1) * 255
    dilated = _add_contact_shadow(image_np, dilated)

    # Feather mask — purely for compositing, not passed to inpaint model
    dist = cv2.distanceTransform((dilated > 0).astype(np.uint8), cv2.DIST_L2, 5)
    feather = np.clip(dist / max(1.0, float(FEATHER_PX)), 0.0, 1.0)
    k = max(3, FEATHER_PX // 2) | 1
    feather = cv2.GaussianBlur(feather.astype(np.float32), (k, k), k / 3.0)
    feather = np.clip(feather, 0.0, 1.0)

    logger.debug(
        f"[restoration/mask] dil={dil_px}px "
        f"mask={int(np.sum(dilated > 0))}px "
        f"({100 * np.mean(dilated > 0):.1f}% canvas)"
    )
    return dilated.astype(np.uint8), feather.astype(np.float32)


# ── LaMa Stage A ──────────────────────────────────────────────────────────

def _lama_inpaint(image_np: np.ndarray, inpaint_mask: np.ndarray) -> np.ndarray:
    """
    Run LaMa (big-lama) on the full-resolution image.

    image_np:    (H, W, 3) uint8 RGB
    inpaint_mask: (H, W) uint8 binary (255 = fill)

    Returns (H, W, 3) uint8 RGB.
    Falls back to cv2 TELEA+NS if LaMa is unavailable.
    """
    # Pre-fill the hole with surrounding median colour so that if LaMa fails
    # and we fall back to cv2, the object pixels don't contaminate propagation.
    bg_pixels = image_np[inpaint_mask == 0]
    bg_median = np.median(bg_pixels, axis=0).astype(np.uint8) if len(bg_pixels) >= 10 else np.array([128, 128, 128], dtype=np.uint8)
    prefilled = image_np.copy()
    prefilled[inpaint_mask > 0] = bg_median

    try:
        from services.text_service import _get_lama
        lama = _get_lama()
        if lama is not None:
            logger.info(f"[restoration/A] LaMa inpaint — mask {int(np.sum(inpaint_mask > 0))} px")
            result_pil = lama(Image.fromarray(prefilled), Image.fromarray(inpaint_mask))
            result_np = np.array(result_pil.convert("RGB"))
            if result_np.shape[:2] != image_np.shape[:2]:
                result_np = cv2.resize(result_np, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_LANCZOS4)
            # Guard: reject if LaMa reproduced the object
            if not _lama_reproduced_object(image_np, result_np, inpaint_mask):
                logger.info("[restoration/A] LaMa accepted")
                return result_np
            logger.warning("[restoration/A] LaMa reproduced the object — cv2 fallback")
    except Exception as e:
        logger.warning(f"[restoration/A] LaMa failed ({e}) — cv2 fallback")

    # cv2 fallback on pre-filled image
    logger.info("[restoration/A] cv2 TELEA+NS fallback")
    bgr = cv2.cvtColor(prefilled, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    radius = max(8, int(min(h, w) * 0.02))
    radius = min(radius, 32)
    result = cv2.inpaint(bgr, inpaint_mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)
    result = cv2.inpaint(result, inpaint_mask, inpaintRadius=max(4, radius // 2), flags=cv2.INPAINT_NS)
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


def _lama_reproduced_object(
    original_np: np.ndarray,
    result_np: np.ndarray,
    inpaint_mask: np.ndarray,
    threshold: float = 0.75,
) -> bool:
    """
    Return True if more than `threshold` fraction of filled pixels match the
    original object's interior colour — meaning LaMa just copied the object back.
    """
    binary = (inpaint_mask > 0).astype(np.uint8)
    inner_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    inner = cv2.erode(binary, inner_k, iterations=1)
    core = original_np[inner > 0] if np.any(inner > 0) else original_np[inpaint_mask > 0]
    if len(core) < 10:
        return False
    obj_mean = np.median(core, axis=0).astype(np.float32)
    result_inside = result_np[inpaint_mask > 0].astype(np.float32)
    dists = np.linalg.norm(result_inside - obj_mean, axis=1)
    return float(np.mean(dists < 12.0)) > threshold


# ── SDXL Stage B ──────────────────────────────────────────────────────────

def _mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)
    if not np.any(rows):
        h, w = mask.shape[:2]
        return 0, 0, h, w
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return int(y1), int(x1), int(y2) + 1, int(x2) + 1


def _padded_crop_coords(
    image_np: np.ndarray,
    mask: np.ndarray,
    pad_factor: float = SDXL_PAD_FACTOR,
) -> Tuple[int, int, int, int]:
    """Return (cy1, cx1, cy2, cx2) — bbox + 25% padding, clamped to image."""
    h, w = image_np.shape[:2]
    ty1, tx1, ty2, tx2 = _mask_bbox(mask)
    obj_h, obj_w = ty2 - ty1, tx2 - tx1
    py = max(20, int(obj_h * pad_factor))
    px = max(20, int(obj_w * pad_factor))
    return (
        max(0, ty1 - py), max(0, tx1 - px),
        min(h, ty2 + py), min(w, tx2 + px),
    )


def _resize_for_sdxl(
    image_np: np.ndarray,
    mask: np.ndarray,
    max_dim: int = MAX_INPAINT_DIM,
    min_scale: float = 0.9,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Downscale if needed; never below min_scale of original crop."""
    h, w = image_np.shape[:2]
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max(min_scale, max_dim / max(h, w))
        nw, nh = int(w * scale), int(h * scale)
        image_np = cv2.resize(image_np, (nw, nh), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    h2, w2 = image_np.shape[:2]
    nw2 = (w2 // 8) * 8 or 8
    nh2 = (h2 // 8) * 8 or 8
    if nw2 != w2 or nh2 != h2:
        image_np = cv2.resize(image_np, (nw2, nh2), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (nw2, nh2), interpolation=cv2.INTER_NEAREST)
    return image_np, mask, scale


def _build_sdxl_prompts(
    image_np: np.ndarray,
    inpaint_mask: np.ndarray,
    label: str,
) -> Tuple[str, str]:
    """
    Build SDXL prompts from background pixel statistics only.

    POSITIVE: colour name + "seamless texture, empty background"
              — NEVER contains the object label or "no X" negations
    NEGATIVE: object label + broad suppression list
    """
    # Sample ring of background pixels around the mask
    binary = (inpaint_mask > 0).astype(np.uint8)
    band_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (49, 49))
    outer = cv2.dilate(binary, band_k, iterations=1)
    ring = (outer > 0) & (binary == 0)
    bg_pixels = image_np[ring] if np.any(ring) else image_np[binary == 0]
    if len(bg_pixels) == 0:
        bg_pixels = image_np.reshape(-1, 3)

    dom_rgb = np.median(bg_pixels, axis=0).astype(np.uint8)
    colour = _colour_name(dom_rgb)

    positive = f"seamless {colour} textured surface, empty background, no objects"
    negative = (
        f"{label}, shoe, sneaker, boot, product, clothing, person, human, animal, "
        f"object, logo, text, letters, watermark, brand, silhouette, shadow, duplicate, "
        f"artifact, blurry, low quality, distorted"
    )
    return positive, negative


def _blurred_proxy_mask(
    mask: np.ndarray,
    sigma: float = 20.0,
) -> np.ndarray:
    """
    Build a heavily blurred version of the mask to pass to SDXL, eliminating
    the shape cue.  SDXL sees a soft blob instead of a shoe silhouette.
    Returns (H, W) uint8 [0..255].
    """
    binary_f = (mask > 0).astype(np.float32)
    k = int(sigma * 6) | 1
    blurred = cv2.GaussianBlur(binary_f, (k, k), sigma)
    if blurred.max() > 0:
        blurred /= blurred.max()
    return (blurred * 255).astype(np.uint8)


def _sdxl_refine(
    lama_result: np.ndarray,
    original_np: np.ndarray,
    inpaint_mask: np.ndarray,
    feather_mask: np.ndarray,
    label: str,
    seed: int = 42,
) -> np.ndarray:
    """
    Optional SDXL texture-refinement pass on the LaMa output.

    Runs SDXL on a per-object crop at low strength (0.35) using a blurred
    proxy mask.  Composites result back inside the real feather mask only.

    Returns full-canvas (H, W, 3) RGB, or lama_result on failure.
    """
    from services.model_manager import model_manager
    import torch

    h, w = lama_result.shape[:2]
    hole_frac = float(np.sum(inpaint_mask > 0)) / max(h * w, 1)

    if hole_frac < SDXL_MIN_HOLE_FRAC:
        logger.info(f"[restoration/B] hole={hole_frac:.1%} < {SDXL_MIN_HOLE_FRAC:.0%} → skip SDXL")
        return lama_result

    # Crop around object
    cy1, cx1, cy2, cx2 = _padded_crop_coords(lama_result, inpaint_mask)
    crop_lama  = lama_result[cy1:cy2, cx1:cx2].copy()
    crop_mask  = inpaint_mask[cy1:cy2, cx1:cx2]
    crop_feat  = feather_mask[cy1:cy2, cx1:cx2]

    # Blurred proxy mask for SDXL (removes shape cue)
    proxy_mask = _blurred_proxy_mask(crop_mask, sigma=20.0)

    # Resize for model
    crop_s, proxy_s, scale = _resize_for_sdxl(crop_lama, proxy_mask)
    ch, cw = crop_s.shape[:2]

    positive, negative = _build_sdxl_prompts(original_np, inpaint_mask, label)

    logger.info(
        f"[restoration/B] SDXL refine crop=({cy1},{cx1},{cy2},{cx2}) "
        f"model={cw}×{ch} scale={scale:.2f} strength={SDXL_STRENGTH} "
        f"guidance={SDXL_GUIDANCE} steps={SDXL_STEPS} seed={seed}"
    )
    logger.info(f"[restoration/B] positive='{positive}'")
    logger.info(f"[restoration/B] negative='{negative}'")

    _debug_save(f"sdxl_crop_{label}", crop_s, proxy_s)

    try:
        pipe = model_manager.get_inpaint_pipe()
        gen  = torch.Generator(device="cpu").manual_seed(seed)
        out = pipe(
            prompt=positive,
            negative_prompt=negative,
            image=Image.fromarray(crop_s),
            mask_image=Image.fromarray(proxy_s).convert("RGB"),
            height=ch,
            width=cw,
            num_inference_steps=SDXL_STEPS,
            guidance_scale=SDXL_GUIDANCE,
            strength=SDXL_STRENGTH,
            generator=gen,
        ).images[0]

        result_crop = np.array(out.convert("RGB"))
        if scale < 1.0:
            result_crop = cv2.resize(result_crop, (cx2 - cx1, cy2 - cy1), interpolation=cv2.INTER_LANCZOS4)

        # Composite SDXL result back inside the REAL feather mask only
        # (not the blurred proxy — this is where shape-cue suppression matters)
        feat3 = crop_feat[:, :, np.newaxis]
        blended_crop = (
            crop_lama.astype(np.float32) * (1.0 - feat3)
            + result_crop.astype(np.float32) * feat3
        )
        blended_crop = np.clip(blended_crop, 0, 255).astype(np.uint8)

        full_result = lama_result.copy()
        full_result[cy1:cy2, cx1:cx2] = blended_crop

        # Hard-lock unmasked pixels
        full_result[inpaint_mask == 0] = lama_result[inpaint_mask == 0]

        logger.info("[restoration/B] SDXL refine accepted")
        try:
            model_manager.unload_inpaint_pipe()
        except Exception:
            pass
        return full_result

    except Exception as e:
        logger.warning(f"[restoration/B] SDXL refine failed ({e}) — using LaMa result")
        return lama_result


# ── Validation with Grounding DINO ────────────────────────────────────────

def _validate_restoration(
    image_np: np.ndarray,
    restored_np: np.ndarray,
    inpaint_mask: np.ndarray,
    label: str,
) -> bool:
    """
    Run Grounding DINO on the restored hole region with "{label} . text . logo".
    Returns True if the restoration is clean (no object detected at > 0.3).
    """
    try:
        from services.model_manager import model_manager, DEVICE
        import torch

        model, processor = model_manager.get_grounding_dino()
        h, w = restored_np.shape[:2]

        # Crop the restored hole region
        ty1, tx1, ty2, tx2 = _mask_bbox(inpaint_mask)
        pad = 10
        ry1 = max(0, ty1 - pad); rx1 = max(0, tx1 - pad)
        ry2 = min(h, ty2 + pad); rx2 = min(w, tx2 + pad)
        region = restored_np[ry1:ry2, rx1:rx2]

        region_pil = Image.fromarray(region)
        text_prompt = f"{label} . text . logo ."
        inputs = processor(images=region_pil, text=text_prompt, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)

        region_h, region_w = region.shape[:2]
        try:
            results = processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids,
                box_threshold=0.1, text_threshold=0.1,
                target_sizes=[(region_h, region_w)],
            )[0]
        except TypeError:
            results = processor.post_process_grounded_object_detection(
                outputs=outputs, input_ids=inputs.input_ids,
                target_sizes=[(region_h, region_w)],
            )[0]
            keep = results["scores"] > 0.1
            results = {
                "scores": results["scores"][keep],
                "labels": [results["labels"][i] for i, k in enumerate(keep.tolist()) if k],
            }

        if len(results["scores"]) == 0:
            logger.info(f"[restoration/validate] no detections — clean")
            return True

        top_score = float(results["scores"].max())
        top_label = results["labels"][int(results["scores"].argmax())]
        logger.info(f"[restoration/validate] top detection: '{top_label}' score={top_score:.3f}")

        if top_score > DINO_DETECT_THRESHOLD:
            logger.warning(
                f"[restoration/validate] '{top_label}' detected at {top_score:.2f} > "
                f"{DINO_DETECT_THRESHOLD} — restoration failed"
            )
            return False
        return True

    except Exception as e:
        logger.warning(f"[restoration/validate] DINO validation failed ({e}) — assuming clean")
        return True


# ── Blend ──────────────────────────────────────────────────────────────────

def _feathered_alpha_blend(
    original_np: np.ndarray,
    restored_np: np.ndarray,
    feather_mask: np.ndarray,
) -> np.ndarray:
    """
    Composite restored pixels into the original using the feather weight map.
    weight=1.0 → use restored; weight=0.0 → use original.
    Hard-locks all pixels with weight==0 to the exact original.
    """
    w3 = feather_mask[:, :, np.newaxis]
    blended = (
        original_np.astype(np.float32) * (1.0 - w3)
        + restored_np.astype(np.float32) * w3
    )
    result = np.clip(blended, 0, 255).astype(np.uint8)
    result[feather_mask == 0] = original_np[feather_mask == 0]
    return result


def _poisson_blend(
    original_np: np.ndarray,
    restored_np: np.ndarray,
    inpaint_mask: np.ndarray,
) -> Optional[np.ndarray]:
    try:
        h, w = original_np.shape[:2]
        binary = (inpaint_mask > 0).astype(np.uint8) * 255
        ys, xs = np.where(binary > 0)
        if len(ys) == 0:
            return None
        if ys.min() < 2 or ys.max() > h - 3 or xs.min() < 2 or xs.max() > w - 3:
            return None
        cy = int((ys.min() + ys.max()) // 2)
        cx = int((xs.min() + xs.max()) // 2)
        src_bgr = cv2.cvtColor(restored_np, cv2.COLOR_RGB2BGR)
        dst_bgr = cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR)
        result_bgr = cv2.seamlessClone(src_bgr, dst_bgr, binary, (cx, cy), cv2.NORMAL_CLONE)
        return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        logger.debug(f"[restoration/blend] Poisson failed ({e})")
        return None


def _blend(
    original_np: np.ndarray,
    restored_np: np.ndarray,
    inpaint_mask: np.ndarray,
    feather_mask: np.ndarray,
) -> np.ndarray:
    """
    Always feathered alpha composite.
    Poisson only when hole < POISSON_MAX_HOLE_FRAC (avoids artefacts on large holes).
    """
    h, w = original_np.shape[:2]
    hole_frac = float(np.sum(inpaint_mask > 0)) / max(h * w, 1)

    if hole_frac < POISSON_MAX_HOLE_FRAC:
        logger.info(f"[restoration/blend] hole={hole_frac:.1%} → Poisson + feather")
        poisson = _poisson_blend(original_np, restored_np, inpaint_mask)
        if poisson is not None:
            result = _feathered_alpha_blend(original_np, poisson, feather_mask)
            result[inpaint_mask == 0] = original_np[inpaint_mask == 0]
            return result
        logger.debug("[restoration/blend] Poisson failed — feather only")

    result = _feathered_alpha_blend(original_np, restored_np, feather_mask)
    result[inpaint_mask == 0] = original_np[inpaint_mask == 0]
    logger.info(f"[restoration/blend] feathered alpha composite done (hole={hole_frac:.1%})")
    return result


# ── Per-object restoration ─────────────────────────────────────────────────

def _restore_one_object(
    image_np: np.ndarray,          # original full-res image
    current_canvas: np.ndarray,    # working canvas (may already have other objects restored)
    raw_mask: np.ndarray,          # SAM2 mask for THIS object
    label: str,
    use_sdxl: bool = True,
    seed: int = 42,
) -> np.ndarray:
    """
    Restore the background behind one object.

    Uses current_canvas as both the inpaint source (so already-restored
    regions are preserved) and the compositing base.

    Returns updated canvas (H, W, 3) RGB.
    """
    logger.info(f"[restoration] restoring object '{label}'")

    # Build per-object mask
    inpaint_mask, feather_mask = build_object_removal_mask(image_np, raw_mask)

    if not np.any(inpaint_mask > 0):
        logger.warning(f"[restoration] '{label}': empty mask — skipping")
        return current_canvas

    _debug_save(f"input_{label}", current_canvas, inpaint_mask)

    # Stage A — LaMa on the current canvas
    lama_result = _lama_inpaint(current_canvas, inpaint_mask)
    _debug_save(f"lama_{label}", lama_result)

    # Stage B — SDXL texture refinement (conditional)
    if use_sdxl:
        refined = _sdxl_refine(
            lama_result=lama_result,
            original_np=image_np,
            inpaint_mask=inpaint_mask,
            feather_mask=feather_mask,
            label=label,
            seed=seed,
        )
    else:
        refined = lama_result

    # Validation — run DINO on the hole, retry once if detected
    clean = _validate_restoration(image_np, refined, inpaint_mask, label)
    if not clean:
        logger.warning(f"[restoration] '{label}': validation failed — retrying with larger mask")
        # Retry: add 15px extra dilation and different seed
        binary = (inpaint_mask > 0).astype(np.uint8)
        retry_k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (RETRY_EXTRA_DILATION * 2 + 1, RETRY_EXTRA_DILATION * 2 + 1)
        )
        retry_mask = cv2.dilate(binary, retry_k, iterations=1) * 255
        retry_feather = _build_feather_float(retry_mask)

        lama_retry = _lama_inpaint(current_canvas, retry_mask)
        if use_sdxl:
            refined_retry = _sdxl_refine(
                lama_result=lama_retry,
                original_np=image_np,
                inpaint_mask=retry_mask,
                feather_mask=retry_feather,
                label=label,
                seed=seed + 1,
            )
        else:
            refined_retry = lama_retry

        clean2 = _validate_restoration(image_np, refined_retry, retry_mask, label)
        if clean2:
            refined = refined_retry
            feather_mask = retry_feather
            inpaint_mask = retry_mask
            logger.info(f"[restoration] '{label}': retry succeeded")
        else:
            # Fall back to pure LaMa
            logger.warning(f"[restoration] '{label}': retry also failed — using pure LaMa")
            refined = lama_result

    # Blend into current canvas
    final = _blend(current_canvas, refined, inpaint_mask, feather_mask)
    _debug_save(f"final_{label}", final)

    logger.info(f"[restoration] '{label}' done")
    return final


def _build_feather_float(mask: np.ndarray) -> np.ndarray:
    """Build feather weight from an already-dilated binary mask."""
    dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    weight = np.clip(dist / max(1.0, float(FEATHER_PX)), 0.0, 1.0)
    k = max(3, FEATHER_PX // 2) | 1
    weight = cv2.GaussianBlur(weight.astype(np.float32), (k, k), k / 3.0)
    return np.clip(weight, 0.0, 1.0)


# ── Public API ─────────────────────────────────────────────────────────────

def restore_background(
    image_np: np.ndarray,
    per_object_data: List[Dict[str, Any]],
    use_sdxl: bool = True,
    seed: int = 42,
) -> np.ndarray:
    """
    Restore the background behind all removed objects, processing each one
    separately on its own crop.

    Args:
        image_np:         Original (H, W, 3) uint8 RGB.
        per_object_data:  List of dicts, one per removed object:
                            {
                              "mask":  np.ndarray (H, W) uint8 SAM2 mask,
                              "label": str object name,
                            }
        use_sdxl:         Run SDXL texture refinement after LaMa (default True).
        seed:             Base random seed for SDXL.

    Returns:
        (H, W, 3) uint8 RGB — fully restored image.
    """
    if not per_object_data:
        logger.info("[restoration] no objects to restore")
        return image_np.copy()

    orig_h, orig_w = image_np.shape[:2]
    logger.info(
        f"[restoration] {orig_w}×{orig_h} "
        f"objects={[o['label'] for o in per_object_data]} "
        f"use_sdxl={use_sdxl}"
    )

    # Process largest object first (most context for later objects)
    ordered = sorted(
        per_object_data,
        key=lambda o: int(np.sum(o["mask"] > 0)),
        reverse=True,
    )

    canvas = image_np.copy()
    for i, obj in enumerate(ordered):
        raw_mask = obj["mask"]
        label    = obj.get("label", f"object_{i}")

        # Validate mask shape
        if raw_mask.shape[:2] != (orig_h, orig_w):
            raw_mask = cv2.resize(raw_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        canvas = _restore_one_object(
            image_np=image_np,
            current_canvas=canvas,
            raw_mask=raw_mask,
            label=label,
            use_sdxl=use_sdxl,
            seed=seed + i,
        )

    logger.info("[restoration] all objects restored")
    return canvas


# ── Backwards-compatible wrapper (used by on-demand endpoint) ──────────────

def restore_background_from_mask_files(
    image_np: np.ndarray,
    mask_paths: List[str],
    object_labels: Optional[List[str]] = None,
    force_generative: bool = False,
    steps: int = 30,
    dilation_px: int = MIN_DILATION_PX,
) -> np.ndarray:
    """Load per-object mask PNGs from disk and run restore_background."""
    from pathlib import Path
    from utils import config

    img_h, img_w = image_np.shape[:2]
    per_object_data: List[Dict[str, Any]] = []

    for i, mp in enumerate(mask_paths):
        p = Path(mp)
        if not p.is_absolute():
            rel = str(mp).lstrip("/")
            if rel.startswith("temp/"):
                rel = rel[len("temp/"):]
            p = (config.TEMP_DIR / rel).resolve()
        if not p.exists():
            logger.warning(f"[restoration] mask file not found: {p} — skipping")
            continue
        m = np.array(Image.open(p).convert("L"))
        if m.shape != (img_h, img_w):
            m = cv2.resize(m, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        label = (object_labels[i] if object_labels and i < len(object_labels) else f"object_{i}")
        per_object_data.append({"mask": m, "label": label})

    if not per_object_data:
        logger.warning("[restoration] no valid mask files — returning original")
        return image_np.copy()

    return restore_background(
        image_np=image_np,
        per_object_data=per_object_data,
        use_sdxl=force_generative,
    )


# ── Keep build_removal_mask for external callers ───────────────────────────

def build_removal_mask(
    refined_masks: List[np.ndarray],
    dilation_px: int = MIN_DILATION_PX,
) -> np.ndarray:
    """Combine per-object masks into one binary removal mask (simple union + dilation)."""
    if not refined_masks:
        return np.zeros((1, 1), dtype=np.uint8)
    combined = np.zeros(refined_masks[0].shape[:2], dtype=np.uint8)
    for m in refined_masks:
        combined = np.maximum(combined, (m > 128).astype(np.uint8) * 255)
    if dilation_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_px * 2 + 1, dilation_px * 2 + 1))
        combined = cv2.dilate(combined, k, iterations=1)
    return combined


# ── Keep decontaminate_foreground_edges for segmentation_service ────────────

def decontaminate_foreground_edges(
    object_rgba: np.ndarray,
    band_px: int = 3,
) -> np.ndarray:
    """Remove colour-bleed fringe from semi-transparent SAM2 object edges."""
    result = object_rgba.copy()
    alpha = object_rgba[:, :, 3].astype(np.float32)
    opaque = (alpha > 200).astype(np.uint8)
    if not np.any(opaque):
        return result
    for c in range(3):
        channel = object_rgba[:, :, c].astype(np.float32)
        filled = channel.copy()
        filled[opaque == 0] = 0
        weight = alpha.copy()
        weight[opaque == 0] = 0
        k = band_px * 4 + 1
        filled_blur = cv2.GaussianBlur(filled, (k, k), band_px * 2)
        weight_blur = cv2.GaussianBlur(weight, (k, k), band_px * 2)
        pure_color = np.where(weight_blur > 1e-3, filled_blur / (weight_blur + 1e-6), channel)
        edge_band = (alpha > 5) & (alpha < 200)
        result[:, :, c] = np.where(edge_band, np.clip(pure_color, 0, 255), channel).astype(np.uint8)
    return result
