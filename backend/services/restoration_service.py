"""
Intelligent background restoration after object cutout.

Fixed bugs vs. previous version
────────────────────────────────
C1  _build_feather_weight produced near-zero weights for large objects because
    it multiplied by (feather_px / max_dist_inside), collapsing everything to ~0.
    Fixed: compute weight purely from distance transform, no normalisation by
    max_dist_inside. Hard overrides removed; smooth ramps used instead.

C2  Stage C blended original object-edge pixels back in (ghost halo) because
    the weight was near-zero in the transition zone.  Fixed by C1.

C3  cv2.inpaint ran on the full canvas with inpaintRadius=12, causing colour
    drift for large holes.  Fixed: crop a padded bounding-box context window,
    run inpaint on that crop (so the radius reaches real background faster),
    paste the result back.

C4  Fixed-radius dilation missed colour-contaminated boundary pixels.
    Fixed: adaptive_removal_mask() adds a colour-contamination ring on top of
    the flat dilation: pixels just outside the mask that contain significant
    object-colour bleed are included in the removal region.

C5  SDXL received a hard binary mask with no softening.  Fixed: the mask
    passed to the generative model is Gaussian-blurred so the model sees a
    soft gradient and generates natural boundary transitions.

C6  SDXL received the full downscaled image.  Fixed: crop a padded bounding
    box around the object, run inpaint at high resolution on the crop, paste
    back.  The model now sees the object at much higher effective resolution.

C7  Hard weight overrides at dist==0 and dist>feather_px created two
    additional ring artefacts.  Fixed: smooth sigmoid/linear ramps replace
    hard steps.

C8  Pure alpha blending for large regions left visible colour patches when
    SDXL generated slightly different illumination.  Fixed: Poisson seamless
    clone is applied for large holes; colour matching is applied before blend.

Pipeline overview
─────────────────
  adaptive_removal_mask()   build scale-aware, colour-contamination-aware mask
  _stage_a()                cv2 TELEA+NS on a padded crop (better context)
  _stage_b()                SDXL/FLUX on a padded crop with soft mask blur
  _color_decontaminate()    replace boundary-band residual object pixels
  _stage_c()                correct feathered blend + Poisson for large holes
  restore_background()      orchestrator
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

GENERATIVE_THRESHOLD: float = 0.05    # hole fraction that triggers Stage B
COMPLEX_SCENE_EDGE_THRESHOLD: float = 0.08  # bg edge density → Stage B

DEFAULT_DILATION_PX: int = 12         # base mask dilation radius
SEAM_FEATHER_PX: int = 24             # feather width in Stage C (was 16)
MAX_INPAINT_DIM: int = 1024           # max dim for generative model
CONTEXT_PAD_FACTOR: float = 0.35      # bounding-box padding factor for crop
MIN_CONTEXT_PAD_PX: int = 80          # minimum crop padding in pixels
MASK_BLUR_SIGMA: float = 8.0          # Gaussian blur sigma for model mask
POISSON_THRESHOLD: float = 0.12       # hole fraction above which Poisson is used

SAME_PIXEL_DIST: float = 10.0    # L2 threshold to call a pixel "same as original"
SAME_PIXEL_FRAC: float = 0.80    # fraction of mask pixels that must differ for result to be accepted
                                  # raised from 0.60 — false rejections occurred when background
                                  # colour happened to match the original's surrounding area

# Color-contamination detection thresholds
COLOR_CONTAM_BAND_PX: int = 6         # pixels outside mask to inspect for bleed
COLOR_CONTAM_MAX_DIST: float = 40.0   # max L2 dist-to-object-mean to flag pixel


# ---------------------------------------------------------------------------
# Adaptive removal mask (fixes BUG C4)
# ---------------------------------------------------------------------------

def adaptive_removal_mask(
    image_np: np.ndarray,
    hard_mask: np.ndarray,
    dilation_px: int = DEFAULT_DILATION_PX,
    protection_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Build a removal mask that is larger than the raw segmentation mask in two ways:

    1. Flat dilation scaled to image resolution (dilation_px, but capped to avoid
       destroying nearby objects protected by protection_mask).

    2. Colour-contamination expansion: pixels in the band just outside the raw
       mask that contain significant object-colour bleed are added to the
       removal region.  This catches SAM2's feathered-edge semi-transparent
       pixels that mix object + background colours.

    The protection_mask (binary, 255 = protect) prevents expansion from
    touching pixels claimed by other independently-segmented objects.

    Args:
        image_np:        (H, W, 3) uint8 RGB — original image.
        hard_mask:       (H, W) uint8 — raw combined object mask (0/255).
        dilation_px:     Base dilation radius.
        protection_mask: (H, W) uint8 — pixels that must not be added (other objects).

    Returns:
        (H, W) uint8 removal mask (0/255) ready for inpainting.
    """
    h, w = image_np.shape[:2]
    binary = (hard_mask > 128).astype(np.uint8)

    # ── 1. Scale-aware dilation ────────────────────────────────────────────
    # Scale dilation_px by the geometric mean of image dimensions so that
    # the same relative expansion is applied at different resolutions.
    scale_factor = np.sqrt(h * w) / 1000.0
    scaled_dil = max(4, int(dilation_px * min(scale_factor, 2.0)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (scaled_dil * 2 + 1, scaled_dil * 2 + 1)
    )
    dilated = cv2.dilate(binary, kernel, iterations=1)

    # ── 2. Colour-contamination expansion ─────────────────────────────────
    # Estimate the object's representative colour from pixels 4–12px inside
    # the original mask (safely interior, no edge contamination).
    inner_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (9, 9)
    )
    inner_core = cv2.erode(binary, inner_kernel, iterations=1)
    core_pixels = image_np[inner_core > 0]

    if len(core_pixels) >= 50:
        obj_color_mean = np.median(core_pixels, axis=0).astype(np.float32)

        # Build a band of pixels: outside the dilated mask but within
        # COLOR_CONTAM_BAND_PX of the outer edge of the original mask
        contam_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (COLOR_CONTAM_BAND_PX * 2 + 1, COLOR_CONTAM_BAND_PX * 2 + 1),
        )
        outer_band_mask = cv2.dilate(binary, contam_kernel, iterations=1)
        band = (outer_band_mask > 0) & (dilated == 0)  # just outside dilated edge

        if np.any(band):
            band_pixels = image_np[band].astype(np.float32)
            dists = np.linalg.norm(band_pixels - obj_color_mean, axis=1)
            contaminated = np.zeros((h, w), dtype=bool)
            contaminated[band] = dists < COLOR_CONTAM_MAX_DIST
            dilated = np.maximum(dilated, contaminated.astype(np.uint8))
            n_contam = int(np.sum(contaminated))
            if n_contam > 0:
                logger.debug(f"[restoration/mask] added {n_contam} colour-contaminated pixels")

    result = (dilated * 255).astype(np.uint8)

    # ── 3. Apply protection mask ───────────────────────────────────────────
    if protection_mask is not None:
        protected = (protection_mask > 128)
        result[protected] = 0

    return result


def build_removal_mask(
    refined_masks: List[np.ndarray],
    dilation_px: int = DEFAULT_DILATION_PX,
) -> np.ndarray:
    """
    Combine all per-object SAM2 masks into one binary removal mask.
    (Simple version used when image_np is not available; adaptive version preferred.)
    """
    if not refined_masks:
        return np.zeros((1, 1), dtype=np.uint8)

    combined = np.zeros(refined_masks[0].shape[:2], dtype=np.uint8)
    for m in refined_masks:
        combined = np.maximum(combined, (m > 128).astype(np.uint8) * 255)

    if dilation_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilation_px * 2 + 1, dilation_px * 2 + 1)
        )
        combined = cv2.dilate(combined, kernel, iterations=1)

    return combined


# ---------------------------------------------------------------------------
# Mask → padded bounding-box crop (fixes BUG C3, C6)
# ---------------------------------------------------------------------------

def _mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Return (y1, x1, y2, x2) tight bounding box of the non-zero mask region."""
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)
    if not np.any(rows):
        h, w = mask.shape[:2]
        return 0, 0, h, w
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return int(y1), int(x1), int(y2) + 1, int(x2) + 1


def _padded_crop(
    image_np: np.ndarray,
    mask: np.ndarray,
    pad_factor: float = CONTEXT_PAD_FACTOR,
    min_pad: int = MIN_CONTEXT_PAD_PX,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """
    Crop image+mask to the mask's bounding box expanded by pad_factor on each side.

    Returns (crop_img, crop_mask, (y1, x1, y2, x2)) where the coordinates
    are in the FULL image coordinate system so the result can be pasted back.
    """
    h, w = image_np.shape[:2]
    ty1, tx1, ty2, tx2 = _mask_bbox(mask)

    obj_h = ty2 - ty1
    obj_w = tx2 - tx1
    pad_y = max(min_pad, int(obj_h * pad_factor))
    pad_x = max(min_pad, int(obj_w * pad_factor))

    cy1 = max(0, ty1 - pad_y)
    cx1 = max(0, tx1 - pad_x)
    cy2 = min(h, ty2 + pad_y)
    cx2 = min(w, tx2 + pad_x)

    return image_np[cy1:cy2, cx1:cx2].copy(), mask[cy1:cy2, cx1:cx2].copy(), (cy1, cx1, cy2, cx2)


def _paste_crop(
    canvas: np.ndarray,
    crop: np.ndarray,
    coords: Tuple[int, int, int, int],
    blend_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Paste a crop back into a canvas at the given coordinates.

    If blend_mask is provided (float32, 0-1, same size as crop), it is used
    for alpha compositing instead of a hard paste.
    """
    result = canvas.copy()
    cy1, cx1, cy2, cx2 = coords
    if blend_mask is None:
        result[cy1:cy2, cx1:cx2] = crop
    else:
        bm = blend_mask[:, :, np.newaxis]
        orig_region = canvas[cy1:cy2, cx1:cx2].astype(np.float32)
        crop_f = crop.astype(np.float32)
        blended = orig_region * (1.0 - bm) + crop_f * bm
        result[cy1:cy2, cx1:cx2] = np.clip(blended, 0, 255).astype(np.uint8)
    return result


# ---------------------------------------------------------------------------
# Complexity routing
# ---------------------------------------------------------------------------

def _needs_generative(
    image_np: np.ndarray,
    removal_mask: np.ndarray,
    force: bool,
) -> bool:
    if force:
        return True

    h, w = image_np.shape[:2]
    hole_fraction = float(np.sum(removal_mask > 0)) / max(h * w, 1)

    if hole_fraction > GENERATIVE_THRESHOLD:
        logger.info(f"[restoration] hole={hole_fraction:.1%} → Stage B")
        return True

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    bg = (removal_mask == 0)
    edge_density = float(np.sum(edges[bg] > 0)) / max(int(np.sum(bg)), 1)

    if edge_density > COMPLEX_SCENE_EDGE_THRESHOLD:
        logger.info(f"[restoration] bg edge_density={edge_density:.3f} → Stage B")
        return True

    logger.info(f"[restoration] hole={hole_fraction:.1%}, edge={edge_density:.3f} → Stage A")
    return False


# ---------------------------------------------------------------------------
# Stage A — Classical inpainting on a padded crop (fixes BUG C3)
# ---------------------------------------------------------------------------

def _stage_a(image_np: np.ndarray, removal_mask: np.ndarray) -> np.ndarray:
    """
    Classical inpainting using cv2 TELEA + NS on a padded crop.

    Critical fix: before running cv2.inpaint, we replace the hole pixels in
    the crop with the median colour of the surrounding background.  cv2.inpaint
    propagates from the hole *boundary* inward — if the hole still contains the
    original object pixels, those colours contaminate the propagation and the
    object can bleed back into the result.  Pre-zeroing forces propagation
    from clean background only.
    """
    logger.info("[restoration/A] cv2 TELEA + NS inpainting (padded crop, pre-zeroed hole)…")

    crop_img, crop_mask, coords = _padded_crop(image_np, removal_mask)
    cy1, cx1, cy2, cx2 = coords

    # ── Pre-zero the hole with median background colour ────────────────────
    # Sample background pixels in the crop (outside the mask)
    bg_pixels = crop_img[crop_mask == 0]
    if len(bg_pixels) >= 10:
        bg_median = np.median(bg_pixels, axis=0).astype(np.uint8)
    else:
        bg_median = np.array([128, 128, 128], dtype=np.uint8)

    crop_prezeroed = crop_img.copy()
    crop_prezeroed[crop_mask > 0] = bg_median  # replace object with flat bg colour

    bgr_crop = cv2.cvtColor(crop_prezeroed, cv2.COLOR_RGB2BGR)

    crop_h, crop_w = crop_img.shape[:2]
    radius = max(8, int(min(crop_h, crop_w) * 0.08))
    radius = min(radius, 32)

    result_bgr = cv2.inpaint(bgr_crop, crop_mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)
    result_bgr = cv2.inpaint(result_bgr, crop_mask, inpaintRadius=max(4, radius // 2), flags=cv2.INPAINT_NS)

    result_crop = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    full_result = image_np.copy()
    full_result[cy1:cy2, cx1:cx2][crop_mask > 0] = result_crop[crop_mask > 0]

    logger.info("[restoration/A] cv2 inpainting complete")
    return full_result


# ---------------------------------------------------------------------------
# Stage B — Generative inpainting on a padded crop with soft mask (fixes C5, C6)
# ---------------------------------------------------------------------------

def _soft_mask(mask: np.ndarray, sigma: float = MASK_BLUR_SIGMA) -> np.ndarray:
    """
    Blur a binary mask to produce a soft gradient edge.
    Returns float32 [0,1] mask.
    The core (pixels far inside) stays near 1; the boundary feathers to 0.
    """
    binary_f = (mask > 0).astype(np.float32)
    k = int(sigma * 6) | 1  # ensure odd
    blurred = cv2.GaussianBlur(binary_f, (k, k), sigma)
    # Re-normalise so the interior stays 1.0
    if blurred.max() > 0:
        blurred = blurred / blurred.max()
    return blurred


def _resize_for_inpaint(
    image_np: np.ndarray,
    mask: np.ndarray,
    max_dim: int = MAX_INPAINT_DIM,
) -> Tuple[np.ndarray, np.ndarray, float]:
    h, w = image_np.shape[:2]
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        nw, nh = int(w * scale), int(h * scale)
        image_np = cv2.resize(image_np, (nw, nh), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    h2, w2 = image_np.shape[:2]
    nw = (w2 // 8) * 8
    nh = (h2 // 8) * 8
    if nw != w2 or nh != h2:
        image_np = cv2.resize(image_np, (nw, nh), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    return image_np, mask, scale


def _guard_original_pixels(
    original_np: np.ndarray,
    result_np: np.ndarray,
    removal_mask: np.ndarray,
) -> bool:
    """
    Return True only when the inpaint model clearly reproduced the original
    object pixels inside the mask (i.e. completely failed to inpaint).

    Previous version: compared result to raw original pixels inside mask.
    Problem: on a shoe ad with a red background, the shoe is also red — the
    background reconstruction might be nearly identical to the non-shoe parts
    of the original, triggering a false rejection and falling back to Stage A,
    which then propagates the shoe back into the hole.

    Fixed approach: compare the result pixels inside the mask to the
    *object interior colour mean* (sampled deep inside the mask from the
    original).  A valid background reconstruction will be far from the object
    colour.  A failed reconstruction (object reproduced) will be close.

    Only fires when the result is objectively identical to the original object
    colour (same_frac > SAME_PIXEL_FRAC at a tighter L2 distance).
    """
    mask_bool = removal_mask > 0
    if not np.any(mask_bool):
        return False

    # Estimate the object's interior colour from the original image.
    # Erode the mask to get the core (avoids boundary background pixels).
    binary = (removal_mask > 0).astype(np.uint8)
    inner_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    inner = cv2.erode(binary, inner_k, iterations=1)
    core_pixels = original_np[inner > 0] if np.any(inner > 0) else original_np[mask_bool]

    if len(core_pixels) < 10:
        return False  # not enough pixels to judge — don't reject

    obj_mean = np.median(core_pixels, axis=0).astype(np.float32)

    # How similar is the result (inside the mask) to the original object colour?
    result_inside = result_np[mask_bool].astype(np.float32)
    dists_to_obj  = np.linalg.norm(result_inside - obj_mean, axis=1)
    same_as_obj_frac = float(np.mean(dists_to_obj < SAME_PIXEL_DIST))

    if same_as_obj_frac > SAME_PIXEL_FRAC:
        logger.warning(
            f"[restoration/B] {same_as_obj_frac:.0%} of result pixels match object colour "
            f"(obj_mean={obj_mean.astype(int).tolist()}) — model reproduced the object, rejecting"
        )
        return True

    logger.debug(
        f"[restoration/B] guard: {same_as_obj_frac:.0%} match object colour — accepted"
    )
    return False


def _build_inpaint_prompt(
    image_np: np.ndarray,
    object_labels: List[str],
    removal_mask: np.ndarray,
) -> str:
    bg_pixels = image_np[(removal_mask == 0)]
    if len(bg_pixels) > 0:
        dom = np.median(bg_pixels, axis=0).astype(int)
        colour_hint = f"dominant background colour approximately rgb({dom[0]},{dom[1]},{dom[2]})"
    else:
        colour_hint = "natural background"
    objects_str = ", ".join(object_labels) if object_labels else "objects"
    return (
        f"Remove the {objects_str} completely. Reconstruct only the background that would "
        f"naturally exist behind them. Continue the surrounding structure, texture, perspective, "
        f"and lighting. {colour_hint}. "
        f"Do not reproduce any part of the removed subjects. "
        f"Clean, seamless, photorealistic background continuation. No outlines, no halos."
    )


def _stage_b_local(
    image_np: np.ndarray,
    removal_mask: np.ndarray,
    prompt: str,
    steps: int,
    stage_a_result: np.ndarray,
    object_labels: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Local generative inpainting on a padded crop with a soft mask.

    Crops a padded region around the object, runs SDXL inpainting at high
    effective resolution, pastes the result back.  The mask fed to SDXL is
    Gaussian-blurred for natural boundary transitions.
    """
    from services.model_manager import model_manager
    import torch

    orig_h, orig_w = image_np.shape[:2]

    # ── Crop context window around hole ───────────────────────────────────
    crop_img, crop_mask_hard, coords = _padded_crop(image_np, removal_mask)
    cy1, cx1, cy2, cx2 = coords

    # Soft mask for SDXL (gradient edge, not hard binary)
    crop_mask_soft_f = _soft_mask(crop_mask_hard, sigma=MASK_BLUR_SIGMA)
    crop_mask_soft = (np.clip(crop_mask_soft_f * 255, 0, 255)).astype(np.uint8)

    # Resize crop for model
    crop_small, mask_small, scale = _resize_for_inpaint(crop_img, crop_mask_soft)
    ch, cw = crop_small.shape[:2]

    image_pil = Image.fromarray(crop_small)
    mask_pil  = Image.fromarray(mask_small).convert("RGB")

    logger.info(f"[restoration/B] SDXL on crop {cw}×{ch} (orig {orig_w}×{orig_h}) steps={steps}")

    # Negative prompt: explicitly suppress the objects we removed
    _labels = object_labels or []
    objects_str = ", ".join(_labels) if _labels else "object"
    negative_prompt = (
        f"{objects_str}, product, shoe, sneaker, clothing, person, animal, "
        f"watermark, text, logo, duplicate, artifact, blurry, low quality"
    )

    try:
        pipe = model_manager.get_inpaint_pipe()
        gen  = torch.Generator(device="cpu").manual_seed(42)
        out = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image_pil,
            mask_image=mask_pil,
            height=ch,
            width=cw,
            num_inference_steps=steps,
            guidance_scale=12.0,   # raised from 7.5 — forces stronger adherence to prompt
            generator=gen,
        ).images[0]

        result_crop = np.array(out.convert("RGB"))
        if scale < 1.0:
            result_crop = cv2.resize(
                result_crop, (cx2 - cx1, cy2 - cy1), interpolation=cv2.INTER_LANCZOS4
            )

        # Only accept pixels where the crop mask is active
        full_result = image_np.copy()
        crop_mask_paste = crop_mask_hard > 0
        full_result[cy1:cy2, cx1:cx2][crop_mask_paste] = result_crop[crop_mask_paste]

        if _guard_original_pixels(image_np, full_result, removal_mask):
            logger.warning("[restoration/B] SDXL returned original pixels → FLUX fallback")
            raise RuntimeError("guard failed")

        logger.info("[restoration/B] SDXL accepted")
        try:
            model_manager.unload_inpaint_pipe()
        except Exception:
            pass
        return full_result

    except Exception as e:
        logger.warning(f"[restoration/B] SDXL failed ({e}), trying FLUX.1 Fill…")

    # Fallback: FLUX.1 Fill (operates on full image, already handles large context)
    try:
        from services.flux_inpaint_service import reconstruct_background_with_flux
        result_pil = reconstruct_background_with_flux(
            original_image=Image.fromarray(image_np),
            combined_mask=removal_mask,
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=3.5,
            max_image_size=MAX_INPAINT_DIM,
        )
        result_np = np.array(result_pil.convert("RGB"))
        if _guard_original_pixels(image_np, result_np, removal_mask):
            logger.warning("[restoration/B] FLUX returned original pixels → Stage A fallback")
            return stage_a_result
        logger.info("[restoration/B] FLUX.1 Fill accepted")
        return result_np
    except Exception as fe:
        logger.warning(f"[restoration/B] FLUX failed ({fe}) → Stage A fallback")
        return stage_a_result


def _stage_b_remote(
    image_np: np.ndarray,
    removal_mask: np.ndarray,
    prompt: str,
    steps: int,
    stage_a_result: np.ndarray,
) -> np.ndarray:
    from services.remote_services.inpaint import inpaint_remote

    orig_h, orig_w = image_np.shape[:2]
    crop_img, crop_mask_hard, coords = _padded_crop(image_np, removal_mask)
    cy1, cx1, cy2, cx2 = coords

    crop_mask_soft = (np.clip(_soft_mask(crop_mask_hard) * 255, 0, 255)).astype(np.uint8)
    crop_small, mask_small, scale = _resize_for_inpaint(crop_img, crop_mask_soft)

    rgba = Image.fromarray(crop_small).convert("RGBA")
    rgba.putalpha(Image.fromarray(mask_small))

    try:
        result_rgba = inpaint_remote(rgba, prompt, strength=0.99, guidance_scale=7.5, steps=steps)
        result_crop = np.array(result_rgba.convert("RGB"))
        if scale < 1.0:
            result_crop = cv2.resize(result_crop, (cx2 - cx1, cy2 - cy1), interpolation=cv2.INTER_LANCZOS4)

        full_result = image_np.copy()
        full_result[cy1:cy2, cx1:cx2][crop_mask_hard > 0] = result_crop[crop_mask_hard > 0]

        if _guard_original_pixels(image_np, full_result, removal_mask):
            logger.warning("[restoration/B/remote] returned original pixels → Stage A fallback")
            return stage_a_result
        return full_result
    except Exception as e:
        logger.warning(f"[restoration/B/remote] failed ({e}) → Stage A fallback")
        return stage_a_result


def _stage_b(
    image_np: np.ndarray,
    removal_mask: np.ndarray,
    object_labels: List[str],
    steps: int,
    stage_a_result: np.ndarray,
) -> np.ndarray:
    from services.remote_config import remote_cfg
    prompt = _build_inpaint_prompt(image_np, object_labels, removal_mask)
    if remote_cfg.COMPUTE_MODE == "remote":
        return _stage_b_remote(image_np, removal_mask, prompt, steps, stage_a_result)
    return _stage_b_local(
        image_np, removal_mask, prompt, steps, stage_a_result,
        object_labels=object_labels,
    )


# ---------------------------------------------------------------------------
# Color decontamination (fixes BUG C4 residual bleed after inpaint)
# ---------------------------------------------------------------------------

def _color_decontaminate(
    original_np: np.ndarray,
    restored_np: np.ndarray,
    removal_mask: np.ndarray,
    band_px: int = COLOR_CONTAM_BAND_PX,
) -> np.ndarray:
    """
    Replace residual object-coloured pixels in the narrow band just inside
    the removal mask boundary with the surrounding restored pixels.

    After inpainting, a few pixels at the very edge of the mask can still
    carry object-colour bleed because:
      a) cv2.inpaint's radius-limited propagation didn't fully overwrite them.
      b) The SDXL model blended a small amount of the original at the edge.

    Strategy: build a thin inner shell (band_px wide) of the removal mask.
    For each pixel in that shell, compare its L2 distance to the object-colour
    mean.  If it's too close to the object colour, replace it with a
    distance-weighted average of its immediate restored neighbours.
    """
    binary = (removal_mask > 0).astype(np.uint8)

    # Inner core: pixels deeply inside the mask (not on the boundary)
    inner_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    inner = cv2.erode(binary, inner_kernel, iterations=1)

    # Estimate object colour from a small interior zone of the ORIGINAL image
    # (before any restoration — we want the object's colours, not the fill).
    core_pixels = original_np[inner > 0]
    if len(core_pixels) < 20:
        # Not enough interior pixels — skip decontamination
        return restored_np

    obj_mean = np.median(core_pixels, axis=0).astype(np.float32)

    # Outer shell of the removal mask
    band_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (band_px * 2 + 1, band_px * 2 + 1)
    )
    dilated_binary = cv2.dilate(binary, band_kernel, iterations=1)
    shell = (dilated_binary > 0) & (inner == 0)  # ring between eroded and dilated

    if not np.any(shell):
        return restored_np

    result = restored_np.copy()
    shell_pixels = restored_np[shell].astype(np.float32)
    dists = np.linalg.norm(shell_pixels - obj_mean, axis=1)

    # Pixels in the shell that are still object-coloured
    contaminated_in_shell = shell.copy()
    contaminated_in_shell[shell] = dists < (COLOR_CONTAM_MAX_DIST * 0.7)

    if not np.any(contaminated_in_shell):
        return restored_np

    # Replace contaminated pixels with Gaussian-blurred restored image
    # (this pulls in clean surrounding background colour)
    blur_sigma = max(4.0, band_px * 1.5)
    k = int(blur_sigma * 4) | 1
    blurred = cv2.GaussianBlur(restored_np, (k, k), blur_sigma)
    result[contaminated_in_shell] = blurred[contaminated_in_shell]

    n = int(np.sum(contaminated_in_shell))
    logger.debug(f"[restoration/decontam] replaced {n} residual object-colour pixels in shell")
    return result


# ---------------------------------------------------------------------------
# Stage C — Correct feathered seam blend + Poisson (fixes BUG C1, C7, C8)
# ---------------------------------------------------------------------------

def _build_feather_weight(
    removal_mask: np.ndarray,
    feather_px: int = SEAM_FEATHER_PX,
) -> np.ndarray:
    """
    Build a float32 weight map [0..1] for compositing:
      0.0 = use original (outside mask)
      1.0 = use reconstructed (inside mask)
      smooth ramp in the feather band

    FIXED vs. previous version:
    - No longer multiplies by (feather_px / max_dist_inside) which collapsed
      all weights to near-zero for large objects.
    - Uses a pure distance-transform ramp clamped to [0, feather_px]:
        weight = min(dist_inside, feather_px) / feather_px
    - Then applies a single Gaussian blur to smooth staircase artefacts,
      but does NOT apply hard overrides at the extremes (those created rings).
    - Interior pixels (dist_inside > feather_px) get weight 1.0 via the clamp.
    - Boundary pixels (dist_inside == 0) get weight 0.0 via the clamp.
    """
    binary = (removal_mask > 0).astype(np.uint8)
    dist_inside = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    # Correct ramp: linear from 0 at boundary to 1.0 at feather_px inside
    weight = np.clip(dist_inside / max(1.0, float(feather_px)), 0.0, 1.0)

    # Smooth the ramp to remove Euclidean-distance staircase artefacts
    # Use a small kernel — we want smoothing of the ramp shape, not blurring
    k = max(3, feather_px // 4) | 1
    weight = cv2.GaussianBlur(weight, (k, k), k / 3.0)

    # Re-clamp after blur to prevent values outside [0,1] from accumulating
    weight = np.clip(weight, 0.0, 1.0)

    return weight.astype(np.float32)


def _colour_match_crop(
    original_np: np.ndarray,
    reconstructed_np: np.ndarray,
    removal_mask: np.ndarray,
    band_px: int = 24,
) -> np.ndarray:
    """
    Adjust the reconstructed image's colour inside the removal mask so that
    its mean and std match the original's background pixels in the boundary band.

    This eliminates gross colour/exposure mismatches from generative models
    (different white balance, tone mapping) without distorting textures.
    """
    # Sample background pixels in a band just outside the removal mask
    binary = (removal_mask > 0).astype(np.uint8)
    band_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (band_px * 2 + 1, band_px * 2 + 1)
    )
    outer_dilated = cv2.dilate(binary, band_kernel, iterations=1)
    outer_band = (outer_dilated > 0) & (binary == 0)

    if not np.any(outer_band) or not np.any(binary):
        return reconstructed_np

    ref_pixels = original_np[outer_band].astype(np.float32)
    gen_pixels  = reconstructed_np[binary > 0].astype(np.float32)

    if len(ref_pixels) < 10 or len(gen_pixels) < 10:
        return reconstructed_np

    ref_mean = ref_pixels.mean(axis=0)
    ref_std  = ref_pixels.std(axis=0) + 1e-6
    gen_mean = gen_pixels.mean(axis=0)
    gen_std  = gen_pixels.std(axis=0) + 1e-6

    result = reconstructed_np.astype(np.float32).copy()
    inside = binary > 0
    # Normalise generated pixels to ref statistics
    for c in range(3):
        result[inside, c] = (result[inside, c] - gen_mean[c]) / gen_std[c] * ref_std[c] + ref_mean[c]

    return np.clip(result, 0, 255).astype(np.uint8)


def _poisson_blend(
    original_np: np.ndarray,
    reconstructed_np: np.ndarray,
    removal_mask: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Poisson / seamless clone blending for large holes.
    Returns None if OpenCV seamlessClone fails (e.g. mask too close to border).
    """
    try:
        h, w = original_np.shape[:2]
        binary = (removal_mask > 0).astype(np.uint8) * 255

        # cv2.seamlessClone needs the destination to be the original and the
        # source to be the reconstructed image.  The mask marks the region to
        # transfer from source → destination.
        # Centre of the mask bounding box
        ys, xs = np.where(binary > 0)
        if len(ys) == 0:
            return None
        cy = int((ys.min() + ys.max()) // 2)
        cx = int((xs.min() + xs.max()) // 2)

        # Poisson requires mask not to touch the image border
        if ys.min() < 2 or ys.max() > h - 3 or xs.min() < 2 or xs.max() > w - 3:
            return None

        src_bgr = cv2.cvtColor(reconstructed_np, cv2.COLOR_RGB2BGR)
        dst_bgr = cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR)
        result_bgr = cv2.seamlessClone(src_bgr, dst_bgr, binary, (cx, cy), cv2.NORMAL_CLONE)
        return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        logger.debug(f"[restoration/C] Poisson blend failed ({e}) — using alpha blend")
        return None


def _stage_c(
    original_np: np.ndarray,
    reconstructed_np: np.ndarray,
    removal_mask: np.ndarray,
) -> np.ndarray:
    """
    Blend reconstructed pixels into the original image.

    Steps:
    1. Colour-match: align reconstructed region's mean/std to surrounding bg.
    2. Weight map: linear distance-transform ramp (FIXED — no broken normalisation).
    3. Poisson blend for large holes (>POISSON_THRESHOLD fraction of canvas).
    4. Weighted alpha blend as primary or fallback.
    5. Hard-clamp: pixels outside removal_mask are exactly the original.
    """
    logger.info("[restoration/C] seam blending…")

    h, w = original_np.shape[:2]
    hole_frac = float(np.sum(removal_mask > 0)) / max(h * w, 1)

    # ── 1. Colour match ────────────────────────────────────────────────────
    recon_matched = _colour_match_crop(original_np, reconstructed_np, removal_mask)

    # ── 2. Try Poisson for large holes ─────────────────────────────────────
    final = None
    if hole_frac > POISSON_THRESHOLD:
        logger.info(f"[restoration/C] hole={hole_frac:.1%} → Poisson blend")
        final = _poisson_blend(original_np, recon_matched, removal_mask)
        if final is not None:
            logger.info("[restoration/C] Poisson blend accepted")

    # ── 3. Fallback: distance-transform feathered alpha blend ──────────────
    if final is None:
        weight = _build_feather_weight(removal_mask, feather_px=SEAM_FEATHER_PX)
        w3 = weight[:, :, np.newaxis]
        blended = (
            original_np.astype(np.float32) * (1.0 - w3)
            + recon_matched.astype(np.float32) * w3
        )
        final = np.clip(blended, 0, 255).astype(np.uint8)

    # ── 4. Hard-lock: unmasked pixels are EXACTLY the original ─────────────
    bg = (removal_mask == 0)
    final[bg] = original_np[bg]

    logger.info("[restoration/C] seam blending complete")
    return final


# ---------------------------------------------------------------------------
# Main public entry point
# ---------------------------------------------------------------------------

def restore_background(
    image_np: np.ndarray,
    union_mask: np.ndarray,
    object_labels: Optional[List[str]] = None,
    refined_masks: Optional[List[np.ndarray]] = None,
    force_generative: bool = False,
    steps: int = 20,
    dilation_px: int = DEFAULT_DILATION_PX,
    protection_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Reconstruct the background behind all removed objects.

    Args:
        image_np:         Original (H, W, 3) uint8 RGB.
        union_mask:       Combined object mask (H, W) uint8 (0/255).
        object_labels:    Removed object names (for prompt).
        refined_masks:    Individual per-object masks (enables adaptive mask).
        force_generative: Always run Stage B.
        steps:            Diffusion steps for Stage B.
        dilation_px:      Base dilation radius.
        protection_mask:  Pixels that must not be removed (other objects).

    Returns:
        (H, W, 3) uint8 RGB fully restored image.
    """
    if object_labels is None:
        object_labels = []

    orig_h, orig_w = image_np.shape[:2]
    logger.info(
        f"[restoration] {orig_w}×{orig_h} objects={object_labels} "
        f"force_gen={force_generative} steps={steps} dil={dilation_px}"
    )

    # ── 0. Validate input ─────────────────────────────────────────────────
    if union_mask.shape[:2] != (orig_h, orig_w):
        union_mask = cv2.resize(union_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # ── 1. Adaptive removal mask ──────────────────────────────────────────
    removal_mask = adaptive_removal_mask(
        image_np=image_np,
        hard_mask=union_mask,
        dilation_px=dilation_px,
        protection_mask=protection_mask,
    )

    if not np.any(removal_mask > 0):
        logger.info("[restoration] mask is empty — returning original")
        return image_np.copy()

    logger.info(
        f"[restoration] removal mask: {int(np.sum(removal_mask > 0))} px "
        f"({100 * np.mean(removal_mask > 0):.1f}% of canvas)"
    )

    # ── 2. Stage A ────────────────────────────────────────────────────────
    stage_a_result = _stage_a(image_np, removal_mask)

    # ── 3. Stage B (conditional) ──────────────────────────────────────────
    use_generative = _needs_generative(image_np, removal_mask, force_generative)
    if use_generative:
        reconstructed = _stage_b(image_np, removal_mask, object_labels, steps, stage_a_result)
    else:
        reconstructed = stage_a_result

    # ── 4. Color decontamination ──────────────────────────────────────────
    reconstructed = _color_decontaminate(image_np, reconstructed, removal_mask)

    # ── 5. Stage C — seam blend ───────────────────────────────────────────
    final = _stage_c(image_np, reconstructed, removal_mask)

    logger.info(
        f"[restoration] done ({'generative' if use_generative else 'classical'})"
    )
    return final


# ---------------------------------------------------------------------------
# Foreground edge decontamination (BUG C4 — object cutout bleed)
# ---------------------------------------------------------------------------

def decontaminate_foreground_edges(
    object_rgba: np.ndarray,
    band_px: int = 3,
) -> np.ndarray:
    """
    Remove residual background-colour bleed from the edges of an object cutout.

    SAM2 masks are feathered: the alpha channel transitions from 255 → 0 over
    several pixels.  In that transition zone, the RGB values are a mix of
    object + background colours.  This creates a coloured fringe on the
    object's edges when composited over a different background.

    Fix: for every pixel in the semi-transparent edge band (0 < alpha < 200),
    estimate the "pure object colour" from nearby fully-opaque pixels and
    replace the RGB with that estimate.  The alpha channel is NOT changed.

    Args:
        object_rgba: (H, W, 4) uint8 RGBA cutout.
        band_px:     Width of the edge band to process.

    Returns:
        (H, W, 4) uint8 RGBA with decontaminated edges.
    """
    result = object_rgba.copy()
    alpha = object_rgba[:, :, 3].astype(np.float32)

    # Opaque core: alpha > 200
    opaque = (alpha > 200).astype(np.uint8)
    if not np.any(opaque):
        return result

    # Dilate the opaque core outward to get candidate reference pixels
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band_px * 2 + 1, band_px * 2 + 1))
    ref_zone = cv2.dilate(opaque, kern, iterations=1)

    for c in range(3):
        channel = object_rgba[:, :, c].astype(np.float32)
        # Build a "pure-object" approximation by Gaussian-blurring the opaque
        # core — this propagates interior colours outward into the edge band
        filled = channel.copy()
        filled[opaque == 0] = 0  # zero out non-opaque pixels before blurring
        weight = alpha.copy()
        weight[opaque == 0] = 0
        k = band_px * 4 + 1
        filled_blur  = cv2.GaussianBlur(filled,  (k, k), band_px * 2)
        weight_blur  = cv2.GaussianBlur(weight, (k, k), band_px * 2)
        pure_color = np.where(weight_blur > 1e-3, filled_blur / (weight_blur + 1e-6), channel)

        # Only apply in the semi-transparent edge band
        edge_band = (alpha > 5) & (alpha < 200)
        result[:, :, c] = np.where(edge_band, np.clip(pure_color, 0, 255), channel).astype(np.uint8)

    return result


# ---------------------------------------------------------------------------
# Utility: restore from saved mask files (on-demand endpoint)
# ---------------------------------------------------------------------------

def restore_background_from_mask_files(
    image_np: np.ndarray,
    mask_paths: List[str],
    object_labels: Optional[List[str]] = None,
    force_generative: bool = False,
    steps: int = 30,
    dilation_px: int = DEFAULT_DILATION_PX,
) -> np.ndarray:
    from pathlib import Path
    from utils import config

    refined_masks: List[np.ndarray] = []
    img_h, img_w = image_np.shape[:2]

    for mp in mask_paths:
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
        refined_masks.append(m)

    if not refined_masks:
        logger.warning("[restoration] no valid mask files — returning original")
        return image_np.copy()

    union_mask = build_removal_mask(refined_masks, dilation_px=0)

    return restore_background(
        image_np=image_np,
        union_mask=union_mask,
        refined_masks=refined_masks,
        object_labels=object_labels,
        force_generative=force_generative,
        steps=steps,
        dilation_px=dilation_px,
    )
