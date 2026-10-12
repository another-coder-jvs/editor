"""
Intelligent background restoration after object cutout.

Pipeline overview
─────────────────
Given the original image and a binary union mask of all removed objects, this
service reconstructs what the background would look like behind those objects.

Stage A — Classical inpainting (always runs first)
    cv2.inpaint TELEA + NS gives a fast, deterministic fill.
    Good enough for small holes and uniform / near-solid backgrounds.

Stage B — Generative inpainting (runs when Stage A is insufficient)
    Threshold: hole covers > GENERATIVE_THRESHOLD fraction of the canvas,
    OR the background has high edge density (complex scene).
    Primary:   SDXL Inpainting (already in model_manager / remote mode).
    Fallback:  FLUX.1 Fill if SDXL fails.
    Guard:     if the model returns pixels that are too similar to the original
               object pixels inside the mask, we fall back to Stage A output.

Stage C — Seam & colour correction (always runs after A or B)
    Gaussian-feathered compositing at the mask boundary eliminates hard seams.
    Optional Laplacian-pyramid or Poisson blending for large regions.

Critical invariant
──────────────────
Pixels INSIDE the removal mask in the output must come from the reconstructed
background, NEVER from the removed object.  The original image is used only as
contextual input to the generative model; the resulting pixels are never pasted
back verbatim into the hole.

Public API
──────────
    restore_background(
        image_np,           # (H, W, 3) uint8 RGB — the original full image
        union_mask,         # (H, W) uint8  0=background 255=objects to remove
        object_labels,      # list[str] — names of removed objects (for prompt)
        force_generative,   # bool — skip complexity check, always use Stage B
        steps,              # int  — inference steps for generative model
        dilation_px,        # int  — mask dilation radius in pixels
    ) -> np.ndarray          # (H, W, 3) uint8 RGB — fully reconstructed image

    build_removal_mask(
        refined_masks,      # list of (H, W) uint8 masks from SAM2
        dilation_px,        # int
    ) -> np.ndarray          # (H, W) uint8 combined + dilated mask
"""
from __future__ import annotations

import gc
import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

# Fraction of canvas area that the removal mask must exceed to trigger Stage B.
# 0.05 = holes larger than 5 % of the canvas get generative treatment.
GENERATIVE_THRESHOLD: float = 0.05

# Edge density (fraction of edge pixels in background border) above which we
# treat the scene as "complex" and use generative inpainting regardless of size.
COMPLEX_SCENE_EDGE_THRESHOLD: float = 0.08

# Dilation radius applied to the removal mask to cover feathered SAM2 edges.
DEFAULT_DILATION_PX: int = 12

# Width of the feather band used for Stage C seam blending.
SEAM_FEATHER_PX: int = 16

# Maximum image dimension before downscaling for generative inference.
MAX_INPAINT_DIM: int = 1024

# If the fraction of mask pixels whose L2 distance to original < this threshold
# exceeds SAME_PIXEL_FRAC, we consider the model returned the original pixels.
SAME_PIXEL_DIST: float = 15.0
SAME_PIXEL_FRAC: float = 0.60


# ---------------------------------------------------------------------------
# Mask construction
# ---------------------------------------------------------------------------

def build_removal_mask(
    refined_masks: List[np.ndarray],
    dilation_px: int = DEFAULT_DILATION_PX,
) -> np.ndarray:
    """
    Combine all per-object SAM2 masks into one binary removal mask.

    The combined mask is dilated by `dilation_px` to ensure feathered / anti-
    aliased object edges do not leave residual object pixels in the background.

    Returns a (H, W) uint8 mask: 255 = region to reconstruct, 0 = keep as-is.
    """
    if not refined_masks:
        shape = refined_masks[0].shape[:2] if refined_masks else (1, 1)
        return np.zeros(shape, dtype=np.uint8)

    combined = np.zeros(refined_masks[0].shape[:2], dtype=np.uint8)
    for m in refined_masks:
        binary = (m > 128).astype(np.uint8) * 255
        combined = np.maximum(combined, binary)

    if dilation_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (dilation_px * 2 + 1, dilation_px * 2 + 1),
        )
        combined = cv2.dilate(combined, kernel, iterations=1)

    logger.debug(
        f"[restoration] removal mask: {int(np.sum(combined > 0))} pixels "
        f"({100 * np.mean(combined > 0):.1f}% of canvas)"
    )
    return combined


# ---------------------------------------------------------------------------
# Complexity analysis
# ---------------------------------------------------------------------------

def _needs_generative(
    image_np: np.ndarray,
    removal_mask: np.ndarray,
    force: bool,
) -> bool:
    """
    Return True if the hole is complex enough to warrant generative inpainting.

    Criteria (any one triggers Stage B):
      • force=True
      • hole area > GENERATIVE_THRESHOLD fraction of canvas
      • background region has high edge density (complex scene)
    """
    if force:
        return True

    h, w = image_np.shape[:2]
    canvas_area = h * w
    hole_area = int(np.sum(removal_mask > 0))
    hole_fraction = hole_area / max(canvas_area, 1)

    if hole_fraction > GENERATIVE_THRESHOLD:
        logger.info(
            f"[restoration] hole={hole_fraction:.1%} > {GENERATIVE_THRESHOLD:.0%} → Stage B"
        )
        return True

    # Measure edge density in the background (non-masked) region
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    bg_region = (removal_mask == 0)
    bg_edge_pixels = int(np.sum(edges[bg_region] > 0))
    bg_total = int(np.sum(bg_region))
    edge_density = bg_edge_pixels / max(bg_total, 1)

    if edge_density > COMPLEX_SCENE_EDGE_THRESHOLD:
        logger.info(
            f"[restoration] background edge density={edge_density:.3f} > "
            f"{COMPLEX_SCENE_EDGE_THRESHOLD} → Stage B"
        )
        return True

    logger.info(
        f"[restoration] hole={hole_fraction:.1%}, edge_density={edge_density:.3f} "
        f"→ Stage A sufficient"
    )
    return False


# ---------------------------------------------------------------------------
# Stage A — Classical inpainting
# ---------------------------------------------------------------------------

def _stage_a(image_np: np.ndarray, removal_mask: np.ndarray) -> np.ndarray:
    """
    Fill the removal mask using cv2 TELEA + NS inpainting.

    Returns (H, W, 3) uint8 RGB with the hole filled.
    """
    logger.info("[restoration/A] running cv2 TELEA + NS inpainting…")
    bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # TELEA: fast propagation — good for initial fill
    result = cv2.inpaint(bgr, removal_mask, inpaintRadius=12, flags=cv2.INPAINT_TELEA)
    # NS: Navier-Stokes — smooths texture transitions
    result = cv2.inpaint(result, removal_mask, inpaintRadius=6, flags=cv2.INPAINT_NS)

    rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    logger.info("[restoration/A] cv2 inpainting complete")
    return rgb


# ---------------------------------------------------------------------------
# Stage B — Generative inpainting
# ---------------------------------------------------------------------------

def _build_inpaint_prompt(
    image_np: np.ndarray,
    object_labels: List[str],
    removal_mask: np.ndarray,
) -> str:
    """
    Construct a prompt that instructs the model to reconstruct the background
    behind the removed objects rather than reproduce them.
    """
    h, w = image_np.shape[:2]

    # Analyse background colours from unmasked border region
    border = min(30, h // 8, w // 8)
    bg_region = (removal_mask == 0)
    bg_pixels = image_np[bg_region]
    if len(bg_pixels) > 0:
        dominant = np.median(bg_pixels, axis=0).astype(int)
        r, g, b = int(dominant[0]), int(dominant[1]), int(dominant[2])
        colour_hint = f"background dominant colour rgb({r},{g},{b})"
    else:
        colour_hint = "natural background"

    objects_str = ", ".join(object_labels) if object_labels else "objects"

    prompt = (
        f"Remove the {objects_str} completely and reconstruct only the background "
        f"that would naturally exist behind them. Continue the surrounding structure, "
        f"texture, perspective, and lighting. {colour_hint}. "
        f"Do not reproduce any part of the removed subjects. "
        f"Preserve all unmasked content exactly as provided. "
        f"Clean, seamless background reconstruction."
    )
    logger.debug(f"[restoration/B] inpaint prompt: '{prompt[:120]}…'")
    return prompt


def _guard_original_pixels(
    original_np: np.ndarray,
    result_np: np.ndarray,
    removal_mask: np.ndarray,
) -> bool:
    """
    Return True if the generative result is too similar to the original inside
    the mask (i.e., the model just copied the original object pixels back).

    Compares per-pixel L2 distance in the masked region.
    """
    mask_bool = removal_mask > 0
    if not np.any(mask_bool):
        return False

    orig_region = original_np[mask_bool].astype(np.float32)
    result_region = result_np[mask_bool].astype(np.float32)
    dists = np.sqrt(np.sum((orig_region - result_region) ** 2, axis=1))
    same_frac = float(np.mean(dists < SAME_PIXEL_DIST))

    if same_frac > SAME_PIXEL_FRAC:
        logger.warning(
            f"[restoration/B] {same_frac:.0%} of masked pixels are too similar to original "
            f"(threshold {SAME_PIXEL_FRAC:.0%}) — model returned original pixels, rejecting result"
        )
        return True
    return False


def _resize_for_inpaint(
    image_np: np.ndarray,
    mask: np.ndarray,
    max_dim: int = MAX_INPAINT_DIM,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Downscale image+mask if needed; return (resized_image, resized_mask, scale)."""
    h, w = image_np.shape[:2]
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        image_np = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    # Ensure dimensions divisible by 8 (SDXL requirement)
    h2, w2 = image_np.shape[:2]
    nw = (w2 // 8) * 8
    nh = (h2 // 8) * 8
    if nw != w2 or nh != h2:
        image_np = cv2.resize(image_np, (nw, nh), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    return image_np, mask, scale


def _stage_b_local(
    image_np: np.ndarray,
    removal_mask: np.ndarray,
    prompt: str,
    steps: int,
    stage_a_result: np.ndarray,
) -> np.ndarray:
    """
    Local generative inpainting using SDXL Inpaint (primary) or FLUX Fill (fallback).
    Returns (H, W, 3) uint8 RGB.
    """
    from services.model_manager import model_manager, DEVICE
    import torch

    orig_h, orig_w = image_np.shape[:2]
    img_small, mask_small, scale = _resize_for_inpaint(image_np, removal_mask)
    h2, w2 = img_small.shape[:2]

    image_pil = Image.fromarray(img_small)
    # SDXL inpaint mask: white = region to fill, black = keep
    mask_pil = Image.fromarray(mask_small).convert("RGB")

    logger.info(f"[restoration/B] SDXL inpaint {w2}×{h2} steps={steps}")

    try:
        pipe = model_manager.get_inpaint_pipe()
        generator = torch.Generator(device="cpu").manual_seed(42)
        out = pipe(
            prompt=prompt,
            image=image_pil,
            mask_image=mask_pil,
            height=h2,
            width=w2,
            num_inference_steps=steps,
            guidance_scale=7.5,
            strength=0.99,
            generator=generator,
        ).images[0]
        result_np = np.array(out.convert("RGB"))
        if scale < 1.0:
            result_np = cv2.resize(result_np, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)

        # Guard: reject if model returned original pixels
        if _guard_original_pixels(image_np, result_np, removal_mask):
            logger.warning("[restoration/B] SDXL returned original pixels → trying FLUX Fill")
            raise RuntimeError("SDXL returned original pixels")

        logger.info("[restoration/B] SDXL inpaint accepted")
        model_manager.unload_inpaint_pipe()
        return result_np

    except Exception as sdxl_err:
        logger.warning(f"[restoration/B] SDXL failed ({sdxl_err}), trying FLUX.1 Fill…")

    # Fallback: FLUX.1 Fill
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
            logger.warning("[restoration/B] FLUX returned original pixels → falling back to Stage A")
            return stage_a_result
        logger.info("[restoration/B] FLUX.1 Fill accepted")
        return result_np
    except Exception as flux_err:
        logger.warning(f"[restoration/B] FLUX failed ({flux_err}), using Stage A result")
        return stage_a_result


def _stage_b_remote(
    image_np: np.ndarray,
    removal_mask: np.ndarray,
    prompt: str,
    steps: int,
    stage_a_result: np.ndarray,
) -> np.ndarray:
    """Remote generative inpainting via configured cloud provider."""
    from services.remote_services.inpaint import inpaint_remote

    orig_h, orig_w = image_np.shape[:2]
    img_small, mask_small, scale = _resize_for_inpaint(image_np, removal_mask)

    # Build RGBA image: alpha = mask (white = fill this area)
    rgba = Image.fromarray(img_small).convert("RGBA")
    rgba.putalpha(Image.fromarray(mask_small))

    try:
        result_rgba = inpaint_remote(rgba, prompt, strength=0.99, guidance_scale=7.5, steps=steps)
        result_np = np.array(result_rgba.convert("RGB"))
        if scale < 1.0:
            result_np = cv2.resize(result_np, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
        if _guard_original_pixels(image_np, result_np, removal_mask):
            logger.warning("[restoration/B/remote] model returned original pixels → Stage A fallback")
            return stage_a_result
        logger.info("[restoration/B/remote] remote inpaint accepted")
        return result_np
    except Exception as e:
        logger.warning(f"[restoration/B/remote] remote inpaint failed ({e}) → Stage A fallback")
        return stage_a_result


def _stage_b(
    image_np: np.ndarray,
    removal_mask: np.ndarray,
    object_labels: List[str],
    steps: int,
    stage_a_result: np.ndarray,
) -> np.ndarray:
    """Dispatch Stage B to local or remote depending on COMPUTE_MODE."""
    from services.remote_config import remote_cfg
    prompt = _build_inpaint_prompt(image_np, object_labels, removal_mask)

    if remote_cfg.COMPUTE_MODE == "remote":
        return _stage_b_remote(image_np, removal_mask, prompt, steps, stage_a_result)
    return _stage_b_local(image_np, removal_mask, prompt, steps, stage_a_result)


# ---------------------------------------------------------------------------
# Stage C — Seam blending
# ---------------------------------------------------------------------------

def _build_feather_weight(
    removal_mask: np.ndarray,
    feather_px: int = SEAM_FEATHER_PX,
) -> np.ndarray:
    """
    Build a float32 weight map [0..1] that is:
      - 1.0 deep inside the mask (use reconstructed pixels)
      - 0.0 deep outside the mask (use original pixels)
      - smooth gradient in the transition band
    """
    binary = (removal_mask > 0).astype(np.uint8)

    # Distance from the mask interior edge
    dist_inside = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    # Distance from the mask exterior edge
    dist_outside = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 5)

    # Weight = dist_inside / (dist_inside + dist_outside)
    total = dist_inside + dist_outside + 1e-6
    weight = dist_inside / total

    # Clamp: pixels more than feather_px inside get weight=1
    weight = np.clip(weight * (feather_px / max(1, np.max(dist_inside))), 0.0, 1.0)
    # Reclamp with Gaussian smoothing on the boundary
    weight_smooth = cv2.GaussianBlur(weight, (feather_px | 1, feather_px | 1), feather_px / 3)

    # Hard-clamp: outside stays 0, deep inside stays 1
    weight_smooth[dist_inside == 0] = 0.0
    weight_smooth[dist_inside > feather_px] = 1.0

    return weight_smooth.astype(np.float32)


def _stage_c(
    original_np: np.ndarray,
    reconstructed_np: np.ndarray,
    removal_mask: np.ndarray,
) -> np.ndarray:
    """
    Blend reconstructed pixels into the original image using feathered masking.

    Deep inside the removal mask → use reconstructed pixels.
    Deep outside the removal mask → use original pixels.
    Transition zone → smooth gradient blend.

    This eliminates hard seams and colour discontinuities at object boundaries.
    """
    logger.info("[restoration/C] applying feathered seam blending…")

    weight = _build_feather_weight(removal_mask)
    w3 = weight[:, :, np.newaxis]  # broadcast over RGB channels

    orig_f   = original_np.astype(np.float32)
    recon_f  = reconstructed_np.astype(np.float32)

    blended = orig_f * (1.0 - w3) + recon_f * w3
    result   = np.clip(blended, 0, 255).astype(np.uint8)

    logger.info("[restoration/C] seam blending complete")
    return result


# ---------------------------------------------------------------------------
# Main public entry point
# ---------------------------------------------------------------------------

def restore_background(
    image_np: np.ndarray,
    union_mask: np.ndarray,
    object_labels: Optional[List[str]] = None,
    force_generative: bool = False,
    steps: int = 20,
    dilation_px: int = DEFAULT_DILATION_PX,
) -> np.ndarray:
    """
    Reconstruct the background behind all removed objects.

    Args:
        image_np:         Original full image (H, W, 3) uint8 RGB.
        union_mask:       Combined mask of all removed objects (H, W) uint8.
                          255 = pixels that belong to objects and must be filled.
                          0   = background pixels that must not be changed.
        object_labels:    Names of the removed objects (used to build the prompt).
        force_generative: Skip the complexity check and always run Stage B.
        steps:            Diffusion steps for Stage B (default 20).
        dilation_px:      Extra dilation applied to union_mask to ensure
                          feathered edges are covered.  Pass 0 if the mask is
                          already dilated upstream.

    Returns:
        (H, W, 3) uint8 RGB — the fully restored image.  Pixels INSIDE
        union_mask come from Stage A/B reconstruction, never from the original
        removed object.  Pixels OUTSIDE union_mask are identical to image_np.
    """
    if object_labels is None:
        object_labels = []

    orig_h, orig_w = image_np.shape[:2]
    logger.info(
        f"[restoration] restore_background: {orig_w}×{orig_h} "
        f"objects={object_labels} force_gen={force_generative} steps={steps}"
    )

    # ── 0. Validate / dilate mask ──────────────────────────────────────────
    if union_mask.shape[:2] != (orig_h, orig_w):
        union_mask = cv2.resize(union_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    if dilation_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (dilation_px * 2 + 1, dilation_px * 2 + 1),
        )
        removal_mask = cv2.dilate(
            (union_mask > 0).astype(np.uint8) * 255, kernel, iterations=1
        )
    else:
        removal_mask = (union_mask > 0).astype(np.uint8) * 255

    if not np.any(removal_mask > 0):
        logger.info("[restoration] mask is empty — returning original image unchanged")
        return image_np.copy()

    # ── Stage A — Classical inpainting ────────────────────────────────────
    stage_a_result = _stage_a(image_np, removal_mask)

    # ── Stage B — Generative inpainting (conditional) ─────────────────────
    use_generative = _needs_generative(image_np, removal_mask, force_generative)

    if use_generative:
        reconstructed = _stage_b(
            image_np, removal_mask, object_labels, steps, stage_a_result
        )
    else:
        reconstructed = stage_a_result
        logger.info("[restoration] using Stage A result (generative not needed)")

    # ── Stage C — Seam blending ───────────────────────────────────────────
    final = _stage_c(image_np, reconstructed, removal_mask)

    # ── Final guard: unmasked pixels must be unchanged ─────────────────────
    # Hard-overwrite background pixels to guarantee no drift outside mask
    bg_mask = (removal_mask == 0)
    final[bg_mask] = image_np[bg_mask]

    hole_px = int(np.sum(removal_mask > 0))
    logger.info(
        f"[restoration] complete: reconstructed {hole_px} pixels "
        f"({'generative' if use_generative else 'classical'})"
    )
    return final


# ---------------------------------------------------------------------------
# Utility: restore from saved mask files (used by on-demand endpoint)
# ---------------------------------------------------------------------------

def restore_background_from_mask_files(
    image_np: np.ndarray,
    mask_paths: List[str],
    object_labels: Optional[List[str]] = None,
    force_generative: bool = False,
    steps: int = 30,
    dilation_px: int = DEFAULT_DILATION_PX,
) -> np.ndarray:
    """
    Load per-object mask PNGs from disk, combine them, and run restoration.
    Used by the /restore-background on-demand endpoint.
    """
    from pathlib import Path
    from utils import config

    refined_masks: List[np.ndarray] = []
    img_h, img_w = image_np.shape[:2]

    for mp in mask_paths:
        # Resolve path the same way segmentation_service does
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
        logger.warning("[restoration] no valid mask files — returning original image")
        return image_np.copy()

    union_mask = build_removal_mask(refined_masks, dilation_px=0)  # already combined

    return restore_background(
        image_np=image_np,
        union_mask=union_mask,
        object_labels=object_labels,
        force_generative=force_generative,
        steps=steps,
        dilation_px=dilation_px,
    )
