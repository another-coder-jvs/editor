"""
FLUX.1 Fill inpainting service — generative background reconstruction.

After objects are segmented and cutout from the original image, this service
uses FLUX.1 Fill to reconstruct the empty/masked regions with NEW generative
content. The original cutout pixels are NEVER used — only context-aware
generation fills the void.
"""
from __future__ import annotations

import gc
import logging
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from services.model_manager import model_manager, DEVICE

logger = logging.getLogger(__name__)


def reconstruct_background_with_flux(
    original_image: Image.Image,
    combined_mask: np.ndarray,
    prompt: str = "clean background, natural scene continuation, empty area",
    num_inference_steps: int = 20,
    guidance_scale: float = 3.5,
    max_image_size: int = 1024,
    seed: int | None = None,
) -> Image.Image:
    """
    Use FLUX.1 Fill to generatively reconstruct masked regions of an image.

    The masked areas (white in combined_mask) are filled with NEW generative
    content — NOT the original pixels from the cutout objects. This ensures
    the background is contextually appropriate without reproducing the
    removed objects.

    Args:
        original_image: The full original RGB image.
        combined_mask: Binary mask where white = area to fill generatively
                       (the union of all object masks).
        prompt: Text prompt guiding the generative fill.
        num_inference_steps: Number of diffusion steps (20 is good quality/speed balance).
        guidance_scale: Classifier-free guidance scale.
        max_image_size: Max dimension for FLUX processing (controls VRAM usage).
        seed: Optional random seed for reproducibility.

    Returns:
        Reconstructed full image with generatively filled regions.
    """
    logger.info(
        f"[flux_inpaint] reconstructing background with FLUX.1 Fill "
        f"(image={original_image.size}, mask_white_pixels={int(np.sum(combined_mask > 0))})"
    )

    # ── Remote mode: use cloud inpainting API ────────────────────────────────
    from services.remote_config import remote_cfg
    if remote_cfg.COMPUTE_MODE == "remote":
        logger.info("[flux_inpaint] COMPUTE_MODE=remote — using remote inpainting API")
        from services.remote_services.inpaint import inpaint_remote
        from PIL import Image as _Image
        mask_bin_remote = (combined_mask > 128).astype(np.uint8) * 255
        if not np.any(mask_bin_remote):
            return original_image.copy()
        mask_alpha = _Image.fromarray(mask_bin_remote, mode="L")
        layer_rgba = original_image.convert("RGBA")
        layer_rgba.putalpha(mask_alpha)
        result = inpaint_remote(layer_rgba, prompt, strength=0.99,
                                guidance_scale=guidance_scale, steps=num_inference_steps)
        return result.convert("RGB").resize(original_image.size, _Image.LANCZOS)

    # ── Local mode (unchanged below) ─────────────────────────────────────────

    # Ensure mask is uint8 binary
    mask_bin = (combined_mask > 128).astype(np.uint8) * 255

    # Skip if mask is empty
    if not np.any(mask_bin):
        logger.info("[flux_inpaint] mask is empty, returning original")
        return original_image.copy()

    # Convert image to RGB if needed
    img_rgb = original_image.convert("RGB")

    # Resize if needed for VRAM safety
    orig_w, orig_h = img_rgb.size
    scale = 1.0
    if max(orig_w, orig_h) > max_image_size:
        scale = max_image_size / max(orig_w, orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        img_rgb = img_rgb.resize((new_w, new_h), Image.LANCZOS)
        mask_pil = Image.fromarray(mask_bin).resize((new_w, new_h), Image.NEAREST)
        logger.info(f"[flux_inpaint] resized to {new_w}x{new_h} for processing")
    else:
        mask_pil = Image.fromarray(mask_bin)

    # Ensure dimensions are divisible by 16 (FLUX requirement)
    w, h = img_rgb.size
    w16 = (w // 16) * 16
    h16 = (h // 16) * 16
    if w16 != w or h16 != h:
        img_rgb = img_rgb.resize((w16, h16), Image.LANCZOS)
        mask_pil = mask_pil.resize((w16, h16), Image.NEAREST)

    # Run FLUX.1 Fill
    pipe = model_manager.get_flux_fill_pipe()

    import torch
    generator = torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None

    logger.info(
        f"[flux_inpaint] running FLUX.1 Fill: steps={num_inference_steps}, "
        f"guidance={guidance_scale}, prompt='{prompt}'"
    )
    result = pipe(
        prompt=prompt,
        image=img_rgb,
        mask_image=mask_pil,
        height=h16,
        width=w16,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

    # Resize back to original dimensions if we scaled down
    if scale < 1.0:
        result = result.resize((orig_w, orig_h), Image.LANCZOS)

    logger.info(f"[flux_inpaint] FLUX.1 Fill complete → {result.size}")

    # Free FLUX model immediately to reclaim VRAM
    model_manager.unload_flux_fill_pipe()

    return result


def create_reasoned_fill_prompt(
    image: Image.Image,
    objects: list[dict],
    background_analysis: dict | None = None,
) -> str:
    """
    Build a FLUX Fill prompt that is more likely to recreate a *plausible*
    background when the removed objects leave large semantically empty regions.

    This is intentionally used by segmentation_service when the canvas still
    contains big empty rectangles / white blocks after the obvious object layers
    have been removed. The goal is to avoid leaving the original placeholder
    pixels in the background layer.
    """
    base = "clean background, natural scene continuation, empty area"

    # If known background analysis says it was a solid colour, bias prompt toward that
    if background_analysis:
        bg_type = str(background_analysis.get("bg_type", "") if isinstance(background_analysis.get("bg_type"), str) else getattr(background_analysis.get("bg_type"), "value", str(background_analysis.get("bg_type"))))
        dominant = background_analysis.get("dominant_color")
        if bg_type == "nearly_solid" and dominant:
            r, g, b = (int(dominant[0]), int(dominant[1]), int(dominant[2]))
            base = (
                f"clean background, natural scene continuation, empty area, "
                f"dominant dark tone near rgb({r},{g},{b})"
            )

    # If there were very few real objects, the image may be mostly graphic design;
    # encourage a plain neutral background instead of inventing scene details.
    if len(objects) <= 2:
        base = "clean background, neutral texture, empty area, no text, no icons"

    return base


def create_object_removal_mask(
    image_shape: tuple[int, int],
    object_masks: list[np.ndarray],
    dilate_pixels: int = 10,
) -> np.ndarray:
    """
    Combine all object masks into a single inpainting mask, dilated to
    cover feathered edges and ensure no object remnants remain.

    Args:
        image_shape: (height, width) of the full image.
        object_masks: List of binary masks (one per detected object).
        dilate_pixels: How many pixels to expand each mask for clean edges.

    Returns:
        Combined binary mask (uint8, 255 = region to fill).
    """
    combined = np.zeros(image_shape[:2], dtype=np.uint8)

    for mask in object_masks:
        binary = (mask > 128).astype(np.uint8) * 255
        if dilate_pixels > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (dilate_pixels * 2 + 1, dilate_pixels * 2 + 1)
            )
            binary = cv2.dilate(binary, kernel, iterations=1)
        combined = np.maximum(combined, binary)

    logger.debug(
        f"[flux_inpaint] combined mask: {len(object_masks)} objects, "
        f"{int(np.sum(combined > 0))} white pixels"
    )
    return combined
