"""
Background detection service – analyzes image to determine background type
and creates appropriate masks for background separation.

Pipeline:
1. Analyze image for background type (solid color, gradient, complex)
2. For solid/color backgrounds: create color-based mask
3. For complex backgrounds: use segmentation-based approach
4. Return background analysis results with mask data
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class BackgroundType(str, Enum):
    SOLID_COLOR = "solid_color"
    NEARLY_SOLID = "nearly_solid"  # Very uniform with minor variations
    GRADIENT = "gradient"
    COMPLEX = "complex"
    TEXTURED = "textured"


@dataclass
class BackgroundAnalysis:
    """Result of background analysis."""
    bg_type: BackgroundType
    dominant_color: Tuple[int, int, int]  # RGB
    color_variance: float  # Lower = more uniform
    gradient_direction: Optional[str] = None  # "horizontal", "vertical", "radial", None
    confidence: float = 0.0  # 0-1, how confident we are in this classification
    edge_density: float = 0.0  # How many edges in background regions
    color_mask_path: Optional[str] = None  # Path to generated color mask if applicable


def analyze_background(image: np.ndarray) -> BackgroundAnalysis:
    """
    Analyze image to determine the type of background.
    
    Strategy:
    1. Sample colors from image edges/borders (likely background)
    2. Check color uniformity (variance)
    3. Detect gradient patterns
    4. Classify as solid/nearly-solid/gradient/complex
    """
    h, w = image.shape[:2]
    logger.info(f"[bg_detect] Analyzing image {w}x{h}")
    
    # Sample border regions (top, bottom, left, right edges)
    border_width = min(20, h // 10, w // 10)
    border_regions = []
    
    # Top edge
    border_regions.append(image[:border_width, :, :])
    # Bottom edge
    border_regions.append(image[h - border_width:, :, :])
    # Left edge
    border_regions.append(image[:, :border_width, :])
    # Right edge
    border_regions.append(image[:, w - border_width:, :])
    
    # Combine all border pixels
    border_pixels = np.vstack([region.reshape(-1, 3) for region in border_regions])
    
    # Also sample corners (very likely background)
    corner_size = min(50, h // 4, w // 4)
    corners = [
        image[:corner_size, :corner_size, :],  # top-left
        image[:corner_size, w - corner_size:, :],  # top-right
        image[h - corner_size:, :corner_size, :],  # bottom-left
        image[h - corner_size:, w - corner_size:, :],  # bottom-right
    ]
    corner_pixels = np.vstack([c.reshape(-1, 3) for c in corners])
    
    # Combine border and corner samples for analysis
    # Weight corners more heavily as they're more likely to be pure background
    all_pixels = np.vstack([border_pixels, corner_pixels, corner_pixels, corner_pixels])
    
    # Calculate dominant color (median is more robust than mean)
    dominant_color = tuple(np.median(all_pixels, axis=0).astype(int).tolist())
    
    # Calculate color variance
    color_std = np.std(all_pixels.astype(float), axis=0)
    color_variance = float(np.mean(color_std))
    
    logger.info(f"[bg_detect] Dominant color: {dominant_color}, Variance: {color_variance:.2f}")
    
    # Check for gradient patterns
    gradient_info = _detect_gradient(image, dominant_color)
    logger.info(f"[bg_detect] Gradient info: {gradient_info}")
    
    # Calculate edge density in border regions
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    border_edge_density = _calculate_border_edge_density(edges, border_width)
    logger.info(f"[bg_detect] Border edge density: {border_edge_density:.4f}")
    
    # Classify background type
    bg_type, confidence = _classify_background(
        color_variance, gradient_info, border_edge_density
    )
    
    logger.info(f"[bg_detect] Classification: {bg_type.value} (confidence: {confidence:.2f})")
    
    return BackgroundAnalysis(
        bg_type=bg_type,
        dominant_color=dominant_color,
        color_variance=color_variance,
        gradient_direction=gradient_info.get("direction"),
        confidence=confidence,
        edge_density=border_edge_density,
    )


def _detect_gradient(
    image: np.ndarray, 
    dominant_color: Tuple[int, int, int]
) -> Dict[str, any]:
    """Detect if image has a gradient background pattern."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(float)
    
    # Sample horizontal and vertical gradients
    # Check if intensity changes smoothly across the image
    
    # Horizontal gradient check: average columns
    col_means = np.mean(gray, axis=0)
    # Smooth the signal using numpy convolution
    kernel_size = max(3, w // 10)
    kernel = np.ones(kernel_size) / kernel_size
    col_smooth = np.convolve(col_means, kernel, mode='same')
    
    # Check for monotonic change (gradient characteristic)
    col_diff = np.diff(col_smooth)
    col_monotonicity = abs(np.sum(np.sign(col_diff))) / len(col_diff) if len(col_diff) > 0 else 0
    
    # Vertical gradient check: average rows
    row_means = np.mean(gray, axis=1)
    kernel_size = max(3, h // 10)
    kernel = np.ones(kernel_size) / kernel_size
    row_smooth = np.convolve(row_means, kernel, mode='same')
    row_diff = np.diff(row_smooth)
    row_monotonicity = abs(np.sum(np.sign(row_diff))) / len(row_diff) if len(row_diff) > 0 else 0
    
    # Radial gradient check: compare corners to center
    center_region = gray[h//3:2*h//3, w//3:2*w//3]
    corner_regions = [
        gray[:h//4, :w//4],
        gray[:h//4, 3*w//4:],
        gray[3*h//4:, :w//4],
        gray[3*h//4:, 3*w//4:],
    ]
    center_mean = np.mean(center_region)
    corner_mean = np.mean([np.mean(r) for r in corner_regions])
    radial_diff = abs(center_mean - corner_mean)
    
    logger.debug(
        f"[bg_detect] Gradient: h_mono={col_monotonicity:.3f} "
        f"v_mono={row_monotonicity:.3f} radial_diff={radial_diff:.1f}"
    )
    
    result = {"direction": None, "strength": 0.0}
    
    # Thresholds for gradient detection
    MONOTONICITY_THRESHOLD = 0.6
    RADIAL_THRESHOLD = 15
    
    if col_monotonicity > MONOTONICITY_THRESHOLD:
        result["direction"] = "horizontal"
        result["strength"] = col_monotonicity
    elif row_monotonicity > MONOTONICITY_THRESHOLD:
        result["direction"] = "vertical"
        result["strength"] = row_monotonicity
    elif radial_diff > RADIAL_THRESHOLD:
        result["direction"] = "radial"
        result["strength"] = min(1.0, radial_diff / 50)
    
    return result


def _calculate_border_edge_density(edges: np.ndarray, border_width: int) -> float:
    """Calculate edge density in the border regions of the image."""
    h, w = edges.shape[:2]
    
    # Extract border regions
    top = edges[:border_width, :]
    bottom = edges[h - border_width:, :]
    left = edges[:, :border_width]
    right = edges[:, w - border_width:]
    
    # Combine border regions
    border_pixels = np.concatenate([
        top.flatten(),
        bottom.flatten(),
        left.flatten(),
        right.flatten(),
    ])
    
    # Edge density = percentage of edge pixels
    return float(np.mean(border_pixels > 0))


def _classify_background(
    color_variance: float,
    gradient_info: Dict,
    edge_density: float,
) -> Tuple[BackgroundType, float]:
    """Classify background based on analysis metrics."""
    
    # Thresholds (tuned for typical images)
    SOLID_VARIANCE_THRESHOLD = 15
    NEARLY_SOLID_VARIANCE_THRESHOLD = 40
    EDGE_THRESHOLD = 0.05
    
    # Complex background: high variance + many edges
    if color_variance > NEARLY_SOLID_VARIANCE_THRESHOLD and edge_density > EDGE_THRESHOLD:
        return BackgroundType.COMPLEX, 0.7 + min(0.3, (color_variance - NEARLY_SOLID_VARIANCE_THRESHOLD) / 100)
    
    # Textured: moderate variance + high edges (but not complex enough)
    if edge_density > 0.1 and color_variance > 30:
        return BackgroundType.TEXTURED, 0.6
    
    # Gradient: smooth color variation with directional pattern
    if gradient_info.get("direction") and gradient_info.get("strength", 0) > 0.5:
        if color_variance < NEARLY_SOLID_VARIANCE_THRESHOLD:
            return BackgroundType.GRADIENT, 0.7 + gradient_info["strength"] * 0.2
    
    # Solid color: very low variance
    if color_variance < SOLID_VARIANCE_THRESHOLD:
        return BackgroundType.SOLID_COLOR, 0.9 - color_variance / 100
    
    # Nearly solid: low variance
    if color_variance < NEARLY_SOLID_VARIANCE_THRESHOLD:
        return BackgroundType.NEARLY_SOLID, 0.7 - (color_variance - SOLID_VARIANCE_THRESHOLD) / 100
    
    # Default: complex
    return BackgroundType.COMPLEX, 0.5


def create_background_mask(
    image: np.ndarray,
    analysis: BackgroundAnalysis,
    tolerance: int = 30,
    gradient_margin: int = 40,
) -> np.ndarray:
    """
    Create a background mask based on the analysis.
    
    For SOLID_COLOR/NEARLY_SOLID: flood-fill + color distance mask
    For GRADIENT: gradient-aware mask
    For COMPLEX/TEXTURED: return None (use SAM2 or other method)
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    if analysis.bg_type in (BackgroundType.SOLID_COLOR, BackgroundType.NEARLY_SOLID):
        logger.info(f"[bg_detect] Creating color-based mask for {analysis.bg_type.value}")
        mask = _create_color_mask(image, analysis.dominant_color, tolerance)
        
    elif analysis.bg_type == BackgroundType.GRADIENT:
        logger.info(f"[bg_detect] Creating gradient-aware mask")
        mask = _create_gradient_mask(image, analysis.dominant_color, gradient_margin)
        
    else:
        # Complex/Textured - can't easily create a simple mask
        logger.info(f"[bg_detect] Complex background - no simple mask created")
        return None
    
    # Refine mask
    mask = _refine_bg_mask(mask)
    
    return mask


def _create_color_mask(
    image: np.ndarray,
    dominant_color: Tuple[int, int, int],
    tolerance: int,
) -> np.ndarray:
    """Create mask by flood-filling from corners with color matching."""
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Convert dominant color to numpy array
    target = np.array(dominant_color, dtype=float)
    
    # Color distance for each pixel
    diff = image.astype(float) - target
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    
    # Binary mask: pixels close to dominant color
    color_mask = (dist < tolerance * 3).astype(np.uint8) * 255
    
    # Flood fill from corners to get connected background region
    corners = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
    filled = np.zeros((h, w), dtype=np.uint8)
    
    for seed in corners:
        if color_mask[seed[1], seed[0]] > 0:
            flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
            cv2.floodFill(
                color_mask.copy(), 
                flood_mask, 
                seed, 
                255,
                loDiff=tolerance,
                upDiff=tolerance,
            )
            filled = np.maximum(filled, flood_mask[1:h+1, 1:w+1])
    
    return filled


def _create_gradient_mask(
    image: np.ndarray,
    dominant_color: Tuple[int, int, int],
    margin: int,
) -> np.ndarray:
    """Create mask for gradient backgrounds by detecting smooth color regions."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Use adaptive threshold to separate regions
    # But gradient backgrounds are smooth, so we look for lack of edges
    edges = cv2.Canny(gray, 30, 100)
    
    # Dilate edges to create boundary
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Flood fill from center of each edge
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Start from corners (background)
    seeds = [
        (w // 2, margin),  # top center
        (w // 2, h - margin),  # bottom center
        (margin, h // 2),  # left center
        (w - margin, h // 2),  # right center
    ]
    
    for seed in seeds:
        if dilated[seed[1], seed[0]] == 0:  # Only if not on an edge
            flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
            temp = dilated.copy()
            cv2.floodFill(temp, flood_mask, seed, 255)
            mask = np.maximum(mask, flood_mask[1:h+1, 1:w+1])
    
    return mask


def _refine_bg_mask(mask: np.ndarray) -> np.ndarray:
    """Clean up background mask: remove noise, fill holes."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Close small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Remove small isolated regions
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find largest connected component (likely the actual background)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    if num_labels > 1:
        # Keep only the largest component
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = ((labels == largest_label) * 255).astype(np.uint8)
    
    return mask


def create_foreground_hint_from_bg_mask(
    bg_mask: np.ndarray, 
    margin: int = 10,
) -> np.ndarray:
    """
    Convert background mask to foreground hints for SAM2.
    Returns points that are definitely NOT background (for negative prompts).
    """
    h, w = bg_mask.shape[:2]
    
    # Erode background mask to get inner region
    kernel = np.ones((margin * 2, margin * 2), np.uint8)
    eroded = cv2.erode(bg_mask, kernel, iterations=1)
    
    # Foreground = not background
    fg_mask = cv2.bitwise_not(eroded)
    
    return fg_mask


def analyze_and_create_masks(
    image: np.ndarray,
    session_dir: str = None,
    session_id: str = None,
) -> Tuple[BackgroundAnalysis, Optional[np.ndarray]]:
    """
    Full pipeline: analyze background and create appropriate masks.
    
    Returns:
        - BackgroundAnalysis with classification info
        - Background mask (or None if complex)
    """
    # Analyze background type
    analysis = analyze_background(image)
    
    # Create mask based on type
    bg_mask = create_background_mask(image, analysis)
    
    # Save mask if path provided
    if bg_mask is not None and session_dir and session_id:
        import os
        os.makedirs(session_dir, exist_ok=True)
        mask_path = os.path.join(session_dir, f"{session_id}_bg_mask.png")
        cv2.imwrite(mask_path, bg_mask)
        analysis.color_mask_path = mask_path
        logger.info(f"[bg_detect] Saved background mask → {mask_path}")
    
    return analysis, bg_mask
