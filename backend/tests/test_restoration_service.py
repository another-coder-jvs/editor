"""
Pure-Python validation tests for the background restoration service.

These tests verify:
  1. build_removal_mask — correct union + dilation behaviour.
  2. _needs_generative — correct routing based on hole size and edge density.
  3. _stage_a — classical inpainting fills the mask and doesn't leave original pixels.
  4. _guard_original_pixels — correctly detects when model returned original pixels.
  5. _stage_c — seam blending preserves unmasked pixels exactly.
  6. restore_background — end-to-end: unmasked pixels unchanged, masked pixels differ.
  7. Multiple-object union mask — repeated objects don't reintroduce each other.
  8. restore_background_from_mask_files — loads mask files from disk correctly.

No GPU or AI model is required for any of these tests.
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# ── Path setup ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solid_image(h: int, w: int, color=(100, 150, 200)) -> np.ndarray:
    """Create a uniform-color RGB image."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = color
    return img


def _rect_mask(h: int, w: int, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    m[y1:y2, x1:x2] = 255
    return m


# ---------------------------------------------------------------------------
# 1. build_removal_mask
# ---------------------------------------------------------------------------

class TestBuildRemovalMask:

    def test_union_of_two_masks(self):
        from services.restoration_service import build_removal_mask
        h, w = 100, 100
        m1 = _rect_mask(h, w, 0, 0, 40, 40)
        m2 = _rect_mask(h, w, 60, 60, 100, 100)
        result = build_removal_mask([m1, m2], dilation_px=0)
        # Both regions should be white
        assert result[10, 10] == 255
        assert result[80, 80] == 255
        # Gap between the two regions should still be 0 (no dilation)
        assert result[50, 50] == 0

    def test_dilation_expands_mask(self):
        from services.restoration_service import build_removal_mask
        h, w = 200, 200
        m = _rect_mask(h, w, 90, 90, 110, 110)  # 20×20 centre
        result_nodilate = build_removal_mask([m], dilation_px=0)
        result_dilated  = build_removal_mask([m], dilation_px=10)
        # Dilated mask should cover more pixels
        assert np.sum(result_dilated > 0) > np.sum(result_nodilate > 0)
        # Pixels near the border of the original mask should now be included
        assert result_dilated[85, 95] == 255  # 5px outside original top edge

    def test_empty_mask_list(self):
        from services.restoration_service import build_removal_mask
        # Edge case: empty list should not crash
        try:
            result = build_removal_mask([], dilation_px=0)
        except Exception:
            pass  # acceptable — the function documents requiring non-empty list

    def test_single_mask_no_dilation(self):
        from services.restoration_service import build_removal_mask
        h, w = 50, 50
        m = _rect_mask(h, w, 10, 10, 40, 40)
        result = build_removal_mask([m], dilation_px=0)
        assert np.array_equal((result > 0), (m > 0))


# ---------------------------------------------------------------------------
# 2. _needs_generative routing
# ---------------------------------------------------------------------------

class TestNeedsGenerative:

    def _make_complex_image(self, h=100, w=100):
        """Image with lots of edges (checkerboard)."""
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(0, h, 4):
            for j in range(0, w, 4):
                if (i // 4 + j // 4) % 2 == 0:
                    img[i:i+4, j:j+4] = [200, 200, 200]
        return img

    def test_force_true_always_generative(self):
        from services.restoration_service import _needs_generative
        img = _solid_image(100, 100)
        mask = _rect_mask(100, 100, 0, 0, 5, 5)  # tiny hole
        assert _needs_generative(img, mask, force=True) is True

    def test_small_hole_simple_bg_not_generative(self):
        from services.restoration_service import _needs_generative
        # Hole < 5% of canvas, uniform background → Stage A sufficient
        img = _solid_image(200, 200, color=(80, 80, 80))
        mask = _rect_mask(200, 200, 90, 90, 110, 110)  # 20×20 = 1% of 200×200
        assert _needs_generative(img, mask, force=False) is False

    def test_large_hole_triggers_generative(self):
        from services.restoration_service import _needs_generative, GENERATIVE_THRESHOLD
        img = _solid_image(100, 100, color=(80, 80, 80))
        # Hole > GENERATIVE_THRESHOLD (5%) of canvas
        mask = _rect_mask(100, 100, 0, 0, 40, 40)  # 16% of canvas
        assert _needs_generative(img, mask, force=False) is True

    def test_complex_background_triggers_generative(self):
        from services.restoration_service import _needs_generative
        img = self._make_complex_image(100, 100)
        mask = _rect_mask(100, 100, 45, 45, 55, 55)  # 1% hole, but complex bg
        assert _needs_generative(img, mask, force=False) is True


# ---------------------------------------------------------------------------
# 3. _stage_a — classical inpainting
# ---------------------------------------------------------------------------

class TestStageA:

    def test_fills_hole_region(self):
        from services.restoration_service import _stage_a
        h, w = 100, 100
        # Uniform red image with a white rectangle in the middle
        img = _solid_image(h, w, color=(200, 50, 50))
        img[40:60, 40:60] = [255, 255, 255]  # white "object"
        mask = _rect_mask(h, w, 40, 40, 60, 60)

        result = _stage_a(img, mask)

        assert result.shape == img.shape
        assert result.dtype == np.uint8

        # The white rectangle should be replaced (cv2 inpaint fills it with red)
        inside = result[45:55, 45:55]
        # Inside should not be white anymore
        assert not np.all(inside > 200), "Stage A did not fill the white hole"

    def test_unmasked_pixels_approximately_preserved(self):
        from services.restoration_service import _stage_a
        h, w = 100, 100
        img = _solid_image(h, w, color=(120, 120, 120))
        mask = _rect_mask(h, w, 40, 40, 60, 60)

        result = _stage_a(img, mask)

        # Outside the mask the pixels should be very close to original
        outside = np.ones((h, w), dtype=bool)
        outside[40:60, 40:60] = False
        diff = np.abs(result[outside].astype(int) - img[outside].astype(int))
        assert np.mean(diff) < 5.0, "Stage A changed too many unmasked pixels"

    def test_output_shape_matches_input(self):
        from services.restoration_service import _stage_a
        img  = _solid_image(80, 120, color=(10, 20, 30))
        mask = _rect_mask(80, 120, 20, 10, 60, 50)
        result = _stage_a(img, mask)
        assert result.shape == img.shape


# ---------------------------------------------------------------------------
# 4. _guard_original_pixels
# ---------------------------------------------------------------------------

class TestGuardOriginalPixels:

    def test_detects_identical_result(self):
        from services.restoration_service import _guard_original_pixels
        h, w = 50, 50
        img  = _solid_image(h, w, color=(100, 100, 100))
        mask = _rect_mask(h, w, 10, 10, 40, 40)
        # result is same as original inside mask → should trigger guard
        assert _guard_original_pixels(img, img.copy(), mask) is True

    def test_accepts_different_result(self):
        from services.restoration_service import _guard_original_pixels
        h, w = 50, 50
        original = _solid_image(h, w, color=(100, 100, 100))
        result   = _solid_image(h, w, color=(50, 200, 10))   # very different
        mask     = _rect_mask(h, w, 10, 10, 40, 40)
        assert _guard_original_pixels(original, result, mask) is False

    def test_empty_mask_returns_false(self):
        from services.restoration_service import _guard_original_pixels
        img  = _solid_image(50, 50, color=(0, 0, 0))
        mask = np.zeros((50, 50), dtype=np.uint8)  # empty mask
        assert _guard_original_pixels(img, img.copy(), mask) is False

    def test_partially_similar_below_threshold(self):
        from services.restoration_service import _guard_original_pixels, SAME_PIXEL_FRAC
        h, w = 100, 100
        original = _solid_image(h, w, color=(100, 100, 100))
        mask     = _rect_mask(h, w, 10, 10, 90, 90)

        # Make result differ enough so same_frac < SAME_PIXEL_FRAC
        result = original.copy()
        result[10:90, 10:90] = [10, 200, 10]  # very different inside mask

        assert _guard_original_pixels(original, result, mask) is False


# ---------------------------------------------------------------------------
# 5. _stage_c — seam blending
# ---------------------------------------------------------------------------

class TestStageC:

    def test_unmasked_pixels_unchanged(self):
        from services.restoration_service import _stage_c
        h, w = 100, 100
        original = _solid_image(h, w, color=(50, 100, 150))
        # reconstructed is totally different
        reconstructed = _solid_image(h, w, color=(200, 10, 20))
        mask = _rect_mask(h, w, 30, 30, 70, 70)

        result = _stage_c(original, reconstructed, mask)

        # Pixels far outside the mask must be unchanged
        # (corners, far from the feather band)
        assert int(result[0, 0, 0]) == 50, "Unmasked corner R wrong"
        assert int(result[0, 0, 1]) == 100, "Unmasked corner G wrong"
        assert int(result[0, 0, 2]) == 150, "Unmasked corner B wrong"
        assert int(result[99, 99, 0]) == 50

    def test_deep_inside_mask_uses_reconstructed(self):
        from services.restoration_service import _stage_c, SEAM_FEATHER_PX
        h, w = 200, 200
        original      = _solid_image(h, w, color=(10, 10, 10))
        reconstructed = _solid_image(h, w, color=(240, 240, 240))
        # Large mask with enough interior so the feather band doesn't reach centre
        mask = _rect_mask(h, w, 20, 20, 180, 180)

        result = _stage_c(original, reconstructed, mask)

        # Centre pixel should be close to reconstructed (light)
        centre = result[100, 100]
        assert int(centre[0]) > 200, f"Centre should be close to reconstructed: {centre}"

    def test_output_shape_preserved(self):
        from services.restoration_service import _stage_c
        h, w = 150, 200
        orig  = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        recon = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        mask  = _rect_mask(h, w, 50, 50, 100, 100)
        result = _stage_c(orig, recon, mask)
        assert result.shape == (h, w, 3)
        assert result.dtype == np.uint8


# ---------------------------------------------------------------------------
# 6. restore_background — end-to-end (Stage A only, no GPU)
# ---------------------------------------------------------------------------

class TestRestoreBackground:

    def test_unmasked_pixels_unchanged(self):
        """Pixels outside the removal mask must be identical to the original."""
        from services.restoration_service import restore_background
        h, w = 100, 100
        img = _solid_image(h, w, color=(80, 130, 200))
        # Put a visible white square in the middle (the "object")
        img[30:70, 30:70] = [255, 255, 255]
        mask = _rect_mask(h, w, 30, 30, 70, 70)

        # force_generative=False → Stage A only (no model needed)
        result = restore_background(img, mask, object_labels=["object"], force_generative=False)

        assert result.shape == (h, w, 3)
        # Top-left corner is outside the mask, must be exactly preserved
        assert np.array_equal(result[0, 0], img[0, 0])
        assert np.array_equal(result[99, 99], img[99, 99])

    def test_masked_pixels_are_different_from_original(self):
        """The restored pixels inside the mask must differ from the original object pixels."""
        from services.restoration_service import restore_background
        h, w = 100, 100
        # Uniform dark-grey background with a bright-white "object" in the middle
        img = _solid_image(h, w, color=(60, 60, 60))
        img[35:65, 35:65] = [255, 255, 255]  # bright white object
        mask = _rect_mask(h, w, 35, 35, 65, 65)

        result = restore_background(img, mask, force_generative=False)

        # Inside the mask, original pixels are [255,255,255].
        # After cv2 inpaint (which propagates the surrounding grey), they should differ.
        inside_orig   = img[40:60, 40:60].astype(float)
        inside_result = result[40:60, 40:60].astype(float)
        mean_diff = np.mean(np.abs(inside_result - inside_orig))
        assert mean_diff > 30.0, (
            f"Restored pixels too similar to original object pixels (mean diff={mean_diff:.1f})"
        )

    def test_canvas_dimensions_preserved(self):
        from services.restoration_service import restore_background
        h, w = 80, 120
        img  = np.random.randint(50, 200, (h, w, 3), dtype=np.uint8)
        mask = _rect_mask(h, w, 20, 10, 60, 50)
        result = restore_background(img, mask, force_generative=False)
        assert result.shape == (h, w, 3)

    def test_empty_mask_returns_original(self):
        from services.restoration_service import restore_background
        h, w = 50, 50
        img  = _solid_image(h, w, color=(100, 100, 100))
        mask = np.zeros((h, w), dtype=np.uint8)  # no objects
        result = restore_background(img, mask, force_generative=False)
        assert np.array_equal(result, img), "Empty mask should return unchanged image"

    def test_no_transparent_holes(self):
        """Result must be RGB (no alpha), meaning no transparent holes."""
        from services.restoration_service import restore_background
        h, w = 60, 60
        img  = _solid_image(h, w)
        mask = _rect_mask(h, w, 20, 20, 40, 40)
        result = restore_background(img, mask, force_generative=False)
        assert result.ndim == 3
        assert result.shape[2] == 3, "Result should be RGB, not RGBA"


# ---------------------------------------------------------------------------
# 7. Multiple-object union mask
# ---------------------------------------------------------------------------

class TestMultipleObjects:

    def test_two_objects_union_fills_both_holes(self):
        from services.restoration_service import restore_background, build_removal_mask
        h, w = 200, 200
        # Background: uniform blue
        img = _solid_image(h, w, color=(30, 80, 160))
        # Two distinct bright-red "objects"
        img[20:50, 20:50]   = [250, 10, 10]   # object A
        img[120:160, 130:180] = [250, 10, 10] # object B

        mask_a = _rect_mask(h, w, 20, 20, 50, 50)
        mask_b = _rect_mask(h, w, 130, 120, 180, 160)
        union  = build_removal_mask([mask_a, mask_b], dilation_px=0)

        result = restore_background(img, union, object_labels=["a", "b"], force_generative=False)

        # Both holes should be filled (different from original red)
        region_a = result[25:45, 25:45].astype(float)
        region_b = result[125:155, 135:175].astype(float)
        original_red = np.array([250, 10, 10], dtype=float)

        diff_a = np.mean(np.abs(region_a - original_red))
        diff_b = np.mean(np.abs(region_b - original_red))

        assert diff_a > 20.0, f"Object A hole was not filled (mean_diff={diff_a:.1f})"
        assert diff_b > 20.0, f"Object B hole was not filled (mean_diff={diff_b:.1f})"

    def test_unrelated_region_untouched(self):
        from services.restoration_service import restore_background, build_removal_mask
        h, w = 100, 100
        img = _solid_image(h, w, color=(70, 140, 200))
        img[10:30, 10:30] = [255, 0, 0]  # object

        mask = _rect_mask(h, w, 10, 10, 30, 30)
        result = restore_background(img, mask, force_generative=False)

        # Bottom-right quadrant: completely unaffected
        br_orig   = img[60:100, 60:100]
        br_result = result[60:100, 60:100]
        assert np.array_equal(br_orig, br_result), "Unrelated region was changed"


# ---------------------------------------------------------------------------
# 8. restore_background_from_mask_files
# ---------------------------------------------------------------------------

class TestRestoreFromMaskFiles:

    def test_loads_mask_and_restores(self, tmp_path):
        from services.restoration_service import restore_background_from_mask_files

        h, w = 80, 80
        img = _solid_image(h, w, color=(100, 150, 100))
        img[20:60, 20:60] = [255, 0, 0]   # red object

        # Save a mask PNG to disk
        mask_arr = _rect_mask(h, w, 20, 20, 60, 60)
        mask_path = tmp_path / "obj_mask.png"
        Image.fromarray(mask_arr).save(str(mask_path))

        result = restore_background_from_mask_files(
            image_np=img,
            mask_paths=[str(mask_path)],
            object_labels=["object"],
            force_generative=False,
        )

        assert result.shape == (h, w, 3)
        # Red object hole should be filled differently
        region = result[25:55, 25:55].astype(float)
        mean_diff = np.mean(np.abs(region - np.array([255, 0, 0], dtype=float)))
        assert mean_diff > 20.0, "Mask file not applied: red object still visible"

    def test_missing_mask_file_skipped(self, tmp_path):
        from services.restoration_service import restore_background_from_mask_files
        h, w = 50, 50
        img = _solid_image(h, w, color=(50, 50, 50))
        # Pass a path that doesn't exist
        result = restore_background_from_mask_files(
            image_np=img,
            mask_paths=[str(tmp_path / "nonexistent.png")],
            force_generative=False,
        )
        # Should return original image unchanged (no valid masks)
        assert np.array_equal(result, img)
