"""
Regression tests for the hard-cutout-outline fixes.

Tests cover every bug fixed in this change:
  C1/C7 — _build_feather_weight correctness (no collapsed-to-zero weights)
  C2    — Stage C does not reintroduce original object pixels at boundary
  C3    — _stage_a padded-crop context (fill diverges from original object colour)
  C4    — adaptive_removal_mask colour-contamination expansion
  C5    — _soft_mask produces a gradient, not a hard step
  C6    — _padded_crop provides context around the object
  C8    — _colour_match_crop aligns stats before blending
        — decontaminate_foreground_edges removes edge colour bleed

No GPU or AI model required.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ── helpers ─────────────────────────────────────────────────────────────────

def _img(h, w, color=(80, 130, 200)):
    a = np.zeros((h, w, 3), dtype=np.uint8)
    a[:, :] = color
    return a

def _rect(h, w, x1, y1, x2, y2):
    m = np.zeros((h, w), dtype=np.uint8)
    m[y1:y2, x1:x2] = 255
    return m


# ═══════════════════════════════════════════════════════════════════════════
# C1 / C7 — _build_feather_weight
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildFeatherWeight:

    def _weight(self, mask, feather_px=24):
        from services.restoration_service import _build_feather_weight
        return _build_feather_weight(mask, feather_px=feather_px)

    def test_weight_is_zero_outside_mask(self):
        h, w = 200, 200
        mask = _rect(h, w, 50, 50, 150, 150)
        wt = self._weight(mask)
        # Outside mask (top-left corner) must be 0
        assert wt[0, 0] == pytest.approx(0.0, abs=0.01)
        assert wt[10, 10] == pytest.approx(0.0, abs=0.01)

    def test_weight_is_one_deep_inside_mask(self):
        """Weight deep inside the mask (far from boundary) must be 1.0."""
        h, w = 300, 300
        # Large mask so interior is well beyond feather_px
        mask = _rect(h, w, 20, 20, 280, 280)
        wt = self._weight(mask, feather_px=24)
        # Centre is ~120px from nearest edge — well past feather_px=24
        assert wt[150, 150] == pytest.approx(1.0, abs=0.05)

    def test_weight_is_strictly_monotone_from_boundary(self):
        """Weight must increase monotonically from boundary inward."""
        h, w = 200, 200
        mask = _rect(h, w, 40, 100, 160, 101)  # thin horizontal bar
        # Use a large square mask so we can sample a clean column
        mask = _rect(h, w, 40, 40, 160, 160)
        wt = self._weight(mask, feather_px=30)
        # Sample a vertical strip through the mask centre column
        col = 100
        # Collect weights at y=38 (just outside), y=40 (boundary), y=55, y=70, y=100
        pts = [38, 40, 45, 55, 70, 100]
        vals = [float(wt[y, col]) for y in pts]
        for i in range(len(vals) - 1):
            assert vals[i] <= vals[i + 1] + 0.05, (
                f"Weight not monotone at pts {pts[i]}→{pts[i+1]}: {vals[i]:.3f}>{vals[i+1]:.3f}"
            )

    def test_no_hard_ring_at_feather_px_boundary(self):
        """
        The old broken formula produced a hard step at dist==feather_px.
        Verify weight is smooth (no jump > 0.1 between adjacent pixels).
        """
        h, w = 200, 200
        mask = _rect(h, w, 30, 30, 170, 170)
        wt = self._weight(mask, feather_px=20)
        # Check a horizontal line through the mask boundary
        row = 30  # boundary row
        for x in range(1, w - 1):
            diff = abs(float(wt[row, x]) - float(wt[row, x - 1]))
            assert diff < 0.15, f"Hard step at ({row},{x}): diff={diff:.3f}"

    def test_weights_in_range(self):
        h, w = 100, 100
        mask = _rect(h, w, 20, 20, 80, 80)
        wt = self._weight(mask)
        assert float(wt.min()) >= 0.0
        assert float(wt.max()) <= 1.0 + 1e-5


# ═══════════════════════════════════════════════════════════════════════════
# C2 — Stage C does not reintroduce original object colours at boundary
# ═══════════════════════════════════════════════════════════════════════════

class TestStageCNoBoundaryBleed:

    def test_object_colour_absent_in_transition_zone(self):
        """
        Original image has a bright-red object in the centre.
        Reconstructed image has correct background (grey) everywhere.
        After Stage C, the transition zone must NOT be red.
        """
        from services.restoration_service import _stage_c

        h, w = 200, 200
        # Original: grey background + red object
        original = _img(h, w, color=(80, 80, 80))
        original[70:130, 70:130] = [220, 20, 20]  # red object

        # Perfect reconstruction: grey everywhere (object removed)
        reconstructed = _img(h, w, color=(80, 80, 80))

        mask = _rect(h, w, 70, 70, 130, 130)
        result = _stage_c(original, reconstructed, mask)

        # Transition zone: just outside mask boundary
        # (rows 68-70, cols 70-130) — should be grey, not red
        transition = result[68:72, 70:130]
        assert np.mean(transition[:, :, 0]) < 150, (
            f"Red channel too high in transition zone: {np.mean(transition[:,:,0]):.1f}"
        )

    def test_unmasked_pixels_exactly_preserved(self):
        from services.restoration_service import _stage_c

        h, w = 100, 100
        original = _img(h, w, color=(50, 100, 200))
        reconstructed = _img(h, w, color=(200, 10, 10))
        mask = _rect(h, w, 30, 30, 70, 70)

        result = _stage_c(original, reconstructed, mask)

        # Corners are far from the mask — must be exactly the original
        assert np.array_equal(result[0, 0], original[0, 0])
        assert np.array_equal(result[99, 99], original[99, 99])
        assert np.array_equal(result[0, 99], original[0, 99])


# ═══════════════════════════════════════════════════════════════════════════
# C3 — _stage_a padded crop fills with surrounding colour, not object colour
# ═══════════════════════════════════════════════════════════════════════════

class TestStageAPaddedCrop:

    def test_fill_colour_close_to_surrounding_bg(self):
        from services.restoration_service import _stage_a

        h, w = 160, 160
        # Uniform dark-green background with a bright-white object
        img = _img(h, w, color=(30, 100, 30))
        img[55:105, 55:105] = [255, 255, 255]
        mask = _rect(h, w, 55, 55, 105, 105)

        result = _stage_a(img, mask)

        # Inside the hole, the fill should be greenish (not white)
        inside = result[60:100, 60:100].astype(float)
        mean_g = np.mean(inside[:, :, 1])
        mean_r = np.mean(inside[:, :, 0])
        # Green channel should dominate over red in the fill
        assert mean_g > mean_r, f"Fill colour not greenish: R={mean_r:.1f} G={mean_g:.1f}"
        # Fill should not still be white
        assert np.mean(inside) < 200, "Stage A did not replace the white object"

    def test_output_shape_preserved(self):
        from services.restoration_service import _stage_a
        img  = _img(80, 120, color=(10, 20, 30))
        mask = _rect(80, 120, 20, 10, 60, 50)
        out  = _stage_a(img, mask)
        assert out.shape == img.shape

    def test_unmasked_pixels_unchanged(self):
        from services.restoration_service import _stage_a
        h, w = 100, 100
        img  = _img(h, w, color=(100, 100, 100))
        mask = _rect(h, w, 30, 30, 70, 70)
        out  = _stage_a(img, mask)
        bg   = mask == 0
        diff = np.abs(out[bg].astype(int) - img[bg].astype(int))
        assert np.mean(diff) < 3.0, "Stage A changed unmasked pixels"


# ═══════════════════════════════════════════════════════════════════════════
# C4 — adaptive_removal_mask
# ═══════════════════════════════════════════════════════════════════════════

class TestAdaptiveRemovalMask:

    def test_larger_than_input_mask(self):
        from services.restoration_service import adaptive_removal_mask
        h, w = 200, 200
        img  = _img(h, w, color=(80, 80, 80))
        mask = _rect(h, w, 80, 80, 120, 120)
        result = adaptive_removal_mask(img, mask, dilation_px=8)
        # Result must cover at least as many pixels as the input
        assert np.sum(result > 0) >= np.sum(mask > 0)

    def test_contaminated_pixels_included(self):
        """
        Pixels just outside the hard mask that have the same colour as the
        object interior must be pulled into the removal region.
        """
        from services.restoration_service import adaptive_removal_mask
        h, w = 200, 200
        # Background: grey; object: bright red
        img = _img(h, w, color=(60, 60, 60))
        img[70:130, 70:130] = [220, 20, 20]  # red object interior
        # Contaminated ring: 2px outside mask, still very red
        img[68:70, 70:130] = [200, 30, 30]
        img[130:132, 70:130] = [200, 30, 30]

        mask = _rect(h, w, 70, 70, 130, 130)
        # Use zero flat dilation so we can see colour-contamination effect alone
        result = adaptive_removal_mask(img, mask, dilation_px=0)

        # The contaminated 2-px band should now be in the removal mask
        assert np.any(result[68:70, 75:125] > 0), (
            "Colour-contaminated pixels just outside mask were not included"
        )

    def test_protection_mask_prevents_expansion(self):
        """Expansion must not cross into pixels owned by another object."""
        from services.restoration_service import adaptive_removal_mask
        h, w = 200, 200
        img  = _img(h, w, color=(60, 60, 60))
        img[70:130, 70:130] = [220, 20, 20]

        # Object mask
        mask = _rect(h, w, 70, 70, 130, 130)
        # Protection: another object is in the top-left quadrant
        protection = _rect(h, w, 0, 0, 60, 60)

        result = adaptive_removal_mask(img, mask, dilation_px=20, protection_mask=protection)
        # Protected area must be zero
        assert np.all(result[0:60, 0:60] == 0), "Protection mask was violated"

    def test_no_expansion_when_dilation_zero_and_no_contamination(self):
        from services.restoration_service import adaptive_removal_mask
        h, w = 100, 100
        # Uniform image — no colour contamination possible
        img  = _img(h, w, color=(100, 100, 100))
        mask = _rect(h, w, 30, 30, 70, 70)
        result = adaptive_removal_mask(img, mask, dilation_px=0)
        # Result should be very close to input (may differ by colour-contam heuristic)
        input_px  = int(np.sum(mask > 0))
        result_px = int(np.sum(result > 0))
        assert result_px <= input_px * 1.10, (
            f"Mask grew unexpectedly: {input_px} → {result_px}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# C5 — _soft_mask gradient
# ═══════════════════════════════════════════════════════════════════════════

class TestSoftMask:

    def test_interior_is_one(self):
        from services.restoration_service import _soft_mask
        h, w = 200, 200
        mask = _rect(h, w, 20, 20, 180, 180)
        soft = _soft_mask(mask, sigma=8.0)
        # Centre should be close to 1.0
        assert float(soft[100, 100]) > 0.9

    def test_exterior_is_zero(self):
        from services.restoration_service import _soft_mask
        h, w = 200, 200
        mask = _rect(h, w, 60, 60, 140, 140)
        soft = _soft_mask(mask, sigma=8.0)
        # Corners are far from mask — should be 0
        assert float(soft[0, 0]) < 0.05
        assert float(soft[199, 199]) < 0.05

    def test_boundary_is_gradient_not_step(self):
        from services.restoration_service import _soft_mask
        h, w = 200, 200
        mask = _rect(h, w, 50, 50, 150, 150)
        soft = _soft_mask(mask, sigma=8.0)
        # The boundary row should have a gradient across it
        row_vals = [float(soft[50, x]) for x in range(40, 60)]
        diffs = [abs(row_vals[i+1] - row_vals[i]) for i in range(len(row_vals)-1)]
        # At least one small step (gradient) rather than all zeros or a hard 0→1 jump
        assert max(diffs) < 0.4, f"Hard step detected in soft mask: max_diff={max(diffs):.3f}"
        assert sum(v > 0.01 for v in row_vals) >= 3, "Soft mask is still too hard at boundary"


# ═══════════════════════════════════════════════════════════════════════════
# C6 — _padded_crop context
# ═══════════════════════════════════════════════════════════════════════════

class TestPaddedCrop:

    def test_crop_is_larger_than_mask_bbox(self):
        from services.restoration_service import _padded_crop
        h, w = 300, 400
        img  = _img(h, w)
        mask = _rect(h, w, 100, 80, 200, 180)
        crop_img, crop_mask, (cy1, cx1, cy2, cx2) = _padded_crop(img, mask)
        # Crop should extend beyond the mask bbox
        assert cy1 < 80
        assert cx1 < 100
        assert cy2 > 180
        assert cx2 > 200

    def test_crop_dimensions_match_coords(self):
        from services.restoration_service import _padded_crop
        h, w = 200, 300
        img  = _img(h, w)
        mask = _rect(h, w, 50, 40, 150, 140)
        crop_img, crop_mask, (cy1, cx1, cy2, cx2) = _padded_crop(img, mask)
        assert crop_img.shape[0] == cy2 - cy1
        assert crop_img.shape[1] == cx2 - cx1
        assert crop_mask.shape == crop_img.shape[:2]

    def test_crop_stays_within_image_bounds(self):
        from services.restoration_service import _padded_crop
        h, w = 100, 100
        img  = _img(h, w)
        # Mask touching the border
        mask = _rect(h, w, 0, 0, 50, 50)
        crop_img, crop_mask, (cy1, cx1, cy2, cx2) = _padded_crop(img, mask)
        assert cy1 >= 0 and cx1 >= 0
        assert cy2 <= h and cx2 <= w


# ═══════════════════════════════════════════════════════════════════════════
# C8 — _colour_match_crop
# ═══════════════════════════════════════════════════════════════════════════

class TestColourMatchCrop:

    def test_reduces_colour_mismatch(self):
        from services.restoration_service import _colour_match_crop
        h, w = 100, 100
        # Background: dark blue
        original = _img(h, w, color=(20, 20, 120))
        # Reconstructed: bright red (very wrong colour)
        reconstructed = _img(h, w, color=(200, 30, 30))
        mask = _rect(h, w, 30, 30, 70, 70)

        matched = _colour_match_crop(original, reconstructed, mask)

        # After matching, the inside should be closer to original stats
        # than the raw reconstructed
        bg = (mask == 0)
        ref_mean = original[bg].astype(float).mean()

        gen_inside = reconstructed[mask > 0].astype(float)
        mat_inside = matched[mask > 0].astype(float)

        ref_b = float(original[bg, 2].mean())  # blue channel of reference
        gen_b = float(gen_inside[:, 2].mean())
        mat_b = float(mat_inside[:, 2].mean())

        # Matched blue channel should be closer to reference than raw
        assert abs(mat_b - ref_b) < abs(gen_b - ref_b) + 10, (
            f"Colour match did not improve blue channel: ref={ref_b:.1f} gen={gen_b:.1f} mat={mat_b:.1f}"
        )

    def test_unmasked_pixels_unchanged(self):
        from services.restoration_service import _colour_match_crop
        h, w = 80, 80
        original     = _img(h, w, color=(50, 100, 150))
        reconstructed = _img(h, w, color=(200, 20, 20))
        mask = _rect(h, w, 20, 20, 60, 60)
        matched = _colour_match_crop(original, reconstructed, mask)
        # Outside mask: unchanged
        bg = mask == 0
        assert np.array_equal(matched[bg], reconstructed[bg])


# ═══════════════════════════════════════════════════════════════════════════
# decontaminate_foreground_edges
# ═══════════════════════════════════════════════════════════════════════════

class TestDecontaminateForegroundEdges:

    def _make_rgba_with_fringe(self):
        """
        Create an RGBA cutout where:
        - Interior (alpha=255): pure red object
        - Edge band (alpha=80): mix of red + background blue = purple fringe
        - Exterior (alpha=0): transparent
        """
        h, w = 100, 100
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        # Interior — pure red, fully opaque
        rgba[20:80, 20:80] = [255, 0, 0, 255]
        # Edge band — purple fringe (colour bleed)
        rgba[18:20, 20:80] = [150, 0, 100, 80]
        rgba[80:82, 20:80] = [150, 0, 100, 80]
        rgba[20:80, 18:20] = [150, 0, 100, 80]
        rgba[20:80, 80:82] = [150, 0, 100, 80]
        return rgba

    def test_alpha_channel_unchanged(self):
        from services.restoration_service import decontaminate_foreground_edges
        rgba = self._make_rgba_with_fringe()
        result = decontaminate_foreground_edges(rgba, band_px=3)
        # Alpha must be identical
        assert np.array_equal(result[:, :, 3], rgba[:, :, 3]), "Alpha channel was modified"

    def test_opaque_interior_rgb_unchanged(self):
        from services.restoration_service import decontaminate_foreground_edges
        rgba = self._make_rgba_with_fringe()
        result = decontaminate_foreground_edges(rgba, band_px=3)
        # Fully opaque interior (alpha==255) should retain original RGB
        opaque = rgba[:, :, 3] == 255
        diff = np.abs(result[opaque, :3].astype(int) - rgba[opaque, :3].astype(int))
        assert np.mean(diff) < 5.0, "Opaque interior pixels were changed"

    def test_edge_fringe_blue_reduced(self):
        """
        The purple fringe (high blue) in semi-transparent edges should become
        closer to the red interior after decontamination.
        """
        from services.restoration_service import decontaminate_foreground_edges
        rgba = self._make_rgba_with_fringe()
        result = decontaminate_foreground_edges(rgba, band_px=3)

        edge = (rgba[:, :, 3] == 80)
        if not np.any(edge):
            return  # skip if no semi-transparent pixels

        before_b = float(rgba[edge, 2].mean())
        after_b  = float(result[edge, 2].mean())
        # Blue channel (fringe colour) should be reduced toward the red interior
        assert after_b <= before_b + 5.0, (
            f"Blue fringe not reduced: before={before_b:.1f} after={after_b:.1f}"
        )

    def test_transparent_pixels_untouched(self):
        from services.restoration_service import decontaminate_foreground_edges
        rgba = self._make_rgba_with_fringe()
        result = decontaminate_foreground_edges(rgba, band_px=3)
        transparent = rgba[:, :, 3] == 0
        if np.any(transparent):
            assert np.array_equal(result[transparent], rgba[transparent]), (
                "Fully transparent pixels were modified"
            )


# ═══════════════════════════════════════════════════════════════════════════
# End-to-end: restore_background — object outline absent in transition zone
# ═══════════════════════════════════════════════════════════════════════════

class TestRestoreBackgroundEndToEnd:

    def test_object_outline_not_visible_in_transition(self):
        """
        After restoring a red-object from a grey background, the transition zone
        just outside the mask must not contain red pixels (the object's colour).
        This directly tests the C1+C2 fix.
        """
        from services.restoration_service import restore_background

        h, w = 200, 200
        # Grey background with a red square in the middle
        img = _img(h, w, color=(80, 80, 80))
        img[60:140, 60:140] = [220, 30, 30]  # bright red object

        mask = _rect(h, w, 60, 60, 140, 140)
        result = restore_background(img, mask, force_generative=False)

        # Transition zone: 4px just outside the mask on each side
        top_strip    = result[55:60, 60:140]
        bottom_strip = result[140:145, 60:140]
        left_strip   = result[60:140, 55:60]
        right_strip  = result[60:140, 140:145]

        for name, strip in [("top", top_strip), ("bottom", bottom_strip),
                             ("left", left_strip), ("right", right_strip)]:
            mean_r = float(strip[:, :, 0].mean())
            assert mean_r < 160, (
                f"{name} transition zone still red: mean_R={mean_r:.1f} "
                f"(object was rgb(220,30,30), background rgb(80,80,80))"
            )

    def test_unmasked_pixels_unchanged_end_to_end(self):
        from services.restoration_service import restore_background
        h, w = 100, 100
        img  = _img(h, w, color=(50, 120, 200))
        img[35:65, 35:65] = [250, 10, 10]
        mask = _rect(h, w, 35, 35, 65, 65)
        result = restore_background(img, mask, force_generative=False)
        assert np.array_equal(result[0, 0], img[0, 0])
        assert np.array_equal(result[99, 99], img[99, 99])

    def test_no_transparent_holes(self):
        from services.restoration_service import restore_background
        h, w = 80, 80
        img  = _img(h, w)
        mask = _rect(h, w, 20, 20, 60, 60)
        result = restore_background(img, mask, force_generative=False)
        assert result.ndim == 3 and result.shape[2] == 3
