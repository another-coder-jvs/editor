"""
Pure-Python validation tests for the segmentation pipeline.

These tests exercise:
  1. Pixel-ownership (priority mask) logic — no GPU required.
  2. Full-canvas layer PNG size preservation.
  3. Background layer correctness (original pixels, no FLUX).
  4. Detection degenerate-box / containment suppression.
  5. merge_service compatibility with full-canvas layers (bbox.x == 0).

All tests run without loading SAM2, Grounding DINO, or any AI model.
"""
from __future__ import annotations

import json
import sys
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# ── path setup so tests import from backend/ ──────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------------------
# Helpers that replicate the core logic without SAM2
# ---------------------------------------------------------------------------

def _build_ownership_map(
    image_shape,
    masks: list[np.ndarray],
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Simulate the ownership-map pass from segmentation_service.segment_objects.

    masks: list ordered small→large (already sorted by bbox area).
    Returns (ownership_map, owned_masks_per_object).
    """
    h, w = image_shape[:2]
    ownership_map = np.full((h, w), -1, dtype=np.int32)
    owned_masks: list[np.ndarray] = []

    for idx, refined in enumerate(masks):
        binary = (refined > 128).astype(bool)
        newly_claimed = binary & (ownership_map < 0)
        ownership_map[newly_claimed] = idx
        owned_mask = np.where(newly_claimed, refined, 0).astype(np.uint8)
        owned_masks.append(owned_mask)

    return ownership_map, owned_masks


def _make_rect_mask(shape, x1, y1, x2, y2, value=255):
    m = np.zeros(shape[:2], dtype=np.uint8)
    m[y1:y2, x1:x2] = value
    return m


# ---------------------------------------------------------------------------
# 1. Pixel-ownership — no two objects share a foreground pixel
# ---------------------------------------------------------------------------

class TestPixelOwnership:

    def test_no_overlap_between_two_objects(self):
        """Two non-overlapping objects → each owns their own pixels, zero intersection."""
        shape = (100, 100)
        mask_a = _make_rect_mask(shape, 0,  0, 40, 40)   # top-left square
        mask_b = _make_rect_mask(shape, 60, 60, 100, 100) # bottom-right square
        # sorted small→large (same area here; order doesn't matter)
        _, owned = _build_ownership_map(shape, [mask_a, mask_b])
        owned_a_bin = owned[0] > 0
        owned_b_bin = owned[1] > 0
        assert not np.any(owned_a_bin & owned_b_bin), "Overlapping ownership detected!"

    def test_smaller_object_retains_pixels_inside_larger(self):
        """
        Small object fully inside large object.
        After ownership resolution the small object must own its pixels
        and the large object must NOT own those same pixels.
        """
        shape = (200, 200)
        # large object: full canvas minus a 10px border
        mask_large = _make_rect_mask(shape, 10, 10, 190, 190)
        # small object: 20×20 box inside the large one — processed FIRST (small→large)
        mask_small = _make_rect_mask(shape, 80, 80, 100, 100)

        # Small → large order
        _, owned = _build_ownership_map(shape, [mask_small, mask_large])
        owned_small = owned[0] > 0
        owned_large = owned[1] > 0

        # Small object retains its pixels
        assert np.all(owned_small[80:100, 80:100]), "Small object lost some of its pixels!"
        # Large object does NOT own the small object's pixels
        assert not np.any(owned_large[80:100, 80:100]), "Large object still owns small object's pixels!"

    def test_total_coverage_equals_union(self):
        """Sum of all owned masks == union of all input masks (no pixel is lost)."""
        shape = (100, 100)
        # Two overlapping masks
        mask_a = _make_rect_mask(shape, 0, 0, 70, 70)
        mask_b = _make_rect_mask(shape, 30, 30, 100, 100)
        _, owned = _build_ownership_map(shape, [mask_a, mask_b])
        combined_owned = (owned[0] > 0) | (owned[1] > 0)
        union = ((mask_a > 0) | (mask_b > 0))
        assert np.array_equal(combined_owned, union), "Some union pixels are missing from owned masks!"

    def test_ownership_map_no_double_claim(self):
        """Each non-negative entry in ownership_map appears for exactly one object."""
        shape = (100, 100)
        mask_a = _make_rect_mask(shape, 0, 0, 80, 80)
        mask_b = _make_rect_mask(shape, 20, 20, 100, 100)
        ownership_map, owned = _build_ownership_map(shape, [mask_a, mask_b])

        claimed_by_a = owned[0] > 0
        claimed_by_b = owned[1] > 0
        assert not np.any(claimed_by_a & claimed_by_b), "Double-claim detected in owned masks"

    def test_three_objects_no_overlap(self):
        """Three objects with partial overlaps — all must be non-overlapping after resolution."""
        shape = (300, 300)
        masks_raw = [
            _make_rect_mask(shape,   0,   0, 150, 150),  # small
            _make_rect_mask(shape, 100, 100, 250, 250),  # medium
            _make_rect_mask(shape,  50,  50, 300, 300),  # large
        ]
        _, owned = _build_ownership_map(shape, masks_raw)
        # Check all pairs
        for i in range(len(owned)):
            for j in range(i + 1, len(owned)):
                assert not np.any((owned[i] > 0) & (owned[j] > 0)), (
                    f"Objects {i} and {j} share pixels!"
                )


# ---------------------------------------------------------------------------
# 2. Full-canvas layer PNG size preservation
# ---------------------------------------------------------------------------

class TestFullCanvasLayers:

    def test_output_png_matches_source_dimensions(self, tmp_path):
        """mask_to_fullcanvas_png writes a PNG that has the same HxW as the source."""
        from services.segmentation_service import mask_to_fullcanvas_png

        img_h, img_w = 480, 640
        image_np = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        image_np[:, :, 0] = 200  # red channel
        mask = _make_rect_mask((img_h, img_w), 100, 100, 400, 300)

        out = tmp_path / "test_layer.png"
        mask_to_fullcanvas_png(image_np, mask, out)

        result = Image.open(out)
        assert result.size == (img_w, img_h), (
            f"Expected ({img_w},{img_h}), got {result.size}"
        )
        assert result.mode == "RGBA"

    def test_alpha_matches_mask(self, tmp_path):
        """The alpha channel of the saved PNG must equal the input mask."""
        from services.segmentation_service import mask_to_fullcanvas_png

        img_h, img_w = 200, 200
        image_np = np.full((img_h, img_w, 3), 128, dtype=np.uint8)
        mask = _make_rect_mask((img_h, img_w), 50, 50, 150, 150)

        out = tmp_path / "test_alpha.png"
        mask_to_fullcanvas_png(image_np, mask, out)

        result = np.array(Image.open(out).convert("RGBA"))
        assert np.array_equal(result[:, :, 3], mask), "Alpha channel doesn't match input mask!"

    def test_rgb_pixels_preserved_inside_mask(self, tmp_path):
        """Pixels inside the mask must have the same RGB values as the source."""
        from services.segmentation_service import mask_to_fullcanvas_png

        img_h, img_w = 100, 100
        image_np = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        image_np[20:80, 20:80] = [255, 0, 0]  # red square
        mask = _make_rect_mask((img_h, img_w), 20, 20, 80, 80)

        out = tmp_path / "test_rgb.png"
        mask_to_fullcanvas_png(image_np, mask, out)

        result = np.array(Image.open(out).convert("RGBA"))
        inside = result[20:80, 20:80]
        assert np.all(inside[:, :, 0] == 255), "Red channel wrong inside mask"
        assert np.all(inside[:, :, 1] == 0), "Green channel wrong inside mask"

    def test_pixels_outside_mask_have_zero_alpha(self, tmp_path):
        """Pixels outside the mask must have alpha=0."""
        from services.segmentation_service import mask_to_fullcanvas_png

        img_h, img_w = 100, 100
        image_np = np.full((img_h, img_w, 3), 200, dtype=np.uint8)
        mask = _make_rect_mask((img_h, img_w), 40, 40, 60, 60)

        out = tmp_path / "test_outside.png"
        mask_to_fullcanvas_png(image_np, mask, out)

        result = np.array(Image.open(out).convert("RGBA"))
        # Corners should be transparent
        assert result[0, 0, 3] == 0
        assert result[99, 99, 3] == 0


# ---------------------------------------------------------------------------
# 3. Background layer correctness
# ---------------------------------------------------------------------------

class TestBackgroundLayer:

    def test_background_uses_original_pixels(self, tmp_path):
        """
        Background layer must contain the original pixel values, not black or
        reconstructed pixels.
        """
        from services.segmentation_service import _build_background_layer

        img_h, img_w = 100, 100
        # Image: left half green, right half blue
        image_np = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        image_np[:, :50] = [0, 200, 0]   # green left
        image_np[:, 50:] = [0, 0, 200]   # blue right

        # Object owns the left half
        ownership_map = np.full((img_h, img_w), -1, dtype=np.int32)
        ownership_map[:, :50] = 0  # object 0 owns left half

        session_id = uuid.uuid4().hex
        session_dir = tmp_path / session_id
        session_dir.mkdir()

        bg_layer = _build_background_layer(
            image_np, ownership_map, session_id, session_dir, img_w, img_h, 1
        )

        assert bg_layer is not None
        assert bg_layer.is_background is True
        assert bg_layer.layer_type == "background"

        # Check the PNG
        bg_png = Image.open(tmp_path / session_id / Path(bg_layer.png_path).name).convert("RGBA")
        bg_arr = np.array(bg_png)

        # Right half should have original blue pixels and alpha=255
        right = bg_arr[:, 50:, :]
        assert np.all(right[:, :, 2] == 200), "Blue channel wrong in background"
        assert np.all(right[:, :, 3] == 255), "Right half should be fully opaque in background"

        # Left half (owned by object) should have alpha=0
        left = bg_arr[:, :50, :]
        assert np.all(left[:, :, 3] == 0), "Object-owned pixels should be transparent in background"

    def test_background_bbox_is_full_canvas(self, tmp_path):
        """Background layer bbox must be (0,0,w,h)."""
        from services.segmentation_service import _build_background_layer

        img_h, img_w = 50, 80
        image_np = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        ownership_map = np.full((img_h, img_w), -1, dtype=np.int32)
        # Leave some unclaimed
        ownership_map[10:40, 10:70] = 0

        session_id = uuid.uuid4().hex
        session_dir = tmp_path / session_id
        session_dir.mkdir()

        bg_layer = _build_background_layer(
            image_np, ownership_map, session_id, session_dir, img_w, img_h, 1
        )
        assert bg_layer is not None
        assert bg_layer.bbox.x == 0.0
        assert bg_layer.bbox.y == 0.0
        assert bg_layer.bbox.width == float(img_w)
        assert bg_layer.bbox.height == float(img_h)

    def test_no_background_when_fully_claimed(self, tmp_path):
        """If every pixel is claimed by objects, no background layer is created."""
        from services.segmentation_service import _build_background_layer

        img_h, img_w = 50, 50
        image_np = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        ownership_map = np.zeros((img_h, img_w), dtype=np.int32)  # all owned by obj 0

        session_id = uuid.uuid4().hex
        session_dir = tmp_path / session_id
        session_dir.mkdir()

        bg_layer = _build_background_layer(
            image_np, ownership_map, session_id, session_dir, img_w, img_h, 1
        )
        assert bg_layer is None


# ---------------------------------------------------------------------------
# 4. Detection degenerate-box / containment suppression
# ---------------------------------------------------------------------------

class TestDetectionFilters:

    def _make_bbox(self, x, y, w, h):
        return {"x": float(x), "y": float(y), "width": float(w), "height": float(h)}

    def _make_obj(self, label, x, y, w, h, score=0.9):
        return {"label": label, "score": score, "bbox": self._make_bbox(x, y, w, h)}

    def test_drops_near_full_canvas_box(self):
        from services.detection_service import _drop_redundant_objects
        img_w, img_h = 1000, 1000
        # 90% canvas box should be dropped
        objects = [self._make_obj("poster", 0, 0, 950, 950)]
        result = _drop_redundant_objects(objects, img_w, img_h)
        assert len(result) == 0, "Near-full-canvas box was not dropped"

    def test_keeps_small_box(self):
        from services.detection_service import _drop_redundant_objects
        img_w, img_h = 1000, 1000
        # 10% canvas box should be kept
        objects = [self._make_obj("shoe", 100, 100, 200, 200)]
        result = _drop_redundant_objects(objects, img_w, img_h)
        assert len(result) == 1, "Small valid box was incorrectly dropped"

    def test_drops_container_box_that_contains_small(self):
        from services.detection_service import _drop_redundant_objects
        img_w, img_h = 1000, 1000
        # Large box covers 70% and fully contains the small box
        small = self._make_obj("logo", 400, 400, 100, 100)
        large = self._make_obj("advertisement", 50, 50, 800, 800)
        result = _drop_redundant_objects([small, large], img_w, img_h)
        labels = [o["label"] for o in result]
        assert "logo" in labels, "Small logo was incorrectly removed"
        assert "advertisement" not in labels, "Large container was not dropped"

    def test_keeps_two_independent_objects(self):
        from services.detection_service import _drop_redundant_objects
        img_w, img_h = 1000, 1000
        obj_a = self._make_obj("shoe", 100, 200, 300, 300)
        obj_b = self._make_obj("text", 600, 50, 200, 100)
        result = _drop_redundant_objects([obj_a, obj_b], img_w, img_h)
        assert len(result) == 2, "Independent objects incorrectly filtered"

    def test_degenerate_threshold_is_85_percent(self):
        from services.detection_service import _is_degenerate_box
        img_w, img_h = 1000, 1000
        # 84% → not degenerate
        assert not _is_degenerate_box(self._make_bbox(0, 0, 916, 916), img_w, img_h)
        # 85% → degenerate
        assert _is_degenerate_box(self._make_bbox(0, 0, 922, 922), img_w, img_h)


# ---------------------------------------------------------------------------
# 5. merge_service compatibility with full-canvas layers (bbox.x == 0)
# ---------------------------------------------------------------------------

class TestMergeCompatibility:

    def test_full_canvas_layers_composite_correctly(self, tmp_path):
        """
        Create two full-canvas RGBA layers (red shoe, dark background) and
        verify that merge_layers composites them into the correct final image.
        """
        from services.merge_service import merge_layers
        from schemas import LayerData, BoundingBox

        img_w, img_h = 100, 100

        # Layer 1: background — uniform dark grey, fully opaque
        bg_arr = np.zeros((img_h, img_w, 4), dtype=np.uint8)
        bg_arr[:, :, :3] = 40  # dark grey RGB
        bg_arr[:, :, 3]  = 255
        bg_path = tmp_path / "bg_layer.png"
        Image.fromarray(bg_arr).save(str(bg_path))

        # Layer 2: foreground object — red rectangle in centre, transparent elsewhere
        fg_arr = np.zeros((img_h, img_w, 4), dtype=np.uint8)
        fg_arr[20:80, 20:80, 0] = 200  # red
        fg_arr[20:80, 20:80, 3] = 255  # opaque inside rect; 0 outside
        fg_path = tmp_path / "fg_layer.png"
        Image.fromarray(fg_arr).save(str(fg_path))

        full_bbox = BoundingBox(x=0.0, y=0.0, width=float(img_w), height=float(img_h))

        layers = [
            LayerData(
                id="bg", name="background",
                mask_path="/tmp/bg_mask.png", png_path=str(bg_path),
                bbox=full_bbox, z_index=0, visible=True, opacity=1.0,
                layer_type="background", is_background=True,
            ),
            LayerData(
                id="fg", name="shoe",
                mask_path="/tmp/fg_mask.png", png_path=str(fg_path),
                bbox=full_bbox, z_index=1, visible=True, opacity=1.0,
                layer_type="object", is_background=False,
            ),
        ]

        result = merge_layers(layers, img_w, img_h)
        result_arr = np.array(result.convert("RGB"))

        # Inside the object bbox: should be red
        inside = result_arr[20:80, 20:80]
        assert np.mean(inside[:, :, 0]) > 150, "Red object not visible after merge"

        # Outside the object bbox: should be background grey
        corner = result_arr[0, 0]
        assert int(corner[0]) == 40, f"Background grey wrong: {corner}"

    def test_position_offset_still_works(self, tmp_path):
        """
        Ensure that even with full-canvas layers, a non-zero position offset
        still shifts the composite correctly (merge_service adds bbox.x + position.x).
        """
        from services.merge_service import merge_layers
        from schemas import LayerData, BoundingBox

        img_w, img_h = 200, 200

        # A 20×20 red square in the top-left corner of the canvas
        arr = np.zeros((img_h, img_w, 4), dtype=np.uint8)
        arr[0:20, 0:20, 0] = 255  # red
        arr[0:20, 0:20, 3] = 255  # opaque
        layer_path = tmp_path / "shift_layer.png"
        Image.fromarray(arr).save(str(layer_path))

        # With a position offset of (50, 50) the red square should land at 50:70, 50:70
        layer = LayerData(
            id="shift", name="test",
            mask_path="/tmp/dummy.png", png_path=str(layer_path),
            bbox=BoundingBox(x=0.0, y=0.0, width=float(img_w), height=float(img_h)),
            z_index=1, visible=True, opacity=1.0,
            position={"x": 50.0, "y": 50.0},
        )

        result = merge_layers([layer], img_w, img_h)
        result_arr = np.array(result.convert("RGBA"))

        # Original position (0,0) should be transparent
        assert result_arr[5, 5, 3] == 0, "Pixel at original position should be transparent"
        # Shifted position should be red + opaque
        assert result_arr[55, 55, 0] == 255, "Shifted pixel should be red"
        assert result_arr[55, 55, 3] == 255, "Shifted pixel should be opaque"


# ---------------------------------------------------------------------------
# 6. Session metadata round-trip
# ---------------------------------------------------------------------------

class TestSessionMeta:

    def test_session_meta_contains_layer_type(self, tmp_path):
        """
        Simulate what segment_objects writes to session_meta.json and verify
        that layer_type and is_background survive the JSON round-trip.
        """
        from schemas import LayerData, BoundingBox

        layer = LayerData(
            id="abc123",
            name="shoe",
            mask_path="/temp/s/mask.png",
            png_path="/temp/s/layer.png",
            bbox=BoundingBox(x=0.0, y=0.0, width=800.0, height=800.0),
            z_index=1,
            layer_type="object",
            is_background=False,
        )
        bg_layer = LayerData(
            id="bg123",
            name="background",
            mask_path="/temp/s/bg_mask.png",
            png_path="/temp/s/bg_layer.png",
            bbox=BoundingBox(x=0.0, y=0.0, width=800.0, height=800.0),
            z_index=0,
            layer_type="background",
            is_background=True,
        )

        meta = {
            "session_id": "testsession",
            "image_path": "temp/testsession/img.jpg",
            "layers": [layer.model_dump(), bg_layer.model_dump()],
        }
        meta_path = tmp_path / "session_meta.json"
        meta_path.write_text(json.dumps(meta))

        loaded = json.loads(meta_path.read_text())
        layers_raw = loaded["layers"]

        obj_layer = next(l for l in layers_raw if not l["is_background"])
        bg_layer_raw = next(l for l in layers_raw if l["is_background"])

        assert obj_layer["layer_type"] == "object"
        assert obj_layer["is_background"] is False
        assert bg_layer_raw["layer_type"] == "background"
        assert bg_layer_raw["is_background"] is True

        # bbox must be full-canvas (x==0, y==0)
        assert obj_layer["bbox"]["x"] == 0.0
        assert obj_layer["bbox"]["y"] == 0.0
