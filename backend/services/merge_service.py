"""
Merge service – alpha-composites all visible layers into a single image.
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np
from PIL import Image

from schemas import LayerData

logger = logging.getLogger(__name__)


def merge_layers(layers: List[LayerData], canvas_width: int, canvas_height: int) -> Image.Image:
    logger.info(f"[merge] canvas={canvas_width}x{canvas_height} total_layers={len(layers)}")
    canvas = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))

    visible = [l for l in layers if l.visible]
    sorted_layers = sorted(visible, key=lambda l: l.z_index)
    logger.info(f"[merge] visible layers: {len(visible)} (hidden: {len(layers)-len(visible)})")

    for layer in sorted_layers:
        logger.debug(f"[merge] compositing layer '{layer.name}' z={layer.z_index} opacity={layer.opacity} path={layer.png_path}")
        try:
            layer_img = Image.open(layer.png_path).convert("RGBA")
        except Exception as e:
            logger.warning(f"[merge] could not open layer '{layer.name}' ({layer.png_path}): {e} — skipping")
            continue

        if layer.opacity < 1.0:
            r, g, b, a = layer_img.split()
            a = a.point(lambda p: int(p * layer.opacity))
            layer_img = Image.merge("RGBA", (r, g, b, a))
            logger.debug(f"[merge] applied opacity {layer.opacity} to '{layer.name}'")

        if layer.rotation != 0:
            layer_img = layer_img.rotate(-layer.rotation, expand=True, resample=Image.BICUBIC)
            logger.debug(f"[merge] rotated '{layer.name}' by {layer.rotation}°")

        if layer.scale["x"] != 1.0 or layer.scale["y"] != 1.0:
            new_w = int(layer_img.width  * layer.scale["x"])
            new_h = int(layer_img.height * layer.scale["y"])
            layer_img = layer_img.resize((new_w, new_h), Image.LANCZOS)
            logger.debug(f"[merge] scaled '{layer.name}' to {new_w}x{new_h}")

        paste_x = int(layer.bbox.x + layer.position["x"])
        paste_y = int(layer.bbox.y + layer.position["y"])
        logger.debug(f"[merge] pasting '{layer.name}' at ({paste_x}, {paste_y})")

        tmp = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))
        tmp.paste(layer_img, (paste_x, paste_y))
        canvas = Image.alpha_composite(canvas, tmp)

    logger.info(f"[merge] done — final canvas {canvas_width}x{canvas_height}")
    return canvas
