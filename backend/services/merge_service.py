"""
Merge service – alpha-composites all visible layers into a single image.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

from schemas import LayerData

logger = logging.getLogger(__name__)

from utils import config


def _resolve(png_path: str) -> Path:
    """Convert a URL path like /temp/sid/file.png to an absolute filesystem path."""
    p = Path(png_path)
    print(f"THEE PAATH FROM RESOLVE : {p} ")
    if p.is_absolute() and p.exists():
        return p
    # strip leading slash and join with project root
    p = Path(png_path)
    if p.is_absolute(): return p
    # strip leading /temp/ and resolve against TEMP_DIR
    rel = png_path.lstrip("/").removeprefix("temp/").removeprefix("temp\\")
    return config.TEMP_DIR / rel


def merge_layers(layers: List[LayerData], canvas_width: int, canvas_height: int) -> Image.Image:
    logger.info(f"[merge] canvas={canvas_width}x{canvas_height} total_layers={len(layers)}")
    canvas = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))

    visible = [l for l in layers if l.visible]
    sorted_layers = sorted(visible, key=lambda l: l.z_index)
    logger.info(f"[merge] visible layers: {len(visible)} (hidden: {len(layers)-len(visible)})")

    for layer in sorted_layers:
        fs_path = _resolve(layer.png_path)
        logger.debug(f"[merge] compositing '{layer.name}' z={layer.z_index} path={fs_path}")
        try:
            layer_img = Image.open(fs_path).convert("RGBA")
        except Exception as e:
            logger.warning(f"[merge] could not open layer '{layer.name}' ({fs_path}): {e} — skipping")
            continue

        if layer.opacity < 1.0:
            r, g, b, a = layer_img.split()
            a = a.point(lambda p: int(p * layer.opacity))
            layer_img = Image.merge("RGBA", (r, g, b, a))

        if layer.rotation != 0:
            layer_img = layer_img.rotate(-layer.rotation, expand=True, resample=Image.BICUBIC)

        if layer.scale["x"] != 1.0 or layer.scale["y"] != 1.0:
            new_w = int(layer_img.width  * layer.scale["x"])
            new_h = int(layer_img.height * layer.scale["y"])
            layer_img = layer_img.resize((new_w, new_h), Image.LANCZOS)

        paste_x = int(layer.bbox.x + layer.position["x"])
        paste_y = int(layer.bbox.y + layer.position["y"])

        tmp = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))
        tmp.paste(layer_img, (paste_x, paste_y))
        canvas = Image.alpha_composite(canvas, tmp)

    logger.info(f"[merge] done — final canvas {canvas_width}x{canvas_height}")
    return canvas
