"""
Vision identification service – uses Ollama + minicpm-v4.6 to identify
the relevant objects in an image, producing a focused prompt for Grounding DINO.

Instead of detecting 50+ hardcoded categories, we ask the vision model what's
actually in the image, then feed only those objects to Grounding DINO.
This saves VRAM and improves accuracy.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
VISION_MODEL = os.environ.get("VISION_MODEL", "llava:7b")  # or bakllava, llava:13b


def _try_start_ollama() -> bool:
    """Try to start Ollama server if installed but not running."""
    import shutil
    import subprocess
    if not shutil.which("ollama"):
        return False
    logger.info("[identify] Ollama installed but not running, starting it...")
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        import time
        time.sleep(3)  # wait for server to start
        import requests
        r = requests.get(f"{OLLAMA_URL}/api/version", timeout=5)
        if r.status_code == 200:
            logger.info("[identify] Ollama started successfully")
            return True
    except Exception as e:
        logger.warning(f"[identify] Failed to start Ollama: {e}")
    return False


def check_ollama() -> bool:
    """Check if Ollama server is running, try to start if installed."""
    try:
        import requests
        r = requests.get(f"{OLLAMA_URL}/api/version", timeout=5)
        if r.status_code == 200:
            return True
    except Exception:
        pass
    # Not running — try to start it
    return _try_start_ollama()


def check_model() -> bool:
    """Check if the vision model is available."""
    try:
        import ollama
        result = ollama.list()
        models = result.get("models", [])
        for model in models:
            name = model.get("model", "")
            if name.startswith(VISION_MODEL):
                return True
    except Exception as e:
        logger.warning(f"[identify] Could not check models: {e}")
    return False


def download_model() -> None:
    """Download the vision model if not present."""
    logger.info(f"[identify] Downloading {VISION_MODEL}...")
    try:
        import ollama
        ollama.pull(VISION_MODEL)
        logger.info(f"[identify] {VISION_MODEL} ready")
    except Exception as e:
        logger.error(f"[identify] Model download failed: {e}")
        raise


def identify_objects(image_path: str) -> str:
    """
    Use Ollama vision model to identify objects in an image.
    
    Returns a dot-separated string like: "person. shoe. pillow. flower."
    This is used as the prompt for Grounding DINO.
    """
    if not check_ollama():
        raise RuntimeError(
            "Ollama is not running. Start it with: ollama serve"
        )
    
    if not check_model():
        download_model()
    
    logger.info(f"[identify] Analyzing image: {image_path}")
    
    import ollama
    import base64
    from pathlib import Path
    
    # Read image as base64 (more reliable than file path across systems)
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    with open(img_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    
    logger.info(f"[identify] Image loaded ({img_path.stat().st_size // 1024}KB), sending to {VISION_MODEL}...")
    
    response = ollama.chat(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": (
                    "Identify the distinct physical objects in this image.\n\n"
                    "Output format: single-word labels separated by commas.\n"
                    "Example output: shoe, chair, lamp, bottle\n\n"
                    "Rules:\n"
                    "- Each object gets exactly one word\n"
                    "- Use the most specific common name (sneaker → shoe, sofa → couch)\n"
                    "- Do not include: text, letters, numbers, words written on objects\n"
                    "- Do not include: background, wall, floor, ceiling, sky\n"
                    "- Do not include: parts of objects (handle, leg, wheel, button)\n"
                    "- Do not include: colors, textures, patterns, shadows\n"
                    "- Maximum 10 objects, most prominent first"
                ),
                "images": [img_b64],
            }
        ],
        options={
            "temperature": 0,
            "num_ctx": 4096,
            "num_predict": 100,
            "repeat_penalty": 1.5,
            "repeat_last_n": 64,
        },
        keep_alive="30m",
    )
    
    logger.info(f"[identify] Full Ollama response: {response}")
    
    answer = response.get("message", {}).get("content", "").strip()
    logger.info(f"[identify] Vision model output: '{answer}'")
    
    if not answer:
        logger.warning("[identify] Vision model returned empty response")
        return ""
    
    return answer
