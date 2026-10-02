"""
Vision identification service – uses Ollama + minicpm-v4.6 to identify
the relevant objects in an image, producing a focused prompt for Grounding DINO.

Instead of detecting 50+ hardcoded categories, we ask the vision model what's
actually in the image, then feed only those objects to Grounding DINO.
This saves VRAM and improves accuracy.
"""
from __future__ import annotations

import json
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


def _ollama_api(method: str, payload: dict) -> dict:
    """
    Call the local Ollama HTTP API directly.

    This is used when the `ollama` Python package is not installed in the
    runtime, but the Ollama server is still available (for example when the
    server was started with `ollama serve` from the shell).
    """
    try:
        import requests
    except Exception as e:
        raise RuntimeError(f"Cannot reach Ollama API: requests unavailable ({e})") from e

    url = f"{OLLAMA_URL}/api/{method}"
    logger.debug(f"[identify] Ollama API {method}: {url}")
    response = requests.post(url, json=payload, timeout=30)
    try:
        response.raise_for_status()
    except Exception as e:
        logger.error(f"[identify] Ollama API {method} failed: {response.status_code} {response.text}")
        raise RuntimeError(f"Ollama API {method} failed: {response.status_code}") from e
    return response.json()


def _ollama_has_model(model_name: str) -> bool:
    """
    Check whether a model is present using the Ollama API.

    Works with or without the `ollama` Python package.
    """
    try:
        # Prefer the Python package when it exists and is usable
        try:
            import ollama as ollama_pkg

            result = ollama_pkg.list()
            models = result.get("models", [])
            for model in models:
                if str(model.get("model", "")).startswith(model_name):
                    return True
            return False
        except Exception as pkg_error:
            logger.debug(f"[identify] ollama package check failed ({pkg_error}), using HTTP API")

        # Fallback: use the Ollama HTTP API directly
        data = _ollama_api("list", {})
        models = data.get("models", [])
        for model in models:
            if str(model.get("model", "")).startswith(model_name):
                return True
        return False
    except Exception as e:
        logger.warning(f"[identify] Could not check models: {e}")
        return False


def check_model() -> bool:
    """Check if the vision model is available."""
    return _ollama_has_model(VISION_MODEL)


def download_model() -> None:
    """Download the vision model if not present."""
    logger.info(f"[identify] Downloading {VISION_MODEL}...")

    # Prefer the Python package when it exists
    try:
        import ollama as ollama_pkg
        ollama_pkg.pull(VISION_MODEL)
        logger.info(f"[identify] {VISION_MODEL} ready")
        return
    except Exception as pkg_error:
        logger.debug(f"[identify] ollama package pull failed ({pkg_error}), using HTTP API")

    # Fallback: stream the pull via the Ollama HTTP API
    try:
        import requests

        url = f"{OLLAMA_URL}/api/pull"
        logger.debug(f"[identify] Ollama pull API: {url}")
        with requests.post(url, json={"name": VISION_MODEL}, stream=True, timeout=300) as response:
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except Exception:
                    # Some lines may not be valid JSON; ignore them
                    continue
                status = chunk.get("status")
                digest = chunk.get("digest")
                progress = chunk.get("progress") or ""
                total = chunk.get("total")
                if status == "success":
                    logger.info(f"[identify] {VISION_MODEL} ready")
                    return
                if status and status not in {"downloading", "extracting"}:
                    logger.info(f"[identify] Ollama pull status: {status} {digest} {progress}/{total}")
                    continue
                if digest and progress is not None and total:
                    logger.info(f"[identify] Ollama pull: {digest} {progress}/{total}")

        logger.warning(f"[identify] Ollama pull completed without explicit success for {VISION_MODEL}")
    except Exception as e:
        logger.error(f"[identify] Model download failed: {e}")
        raise


def identify_objects(image_path: str) -> str:
    """
    Use Ollama vision model to identify objects in an image.

    Returns a text response from the vision model.
    This is parsed by `_clean_ollama_prompt()` in detection_service.py into a
    dot-separated Grounding DINO prompt.
    """
    if not check_ollama():
        raise RuntimeError(
            "Ollama is not running. Start it with: ollama serve"
        )

    if not check_model():
        download_model()

    logger.info(f"[identify] Analyzing image: {image_path}")

    from pathlib import Path
    import base64

    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with open(img_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    logger.info(f"[identify] Image loaded ({img_path.stat().st_size // 1024}KB), sending to {VISION_MODEL}...")

    payload = {
        "model": VISION_MODEL,
        "messages": [
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
        "options": {
            "temperature": 0,
            "num_ctx": 4096,
            "num_predict": 100,
            "repeat_penalty": 1.5,
            "repeat_last_n": 64,
        },
        "keep_alive": "30m",
    }

    response = _ollama_api("chat", payload)
    
    logger.info(f"[identify] Full Ollama response: {response}")
    
    answer = response.get("message", {}).get("content", "").strip()
    logger.info(f"[identify] Vision model output: '{answer}'")
    
    if not answer:
        logger.warning("[identify] Vision model returned empty response")
        return ""
    
    return answer
