"""
Remote vision service — replaces local Ollama + llava:7b for object identification.
Supports: OpenAI (GPT-4o-mini vision), Anthropic (Claude Haiku vision)
"""
from __future__ import annotations

import base64
import logging
from pathlib import Path

import requests

from services.remote_config import remote_cfg

logger = logging.getLogger(__name__)

_VISION_PROMPT = (
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
)


def identify_objects_remote(image_path: str) -> str:
    """
    Identify objects in an image using a remote vision LLM.
    Returns a comma-separated string like "person, chair, lamp".
    """
    provider = remote_cfg.VISION_PROVIDER
    logger.info(f"[remote/vision] provider={provider} image={image_path}")

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    ext = Path(image_path).suffix.lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"

    if provider == "openai":
        return _identify_openai(b64, mime)
    elif provider == "anthropic":
        return _identify_anthropic(b64, mime)
    else:
        raise ValueError(f"Unknown vision provider: {provider!r}. Choose 'openai' or 'anthropic'.")


def _identify_openai(b64: str, mime: str) -> str:
    model = remote_cfg.VISION_MODELS["openai"]
    headers = {
        "Authorization": f"Bearer {remote_cfg.OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 100,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _VISION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                ],
            }
        ],
    }
    resp = requests.post(
        remote_cfg.ENDPOINTS["vision_openai"],
        json=payload,
        headers=headers,
        timeout=remote_cfg.HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"].strip()
    logger.info(f"[remote/vision] openai result: '{text}'")
    return text


def _identify_anthropic(b64: str, mime: str) -> str:
    model = remote_cfg.VISION_MODELS["anthropic"]
    headers = {
        "x-api-key": remote_cfg.ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 100,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime,
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": _VISION_PROMPT},
                ],
            }
        ],
    }
    resp = requests.post(
        remote_cfg.ENDPOINTS["vision_anthropic"],
        json=payload,
        headers=headers,
        timeout=remote_cfg.HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    text = resp.json()["content"][0]["text"].strip()
    logger.info(f"[remote/vision] anthropic result: '{text}'")
    return text
