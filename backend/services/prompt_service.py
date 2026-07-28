"""
Prompt understanding service – LLM parses edit prompts, heuristic fallback.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict

from services.model_manager import model_manager

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an image editing assistant.
Given a user's edit instruction, extract:
- target_object: the main object being edited
- target_region: the specific part (or "whole")
- edit_type: one of [recolor, replace, style_transfer, blur, sharpen, brightness,
  contrast, saturation, background_remove, generative_fill, erase, upscale,
  cartoon, anime, oil_painting, sketch, pixel_art, expand, clone, other]
- edit_params: dict of relevant parameters (e.g. {"color": "blue"})
- inpaint_prompt: short positive prompt for the inpainting model
Respond ONLY with valid JSON."""


def parse_edit_prompt(user_prompt: str, layer_name: str) -> Dict[str, Any]:
    logger.info(f"[prompt] parsing: '{user_prompt}' for layer '{layer_name}'")
    try:
        logger.info("[prompt] calling LLM…")
        llm = model_manager.get_llm()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Layer: {layer_name}\nUser instruction: {user_prompt}"},
        ]
        output = llm(messages, max_new_tokens=256, do_sample=False)
        raw = output[0]["generated_text"]
        if isinstance(raw, list):
            raw = raw[-1].get("content", "")
        logger.debug(f"[prompt] LLM raw output: {raw[:200]}")
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            logger.info(f"[prompt] LLM parsed: edit_type={result.get('edit_type')} params={result.get('edit_params')}")
            return result
        logger.warning("[prompt] LLM output had no JSON, falling back to heuristic")
    except Exception as e:
        logger.warning(f"[prompt] LLM unavailable ({e}), using heuristic fallback")

    result = _heuristic_parse(user_prompt, layer_name)
    logger.info(f"[prompt] heuristic result: edit_type={result['edit_type']} params={result['edit_params']}")
    return result


def _heuristic_parse(prompt: str, layer_name: str) -> Dict[str, Any]:
    p = prompt.lower()
    edit_type = "other"
    edit_params: Dict[str, Any] = {}

    color_words = ["red","blue","green","black","white","yellow","orange","purple","pink","brown","gray","grey"]
    for color in color_words:
        if color in p:
            edit_type = "recolor"
            edit_params["color"] = color
            break

    if any(w in p for w in ["replace", "change to", "convert", "swap"]):
        edit_type = "replace"
    elif any(w in p for w in ["blur", "soften"]):
        edit_type = "blur"
    elif any(w in p for w in ["sharpen", "sharp"]):
        edit_type = "sharpen"
    elif "bright" in p or "lighten" in p:
        edit_type = "brightness"; edit_params["value"] = 1.3
    elif "dark" in p or "darken" in p:
        edit_type = "brightness"; edit_params["value"] = 0.7
    elif "cartoon" in p:
        edit_type = "cartoon"
    elif "anime" in p:
        edit_type = "anime"
    elif "oil" in p:
        edit_type = "oil_painting"
    elif "sketch" in p:
        edit_type = "sketch"
    elif "pixel" in p:
        edit_type = "pixel_art"
    elif any(w in p for w in ["remove background", "remove bg", "transparent"]):
        edit_type = "background_remove"
    elif any(w in p for w in ["erase", "delete", "remove"]):
        edit_type = "erase"

    logger.debug(f"[prompt] heuristic: '{p}' → edit_type={edit_type} params={edit_params}")
    return {
        "target_object": layer_name,
        "target_region": "whole",
        "edit_type": edit_type,
        "edit_params": edit_params,
        "inpaint_prompt": prompt,
    }
