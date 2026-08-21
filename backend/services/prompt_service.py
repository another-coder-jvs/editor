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

SYSTEM_PROMPT = """You are an image editing assistant. Extract the edit intent from the instruction.

IMPORTANT: If the instruction contains MULTIPLE separate edits (e.g. "change color to X AND add Y"), return a JSON array of edits, one per operation, in the order they should be applied.

Single edit format:
{"target_object": "...", "target_region": "...", "edit_type": "...", "edit_params": {}, "inpaint_prompt": "..."}

Multiple edits format:
[{"target_object": "...", "edit_type": "recolor", "edit_params": {"color": "yellow"}, ...}, {"target_object": "...", "edit_type": "other", "edit_params": {}, "inpaint_prompt": "add flower design on it", ...}]

Rules:
- If the instruction asks to change/recolor/make a color (e.g. "make blue", "change color to red") → edit_type = "recolor", edit_params = {"color": "<color>"}
- If the instruction asks to ADD something (e.g. "add flower design", "add pattern") → edit_type = "other", inpaint_prompt = the full add instruction
- For recolor, always normalize the color value to the closest standard English color word (e.g. "chocolaty" → "brown", "navy" → "blue", "scarlet" → "red")
- NEVER use style_transfer or other for simple color changes alone
- edit_type must be one of: recolor, replace, blur, sharpen, brightness, contrast, saturation, background_remove, generative_fill, erase, upscale, cartoon, anime, oil_painting, sketch, pixel_art, style_transfer, text_edit, other
- "add" / "add pattern" / "add design" / "draw" → use edit_type = "other" with inpaint_prompt describing what to add
- If the instruction asks to change/edit/replace text content or text color → edit_type = "text_edit", edit_params = {"new_text": "<new text>", "color": "<color if specified>", "target_text": "<original text if specified>"}

Respond ONLY with valid JSON (single object or array):"""


def _is_compound_prompt(prompt: str) -> bool:
    """Check if the prompt contains multiple separate edit instructions."""
    separators = [" and ", " also ", "; ", ", then ", ", and ", " plus "]
    prompt_lower = prompt.lower()
    for sep in separators:
        if sep in prompt_lower:
            # Check if both parts contain action words
            parts = prompt_lower.split(sep, 1)
            if len(parts) == 2 and len(parts[0].strip()) > 5 and len(parts[1].strip()) > 5:
                return True
    return False


def _split_compound_prompt(user_prompt: str) -> list:
    """Split a compound prompt into individual edit instructions."""
    separators = [" and ", " also ", "; ", ", then ", ", and ", " plus "]
    prompt_lower = user_prompt.lower()
    
    for sep in separators:
        if sep in prompt_lower:
            parts = user_prompt.split(sep, 1)
            if len(parts) == 2 and len(parts[0].strip()) > 5 and len(parts[1].strip()) > 5:
                return [parts[0].strip(), parts[1].strip()]
    
    return [user_prompt]


def parse_edit_prompt(user_prompt: str, layer_name: str) -> Dict[str, Any]:
    """Parse edit prompt, returning either a single edit or multiple edits.
    
    For compound prompts, returns:
    {"multi_edit": True, "edits": [{edit1}, {edit2}, ...]}
    
    For single edits, returns:
    {"edit_type": ..., "edit_params": ..., ...}
    """
    logger.info(f"[prompt] parsing: '{user_prompt}' for layer '{layer_name}'")
    
    # Check if this is a compound prompt
    is_compound = _is_compound_prompt(user_prompt)
    logger.info(f"[prompt] compound prompt detected: {is_compound}")
    
    # Try LLM first
    try:
        logger.info("[prompt] calling LLM…")
        llm = model_manager.get_llm()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Layer: {layer_name}\nUser instruction: {user_prompt}"},
        ]
        output = llm(messages, max_new_tokens=512, do_sample=False, max_length=None)
        raw = output[0]["generated_text"]
        if isinstance(raw, list):
            raw = raw[-1].get("content", "")
        logger.debug(f"[prompt] LLM raw output: {raw[:300]}")
        
        # Try to parse as array first (compound edit)
        array_match = re.search(r"\[\s*\{.*?\}\s*(?:,\s*\{.*?\}\s*)*\]", raw, re.DOTALL)
        if array_match:
            try:
                edits = json.loads(array_match.group())
                if isinstance(edits, list) and len(edits) > 1:
                    # Normalize each edit
                    for edit in edits:
                        _normalize_edit(edit, user_prompt)
                    logger.info(f"[prompt] LLM parsed {len(edits)} edits: {[e.get('edit_type') for e in edits]}")
                    return {"multi_edit": True, "edits": edits}
            except json.JSONDecodeError:
                pass
        
        # Try to parse as single object
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            _normalize_edit(result, user_prompt)
            
            # If LLM returned single edit but prompt is compound, try splitting
            if is_compound and not result.get("multi_edit"):
                logger.info("[prompt] LLM returned single edit for compound prompt, splitting...")
                return _split_and_parse_compound(user_prompt, layer_name, first_edit=result)
            
            logger.info(f"[prompt] LLM parsed: edit_type={result.get('edit_type')} params={result.get('edit_params')}")
            return result
        
        logger.warning("[prompt] LLM output had no JSON, falling back to heuristic")
    except Exception as e:
        logger.warning(f"[prompt] LLM unavailable ({e}), using heuristic fallback")
    
    # If LLM failed and prompt is compound, use heuristic splitting
    if is_compound:
        return _split_and_parse_compound(user_prompt, layer_name)
    
    result = _heuristic_parse(user_prompt, layer_name)
    logger.info(f"[prompt] heuristic result: edit_type={result['edit_type']} params={result['edit_params']}")
    return result


def _normalize_edit(edit: Dict[str, Any], user_prompt: str) -> None:
    """Normalize a single edit dict: fix color words, add inpaint_prompt for 'other' type, etc."""
    color_words = ["red","blue","green","black","white","yellow","orange","purple","pink","brown","gray","grey","navy","royal blue","teal","cyan","magenta","violet","indigo"]
    
    if edit.get("edit_type") not in ("recolor",):
        for c in color_words:
            if c in user_prompt.lower():
                edit["edit_type"] = "recolor"
                edit.setdefault("edit_params", {})["color"] = edit.get("edit_params", {}).get("color", c)
                break
    
    # For 'other' type, ensure inpaint_prompt is set
    if edit.get("edit_type") == "other" and not edit.get("inpaint_prompt"):
        edit["inpaint_prompt"] = user_prompt
    
    edit.setdefault("target_object", "")
    edit.setdefault("target_region", "whole")
    edit.setdefault("edit_params", {})
    edit.setdefault("inpaint_prompt", user_prompt)


def _split_and_parse_compound(
    user_prompt: str, 
    layer_name: str,
    first_edit: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Split a compound prompt and parse each part."""
    parts = _split_compound_prompt(user_prompt)
    logger.info(f"[prompt] split compound prompt into {len(parts)} parts: {parts}")
    
    edits = []
    for i, part in enumerate(parts):
        if i == 0 and first_edit:
            edits.append(first_edit)
        else:
            edit = _heuristic_parse(part, layer_name)
            edit["inpaint_prompt"] = part
            edits.append(edit)
    
    logger.info(f"[prompt] compound result: {len(edits)} edits: {[e.get('edit_type') for e in edits]}")
    return {"multi_edit": True, "edits": edits}


def _heuristic_parse(prompt: str, layer_name: str) -> Dict[str, Any]:
    p = prompt.lower()
    edit_type = "other"
    edit_params: Dict[str, Any] = {}

    color_words = ["red","blue","green","black","white","yellow","orange","purple","pink","brown","gray","grey","navy","teal","cyan","magenta","violet","indigo"]
    color_aliases = {"navy": "blue", "royal blue": "blue", "teal": "green", "indigo": "purple", "violet": "purple", "magenta": "pink", "cyan": "blue"}
    for color in color_words:
        if color in p:
            edit_type = "recolor"
            edit_params["color"] = color_aliases.get(color, color)
            break

    # Check for 'add' operations (generative fill / other)
    if any(w in p for w in ["add ", "draw ", "paint ", "create ", "insert ", "place ", "put "]):
        edit_type = "other"  # Will use inpainting
    elif any(w in p for w in ["replace", "change to", "convert", "swap"]):
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
    elif any(w in p for w in ["change text", "edit text", "replace text", "text color", "rewrite"]):
        edit_type = "text_edit"

    logger.debug(f"[prompt] heuristic: '{p}' → edit_type={edit_type} params={edit_params}")
    return {
        "target_object": layer_name,
        "target_region": "whole",
        "edit_type": edit_type,
        "edit_params": edit_params,
        "inpaint_prompt": prompt,
    }
