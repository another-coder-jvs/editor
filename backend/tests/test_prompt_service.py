"""Unit tests for prompt_service heuristic fallback."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import patch
from services.prompt_service import _heuristic_parse


def test_recolor_blue():
    result = _heuristic_parse("make the shirt blue", "person")
    assert result["edit_type"] == "recolor"
    assert result["edit_params"]["color"] == "blue"


def test_blur():
    result = _heuristic_parse("blur the background", "background")
    assert result["edit_type"] == "blur"


def test_replace():
    result = _heuristic_parse("replace car with Ferrari", "car")
    assert result["edit_type"] == "replace"


def test_cartoon():
    result = _heuristic_parse("make it cartoon style", "person")
    assert result["edit_type"] == "cartoon"


def test_erase():
    result = _heuristic_parse("erase this object", "tree")
    assert result["edit_type"] == "erase"


def test_brightness_up():
    result = _heuristic_parse("brighten the sky", "sky")
    assert result["edit_type"] == "brightness"
    assert result["edit_params"]["value"] > 1.0


def test_brightness_down():
    result = _heuristic_parse("darken the background", "background")
    assert result["edit_type"] == "brightness"
    assert result["edit_params"]["value"] < 1.0
