"""
Shared HTTP helpers for remote service calls.
Handles fal.ai queue polling, Replicate polling, and plain REST calls.
"""
from __future__ import annotations

import base64
import logging
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from services.remote_config import remote_cfg

logger = logging.getLogger(__name__)


# ── Encoding helpers ──────────────────────────────────────────────────────────

def image_to_data_uri(img: Image.Image, fmt: str = "PNG") -> str:
    """Convert a PIL image to a base64 data URI."""
    buf = BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode()
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def path_to_data_uri(path: str | Path, fmt: str = "PNG") -> str:
    """Read an image file and return as base64 data URI."""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    mime = "image/png" if str(path).lower().endswith(".png") else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def bytes_to_pil(data: bytes) -> Image.Image:
    return Image.open(BytesIO(data)).convert("RGBA")


# ── fal.ai queue-based calls ──────────────────────────────────────────────────

def fal_submit(endpoint: str, payload: dict) -> str:
    """Submit a job to a fal.ai queue endpoint. Returns the request_id."""
    headers = {
        "Authorization": f"Key {remote_cfg.FAL_API_KEY}",
        "Content-Type": "application/json",
    }
    resp = requests.post(endpoint, json=payload, headers=headers, timeout=remote_cfg.HTTP_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    request_id = data.get("request_id") or data.get("id")
    if not request_id:
        raise RuntimeError(f"fal.ai submit returned no request_id: {data}")
    logger.debug(f"[fal] submitted request_id={request_id}")
    return request_id


def fal_poll(endpoint_base: str, request_id: str) -> dict:
    """Poll a fal.ai queue until complete. Returns the result dict."""
    status_url = f"{endpoint_base}/requests/{request_id}/status"
    result_url = f"{endpoint_base}/requests/{request_id}"
    headers = {"Authorization": f"Key {remote_cfg.FAL_API_KEY}"}

    deadline = time.time() + remote_cfg.ASYNC_TIMEOUT
    while time.time() < deadline:
        resp = requests.get(status_url, headers=headers, timeout=30)
        resp.raise_for_status()
        status = resp.json().get("status", "")
        logger.debug(f"[fal] request_id={request_id} status={status}")
        if status in ("COMPLETED", "completed"):
            result = requests.get(result_url, headers=headers, timeout=30)
            result.raise_for_status()
            return result.json()
        if status in ("FAILED", "failed", "ERROR", "error"):
            raise RuntimeError(f"fal.ai job failed: {resp.json()}")
        time.sleep(remote_cfg.POLL_INTERVAL)

    raise TimeoutError(f"fal.ai job timed out after {remote_cfg.ASYNC_TIMEOUT}s")


def fal_call(endpoint: str, payload: dict) -> dict:
    """
    High-level fal.ai call: submit → poll → return result.
    endpoint should be the queue URL (e.g. https://queue.fal.run/fal-ai/esrgan).
    """
    # Derive the base URL for status/result polling
    # queue.fal.run/fal-ai/esrgan  →  base = queue.fal.run/fal-ai/esrgan
    request_id = fal_submit(endpoint, payload)
    return fal_poll(endpoint, request_id)


def fal_image_result(result: dict) -> Image.Image:
    """Extract the first image from a fal.ai result dict."""
    # fal returns {"images": [{"url": "...", ...}]} or {"image": {"url": "..."}}
    images = result.get("images") or [result.get("image")]
    if not images or not images[0]:
        raise RuntimeError(f"No image in fal.ai result: {result}")
    img_data = images[0]
    url = img_data.get("url") if isinstance(img_data, dict) else img_data
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGBA")


# ── Replicate polling ─────────────────────────────────────────────────────────

def replicate_call(endpoint: str, payload: dict) -> dict:
    """Submit a Replicate prediction and poll until done."""
    headers = {
        "Authorization": f"Bearer {remote_cfg.REPLICATE_API_TOKEN}",
        "Content-Type": "application/json",
        "Prefer": "wait",  # ask for synchronous response (up to 60s)
    }
    resp = requests.post(endpoint, json=payload, headers=headers, timeout=remote_cfg.HTTP_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    # If we got a result straight away (Prefer: wait worked)
    if data.get("status") == "succeeded":
        return data

    prediction_url = data.get("urls", {}).get("get") or data.get("url")
    if not prediction_url:
        raise RuntimeError(f"Replicate returned no polling URL: {data}")

    deadline = time.time() + remote_cfg.ASYNC_TIMEOUT
    poll_headers = {"Authorization": f"Bearer {remote_cfg.REPLICATE_API_TOKEN}"}
    while time.time() < deadline:
        r = requests.get(prediction_url, headers=poll_headers, timeout=30)
        r.raise_for_status()
        d = r.json()
        status = d.get("status", "")
        logger.debug(f"[replicate] status={status}")
        if status == "succeeded":
            return d
        if status in ("failed", "canceled"):
            raise RuntimeError(f"Replicate prediction failed: {d.get('error')}")
        time.sleep(remote_cfg.POLL_INTERVAL)

    raise TimeoutError(f"Replicate prediction timed out after {remote_cfg.ASYNC_TIMEOUT}s")


def replicate_image_result(data: dict) -> Image.Image:
    """Download the first output image from a Replicate result."""
    output = data.get("output")
    if isinstance(output, list):
        url = output[0]
    elif isinstance(output, str):
        url = output
    else:
        raise RuntimeError(f"Unexpected Replicate output format: {output}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGBA")
