"""
Remote LLM service — replaces local Qwen/Llama for prompt parsing.
Supports: Groq (Llama 3.1-8B), OpenAI (GPT-4o-mini), Anthropic (Claude Haiku)

Returns the same text-generation pipeline interface that prompt_service.py
uses: a callable that accepts messages and returns [{generated_text: ...}].
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

import requests

from services.remote_config import remote_cfg

logger = logging.getLogger(__name__)


def call_llm_remote(messages: List[Dict[str, str]], max_new_tokens: int = 512) -> str:
    """
    Call the configured remote LLM with a list of chat messages.
    Returns the assistant's text response.
    """
    provider = remote_cfg.LLM_PROVIDER
    logger.info(f"[remote/llm] provider={provider} max_tokens={max_new_tokens}")

    if provider == "groq":
        return _call_groq(messages, max_new_tokens)
    elif provider == "openai":
        return _call_openai(messages, max_new_tokens)
    elif provider == "anthropic":
        return _call_anthropic(messages, max_new_tokens)
    else:
        raise ValueError(f"Unknown LLM provider: {provider!r}. Choose 'groq', 'openai', or 'anthropic'.")


def _call_groq(messages: List[Dict], max_tokens: int) -> str:
    model = remote_cfg.LLM_MODELS["groq"]
    headers = {
        "Authorization": f"Bearer {remote_cfg.GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "response_format": {"type": "json_object"},  # force JSON output
    }
    resp = requests.post(
        remote_cfg.ENDPOINTS["groq_chat"],
        json=payload,
        headers=headers,
        timeout=remote_cfg.HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    logger.debug(f"[remote/llm] groq response: {text[:200]}")
    return text


def _call_openai(messages: List[Dict], max_tokens: int) -> str:
    model = remote_cfg.LLM_MODELS["openai"]
    headers = {
        "Authorization": f"Bearer {remote_cfg.OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    resp = requests.post(
        remote_cfg.ENDPOINTS["openai_chat"],
        json=payload,
        headers=headers,
        timeout=remote_cfg.HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    logger.debug(f"[remote/llm] openai response: {text[:200]}")
    return text


def _call_anthropic(messages: List[Dict], max_tokens: int) -> str:
    model = remote_cfg.LLM_MODELS["anthropic"]
    # Anthropic expects system message separately
    system_msg = ""
    user_messages = []
    for m in messages:
        if m["role"] == "system":
            system_msg = m["content"]
        else:
            user_messages.append(m)

    headers = {
        "x-api-key": remote_cfg.ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "messages": user_messages,
    }
    if system_msg:
        payload["system"] = system_msg

    resp = requests.post(
        remote_cfg.ENDPOINTS["anthropic_chat"],
        json=payload,
        headers=headers,
        timeout=remote_cfg.HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    text = data["content"][0]["text"]
    logger.debug(f"[remote/llm] anthropic response: {text[:200]}")
    return text


class RemoteLLMPipeline:
    """
    Wraps remote LLM calls with the same interface as the local HuggingFace
    pipeline, so prompt_service.py can call it identically:

        output = llm(messages, max_new_tokens=512, do_sample=False)
        raw = output[0]["generated_text"]
    """

    def __call__(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        **kwargs,
    ) -> List[Dict]:
        text = call_llm_remote(messages, max_new_tokens)
        # Mirror the HF pipeline output format
        return [{"generated_text": messages + [{"role": "assistant", "content": text}]}]
