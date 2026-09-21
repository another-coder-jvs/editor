"""
Remote compute configuration — single source of truth for all cloud API settings.

Set COMPUTE_MODE=remote in your environment (or .env file) to activate.
Every key, endpoint, and provider preference lives here.

Usage:
    from services.remote_config import remote_cfg

    token = remote_cfg.fal_api_key
    url   = remote_cfg.ENDPOINTS["flux_inpaint"]
"""
from __future__ import annotations

import os
import logging

logger = logging.getLogger(__name__)


class RemoteConfig:
    """All cloud API credentials and endpoint URLs in one place."""

    # ── Compute mode ──────────────────────────────────────────────────────────
    # Set COMPUTE_MODE=local  → download + run models locally (default)
    # Set COMPUTE_MODE=remote → route every AI call to cloud APIs
    COMPUTE_MODE: str = os.environ.get("COMPUTE_MODE", "local").lower().strip()

    # ── Provider API keys ─────────────────────────────────────────────────────
    # fal.ai  — used for: SAM2 segmentation, FLUX inpainting, SDXL inpainting,
    #           SDXL img2img, rembg background removal, Real-ESRGAN upscaling
    FAL_API_KEY: str = os.environ.get("FAL_API_KEY", "")

    # Replicate — alternative for: Grounding DINO detection, SAM2, SDXL, ESRGAN
    REPLICATE_API_TOKEN: str = os.environ.get("REPLICATE_API_TOKEN", "")

    # Groq — used for: LLM prompt parsing (Llama 3.1-8B, ultra-fast, cheap)
    GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")

    # OpenAI — fallback LLM + optional vision for object identification
    OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")

    # Anthropic — alternative LLM (Claude) if preferred
    ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")

    # remove.bg — dedicated background removal API (higher quality than rembg)
    REMOVEBG_API_KEY: str = os.environ.get("REMOVEBG_API_KEY", "")

    # Stability AI — alternative inpainting / background remove / upscale
    STABILITY_API_KEY: str = os.environ.get("STABILITY_API_KEY", "")

    # ── Provider preference per task ─────────────────────────────────────────
    # Change these to swap which provider handles each task without touching code.
    #
    # detection_provider:    "replicate" | "fal"
    # segmentation_provider: "fal" | "replicate"
    # inpaint_provider:      "fal" | "replicate" | "stability"
    # llm_provider:          "groq" | "openai" | "anthropic"
    # rembg_provider:        "fal" | "removebg" | "stability"
    # upscale_provider:      "fal" | "replicate"
    # vision_provider:       "openai" | "anthropic" (for object identification)

    DETECTION_PROVIDER:    str = os.environ.get("DETECTION_PROVIDER",    "replicate")
    SEGMENTATION_PROVIDER: str = os.environ.get("SEGMENTATION_PROVIDER", "fal")
    INPAINT_PROVIDER:      str = os.environ.get("INPAINT_PROVIDER",      "fal")
    LLM_PROVIDER:          str = os.environ.get("LLM_PROVIDER",          "groq")
    REMBG_PROVIDER:        str = os.environ.get("REMBG_PROVIDER",        "fal")
    UPSCALE_PROVIDER:      str = os.environ.get("UPSCALE_PROVIDER",      "fal")
    VISION_PROVIDER:       str = os.environ.get("VISION_PROVIDER",       "openai")

    # ── API Endpoints ─────────────────────────────────────────────────────────
    ENDPOINTS: dict = {
        # --- Detection ---
        # Replicate: Grounding DINO (same model as local)
        "grounding_dino_replicate": "https://api.replicate.com/v1/models/adirik/grounding-dino/predictions",

        # --- Segmentation ---
        # fal.ai: SAM2 image segmentation
        "sam2_fal": "https://queue.fal.run/fal-ai/sam2/image",

        # Replicate: SAM2 (alternative)
        "sam2_replicate": "https://api.replicate.com/v1/models/meta/sam-2/predictions",

        # --- Inpainting / Generative edits ---
        # fal.ai: FLUX.1 Pro Fill (best quality, $0.05/MP)
        "flux_fill_fal": "https://queue.fal.run/fal-ai/flux-pro/v1/fill",

        # fal.ai: FLUX.1 Fill Dev (open-weight variant)
        "flux_fill_dev_fal": "https://queue.fal.run/fal-ai/flux/dev/image-to-image",

        # fal.ai: SDXL inpainting (fallback)
        "sdxl_inpaint_fal": "https://queue.fal.run/fal-ai/stable-diffusion-xl/inpainting",

        # fal.ai: SDXL img2img for style transfer
        "sdxl_img2img_fal": "https://queue.fal.run/fal-ai/stable-diffusion-xl",

        # Replicate: SDXL inpainting (alternative)
        "sdxl_inpaint_replicate": "https://api.replicate.com/v1/models/stability-ai/sdxl/predictions",

        # Stability AI: inpainting
        "inpaint_stability": "https://api.stability.ai/v2beta/stable-image/edit/inpaint",

        # --- LLM prompt parsing ---
        # Groq: Llama 3.1-8B (fastest, cheapest for JSON extraction)
        "groq_chat": "https://api.groq.com/openai/v1/chat/completions",

        # OpenAI: GPT-4o-mini (fallback LLM)
        "openai_chat": "https://api.openai.com/v1/chat/completions",

        # Anthropic: Claude (alternative LLM)
        "anthropic_chat": "https://api.anthropic.com/v1/messages",

        # --- Background removal ---
        # fal.ai: rembg (fast, cheap)
        "rembg_fal": "https://queue.fal.run/fal-ai/imageutils/rembg",

        # fal.ai: Bria RMBG 2.0 (higher accuracy)
        "bria_rmbg_fal": "https://queue.fal.run/fal-ai/bria/background/removal",

        # remove.bg: dedicated API (best for hair/fur)
        "removebg": "https://api.remove.bg/v1.0/removebg",

        # Stability AI: background remove
        "rembg_stability": "https://api.stability.ai/v2beta/stable-image/edit/remove-background",

        # --- Upscaling ---
        # fal.ai: Real-ESRGAN
        "esrgan_fal": "https://queue.fal.run/fal-ai/esrgan",

        # Replicate: Real-ESRGAN
        "esrgan_replicate": "https://api.replicate.com/v1/models/nightmareai/real-esrgan/predictions",

        # --- Vision (object identification, optional) ---
        # OpenAI: GPT-4o-mini with vision
        "vision_openai": "https://api.openai.com/v1/chat/completions",

        # Anthropic: Claude Haiku with vision
        "vision_anthropic": "https://api.anthropic.com/v1/messages",
    }

    # ── Model names for LLM providers ─────────────────────────────────────────
    LLM_MODELS: dict = {
        "groq":      os.environ.get("GROQ_MODEL",      "llama-3.1-8b-instant"),
        "openai":    os.environ.get("OPENAI_MODEL",    "gpt-4o-mini"),
        "anthropic": os.environ.get("ANTHROPIC_MODEL", "claude-haiku-3-5"),
    }

    VISION_MODELS: dict = {
        "openai":    os.environ.get("OPENAI_VISION_MODEL",    "gpt-4o-mini"),
        "anthropic": os.environ.get("ANTHROPIC_VISION_MODEL", "claude-haiku-3-5"),
    }

    # ── Timeouts & retries ────────────────────────────────────────────────────
    # Seconds to wait for a synchronous API response
    HTTP_TIMEOUT:         int = int(os.environ.get("REMOTE_HTTP_TIMEOUT",  "120"))
    # Seconds between polling for async queue-based APIs (fal.ai, Replicate)
    POLL_INTERVAL:      float = float(os.environ.get("REMOTE_POLL_INTERVAL", "2.0"))
    # Max seconds to wait for an async job before giving up
    ASYNC_TIMEOUT:        int = int(os.environ.get("REMOTE_ASYNC_TIMEOUT", "300"))

    def validate(self) -> None:
        """Warn if COMPUTE_MODE=remote but required keys are missing."""
        if self.COMPUTE_MODE != "remote":
            return

        missing = []

        if self.DETECTION_PROVIDER == "replicate" and not self.REPLICATE_API_TOKEN:
            missing.append("REPLICATE_API_TOKEN (needed for detection_provider=replicate)")
        if self.SEGMENTATION_PROVIDER == "fal" and not self.FAL_API_KEY:
            missing.append("FAL_API_KEY (needed for segmentation_provider=fal)")
        if self.INPAINT_PROVIDER == "fal" and not self.FAL_API_KEY:
            missing.append("FAL_API_KEY (needed for inpaint_provider=fal)")
        if self.INPAINT_PROVIDER == "stability" and not self.STABILITY_API_KEY:
            missing.append("STABILITY_API_KEY (needed for inpaint_provider=stability)")
        if self.LLM_PROVIDER == "groq" and not self.GROQ_API_KEY:
            missing.append("GROQ_API_KEY (needed for llm_provider=groq)")
        if self.LLM_PROVIDER == "openai" and not self.OPENAI_API_KEY:
            missing.append("OPENAI_API_KEY (needed for llm_provider=openai)")
        if self.REMBG_PROVIDER == "fal" and not self.FAL_API_KEY:
            missing.append("FAL_API_KEY (needed for rembg_provider=fal)")
        if self.REMBG_PROVIDER == "removebg" and not self.REMOVEBG_API_KEY:
            missing.append("REMOVEBG_API_KEY (needed for rembg_provider=removebg)")
        if self.UPSCALE_PROVIDER == "fal" and not self.FAL_API_KEY:
            missing.append("FAL_API_KEY (needed for upscale_provider=fal)")

        if missing:
            logger.warning(
                "[remote_config] COMPUTE_MODE=remote but the following env vars are not set:\n"
                + "\n".join(f"  - {m}" for m in missing)
            )


# Singleton
remote_cfg = RemoteConfig()
remote_cfg.validate()

logger.info(f"[remote_config] COMPUTE_MODE={remote_cfg.COMPUTE_MODE}")
