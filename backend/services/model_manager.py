"""
Central model manager – downloads, caches, and provides access to all AI models.
Includes idle-unload: all heavy models are freed after IDLE_SECONDS of inactivity.

COMPUTE_MODE switch
-------------------
Set the environment variable COMPUTE_MODE=remote to route all AI calls to cloud
APIs instead of running models locally.  Set COMPUTE_MODE=local (default) to
keep the existing behaviour: download weights and run on your GPU/CPU.

All remote credentials and endpoint URLs live in:
    backend/services/remote_config.py   ← single config file
"""
from __future__ import annotations

import gc
import logging
import os
import time
import threading
from pathlib import Path
from typing import Optional
from diffusers import StableDiffusionXLImg2ImgPipeline

import torch

logger = logging.getLogger(__name__)

from utils import config
from services.remote_config import remote_cfg  # COMPUTE_MODE + all remote settings

WEIGHTS_DIR = config.WEIGHT_DIR
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32

# Idle timeout: unload all models after this many seconds of no activity
IDLE_SECONDS = int(os.environ.get("MODEL_IDLE_SECONDS", "120"))  # default 2 min

# Convenience flag checked throughout this file
_REMOTE = remote_cfg.COMPUTE_MODE == "remote"

logger.info(f"[model_manager] COMPUTE_MODE={remote_cfg.COMPUTE_MODE}"
            + (" — all AI calls routed to cloud APIs" if _REMOTE else " — running models locally"))


def _from_pretrained_with_fallback(cls, repo_id: str, **kwargs):
    """Try loading from HF (online), fall back to local snapshot folder if auth/network fails."""
    try:
        logger.info(f"[model_manager] Loading {repo_id} from HF (online)…")
        return cls.from_pretrained(repo_id, **kwargs)
    except Exception as e:
        logger.warning(f"[model_manager] Online load failed ({e}), retrying local cache…")
        # Try direct snapshot folder to bypass incomplete-snapshot check
        cache_dir = kwargs.get("cache_dir")
        if cache_dir:
            from pathlib import Path
            snapshots_dir = Path(cache_dir)
            # Find snapshot folders: cache_dir/models--*/snapshots/*/
            candidates = sorted(snapshots_dir.glob("models--*/snapshots/*/"), reverse=True)
            for snap in candidates:
                if snap.is_dir() and (snap / "unet").exists():
                    logger.info(f"[model_manager] Loading from local snapshot: {snap}")
                    kw = {k: v for k, v in kwargs.items() if k != "cache_dir"}
                    return cls.from_pretrained(str(snap), **kw)
        return cls.from_pretrained(repo_id, local_files_only=True, **kwargs)

logger.info(f"[model_manager] device={DEVICE} dtype={DTYPE}")
logger.info(f"[model_manager] weights dir={WEIGHTS_DIR}")
if DEVICE == "cuda":
    logger.info(f"[model_manager] GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**2} MB")


class ModelManager:
    """Singleton that lazily loads and caches every AI model.
    All models auto-unload after IDLE_SECONDS of inactivity.
    Call touch() on every model access to reset the timer.
    """

    _instance: Optional["ModelManager"] = None

    def __new__(cls) -> "ModelManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._grounding_dino = None
        self._grounding_dino_processor = None
        self._sam2_predictor = None
        self._inpaint_pipe = None
        self._llm_pipe = None
        self._rembg_session = None
        self._realesrgan = None
        self._img2img_pipe = None
        self._flux_fill_pipe = None
        self._last_used = time.time()
        self._idle_timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()
        logger.info(f"[model_manager] ModelManager singleton initialized (all models lazy, idle unload={IDLE_SECONDS}s)")

    # ── Idle-unload timer ───────────────────────────────────────────────────────
    def touch(self) -> None:
        """Reset the idle timer. Call after every model access."""
        with self._lock:
            self._last_used = time.time()
            if self._idle_timer is not None:
                self._idle_timer.cancel()
            self._idle_timer = threading.Timer(IDLE_SECONDS, self._idle_unload)
            self._idle_timer.daemon = True
            self._idle_timer.start()

    def _idle_unload(self) -> None:
        """Unload all models if idle for IDLE_SECONDS."""
        with self._lock:
            elapsed = time.time() - self._last_used
            if elapsed < IDLE_SECONDS:
                return  # timer was reset, skip
            logger.info(f"[model_manager] Idle {elapsed:.0f}s > {IDLE_SECONDS}s — unloading all models")
            self.unload_all()

    def unload_all(self) -> None:
        """Unload every model to free RAM + VRAM."""
        with self._lock:
            unloaded = []
            if self._inpaint_pipe is not None:
                del self._inpaint_pipe; self._inpaint_pipe = None; unloaded.append("inpaint")
            if self._flux_fill_pipe is not None:
                del self._flux_fill_pipe; self._flux_fill_pipe = None; unloaded.append("flux_fill")
            if self._img2img_pipe is not None:
                del self._img2img_pipe; self._img2img_pipe = None; unloaded.append("img2img")
            if self._grounding_dino is not None:
                del self._grounding_dino; self._grounding_dino = None; unloaded.append("grounding_dino")
            if self._grounding_dino_processor is not None:
                del self._grounding_dino_processor; self._grounding_dino_processor = None; unloaded.append("gdino_processor")
            if self._sam2_predictor is not None:
                del self._sam2_predictor; self._sam2_predictor = None; unloaded.append("sam2")
            if self._llm_pipe is not None:
                del self._llm_pipe; self._llm_pipe = None; unloaded.append("llm")
            if self._rembg_session is not None:
                del self._rembg_session; self._rembg_session = None; unloaded.append("rembg")
            if self._realesrgan is not None:
                del self._realesrgan; self._realesrgan = None; unloaded.append("realesrgan")
            if unloaded:
                gc.collect()
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                logger.info(f"[model_manager] Unloaded: {unloaded} — VRAM/RAM freed")
            else:
                logger.debug("[model_manager] Nothing to unload")

    def get_img2img_pipe(self):
        if self._img2img_pipe is None:
            logger.info("[model_manager] Loading SDXL img2img pipeline...")
            self._img2img_pipe = self._load_sdxl_img2img()
        self.touch()
        return self._img2img_pipe

    def unload_img2img_pipe(self):
        if self._img2img_pipe is not None:
            del self._img2img_pipe
            self._img2img_pipe = None
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            logger.info("[model_manager] SDXL img2img pipeline unloaded")

    def unload_inpaint_pipe(self):
        if self._inpaint_pipe is not None:
            del self._inpaint_pipe
            self._inpaint_pipe = None
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            logger.info("[model_manager] SDXL inpaint pipeline unloaded")

    # ── FLUX.1 Fill ────────────────────────────────────────────────────────────
    def get_flux_fill_pipe(self):
        if self._flux_fill_pipe is None:
            logger.info("[model_manager] Loading FLUX.1 Fill pipeline…")
            self._flux_fill_pipe = self._load_flux_fill()
        self.touch()
        return self._flux_fill_pipe

    def _load_flux_fill(self):
        from diffusers import FluxFillPipeline
        logger.info("[model_manager] Loading FLUX.1 Fill (black-forest-labs/FLUX.1-fill-dev)…")
        pipe = FluxFillPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-fill-dev",
            torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
            cache_dir=str(WEIGHTS_DIR / "flux_fill"),
        )
        if DEVICE == "cuda":
            pipe.to(DEVICE)
            # Enable memory optimizations for FLUX (large model ~12B params)
            try:
                pipe.enable_model_cpu_offload()
                logger.info("[model_manager] FLUX.1 Fill: model_cpu_offload enabled")
            except Exception:
                logger.info("[model_manager] FLUX.1 Fill: using full GPU")
        else:
            pipe.to("cpu")
        logger.info("[model_manager] FLUX.1 Fill loaded")
        return pipe

    def unload_flux_fill_pipe(self):
        if self._flux_fill_pipe is not None:
            del self._flux_fill_pipe
            self._flux_fill_pipe = None
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            logger.info("[model_manager] FLUX.1 Fill pipeline unloaded")

    # ── Grounding DINO ────────────────────────────────────────────────────────
    def get_grounding_dino(self):
        if self._grounding_dino is None:
            logger.info("[model_manager] Loading Grounding DINO (IDEA-Research/grounding-dino-base)…")
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            model_id = "IDEA-Research/grounding-dino-base"
            cache    = str(WEIGHTS_DIR / "grounding_dino")
            logger.info(f"[model_manager] cache dir: {cache}")
            self._grounding_dino_processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache)
            logger.info("[model_manager] Grounding DINO processor loaded")
            self._grounding_dino = AutoModelForZeroShotObjectDetection.from_pretrained(
                model_id, cache_dir=cache
            ).to(DEVICE)
            logger.info(f"[model_manager] Grounding DINO model loaded → {DEVICE}")
        self.touch()
        return self._grounding_dino, self._grounding_dino_processor

    # ── SAM2 ──────────────────────────────────────────────────────────────────
    def get_sam2(self):
        if self._sam2_predictor is None:
            logger.info("[model_manager] Loading SAM2…")
            try:
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor
                checkpoint = WEIGHTS_DIR / "sam2" / "sam2_hiera_large.pt"
                config     = "sam2_hiera_l.yaml"
                if not checkpoint.exists():
                    logger.info(f"[model_manager] SAM2 weights not found, downloading → {checkpoint}")
                    self._download_sam2(checkpoint)
                logger.info(f"[model_manager] Building SAM2 from {checkpoint}")
                sam2_model = build_sam2(config, str(checkpoint), device=DEVICE)
                self._sam2_predictor = SAM2ImagePredictor(sam2_model)
                logger.info("[model_manager] SAM2 loaded")
            except Exception as e:
                logger.warning(f"[model_manager] SAM2 unavailable ({e}), falling back to SAM")
                self._sam2_predictor = self._load_sam_fallback()
        self.touch()
        return self._sam2_predictor

    def _download_sam2(self, checkpoint: Path):
        import urllib.request
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
        logger.info(f"[model_manager] Downloading SAM2 from {url}")
        urllib.request.urlretrieve(url, str(checkpoint))
        logger.info(f"[model_manager] SAM2 download complete → {checkpoint}")

    def _load_sam_fallback(self):
        from segment_anything import sam_model_registry, SamPredictor
        checkpoint = WEIGHTS_DIR / "sam" / "sam_vit_h_4b8939.pth"
        if not checkpoint.exists():
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            import urllib.request
            url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
            logger.info(f"[model_manager] Downloading SAM fallback from {url}")
            urllib.request.urlretrieve(url, str(checkpoint))
            logger.info("[model_manager] SAM download complete")
        logger.info(f"[model_manager] Loading SAM vit_h from {checkpoint}")
        sam = sam_model_registry["vit_h"](checkpoint=str(checkpoint))
        sam.to(DEVICE)
        predictor = SamPredictor(sam)
        logger.info("[model_manager] SAM fallback loaded")
        return predictor
    def _load_sdxl_img2img(self):
        logger.info("[model_manager] Loading SDXL img2img...")
        pipe = _from_pretrained_with_fallback(
            StableDiffusionXLImg2ImgPipeline,
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float32 if DEVICE == "cpu" else DTYPE,
            low_cpu_mem_usage=True,
            cache_dir=str(WEIGHTS_DIR / "sdxl_img2img"),
        )
        if DEVICE == "cuda":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cpu")
        logger.info("[model_manager] SDXL img2img loaded")
        return pipe
    # ── Inpainting ────────────────────────────────────────────────────────────
    def get_inpaint_pipe(self):
        if self._inpaint_pipe is None:
            logger.info("[model_manager] Loading SDXL inpainting pipeline…")
            self._inpaint_pipe = self._load_sdxl_inpaint()
        self.touch()
        return self._inpaint_pipe

    def _load_sdxl_inpaint(self):
        from diffusers import StableDiffusionXLInpaintPipeline
        logger.info("[model_manager] Loading SDXL inpaint…")
        pipe = _from_pretrained_with_fallback(
            StableDiffusionXLInpaintPipeline,
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            torch_dtype=torch.float32 if DEVICE == "cpu" else DTYPE,
            low_cpu_mem_usage=True,
            cache_dir=str(WEIGHTS_DIR / "sdxl_inpaint"),
        )
        if DEVICE == "cuda":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cpu")
        logger.info("[model_manager] SDXL inpaint loaded")
        return pipe

    # ── LLM ───────────────────────────────────────────────────────────────────
    def get_llm(self):
        if self._llm_pipe is None:
            logger.info("[model_manager] Loading LLM…")
            try:
                self._llm_pipe = self._load_qwen()
            except Exception as e:
                logger.warning(f"[model_manager] Qwen unavailable ({e}), trying Llama 3.1")
                self._llm_pipe = self._load_llama()
        self.touch()
        return self._llm_pipe
    def _load_qwen(self):
        from transformers import pipeline as hf_pipeline
        import torch

        logger.info('[model_manager] Loading Qwen2.5-0.5B-Instruct on CPU...')

        pipe = hf_pipeline(
            'text-generation',
            model='Qwen/Qwen2.5-0.5B-Instruct',
            torch_dtype=torch.float32,   # CPU safe
            device_map=None,             # IMPORTANT: no auto offload
            model_kwargs={'cache_dir': str(WEIGHTS_DIR / 'qwen')},
        )

        logger.info('[model_manager] Qwen2.5 loaded')
        return pipe

    # def _load_qwen(self):
    #     from transformers import pipeline as hf_pipeline
    #     logger.info("[model_manager] Loading Qwen2.5-7B-Instruct…")
    #     pipe = hf_pipeline(
    #         "text-generation",
    #         model="Qwen/Qwen2.5-7B-Instruct",
    #         torch_dtype=DTYPE,
    #         device_map="auto",
    #         model_kwargs={"cache_dir": str(WEIGHTS_DIR / "qwen")},
    #     )
    #     logger.info("[model_manager] Qwen2.5 loaded")
    #     return pipe

    def _load_llama(self):
        from transformers import pipeline as hf_pipeline
        logger.info("[model_manager] Loading Llama-3.1-8B-Instruct…")
        pipe = hf_pipeline(
            "text-generation",
            model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            torch_dtype=DTYPE,
            device_map="auto",
            model_kwargs={"cache_dir": str(WEIGHTS_DIR / "llama")},
        )
        logger.info("[model_manager] Llama 3.1 loaded")
        return pipe

    # ── Rembg ─────────────────────────────────────────────────────────────────
    def get_rembg_session(self):
        if self._rembg_session is None:
            logger.info("[model_manager] Loading rembg u2net…")
            from rembg import new_session
            self._rembg_session = new_session(
                "u2net",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            logger.info("[model_manager] rembg loaded")
        self.touch()
        return self._rembg_session

    # ── Real-ESRGAN ───────────────────────────────────────────────────────────
    def get_realesrgan(self, scale: int = 2):
        if self._realesrgan is None:
            logger.info(f"[model_manager] Loading Real-ESRGAN x{scale}…")
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
            weight_path = WEIGHTS_DIR / "realesrgan" / f"RealESRGAN_x{scale}plus.pth"
            if not weight_path.exists():
                logger.info(f"[model_manager] Real-ESRGAN weights not found, downloading…")
                self._download_realesrgan(weight_path, scale)
            self._realesrgan = RealESRGANer(
                scale=scale, model_path=str(weight_path), model=model,
                tile=512, tile_pad=10, pre_pad=0, half=(DEVICE == "cuda"),
            )
            logger.info(f"[model_manager] Real-ESRGAN x{scale} loaded")
        self.touch()
        return self._realesrgan

    def _download_realesrgan(self, path: Path, scale: int):
        import urllib.request
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x{scale}plus.pth"
        logger.info(f"[model_manager] Downloading Real-ESRGAN from {url}")
        urllib.request.urlretrieve(url, str(path))
        logger.info(f"[model_manager] Real-ESRGAN download complete → {path}")


model_manager = ModelManager()


# ══════════════════════════════════════════════════════════════════════════════
#  Remote routing helpers
#  ─────────────────────────────────────────────────────────────────────────────
#  These functions are the single call-site for every AI operation in the app.
#  When COMPUTE_MODE=local  → delegates to model_manager (original behaviour).
#  When COMPUTE_MODE=remote → delegates to the matching remote_services module.
#
#  Callers (detection_service, segmentation_service, editing_service, etc.)
#  should use these helpers rather than calling model_manager directly so the
#  routing decision stays in one place.
# ══════════════════════════════════════════════════════════════════════════════

# ── Detection ─────────────────────────────────────────────────────────────────

def get_detection_models():
    """Return (model, processor) for Grounding DINO.  Local mode only."""
    if _REMOTE:
        raise RuntimeError(
            "get_detection_models() called in remote mode — use run_detection() instead."
        )
    return model_manager.get_grounding_dino()


def run_detection(image_path: str, prompt=None, box_threshold=0.25, text_threshold=0.25):
    """
    Detect objects in an image.
    Local:  runs Grounding DINO locally.
    Remote: calls the configured detection API (Replicate / fal.ai).
    Returns the same list-of-dicts as detection_service.detect_objects().
    """
    if _REMOTE:
        from services.remote_services.detection import detect_objects_remote
        return detect_objects_remote(image_path, prompt, box_threshold, text_threshold)
    # Local: detection_service handles the full pipeline itself via model_manager
    raise NotImplementedError(
        "run_detection() in local mode is handled by detection_service.detect_objects(). "
        "Call that function directly."
    )


# ── Segmentation ──────────────────────────────────────────────────────────────

def get_sam2():
    """Return the SAM2 predictor.  Local mode only."""
    if _REMOTE:
        raise RuntimeError(
            "get_sam2() called in remote mode — use run_segmentation() instead."
        )
    return model_manager.get_sam2()


def run_segmentation(image_path: str, bbox: dict, image_size: tuple):
    """
    Segment a single object.
    Local:  runs SAM2 locally via model_manager.
    Remote: calls the configured segmentation API (fal.ai SAM2 / Replicate SAM2).
    Returns a numpy uint8 mask (0/255), same size as the image.
    """
    if _REMOTE:
        from services.remote_services.segmentation import segment_object_remote
        return segment_object_remote(image_path, bbox, image_size)
    raise NotImplementedError(
        "run_segmentation() in local mode is handled by segmentation_service. "
        "Call model_manager.get_sam2() directly."
    )


# ── Inpainting ────────────────────────────────────────────────────────────────

def get_inpaint_pipe():
    """Return the SDXL inpaint pipeline.  Local mode only."""
    if _REMOTE:
        raise RuntimeError(
            "get_inpaint_pipe() called in remote mode — use run_inpaint() instead."
        )
    return model_manager.get_inpaint_pipe()


def get_flux_fill_pipe():
    """Return the FLUX.1 Fill pipeline.  Local mode only."""
    if _REMOTE:
        raise RuntimeError(
            "get_flux_fill_pipe() called in remote mode — use run_flux_fill() instead."
        )
    return model_manager.get_flux_fill_pipe()


def get_img2img_pipe():
    """Return the SDXL img2img pipeline.  Local mode only."""
    if _REMOTE:
        raise RuntimeError(
            "get_img2img_pipe() called in remote mode — use run_img2img() instead."
        )
    return model_manager.get_img2img_pipe()


def run_inpaint(layer_img, prompt: str, strength=0.75, guidance_scale=7.5, steps=20):
    """
    Run generative inpainting on a layer image.
    Local:  uses SDXL inpaint pipeline (or FLUX via editing_service).
    Remote: calls the configured inpaint API (fal.ai FLUX Pro Fill / Replicate / Stability).
    Returns an RGBA PIL image.
    """
    if _REMOTE:
        from services.remote_services.inpaint import inpaint_remote
        return inpaint_remote(layer_img, prompt, strength, guidance_scale, steps)
    # Local path: editing_service calls model_manager.get_inpaint_pipe() directly
    raise NotImplementedError(
        "run_inpaint() in local mode is handled by editing_service._inpaint(). "
        "Call model_manager.get_inpaint_pipe() directly."
    )


def run_img2img(layer_img, prompt: str, strength=0.75, guidance_scale=7.5, steps=20):
    """
    Run img2img (style transfer) on a layer image.
    Local:  uses SDXL img2img pipeline.
    Remote: calls the configured img2img API.
    Returns an RGBA PIL image.
    """
    if _REMOTE:
        from services.remote_services.inpaint import img2img_remote
        return img2img_remote(layer_img, prompt, strength, guidance_scale, steps)
    raise NotImplementedError(
        "run_img2img() in local mode is handled by editing_service._style_transfer(). "
        "Call model_manager.get_img2img_pipe() directly."
    )


def run_flux_fill(original_image, combined_mask, prompt: str, steps=20, guidance=3.5, seed=None):
    """
    Run FLUX.1 Fill for background reconstruction.
    Local:  uses FLUX.1 Fill pipeline.
    Remote: calls fal.ai FLUX Pro Fill.
    Returns a PIL RGB image.
    """
    if _REMOTE:
        from services.remote_services.inpaint import inpaint_remote
        from PIL import Image
        # Convert mask to RGBA layer format expected by inpaint_remote
        import numpy as np
        rgb = original_image.convert("RGB")
        mask_alpha = Image.fromarray((combined_mask > 128).astype("uint8") * 255, mode="L")
        layer_rgba = rgb.convert("RGBA")
        layer_rgba.putalpha(mask_alpha)
        result = inpaint_remote(layer_rgba, prompt, strength=0.99, guidance_scale=guidance, steps=steps)
        return result.convert("RGB")
    raise NotImplementedError(
        "run_flux_fill() in local mode is handled by flux_inpaint_service. "
        "Call model_manager.get_flux_fill_pipe() directly."
    )


# ── LLM ───────────────────────────────────────────────────────────────────────

def get_llm():
    """
    Return an LLM pipeline-like object for prompt parsing.
    Local:  returns the loaded Qwen/Llama HuggingFace pipeline.
    Remote: returns a RemoteLLMPipeline that calls Groq/OpenAI/Anthropic.
    """
    if _REMOTE:
        from services.remote_services.llm import RemoteLLMPipeline
        return RemoteLLMPipeline()
    return model_manager.get_llm()


# ── Background removal ────────────────────────────────────────────────────────

def get_rembg_session():
    """Return the rembg session.  Local mode only."""
    if _REMOTE:
        raise RuntimeError(
            "get_rembg_session() called in remote mode — use run_rembg() instead."
        )
    return model_manager.get_rembg_session()


def run_rembg(img):
    """
    Remove background from a PIL image.
    Local:  uses rembg u2net session.
    Remote: calls the configured rembg API (fal.ai / remove.bg / Stability).
    Returns an RGBA PIL image.
    """
    if _REMOTE:
        from services.remote_services.rembg import remove_background_remote
        return remove_background_remote(img)
    from rembg import remove
    session = model_manager.get_rembg_session()
    return remove(img, session=session)


# ── Upscaling ─────────────────────────────────────────────────────────────────

def get_realesrgan(scale: int = 2):
    """Return the Real-ESRGAN upsampler.  Local mode only."""
    if _REMOTE:
        raise RuntimeError(
            "get_realesrgan() called in remote mode — use run_upscale() instead."
        )
    return model_manager.get_realesrgan(scale)


def run_upscale(img, scale: int = 2):
    """
    Upscale a PIL image.
    Local:  uses Real-ESRGAN.
    Remote: calls the configured upscale API (fal.ai / Replicate ESRGAN).
    Returns an RGB PIL image.
    """
    if _REMOTE:
        from services.remote_services.upscale import upscale_remote
        return upscale_remote(img, scale)
    import numpy as np
    upsampler = model_manager.get_realesrgan(scale)
    arr = np.array(img.convert("RGB"))
    out_arr, _ = upsampler.enhance(arr, outscale=scale)
    from PIL import Image
    return Image.fromarray(out_arr)


# ── Vision (object identification) ────────────────────────────────────────────

def run_vision_identify(image_path: str) -> str:
    """
    Identify objects in an image to build a Grounding DINO prompt.
    Local:  uses Ollama + llava:7b (identify_service.py, already optional).
    Remote: calls GPT-4o-mini or Claude Haiku vision API.
    Returns a comma-separated string of object names.
    """
    if _REMOTE:
        from services.remote_services.vision import identify_objects_remote
        return identify_objects_remote(image_path)
    # Local: identify_service handles this itself via ollama
    from services.identify_service import identify_objects
    return identify_objects(image_path)
