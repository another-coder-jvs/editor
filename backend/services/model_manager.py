"""
Central model manager – downloads, caches, and provides access to all AI models.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional
from diffusers import StableDiffusionXLImg2ImgPipeline

import torch

logger = logging.getLogger(__name__)

from utils import config
WEIGHTS_DIR = config.WEIGHT_DIR
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32


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
    """Singleton that lazily loads and caches every AI model."""

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
        logger.info("[model_manager] ModelManager singleton initialized (all models lazy)")

    def get_img2img_pipe(self):
        if self._img2img_pipe is None:
            logger.info("[model_manager] Loading SDXL img2img pipeline...")
            self._img2img_pipe = self._load_sdxl_img2img()
        return self._img2img_pipe

    def unload_img2img_pipe(self):
        if self._img2img_pipe is not None:
            del self._img2img_pipe
            self._img2img_pipe = None
            import gc; gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            logger.info("[model_manager] SDXL img2img pipeline unloaded")

    def unload_inpaint_pipe(self):
        if self._inpaint_pipe is not None:
            del self._inpaint_pipe
            self._inpaint_pipe = None
            import gc; gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            logger.info("[model_manager] SDXL inpaint pipeline unloaded")

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
        return self._realesrgan

    def _download_realesrgan(self, path: Path, scale: int):
        import urllib.request
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x{scale}plus.pth"
        logger.info(f"[model_manager] Downloading Real-ESRGAN from {url}")
        urllib.request.urlretrieve(url, str(path))
        logger.info(f"[model_manager] Real-ESRGAN download complete → {path}")


model_manager = ModelManager()
