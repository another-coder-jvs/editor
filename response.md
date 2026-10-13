# AI Models in This Project — and How to Replace Them with APIs

This document covers every AI model currently running locally in this project,
what each one does, and what cloud API alternatives exist so you can offload the
heavy GPU/RAM work to a third-party service instead of running it yourself.

---

## Current Model Inventory

The project runs **10 models** across the full pipeline:

| # | Model | File / HF ID | Role |
|---|-------|--------------|------|
| 1 | Grounding DINO Base | `IDEA-Research/grounding-dino-base` | Object detection (bounding boxes) |
| 2 | SAM2 Hiera Large | `sam2_hiera_large.pt` | Pixel-level segmentation masks |
| 3 | SAM ViT-H (fallback) | `sam_vit_h_4b8939.pth` | Segmentation if SAM2 not installed |
| 4 | FLUX.1 Fill | `black-forest-labs/FLUX.1-fill-dev` | Generative inpainting (primary) |
| 5 | SDXL Inpainting | `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` | Inpainting fallback |
| 6 | SDXL Img2Img | `stabilityai/stable-diffusion-xl-base-1.0` | Style transfer |
| 7 | Qwen2.5-0.5B-Instruct | `Qwen/Qwen2.5-0.5B-Instruct` | Prompt parsing LLM (active) |
| 8 | Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct` | Prompt parsing LLM (commented out) |
| 9 | Llama 3.1-8B-Instruct | `meta-llama/Meta-Llama-3.1-8B-Instruct` | LLM fallback if Qwen fails |
| 10 | rembg u2net | downloaded by `rembg` library | Background removal |
| 11 | Real-ESRGAN x2 / x4 | `RealESRGAN_x2plus.pth` / `RealESRGAN_x4plus.pth` | Image upscaling |
| 12 | Ollama + llava:7b | local Ollama server (optional) | Vision — identifies objects to seed DINO prompt |

**Local resource cost:** 16–24+ GB VRAM, 32 GB RAM, ~50 GB disk for weights.
The biggest offenders are FLUX.1 Fill (~24 GB), Qwen2.5-7B (~15 GB),
and Llama 3.1-8B (~16 GB).

---

## Can You Use APIs Instead?

**Yes, entirely.** Every single model has a hosted API equivalent. You would
replace local `model_manager.py` calls with HTTP requests to the chosen provider.
No GPU needed on your server — just a CPU box or even a serverless function.

The tradeoff: you pay per call instead of paying upfront for hardware. For
low-to-medium usage this is almost always cheaper than renting a GPU 24/7.

---

## Per-Model API Alternatives

---

### 1. Grounding DINO — Object Detection

**What it does:** Scans the uploaded image and returns bounding boxes +
labels for every object found. Used in `detection_service.py`.

| Provider | Endpoint / Notes | Price |
|----------|-----------------|-------|
| **Replicate** | `adirik/grounding-dino` | ~$0.001 per call |
| **DINO-X API** (IDEA-Research) | Official API from the same team, stronger model (DINO-X Pro: 56 AP on COCO) | Pay-as-you-go, similar range |
| **AWS Rekognition** | `DetectLabels` — closed-vocabulary but covers 1000s of labels | $0.001 per image (first 1M/mo) |
| **Azure Computer Vision** | `Analyze Image` — similar closed-vocab coverage | $0.001–$0.002 per image |
| **Google Cloud Vision** | `OBJECT_LOCALIZATION` | $0.0015 per image |

**Recommendation:** Replicate's Grounding DINO is the direct drop-in replacement
(same model, same open-vocabulary behaviour). DINO-X is better quality if you
need it. AWS/Azure/Google are cheaper at scale but only detect predefined
categories — you lose the ability to detect any arbitrary object with a text prompt.

---

### 2. SAM2 / SAM — Segmentation

**What it does:** Takes each bounding box from DINO and produces a precise
pixel mask for that object. Used in `segmentation_service.py`.

| Provider | Endpoint / Notes | Price |
|----------|-----------------|-------|
| **fal.ai** | `fal-ai/sam2/image` — SAM2 image segmentation endpoint | GPU-time based, very cheap per call |
| **Replicate** | `meta/sam-2` | ~$0.001–$0.003 per run |
| **Roboflow** | Hosted SAM2 via their inference server | Free tier available |

**Recommendation:** fal.ai is the cheapest and fastest. The API takes an image +
bounding boxes and returns masks — exactly what this project needs.

---

### 3. FLUX.1 Fill — Generative Inpainting (Primary)

**What it does:** The main AI edit engine. Takes a layer image + mask and
generates new content via diffusion. Used in `flux_inpaint_service.py` and
`editing_service.py` for replace/anime/oil_painting/generative_fill edits.

| Provider | Endpoint / Notes | Price |
|----------|-----------------|-------|
| **fal.ai** | `fal-ai/flux-pro/v1/fill` — FLUX.1 Pro Fill | $0.05 per megapixel |
| **Replicate** | `black-forest-labs/flux-fill-dev` | ~$0.01–$0.03 per image |
| **Segmind** | `flux-fill-dev` API | ~$0.05 per generation |
| **BFL API** (Black Forest Labs) | Official API — FLUX.1 Pro/Dev variants | Subscription + per-image |

**Recommendation:** fal.ai for best price/quality balance. At 1024×1024 (1 MP)
that's $0.05 per inpainting call. For a typical editing session of 5–10 edits
that's $0.25–$0.50 per user session — very reasonable.

---

### 4. SDXL Inpainting — Fallback Inpainting

**What it does:** Fallback inpainting pipeline when FLUX is unavailable or too
slow. `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`.

| Provider | Endpoint / Notes | Price |
|----------|-----------------|-------|
| **fal.ai** | `fal-ai/stable-diffusion-xl/inpainting` | ~$0.02–$0.04 per image |
| **Replicate** | `stability-ai/sdxl` with inpainting | ~$0.01–$0.02 per run |
| **Stability AI API** | Direct from Stability AI — Stable Image Edit | $0.03–$0.06 per image |

If you're already using FLUX.1 Fill via API, you can drop the SDXL fallback
entirely — FLUX is strictly better quality.

---

### 5. SDXL Img2Img — Style Transfer

**What it does:** img2img pipeline for style_transfer edits
(`stabilityai/stable-diffusion-xl-base-1.0`). Used in `_style_transfer()`.

| Provider | Endpoint / Notes | Price |
|----------|-----------------|-------|
| **fal.ai** | `fal-ai/stable-diffusion-xl` img2img | ~$0.02–$0.04 per image |
| **Replicate** | `stability-ai/sdxl` | ~$0.01–$0.02 per run |
| **Stability AI** | `stable-image/transform` endpoint | $0.04–$0.08 per image |

---

### 6 & 7. LLMs — Qwen2.5 + Llama 3.1 (Prompt Parsing)

**What they do:** Parse natural language edit prompts into structured JSON
`{edit_type, edit_params, inpaint_prompt}`. This is a small text-in / JSON-out
task — the LLM only needs to be good at instruction following and JSON output.
Used in `prompt_service.py`.

Currently active: **Qwen2.5-0.5B-Instruct** (very small, CPU-only, lower quality).
Commented out: **Qwen2.5-7B-Instruct** (better quality, needs GPU).
Fallback: **Llama 3.1-8B-Instruct**.

| Provider | Model | Price |
|----------|-------|-------|
| **Groq** | `llama-3.1-8b-instant` | $0.05 input / $0.08 output per million tokens |
| **Groq** | `llama-3.3-70b-versatile` | $0.59 / $0.79 per million tokens |
| **Together AI** | `Qwen/Qwen2.5-7B-Instruct` or `meta-llama/Llama-3.1-8B` | ~$0.10–$0.20 per million tokens |
| **Fireworks AI** | `qwen2p5-7b-instruct` | ~$0.20 per million tokens |
| **OpenAI** | `gpt-4o-mini` | $0.15 input / $0.60 output per million tokens |
| **Google Gemini** | `gemini-2.0-flash` | $0.10 input / $0.40 output per million tokens |

**Why this is cheap:** A single prompt parse call uses roughly 300–500 tokens.
At Groq's rate, that's **less than $0.0001 per edit** (under a tenth of a cent).
You could do 10,000 edits for $1. This is by far the cheapest model to replace.

**Recommendation:** Groq is the fastest and cheapest for this use case.
Their free tier is generous enough for development. For production, Llama 3.1-8B
on Groq at $0.05/M input tokens is effectively free at any realistic usage scale.
Switch the `_load_qwen()` method to call the Groq API instead.

---

### 8. rembg u2net — Background Removal

**What it does:** Removes backgrounds from images. Called in `_background_remove()`
inside `editing_service.py` when the user types something like "remove background".

| Provider | Endpoint / Notes | Price |
|----------|-----------------|-------|
| **fal.ai** | `fal-ai/imageutils/rembg` | Very cheap, per-image |
| **remove.bg** | Industry standard, best quality | $0.02–$0.20 per image depending on plan |
| **Photoroom** | Background removal + editing API | $0.05–$0.10 per image |
| **Stability AI** | `stable-image/edit/remove-background` | $0.02 per image |
| **Bria RMBG 2.0** (fal.ai) | `fal-ai/bria/background/removal` | ~$0.01–$0.02 per image |

**Recommendation:** fal.ai's rembg or Bria RMBG for lowest cost. remove.bg
if you need the highest accuracy on complex hair/fur edges.

---

### 9. Real-ESRGAN — Upscaling

**What it does:** AI upscaling (2x or 4x) for export. Called in `_upscale()`
when the user applies an upscale edit or exports with upscaling enabled.

| Provider | Endpoint / Notes | Price |
|----------|-----------------|-------|
| **fal.ai** | `fal-ai/esrgan` — Real-ESRGAN endpoint | ~$0.001–$0.005 per image |
| **Replicate** | `nightmareai/real-esrgan` | ~$0.001–$0.003 per run |
| **Stability AI** | `stable-image/upscale` | $0.02–$0.05 per image |
| **Topaz** | Topaz Gigapixel API | Higher cost, best quality |

**Recommendation:** fal.ai ESRGAN is essentially free at this price. Real-ESRGAN
is also one of the lightest models locally (only 67 MB), so keeping it local
is also perfectly fine if you're already running a small server.

---

### 10. Ollama + llava:7b — Vision Model (Optional)

**What it does:** Optional step that analyses the uploaded image and outputs a
list of objects ("person, chair, lamp") to use as the Grounding DINO prompt.
Without it, the project falls back to a hardcoded 50-category list.

The code is in `identify_service.py`. It checks if Ollama is running; if not,
it skips this step entirely — so this is already optional.

| Provider | Model | Price |
|----------|-------|-------|
| **OpenAI** | `gpt-4o-mini` with vision | $0.15 input + $0.60 output per million tokens + $0.003 per image |
| **Anthropic** | `claude-haiku-3.5` with vision | $0.80 input / $4.00 output per million tokens |
| **Google Gemini** | `gemini-2.0-flash` with vision | $0.10 input / $0.40 output per million tokens |
| **Together AI** | `llava-1.5-13b-hf` | ~$0.20 per million tokens |

Since the vision call only needs to return 5–10 words ("person, chair, lamp"),
cost is negligible — about $0.001 per image. GPT-4o-mini is the easiest to
integrate and has excellent vision understanding.

---

## Summary: Going Fully API-Based

Here's what a fully API-based version of the pipeline looks like:

```
Upload image
    ↓
[OPTIONAL] GPT-4o-mini vision → identify objects
    ↓
Replicate / DINO-X  →  bounding boxes
    ↓
fal.ai SAM2         →  pixel masks per object
    ↓
[Layer editing]
  - recolor / blur / brightness / cartoon / sketch
    → PIL/OpenCV (no GPU, runs locally, free)
  - replace / generative_fill / anime / oil_painting
    → fal.ai FLUX.1 Fill API
  - style_transfer
    → fal.ai SDXL img2img API
  - background_remove
    → fal.ai rembg or remove.bg API
  - upscale
    → fal.ai ESRGAN API
    ↓
Natural language prompt → Groq Llama 3.1-8B → JSON parse
    ↓
Export → local PIL compositing (free)
```

With this setup, your backend server needs **zero GPU**. A small CPU-only
instance (2 vCPU, 4 GB RAM) is enough to serve the FastAPI app and handle
PIL/OpenCV transforms. All heavy AI work is delegated to APIs.

---

## Estimated API Cost per User Session

A typical user session: upload 1 image → auto-detect → 5 edits → export.

| Step | API Call | Estimated Cost |
|------|----------|----------------|
| Object detection (DINO) | 1 × Replicate call | $0.001 |
| Segmentation (SAM2) | ~8 objects × fal.ai | ~$0.005 |
| Prompt parsing (LLM) | 5 × Groq Llama | ~$0.0005 |
| AI edits (FLUX inpaint) | ~3 edits × fal.ai | ~$0.15 |
| Background removal | 1 × fal.ai rembg | ~$0.01 |
| Upscaling | 1 × fal.ai ESRGAN | ~$0.003 |
| **Total per session** | | **~$0.17** |

So roughly **$0.15–$0.25 per user session** depending on how many generative
edits they make. Non-AI edits (recolor, blur, brightness, cartoon, sketch,
pixel art) are free — they run as fast PIL/OpenCV transforms on your server.

Compare this to renting a GPU server 24/7:
- RunPod RTX 3090 community: ~$0.22/hr = ~$160/month
- That GPU cost is recovered once you hit ~950 sessions/month (~32/day)
- Below that threshold, APIs are cheaper

---

## Recommended Providers Summary

| Model Category | Best API Option | Why |
|---------------|----------------|-----|
| Object detection | Replicate (Grounding DINO) | Same model, $0.001/call |
| Segmentation | fal.ai SAM2 | Cheapest, fast |
| Generative inpainting | fal.ai FLUX.1 Fill | Best quality/price |
| LLM prompt parsing | Groq (Llama 3.1-8B) | Free tier + $0.05/M — basically free |
| Background removal | fal.ai RMBG or Bria | Cheap, good quality |
| Upscaling | fal.ai ESRGAN | Negligible cost |
| Vision (optional) | GPT-4o-mini | Best understanding, low cost |

All the above providers use simple REST APIs with JSON — straightforward
to integrate by modifying `model_manager.py` to add HTTP client methods
alongside the existing local loaders.

---

## What You'd Need to Change in the Code

The architecture is already well-designed for this. The main changes are in
`backend/services/model_manager.py` — replace the `_load_*` methods with HTTP
calls to the respective APIs. The rest of the code (`detection_service.py`,
`segmentation_service.py`, `editing_service.py`, `prompt_service.py`) calls
`model_manager` through clean interfaces and wouldn't need changes.

You'd also want to:
1. Add API keys to environment variables (create a `.env` file)
2. Add `httpx` or `requests` calls in model_manager for each provider
3. Keep the PIL/OpenCV handlers as-is — they need no API and are instant

The `model_manager.py` already has the right separation to make this clean.
