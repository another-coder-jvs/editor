# User Flows

## 0. Compute Mode — Local vs Remote

```
  ┌─────────────────────────────────────────────────────────┐
  │              .env  (or environment variable)            │
  │                                                         │
  │     COMPUTE_MODE=local   (default)                      │
  │     COMPUTE_MODE=remote                                 │
  └──────────────────────────┬──────────────────────────────┘
                             │
             ┌───────────────┴───────────────┐
             │                               │
             ▼                               ▼
  ┌────────────────────┐         ┌────────────────────────┐
  │   LOCAL mode       │         │   REMOTE mode          │
  │                    │         │                        │
  │ • Download model   │         │ • Zero local models    │
  │   weights to       │         │ • All AI calls go to   │
  │   ./weights/       │         │   cloud APIs via HTTP  │
  │ • Run on your      │         │ • Needs only a CPU     │
  │   GPU / CPU        │         │   server (no GPU)      │
  │ • No API keys      │         │ • Pay per call         │
  │   needed           │         │                        │
  └────────┬───────────┘         └───────────┬────────────┘
           │                                 │
           │                                 ▼
           │                    ┌────────────────────────┐
           │                    │  remote_config.py       │
           │                    │  (single config file)   │
           │                    │                        │
           │                    │  FAL_API_KEY           │
           │                    │  REPLICATE_API_TOKEN   │
           │                    │  GROQ_API_KEY          │
           │                    │  OPENAI_API_KEY        │
           │                    │  ANTHROPIC_API_KEY     │
           │                    │  REMOVEBG_API_KEY      │
           │                    │  STABILITY_API_KEY     │
           │                    │                        │
           │                    │  DETECTION_PROVIDER    │
           │                    │  SEGMENTATION_PROVIDER │
           │                    │  INPAINT_PROVIDER      │
           │                    │  LLM_PROVIDER          │
           │                    │  REMBG_PROVIDER        │
           │                    │  UPSCALE_PROVIDER      │
           │                    │  VISION_PROVIDER       │
           │                    └────────────────────────┘
           │
           ▼
  All AI calls route through model_manager.py
  which checks _REMOTE at every operation.
```

---

## 1. Upload → Detection → Segmentation

```
                    ORIGINAL IMAGE
                          │
                          ▼
              ┌───────────────────────┐
              │   Upload Image        │
              │   (drag & drop /      │
              │    file picker)       │
              └───────────┬───────────┘
                          │
                ┌─────────┴─────────┐
                │                   │
                ▼                   ▼
        ┌──────────────┐   ┌──────────────┐
        │  Auto Mode   │   │ Manual Mode  │
        │  (default)   │   │              │
        └──────┬───────┘   └──────┬───────┘
               │                   │
               ▼                   │
   ┌────────────────────────────────────────────┐
   │  Object Identification                      │
   │                                            │
   │  LOCAL:  Ollama + llava:7b                 │
   │  REMOTE: GPT-4o-mini / Claude Haiku vision │
   │          (run_vision_identify)             │
   └────────────────────┬───────────────────────┘
                        │
                dot-separated labels
                        │
                        ▼
   ┌────────────────────────────────────────────┐
   │  Object Detection                          │
   │                                            │
   │  LOCAL:  Grounding DINO Base               │
   │          (IDEA-Research/grounding-dino-base│
   │           running on GPU/CPU)              │
   │  REMOTE: Replicate grounding-dino   or     │
   │          fal.ai grounding-dino             │
   │          (detect_objects_remote)           │
   └────────────────────┬───────────────────────┘
                        │
                 object bboxes
                        │
                        ▼
   ┌────────────────────────────────────────────┐
   │  Segmentation                              │
   │                                            │
   │  LOCAL:  SAM2 Hiera Large → predict()      │
   │          (sam2_hiera_large.pt)             │
   │          Fallback: SAM ViT-H               │
   │  REMOTE: fal.ai SAM2 image endpoint  or    │
   │          Replicate SAM2                    │
   │          (_segment_objects_remote)         │
   └────────────────────┬───────────────────────┘
                        │
                   pixel masks
                        │
                        ▼
   ┌────────────────────────────────────────────┐
   │  Mask Refinement  (always local — CPU only)│
   │  MORPH_CLOSE → flood-fill → MORPH_OPEN →  │
   │  distance-based edge feathering            │
   └────────────────────┬───────────────────────┘
                        │
                        ▼
   ┌────────────────────┐
   │   Layer Extraction │
   │   → RGBA PNG layers│
   │   → grayscale masks│
   └────────┬───────────┘
            │
            ▼
   ┌────────────────────┐
   │   Layer Panel      │
   │   (left sidebar)   │
   │   - visibility     │
   │   - reorder        │
   │   - selection      │
   └────────────────────┘
```

---

## 2. AI Edit Pipeline (Layer Editing)

```
                    SELECTED LAYER
                          │
                          ▼
            ┌──────────────────────┐
            │  User enters prompt  │
            │  in AI Edit Panel    │
            │  (e.g. "make it     │
            │   blue and bigger")  │
            └──────────┬───────────┘
                       │
                       ▼
            ┌──────────────────────────────────────┐
            │  Prompt Service — parse_edit_prompt  │
            │                                      │
            │  LLM call:                           │
            │    LOCAL:  Qwen2.5-0.5B-Instruct     │
            │            (or Qwen2.5-7B / Llama)   │
            │    REMOTE: Groq Llama-3.1-8B   or    │
            │            OpenAI GPT-4o-mini  or    │
            │            Anthropic Claude Haiku    │
            │            (RemoteLLMPipeline)       │
            └──────────────┬───────────────────────┘
                           │
                  ┌────────┴────────┐
                  │  compound?      │
                  │  ("and","also", │
                  │   ";")          │
                  └───┬─────────┬───┘
                      │         │
                    YES         NO
                      │         │
                      ▼         ▼
              ┌──────────┐ ┌──────────────┐
              │  Split   │ │   LLM Call   │
              │  into    │ └──────┬───────┘
              │  sub-    │        │
              │  prompts │   ┌────┴────┐
              └────┬─────┘   │success? │
                   │         └──┬───┬──┘
                   │          YES   NO
                   │           │     │
                   │           ▼     ▼
                   │    ┌────────┐ ┌────────────┐
                   │    │ JSON   │ │ Heuristic  │
                   │    │ extract│ │ fallback   │
                   │    └───┬────┘ └─────┬──────┘
                   │        │            │
                   └────────┴────────────┘
                                │
                      edit_type + params
                                │
                                ▼
                   ┌────────────────────────┐
                   │  Editing Service       │
                   │  Dispatcher            │
                   └────────────┬───────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ Deterministic    │  │ Generative       │  │ Style Transfer   │
│ (always local,   │  │ Inpainting       │  │                  │
│  no GPU needed)  │  │                  │  │  LOCAL:  SDXL    │
│                  │  │  LOCAL:  SDXL    │  │          Img2Img │
│ • recolor (HSV)  │  │          Inpaint │  │  REMOTE: fal.ai  │
│ • blur           │  │  REMOTE: fal.ai  │  │          SDXL    │
│ • sharpen        │  │    FLUX.1 Fill   │  │          img2img │
│ • brightness     │  │    or Replicate  │  │                  │
│ • contrast       │  │    or Stability  │  │  edit_types:     │
│ • saturation     │  │                  │  │  • anime         │
│ • erase          │  │  edit_types:     │  │  • oil_painting  │
│ • cartoon (CV2)  │  │  • replace       │  │  • style_xfer    │
│ • sketch (CV2)   │  │  • generative    │  └──────────────────┘
│ • pixel_art      │  │    _fill         │
│ • text_edit      │  │  • other         │
│ • bg_remove →    │  └──────────────────┘
│                  │
│   LOCAL:  rembg  │
│   REMOTE: fal.ai │
│     rembg / Bria │
│     or remove.bg │
│     or Stability │
│                  │
│ • upscale →      │
│   LOCAL:  ESRGAN │
│   REMOTE: fal.ai │
│     ESRGAN       │
│     or Replicate │
└──────────────────┘
        │
        └───────────────────────┬───────────────────────┘
                                │
                                ▼
                   ┌────────────────────┐
                   │  Save edited PNG   │
                   │  Update            │
                   │  session_meta.json │
                   └────────┬───────────┘
                            │
                            ▼
                   ┌────────────────────┐
                   │  Canvas re-renders │
                   │  Layer updated     │
                   └────────────────────┘
```

---

## 3. Layer Operations (Manual Editing)

```
                    SELECTED LAYER
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Draw Tools  │  │ Shape Tools  │  │ Adjust Tools │
├──────────────┤  ├──────────────┤  ├──────────────┤
│ • Brush (B)  │  │ • Rect       │  │ • Brightness │
│ • Pencil (P) │  │ • Ellipse    │  │ • Contrast   │
│ • Marker     │  │ • Line       │  │ • Saturation │
│ • Eraser (E) │  │ • Arrow      │  │ • Exposure   │
│ • Clone (S)  │  │ • Triangle   │  │ • Highlights │
│ • Heal (J)   │  │ • Star       │  │ • Shadows    │
│ • Color Pick │  │              │  │ • Temperature│
│   (I)        │  │              │  │ • Tint       │
└──────┬───────┘  └──────┬───────┘  │ • Hue        │
       │                 │          │ • Sharpness  │
       ▼                 ▼          │ • Clarity    │
┌──────────────┐  ┌──────────────┐  │ • Fade       │
│  Paint on    │  │  Add shape   │  │ • Vignette   │
│  layer pixels│  │  as new layer│  │ • Grain      │
│  (direct PNG │  │              │  └──────┬───────┘
│   editing)   │  │              │         │
└──────────────┘  └──────────────┘         ▼
                                  ┌──────────────┐
                                  │  Per-layer   │
                                  │  adjustments │
                                  │  (Konva      │
                                  │   filters)   │
                                  └──────────────┘
```

---

## 4. Selection Tools → Mask Operations

```
                ┌─────────────────────┐
                │  Selection Tool     │
                │  (one of 7 tools)   │
                └──────────┬──────────┘
                           │
     ┌─────────────────────┼─────────────────────┐
     │                     │                     │
     ▼                     ▼                     ▼
┌──────────────┐  ┌────────────────┐  ┌────────────────┐
│ Geometric    │  │ Freeform       │  │ AI-Powered     │
├──────────────┤  ├────────────────┤  ├────────────────┤
│ • Rect (M)   │  │ • Lasso (L)    │  │ • Magic (W)    │
│ • Ellipse    │  │ • Free select  │  │   (SAM2 mask)  │
│              │  │                │  │ • Object (DINO)│
└──────┬───────┘  └───────┬────────┘  └───────┬────────┘
       │                  │                   │
       └──────────────────┼───────────────────┘
                          │
                    selection mask
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
     ┌────────────────┐     ┌────────────────┐
     │ Copy to new    │     │ Delete within  │
     │ layer          │     │ selection      │
     └────────────────┘     └────────────────┘
```

---

## 5. Export Flow

```
                ┌─────────────────────┐
                │  User clicks        │
                │  Export button      │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │  Export Modal       │
                │  - Format (PNG/     │
                │    JPEG/WebP)       │
                │  - Quality          │
                │  - Upscale (2x/4x)  │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │  POST /export       │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │  Merge Service      │
                │  Alpha-composite    │
                │  all visible layers │
                │  onto base image    │
                │  (always local PIL) │
                └──────────┬──────────┘
                           │
                    ┌──────┴──────┐
                    │  upscale?   │
                    └──┬───────┬──┘
                     YES       NO
                      │        │
                      ▼        │
           ┌────────────────────────────────┐
           │ Upscaling                      │
           │                               │
           │  LOCAL:  Real-ESRGAN 2x / 4x  │
           │  REMOTE: fal.ai ESRGAN   or   │
           │          Replicate ESRGAN     │
           └──────────────┬────────────────┘
                          │
                          └────────────┘
                                 │
                                 ▼
                ┌─────────────────────┐
                │  Save to            │
                │  outputs/           │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │  Browser download   │
                │  (blob response)    │
                └─────────────────────┘
```

---

## 6. Session Lifecycle

```
         ┌──────────────┐
         │  Page Load   │
         └──────┬───────┘
                │
                ▼
         ┌─────────────────────┐
         │ GET /session/latest │
         └──────────┬──────────┘
                    │
              ┌─────┴─────┐
              │ session    │
              │ exists?    │
              └──┬─────┬──┘
               YES      NO
                │       │
                ▼       ▼
         ┌──────────┐ ┌──────────────┐
         │ Restore  │ │ Show Upload  │
         │ layers + │ │ Screen       │
         │ canvas   │ │ (empty state)│
         └──────────┘ └──────────────┘

         ┌──────────────┐
         │  Page Close  │
         └──────┬───────┘
                │
                ▼
         ┌─────────────────────┐
         │ Zustand persisted   │
         │ to localStorage     │
         │                     │
         │ Backend: session    │
         │ files remain in     │
         │ temp/{session_id}/  │
         └─────────────────────┘

         ┌──────────────────────┐
         │  Idle Timer          │
         │  (LOCAL mode only)   │
         └──────┬───────────────┘
                │
                ▼
         ┌─────────────────────┐
         │ After MODEL_IDLE_   │
         │ SECONDS (120s)      │
         │ → local models      │
         │   unloaded          │
         │ → VRAM freed        │
         │                     │
         │ REMOTE mode: no     │
         │ local models to     │
         │ unload (no-op)      │
         └─────────────────────┘
```

---

## 7. Prompt Routing Decision Tree

```
         User prompt: "make it blue and add flower pattern"
                            │
                            ▼
                ┌───────────────────────┐
                │ Compound detection    │
                │ (split on "and",      │
                │  "also", ";", ", then")│
                └───────────┬───────────┘
                            │
                      2 sub-prompts
                            │
                ┌───────────┴───────────┐
                ▼                       ▼
     ┌─────────────────┐     ┌─────────────────┐
     │ "make it blue"  │     │ "add flower     │
     │                 │     │  pattern"        │
     └────────┬────────┘     └────────┬────────┘
              │                       │
              ▼                       ▼
     ┌─────────────────────────────────────────┐
     │  LLM call  (parse_edit_prompt)          │
     │                                         │
     │  LOCAL:  Qwen2.5 (local HF pipeline)   │
     │  REMOTE: Groq / OpenAI / Anthropic      │
     │          (RemoteLLMPipeline)            │
     └──────────┬──────────────────────────────┘
                │
     ┌──────────┴──────────┐
     │                     │
     ▼                     ▼
┌──────────────┐  ┌──────────────────┐
│ edit_type:   │  │ edit_type:       │
│ "recolor"    │  │ "other"          │
│ color: "blue"│  │ inpaint_prompt:  │
└──────┬───────┘  │ "flower pattern" │
       │          └──────┬───────────┘
       │                 │
       ▼                 ▼
┌──────────────┐  ┌────────────────────────────┐
│ HSV recolor  │  │  Inpainting                │
│ (PIL/OpenCV, │  │                            │
│  no GPU,     │  │  LOCAL:  SDXL Inpaint      │
│  no API)     │  │  REMOTE: fal.ai FLUX Fill  │
└──────┬───────┘  └──────┬─────────────────────┘
       │                 │
       └────────┬────────┘
                │
          sequential
          execution
                │
                ▼
       ┌────────────────┐
       │ Final edited   │
       │ layer PNG      │
       └────────────────┘
```

---

## 8. Background Inpaint Flow

```
              Layer moved/removed
                    │
                    ▼
         ┌─────────────────────┐
         │ POST /inpaint-bg    │
         │ (layer_id, mask)    │
         └──────────┬──────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │ Background Detector │
         │ classify type       │
         └──────────┬──────────┘
                    │
       ┌────────────┼────────────┐
       │            │            │
       ▼            ▼            ▼
 ┌──────────┐ ┌──────────┐ ┌──────────┐
 │ SOLID    │ │ GRADIENT │ │ COMPLEX  │
 │ COLOR    │ │          │ │          │
 └────┬─────┘ └────┬─────┘ └────┬─────┘
      │             │            │
      ▼             ▼            │
 ┌──────────┐ ┌──────────┐      │
 │ Color    │ │ Gradient │      │
 │ distance │ │ aware    │      │
 │ mask     │ │ mask     │      │
 └────┬─────┘ └────┬─────┘      │
      │             │            │
      └──────┬──────┘            │
             │                   │
             ▼                   ▼
  ┌─────────────────────────────────────────┐
  │  Background Reconstruction              │
  │                                         │
  │  LOCAL:  FLUX.1 Fill pipeline           │
  │          → fallback: LaMa inpainting    │
  │          → fallback: cv2.inpaint        │
  │  REMOTE: fal.ai FLUX.1 Fill  or         │
  │          Replicate / Stability AI       │
  │          (reconstruct_background_with_  │
  │           flux → remote branch)         │
  └──────────────────┬──────────────────────┘
                     │
                     ▼
            ┌────────────────┐
            │ Reconstructed  │
            │ background PNG │
            └────────────────┘
```

---

## 9. Remote Services Architecture

```
  ┌──────────────────────────────────────────────────────────┐
  │                     model_manager.py                     │
  │                                                          │
  │   _REMOTE = (COMPUTE_MODE == "remote")                   │
  │                                                          │
  │   get_llm()          run_detection()   run_segmentation()│
  │   run_inpaint()      run_img2img()     run_flux_fill()   │
  │   run_rembg()        run_upscale()     run_vision_identify│
  └────────────┬─────────────────────────────────────────────┘
               │
     ┌─────────┴──────────┐
     │                    │
     ▼                    ▼
  _REMOTE=False        _REMOTE=True
  (local pipeline)     │
                        ▼
         ┌──────────────────────────────┐
         │   remote_services/           │
         │                              │
         │   detection.py               │
         │   ├─ Replicate grounding-dino│
         │   └─ fal.ai grounding-dino   │
         │                              │
         │   segmentation.py            │
         │   ├─ fal.ai SAM2             │
         │   └─ Replicate SAM2          │
         │                              │
         │   inpaint.py                 │
         │   ├─ fal.ai FLUX.1 Fill      │
         │   ├─ Replicate SDXL          │
         │   └─ Stability AI            │
         │                              │
         │   llm.py                     │
         │   ├─ Groq Llama-3.1-8B       │
         │   ├─ OpenAI GPT-4o-mini      │
         │   └─ Anthropic Claude Haiku  │
         │                              │
         │   rembg.py                   │
         │   ├─ fal.ai rembg / Bria     │
         │   ├─ remove.bg               │
         │   └─ Stability AI            │
         │                              │
         │   upscale.py                 │
         │   ├─ fal.ai ESRGAN           │
         │   └─ Replicate ESRGAN        │
         │                              │
         │   vision.py                  │
         │   ├─ OpenAI GPT-4o-mini      │
         │   └─ Anthropic Claude Haiku  │
         │                              │
         │   _http.py  (shared helpers) │
         │   ├─ fal_call() + polling    │
         │   ├─ replicate_call()        │
         │   └─ image_to_data_uri()     │
         └──────────────────────────────┘
                        │
                        ▼
         ┌──────────────────────────────┐
         │   remote_config.py           │
         │   (single source of truth)   │
         │                              │
         │   API keys (7 providers)     │
         │   Provider preferences       │
         │   Endpoint URLs              │
         │   Model name overrides       │
         │   Timeout settings           │
         └──────────────────────────────┘
```

---

## 10. Models Reference

### LOCAL mode — downloaded to ./weights/

```
┌──────────────────┬─────────┬────────────────────────────┐
│ Model            │ Size    │ Purpose                    │
├──────────────────┼─────────┼────────────────────────────┤
│ grounding_dino   │ ~700 MB │ Object detection           │
│ sam2_hiera_large │ ~900 MB │ Segmentation (primary)     │
│ sam_vit_h        │ ~2.4 GB │ Segmentation fallback      │
│ qwen2.5-0.5B     │ ~1 GB   │ LLM prompt parser (active) │
│ qwen2.5-7B       │ ~15 GB  │ LLM (commented out)        │
│ llama-3.1-8B     │ ~16 GB  │ LLM fallback               │
│ sdxl_inpaint     │ ~6 GB   │ Generative edits           │
│ sdxl_img2img     │ ~6 GB   │ Style transfer             │
│ flux_fill        │ ~24 GB  │ Background reconstruction  │
│ realesrgan x2/x4 │ ~65 MB  │ Upscaling                 │
│ rembg u2net      │ ~170 MB │ Background removal         │
│ lama             │ ~200 MB │ BG inpaint fallback        │
│ easyocr          │ ~100 MB │ Text detection             │
│ ollama + llava:7b│ ~4 GB   │ Object ID (optional)       │
└──────────────────┴─────────┴────────────────────────────┘
```

### REMOTE mode — no local downloads, pay-per-call APIs

```
┌──────────────────┬────────────────────────┬──────────────┐
│ Task             │ Default Provider       │ ~Cost        │
├──────────────────┼────────────────────────┼──────────────┤
│ Object detection │ Replicate grounding-   │ ~$0.001/call │
│                  │ dino                   │              │
│ Segmentation     │ fal.ai SAM2            │ ~$0.005/sess │
│ Inpainting       │ fal.ai FLUX.1 Fill     │ $0.05/MP     │
│ LLM parsing      │ Groq Llama-3.1-8B      │ ~$0.0001/    │
│                  │                        │  edit        │
│ Background rm    │ fal.ai rembg           │ ~$0.01/call  │
│ Upscaling        │ fal.ai ESRGAN          │ ~$0.003/img  │
│ Vision (opt)     │ OpenAI GPT-4o-mini     │ ~$0.001/img  │
├──────────────────┼────────────────────────┼──────────────┤
│ Typical session  │ (5 edits, 1 bg remove) │ ~$0.17–$0.25 │
└──────────────────┴────────────────────────┴──────────────┘
```

---

## 11. Configuration Files

```
.env.example          ← copy to .env and fill in values
  COMPUTE_MODE        ← "local" or "remote"
  ENV                 ← "local" or "server" (path config)
  FAL_API_KEY
  REPLICATE_API_TOKEN
  GROQ_API_KEY
  OPENAI_API_KEY
  ANTHROPIC_API_KEY
  REMOVEBG_API_KEY
  STABILITY_API_KEY
  DETECTION_PROVIDER
  SEGMENTATION_PROVIDER
  INPAINT_PROVIDER
  LLM_PROVIDER
  REMBG_PROVIDER
  UPSCALE_PROVIDER
  VISION_PROVIDER

backend/
  services/
    remote_config.py  ← all remote settings as Python class
    model_manager.py  ← routes local ↔ remote based on _REMOTE flag
    remote_services/
      _http.py        ← shared HTTP helpers
      detection.py
      segmentation.py
      inpaint.py
      llm.py
      rembg.py
      upscale.py
      vision.py
  utils/
    config.py         ← path config (TEMP_DIR, WEIGHT_DIR, etc.)
```
