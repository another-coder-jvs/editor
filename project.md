# AI Image Editor — Project Reference

## Overview

A fully local AI image editor that auto-segments any image into per-object layers and lets you edit each layer with natural language prompts. Everything runs on your GPU (or CPU fallback). No API keys, no cloud.

---

## Tech Stack

### Backend
- Python 3.11+, FastAPI, Uvicorn
- PyTorch (CUDA / CPU), Transformers, Diffusers
- Grounding DINO (object detection)
- SAM2 / SAM fallback (segmentation)
- SDXL Inpainting / FLUX.1-dev (generative edits)
- SDXL Img2Img (style transfer)
- Qwen2.5-0.5B-Instruct (active) / Qwen2.5-7B / Llama 3.1-8B (fallback LLMs)
- Real-ESRGAN (upscaling)
- rembg u2net (background removal)
- EasyOCR + OpenCV (text detection & editing)
- Pillow, NumPy, OpenCV

### Frontend
- React 18 + TypeScript + Vite
- Zustand (state management)
- react-konva / Konva (canvas rendering)
- Tailwind CSS
- Axios (API client)
- lucide-react (icons)
- react-toastify (notifications)

---

## Folder Structure

```
editor/
├── backend/
│   ├── main.py                  # FastAPI app, router registration, lifespan
│   ├── schemas.py               # Pydantic models (LayerData, EditRequest, etc.)
│   ├── requirements.txt
│   ├── routers/
│   │   ├── detect.py            # POST /detect  — upload image, run Grounding DINO
│   │   ├── segment.py           # POST /segment — run SAM2 on detected objects
│   │   ├── layers.py            # POST /layers  — rebuild layers for existing session
│   │   ├── edit.py              # POST /edit    — apply AI/deterministic edit to layer
│   │   ├── merge.py             # POST /merge   — composite layers → output image
│   │   ├── export.py            # POST /export  — merge + optional upscale → file download
│   │   ├── project.py           # POST /project/save, GET /project/list, DELETE /project/{name}
│   │   ├── session.py           # GET /session/latest, /session/list, /session/{id}, DELETE
│   │   ├── text.py              # POST /text/detect — EasyOCR text detection on layer
│   │   └── progress.py         # GET /progress/{session_id}
│   ├── services/
│   │   ├── model_manager.py     # Singleton lazy-loader for all AI models
│   │   ├── detection_service.py # Grounding DINO inference
│   │   ├── segmentation_service.py # SAM2 masks, refinement, transparent PNG layers
│   │   ├── editing_service.py   # Edit dispatcher (recolor, blur, inpaint, etc.)
│   │   ├── prompt_service.py    # LLM prompt parser + heuristic fallback
│   │   ├── merge_service.py     # Alpha-composite layers into final canvas
│   │   ├── text_service.py      # EasyOCR detect + cv2.inpaint erase + perspective re-render
│   │   └── progress_store.py   # In-memory progress tracking dict
│   ├── utils/
│   │   └── config.py            # Path config (ENV=local → ./temp|outputs|projects|weights)
│   └── tests/
│       ├── test_editing_service.py
│       ├── test_merge_service.py
│       └── test_prompt_service.py
├── frontend/
│   ├── src/
│   │   ├── App.tsx              # Root layout, session restore on mount
│   │   ├── main.tsx             # React entry point
│   │   ├── config.ts            # baseUrl (ngrok or localhost)
│   │   ├── api/client.ts        # Axios API functions for all endpoints
│   │   ├── store/editorStore.ts # Zustand store (layers, session, undo/redo, canvas)
│   │   ├── types/index.ts       # TypeScript interfaces (LayerData, Tool, etc.)
│   │   ├── components/
│   │   │   ├── Toolbar.tsx          # Top bar: tools, undo/redo, export, projects
│   │   │   ├── LayerPanel.tsx       # Left sidebar: layer list, visibility, reorder
│   │   │   ├── Canvas.tsx           # Konva canvas: image + layer compositing, pan/zoom
│   │   │   ├── PropertiesPanel.tsx  # Right panel: edit prompt, opacity, transform, text edit
│   │   │   ├── ImageUploader.tsx    # Drag-and-drop upload → detect → segment flow
│   │   │   ├── ExportModal.tsx      # Export format/upscale options
│   │   │   ├── ProjectManager.tsx   # Save/load/delete projects + session list
│   │   │   ├── HistoryPanel.tsx     # Undo/redo history display
│   │   │   └── ProgressBar.tsx      # Progress polling overlay
│   │   ├── hooks/
│   │   │   ├── useKeyboardShortcuts.ts  # Ctrl+Z/Y, Delete, Ctrl+D, Ctrl+Enter
│   │   │   ├── useProgressPoller.ts     # Polls /progress/{id} during long tasks
│   │   │   └── useBlobUrl.ts            # Converts /temp URLs to blob URLs (ngrok-safe)
│   │   └── utils/               # (utility helpers)
│   ├── vite.config.ts           # Vite dev server, proxy all API routes to :8000
│   ├── tailwind.config.js
│   └── package.json
├── weights/                     # Downloaded model weights (auto on first use)
│   ├── grounding_dino/          # IDEA-Research/grounding-dino-base (~700 MB)
│   ├── sam2/sam2_hiera_large.pt (~900 MB)
│   ├── sam/sam_vit_h_4b8939.pth (SAM fallback, ~2.4 GB)
│   ├── qwen/                    # Qwen2.5-0.5B-Instruct (active, ~1 GB)
│   ├── sdxl_inpaint/            # diffusers/stable-diffusion-xl-1.0-inpainting-0.1 (~6 GB)
│   ├── sdxl_img2img/            # stabilityai/stable-diffusion-xl-base-1.0 (~6 GB)
│   └── realesrgan/RealESRGAN_x4plus.pth (~65 MB)
├── models/                      # Local model files (grounding_dino_weights.pth, sam_vit_b/h)
├── temp/                        # Per-session working files (layer PNGs, masks, session_meta.json)
├── outputs/                     # Merged/exported images
├── projects/                    # Saved project JSON files
├── README.md
├── project.md                   # ← this file
├── docker-compose.yml
├── Dockerfile
├── environment.yml              # Conda env
├── setup.py
├── install.sh / install.bat
├── start.sh / start.bat
├── start_ngrok.py               # Starts ngrok tunnel for remote access (Colab)
└── steps_to_setup.sh            # Quick setup steps reference
```

---

## Environment / Config

`backend/utils/config.py` reads `ENV` env var:

| ENV value | Paths used |
|-----------|-----------|
| `local` (default for dev) | `./temp`, `./outputs`, `./projects`, `./weights` |
| `server` (default) | `/content/drive/MyDrive/project_folders/...` (Google Colab/Drive) |

Set `ENV=local` when running locally:
```bash
ENV=local uvicorn main:app --reload
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/detect` | Upload image (multipart), run Grounding DINO, returns objects + session_id |
| POST | `/segment` | Run SAM2 on detected objects, returns LayerData list |
| POST | `/layers` | Rebuild layers for an existing session (re-segments) |
| POST | `/edit` | Apply edit to a single layer (prompt-driven or explicit edit_type) |
| POST | `/merge` | Alpha-composite all visible layers → output image |
| POST | `/export` | Merge + optional Real-ESRGAN upscale → file download |
| POST | `/project/save` | Save project as JSON |
| GET | `/project/list` | List saved projects |
| DELETE | `/project/{name}` | Delete a project |
| GET | `/session/latest` | Return most recently modified valid session |
| GET | `/session/list` | List all sessions |
| GET | `/session/{id}` | Get specific session metadata |
| DELETE | `/session/{id}` | Delete session directory |
| POST | `/text/detect` | EasyOCR text detection on a layer |
| GET | `/progress/{id}` | Poll task progress (0.0–1.0) |
| GET | `/health` | Health check |
| GET | `/temp/{path}` | Browse/serve temp files (HTML dir listing or FileResponse) |

---

## Edit Types

| edit_type | Handler | Notes |
|-----------|---------|-------|
| `recolor` | `_inpaint` | Routes to inpaint pipeline with color prompt |
| `blur` | PIL GaussianBlur | `params.radius` (default 5) |
| `sharpen` | PIL Sharpness | `params.factor` (default 2.0) |
| `brightness` | PIL Brightness | `params.value` (default 1.3) |
| `contrast` | PIL Contrast | `params.value` (default 1.5) |
| `saturation` | PIL Color | `params.value` (default 1.5) |
| `background_remove` | rembg u2net | No params |
| `erase` | Alpha → 0 | Makes layer fully transparent |
| `cartoon` | OpenCV bilateral + adaptive threshold | No params |
| `sketch` | OpenCV pencil sketch | No params |
| `pixel_art` | Nearest-neighbor resize | `params.size` (default 16) |
| `upscale` | Real-ESRGAN | `params.scale` (default 2) |
| `text_edit` | EasyOCR + cv2.inpaint + perspective warp | `params.replacements` JSON or `new_text`/`target_text` |
| `replace` | SDXL/FLUX inpaint | Generative replacement |
| `generative_fill` | SDXL/FLUX inpaint | Fill with generated content |
| `anime` | SDXL/FLUX inpaint | Anime style |
| `oil_painting` | SDXL/FLUX inpaint | Oil painting style |
| `style_transfer` | SDXL Img2Img | Full style transfer via img2img |
| `other` | SDXL/FLUX inpaint | Catch-all generative |

Generative edits require 6 GB+ free RAM/VRAM. On low-memory machines, only deterministic edits (blur, sharpen, brightness, contrast, cartoon, sketch, pixel_art) are allowed.

---

## Model Manager

`services/model_manager.py` — singleton (`ModelManager`) with lazy loading:

- `get_grounding_dino()` → IDEA-Research/grounding-dino-base
- `get_sam2()` → SAM2 hiera large (falls back to SAM vit_h)
- `get_inpaint_pipe()` → FLUX.1-dev inpaint (falls back to SDXL inpaint)
- `get_img2img_pipe()` → FLUX img2img (falls back to SDXL img2img)
- `get_llm()` → Qwen2.5-0.5B-Instruct on CPU (falls back to Llama 3.1-8B)
- `get_rembg_session()` → rembg u2net
- `get_realesrgan(scale)` → RealESRGAN_x4plus

After generative edits, `unload_inpaint_pipe()` / `unload_img2img_pipe()` are called to free VRAM.

Currently active LLM: **Qwen2.5-0.5B-Instruct** (CPU, float32). The 7B variant is commented out.

---

## Session & File Layout

Each upload creates a session directory under `temp/`:
```
temp/{session_id}/
├── {original_filename}.jpg          # uploaded image
├── {session_id}_{idx}_{hex}_layer.png   # transparent RGBA layer crop
├── {session_id}_{idx}_{hex}_mask.png    # grayscale mask
├── {session_id}_{idx}_{hex}_edited_{hex}.png  # edited versions (history)
└── session_meta.json                # {session_id, image_path, layers[]}
```

`session_meta.json` is updated after every edit so the session can be restored on page reload via `GET /session/latest`.

---

## Frontend State (Zustand)

`editorStore.ts` manages:
- `sessionId`, `originalImagePath`, `originalImageUrl`, `canvasWidth`, `canvasHeight`
- `layers: LayerData[]` — full layer list
- `selectedLayerIds: string[]`
- `activeTool: Tool` — move | crop | brush | eraser | magic_select | object_select | text_prompt
- `undoStack / redoStack` — full layer snapshots per action
- `canvasScale`, `canvasOffset` — pan/zoom state

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Z` | Undo |
| `Ctrl+Y` / `Ctrl+Shift+Z` | Redo |
| `Ctrl+D` | Duplicate selected layer |
| `Delete` / `Backspace` | Delete selected layer |
| `Ctrl+Enter` | Apply edit (in prompt box) |
| Mouse wheel | Zoom canvas |
| Alt+drag / Middle mouse+drag | Pan canvas |

---

## Running Locally

```bash
# Backend
cd backend
ENV=local uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Frontend (separate terminal)
cd frontend
npm run dev
# → http://localhost:5173
```

Or use the scripts:
```bash
./start.sh        # Linux/macOS
start.bat         # Windows
```

---

## Running on Google Colab / Remote

1. Set `ENV=server` (default) so paths point to Google Drive.
2. Run `start_ngrok.py` to get a public ngrok URL.
3. Update `frontend/src/config.ts` → `baseUrl` with the ngrok URL.
4. Rebuild frontend or use the pre-built version.

---

## Known Issues / Notes

- `project.py` router: `load_project` function is defined but the `@router.post("/load")` decorator is missing — the endpoint is unreachable.
- `prompt_service.py` heuristic fallback: references `params` variable that doesn't exist in `_heuristic_parse` scope (line with `text_edit` branch).
- `config.ts` has the ngrok URL hardcoded — needs updating when the tunnel restarts.
- `vite.config.ts` proxy `baseUrl` is `null` (falls back to `localhost:8000`) — correct for local dev.
- The `models/` directory at root contains duplicate weight files separate from `weights/` — only `weights/` is used by the backend.
- `backend/.venv` is a Windows venv (Scripts/ folder) — not used on Linux; use root `.venv` or system Python.


Step 1 — Detect Text
• Calls POST /text/detect with session_id, layer_id, image_path (original full image)
• Stores detected regions: {bbox, text, color, font_size}

Step 2 — Apply (handleApplyTextEdits)

1. Fetches the layer's PNG (the cropped RGBA layer, e.g. a person crop) as a blob
2. Draws it onto bgCanvas (same W×H as the layer crop)
3. For each edited text region:
   • Converts bbox from full-image coords → layer-local coords by subtracting bbox.x / bbox.y
   • **Erases** old text: bgCtx.clearRect(lx1, ly1, w, h) — makes that area transparent (this is the bug — it punches a hole)
   • Draws new text onto txtCanvas using fillText with detected color/font_size
4. Saves bgCanvas → blob URL → replaces layer's png_path (background with transparent hole)
5. Saves txtCanvas → blob URL → adds as a new layer on top




Canva's AI Object Detection (Magic Grab / Magic Edit) likely uses a combination of:
1. SAM 2 (Segment Anything Model 2) — Meta's model that segments any object in an image given a prompt (point, box, or text). Same as what your project uses.
2. Grounding DINO — Text-to-object detection. You describe what you want ("sneaker", "logo", "phone"), it finds bounding boxes. Same as your project.
3. CLIP — OpenAI's vision-language model for understanding what's in an image and matching text descriptions to visual content.
4. Custom fine-tuned models — Canva has a large design corpus, so they likely have fine-tuned versions of these models specifically for design elements (text, logos, shapes, backgrounds).  ( only difference ) 
 