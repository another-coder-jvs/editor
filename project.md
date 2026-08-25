# AI Image Editor — Project Reference

## Overview

A fully local, browser-based AI image editor that auto-segments any image into per-object layers and lets you edit each layer with natural language prompts. Everything runs on your GPU (or CPU fallback). No API keys, no cloud.

**Key Capabilities:**
- Auto-segmentation via Grounding DINO + SAM2
- Ollama vision model for smart object identification (feeds Grounding DINO only relevant labels)
- Natural language editing via LLM (Qwen2.5) with compound prompt support
- Layer-based editing with 16 blend modes, 14 adjustment sliders, and per-layer transforms
- Canvas drawing tools (brush, pencil, marker, eraser, clone stamp, heal brush, color picker)
- Selection tools (rect, ellipse, lasso, free, magic/SAM2, object/DINO)
- Shape tools (rect, ellipse, line, arrow, triangle, star)
- Text detection & editing (EasyOCR + perspective warp)
- Background reconstruction (LaMa inpainting / cv2 fallback)
- Smart background detection (solid, gradient, textured, complex classification)
- Export with optional Real-ESRGAN upscaling
- Session persistence + project save/load
- Idle model unloading (2 min default, configurable via `MODEL_IDLE_SECONDS`)

---

## Tech Stack

### Backend
- Python 3.11+, FastAPI, Uvicorn
- PyTorch (CUDA / CPU), Transformers, Diffusers
- Grounding DINO (object detection)
- SAM2 / SAM fallback (segmentation)
- SDXL Inpainting (generative edits, primary)
- SDXL Img2Img (style transfer)
- Qwen2.5-0.5B-Instruct (active) / Llama 3.1-8B-Instruct (fallback LLM)
- Real-ESRGAN (upscaling)
- rembg u2net (background removal)
- LaMa (background reconstruction)
- EasyOCR + OpenCV (text detection & editing)
- Ollama + llava:7b (vision object identification)
- Pillow, NumPy, OpenCV, scikit-image

### Frontend
- React 18 + TypeScript + Vite
- Zustand (state management, persisted to localStorage)
- react-konva / Konva (canvas rendering)
- Tailwind CSS
- Axios (API client with loading spinner)
- lucide-react (icons)
- react-toastify (notifications)

---

## Folder Structure

```
ai-image-editor/
├── backend/
│   ├── main.py                  # FastAPI app, router registration, CORS, lifespan
│   ├── schemas.py               # All Pydantic request/response models
│   ├── requirements.txt         # Python dependencies
│   ├── routers/
│   │   ├── detect.py            # POST /detect — upload image, run Grounding DINO
│   │   ├── redetect.py          # POST /detect/re — re-detect with custom prompt
│   │   ├── segment.py           # POST /segment — run SAM2 on detected objects
│   │   ├── layers.py            # POST /layers — rebuild layers for existing session
│   │   ├── edit.py              # POST /edit — apply AI/deterministic edit to layer
│   │   ├── merge.py             # POST /merge — composite layers → output image
│   │   ├── export.py            # POST /export — merge + optional upscale → download
│   │   ├── project.py           # POST /project/save, GET /project/list, DELETE /project/{name}
│   │   ├── session.py           # GET /session/latest, /session/list, /session/{id}, DELETE
│   │   ├── text.py              # POST /text/detect — EasyOCR text detection on layer
│   │   ├── progress.py          # GET /progress/{session_id}
│   │   ├── upload.py            # POST /upload — save image without detection
│   │   ├── identify.py          # POST /identify — Ollama vision object identification
│   │   └── inpaint_bg.py        # POST /inpaint-bg — reconstruct background behind layer
│   ├── services/
│   │   ├── model_manager.py     # Singleton lazy-loader for all AI models (idle-unload)
│   │   ├── detection_service.py # Grounding DINO inference
│   │   ├── segmentation_service.py # SAM2 masks, refinement, transparent PNG layers
│   │   ├── editing_service.py   # Edit dispatcher (recolor, blur, inpaint, compound edits)
│   │   ├── prompt_service.py    # LLM prompt parser + heuristic fallback + compound split
│   │   ├── merge_service.py     # Alpha-composite layers into final canvas
│   │   ├── text_service.py      # EasyOCR detect + cv2.inpaint erase + perspective re-render
│   │   ├── identify_service.py  # Ollama vision → object labels for Grounding DINO
│   │   ├── inpaint_service.py   # Background inpainting (LaMa + cv2 fallback)
│   │   ├── background_detector.py # Smart background type classification + mask generation
│   │   └── progress_store.py    # In-memory progress tracking dict
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
│   │   ├── index.css            # Global styles, animations, dark theme
│   │   ├── api/client.ts        # Axios API functions + loading spinner tracking
│   │   ├── store/editorStore.ts # Zustand store (layers, session, undo/redo, tools, canvas)
│   │   ├── types/index.ts       # TypeScript interfaces (LayerData, Tool, BlendMode, etc.)
│   │   ├── components/
│   │   │   ├── Toolbar.tsx          # Top bar: undo/redo, save, projects, export
│   │   │   ├── LayerPanel.tsx       # Left sidebar: layer list, visibility, reorder
│   │   │   ├── Canvas.tsx           # Konva canvas: image + layer compositing, pan/zoom
│   │   │   ├── PropertiesPanel.tsx  # Right panel: edit prompt, opacity, transform, text edit
│   │   │   ├── ImageUploader.tsx    # Drag-and-drop upload → detect → segment flow
│   │   │   ├── DetectionModeSelector.tsx # Auto vs manual detection mode toggle
│   │   │   ├── ExportModal.tsx      # Export format/upscale options
│   │   │   ├── ProjectManager.tsx   # Save/load/delete projects + session list
│   │   │   ├── HistoryPanel.tsx     # Undo/redo history display
│   │   │   ├── ProgressBar.tsx      # Progress polling overlay
│   │   │   ├── ToolPanel.tsx        # Grouped tool grid (Select, Draw, Shapes, Edit)
│   │   │   ├── ToolSidebar.tsx      # Icon strip with flyout menus per tool group
│   │   │   ├── ToolOptionsPanel.tsx # Context-sensitive options for active tool
│   │   │   ├── LeftPanel.tsx        # Combined tools + layers layout
│   │   │   ├── AdjustPanel.tsx      # Layer adjustment sliders (brightness, contrast, etc.)
│   │   │   ├── AIEditPanel.tsx      # AI prompt editing UI
│   │   │   ├── TransformPanel.tsx   # Layer position, scale, rotation controls
│   │   │   ├── ResizablePanel.tsx   # Resizable container with drag handles
│   │   │   ├── ApiSpinner.tsx       # Loading spinner during API requests
│   │   │   └── TopBarLoader.tsx     # Top progress bar during API requests
│   │   ├── hooks/
│   │   │   ├── useKeyboardShortcuts.ts  # Ctrl+Z/Y, Delete, Ctrl+D, Ctrl+Enter, tool keys
│   │   │   ├── useProgressPoller.ts     # Polls /progress/{id} during long tasks
│   │   │   └── useBlobUrl.ts            # Converts /temp URLs to blob URLs (ngrok-safe)
│   │   └── utils/               # (utility helpers)
│   ├── vite.config.ts           # Vite dev server, proxy all API routes to :8000
│   ├── tailwind.config.js       # Dark theme config, custom colors
│   └── package.json
├── weights/                     # Downloaded model weights (auto on first use)
│   ├── grounding_dino/          # IDEA-Research/grounding-dino-base (~700 MB)
│   ├── sam2/sam2_hiera_large.pt (~900 MB)
│   ├── sam/sam_vit_h_4b8939.pth (SAM fallback, ~2.4 GB)
│   ├── qwen/                    # Qwen2.5-0.5B-Instruct (active, ~1 GB)
│   ├── sdxl_inpaint/            # diffusers/stable-diffusion-xl-1.0-inpainting-0.1 (~6 GB)
│   ├── sdxl_img2img/            # stabilityai/stable-diffusion-xl-base-1.0 (~6 GB)
│   └── realesrgan/RealESRGAN_x4plus.pth (~65 MB)
├── models/                      # Local model files (duplicate, not used by backend)
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
├── start_colab.py               # Google Colab launcher
├── setup_colab.py               # Colab environment setup
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

**Ollama config** (env vars):
- `OLLAMA_URL` — default `http://127.0.0.1:11434`
- `VISION_MODEL` — default `llava:7b` (for auto-identification)
- `MODEL_IDLE_SECONDS` — default `120` (auto-unload timeout)

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/upload` | Save image without running detection → session_id + image_path |
| POST | `/detect` | Upload image (multipart), run Grounding DINO, returns objects + session_id |
| POST | `/detect/re` | Re-detect objects with custom prompt on existing session image |
| POST | `/segment` | Run SAM2 on detected objects, returns LayerData list |
| POST | `/layers` | Rebuild layers for an existing session (re-segments) |
| POST | `/edit` | Apply edit to a single layer (prompt-driven or explicit edit_type) |
| POST | `/merge` | Alpha-composite all visible layers → output image |
| POST | `/export` | Merge + optional Real-ESRGAN upscale → file download |
| POST | `/project/save` | Save project as JSON |
| POST | `/project/load` | Load a saved project |
| GET | `/project/list` | List saved projects |
| DELETE | `/project/{name}` | Delete a project |
| GET | `/session/latest` | Return most recently modified valid session |
| GET | `/session/list` | List all sessions |
| GET | `/session/{id}` | Get specific session metadata |
| DELETE | `/session/{id}` | Delete session directory |
| POST | `/text/detect` | EasyOCR text detection on a layer |
| POST | `/identify` | Ollama vision → identify objects in image (returns dot-separated labels) |
| POST | `/inpaint-bg` | Reconstruct background behind a layer using its mask |
| GET | `/progress/{id}` | Poll task progress (0.0–1.0) |
| GET | `/health` | Health check |
| GET | `/temp/{path}` | Browse/serve temp files (HTML dir listing or FileResponse) |
| POST | `/models/unload` | Force-unload all AI models to free RAM + VRAM |
| GET | `/identify/status` | Check if Ollama vision model is available |

---

## Edit Types

| edit_type | Handler | Notes |
|-----------|---------|-------|
| `recolor` | `_recolor` (HSV manipulation) | No GPU needed. Maps color name → HSV hue. Supports 30+ colors + fuzzy matching. |
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
| `replace` | SDXL inpaint | Generative replacement |
| `generative_fill` | SDXL inpaint | Fill with generated content |
| `anime` | SDXL inpaint | Anime style |
| `oil_painting` | SDXL inpaint | Oil painting style |
| `style_transfer` | SDXL Img2Img | Full style transfer via img2img |
| `other` | SDXL inpaint | Catch-all generative (add patterns, etc.) |

**Compound edits:** The prompt service detects multi-instruction prompts (e.g. "make it blue AND add flower pattern") and executes each edit sequentially.

Generative edits require 6 GB+ free RAM/VRAM. On low-memory machines, only deterministic edits are allowed.

---

## Model Manager

`services/model_manager.py` — singleton (`ModelManager`) with lazy loading:

| Model | Getter | Source | Fallback |
|-------|--------|--------|----------|
| Grounding DINO | `get_grounding_dino()` | IDEA-Research/grounding-dino-base | — |
| SAM2 | `get_sam2()` | SAM2 hiera large | SAM vit_h |
| SDXL Inpaint | `get_inpaint_pipe()` | diffusers/stable-diffusion-xl-1.0-inpainting-0.1 | — |
| SDXL Img2Img | `get_img2img_pipe()` | stabilityai/stable-diffusion-xl-base-1.0 | — |
| LLM | `get_llm()` | Qwen2.5-0.5B-Instruct (CPU, float32) | Llama 3.1-8B |
| rembg | `get_rembg_session()` | rembg u2net (ONNX) | — |
| Real-ESRGAN | `get_realesrgan(scale)` | RealESRGAN_x{scale}plus | — |

**Idle unload:** All models auto-unload after `MODEL_IDLE_SECONDS` (default 120s) of inactivity. The `touch()` method resets the timer on every access. Call `unload_all()` via `POST /models/unload` to force-free memory.

After generative edits, `unload_inpaint_pipe()` / `unload_img2img_pipe()` are called to free VRAM immediately.

---

## Smart Detection Pipeline

The system supports two detection modes, selectable via `DetectionModeSelector`:

### Auto Mode
1. **Upload image** → `POST /detect` (no prompt)
2. **Ollama vision** (`POST /identify`) analyzes the image using `llava:7b`
   - Returns dot-separated object labels (e.g. "person, shoe, chair, lamp")
   - Only prominent objects, max 10, no background/texture/color labels
   - Ollama auto-starts if installed but not running
3. **Grounding DINO** uses the vision model's output as its detection prompt
   - Much more accurate than hardcoded "everything" prompts
   - Feeds only relevant labels → less VRAM, better precision
4. **SAM2** segments each detected object

### Manual Mode
1. User provides a custom detection prompt
2. Grounding DINO uses that prompt directly
3. SAM2 segments the results

### Re-detection
- `POST /detect/re` — re-run detection on an existing session image with a new prompt
- No need to re-upload the image

---

## Background Services

### Background Detection (`background_detector.py`)
Analyzes image to determine background type and create masks:
- **SOLID_COLOR** — uniform background → color-distance mask with flood-fill
- **NEARLY_SOLID** — minor variations → same as solid with higher tolerance
- **GRADIENT** — smooth directional change → gradient-aware mask
- **COMPLEX** / **TEXTURED** — returns None (use SAM2 or manual approach)

Sampling strategy: borders + corners (weighted 4x) for dominant color detection.

### Background Inpainting (`inpaint_service.py`)
Reconstructs background behind a removed/moved layer:
- Primary: LaMa (Large Mask Inpainting)
- Fallback: cv2.inpaint (Telea + Navier-Stokes)

---

## Session & File Layout

Each upload creates a session directory under `temp/`:
```
temp/{session_id}/
├── {original_filename}.jpg              # uploaded image
├── {session_id}_{idx}_{hex}_layer.png   # transparent RGBA layer crop
├── {session_id}_{idx}_{hex}_mask.png    # grayscale mask
├── {session_id}_{idx}_{hex}_edited_{hex}.png  # edited versions (history)
├── {session_id}_bg_reconstructed_{hex}.png    # background reconstruction
├── {session_id}_bg_mask.png            # background detection mask
└── session_meta.json                    # {session_id, image_path, layers[]}
```

`session_meta.json` is updated after every edit so the session can be restored on page reload via `GET /session/latest`.

---

## Frontend Architecture

### Layout (App.tsx)
```
┌─────────────────────────────────────────────────────────┐
│ Toolbar (undo/redo, save, projects, export)             │
├──────────┬──────────────────────────────┬───────────────┤
│ Layers   │        Canvas (Konva)        │ Tools &       │
│ Panel    │                              │ Inspector     │
│          │                              │               │
│          │  (or ImageUploader when       │  ToolPanel    │
│          │   no image loaded)           │  +            │
│          │                              │  Properties   │
│          │                              │  Panel        │
├──────────┴──────────────────────────────┴───────────────┤
│ ProgressBar (polling overlay)                           │
└─────────────────────────────────────────────────────────┘
```

All panels are resizable via `ResizablePanel` drag handles.

### State (Zustand — `editorStore.ts`)
- `sessionId`, `originalImagePath`, `originalImageUrl`, `canvasWidth`, `canvasHeight`
- `layers: LayerData[]` — full layer list with adjustments, blend modes, group_id
- `selectedLayerIds: string[]`
- `activeTool: Tool` — 22 tools (see below)
- `toolOptions: ToolOptions` — brush color/size/opacity/hardness, shape stroke/fill, selection feather
- `undoStack / redoStack` — full layer snapshots per action
- `canvasScale`, `canvasOffset` — pan/zoom state
- `currentProjectName` — for save/load

### Tools (22 total)

**Select & Move:**
| Tool | Shortcut | Description |
|------|----------|-------------|
| `move` | V | Drag to move layer, arrow keys to nudge |
| `rect_select` | M | Rectangle selection |
| `ellipse_select` | — | Ellipse selection |
| `lasso_select` | L | Freehand lasso selection |
| `free_select` | — | Scissors-style free selection |
| `magic_select` | W | SAM2-powered smart selection |
| `object_select` | — | Grounding DINO object click selection |

**Draw:**
| Tool | Shortcut | Description |
|------|----------|-------------|
| `brush` | B | Soft/hard brush (color, size, opacity, hardness) |
| `pencil` | P | Hard-edge pencil |
| `marker` | — | Semi-transparent marker |
| `eraser` | E | Erase pixels (size adjustable) |
| `color_picker` | I | Pick color from canvas |
| `clone` | S | Clone stamp (Alt+Click to set source) |
| `heal` | J | Heal brush |

**Shapes:**
| Tool | Description |
|------|-------------|
| `shape_rect` | Rectangle (stroke + fill) |
| `shape_ellipse` | Ellipse |
| `shape_line` | Line |
| `shape_arrow` | Arrow |
| `shape_triangle` | Triangle |
| `shape_star` | Star |

**Edit:**
| Tool | Shortcut | Description |
|------|----------|-------------|
| `crop` | C | Crop canvas (Free, 1:1, 4:3, 16:9, 3:2) |
| `text_add` | T | Add text layer |
| `text_prompt` | — | AI prompt editing mode |

### Layer Data (TypeScript)
```typescript
interface LayerData {
  id: string              // "{session_id}_{idx}_{hex6}"
  name: string            // Grounding DINO label
  mask_path: string       // URL path to mask PNG
  png_path: string        // URL path to layer PNG
  bbox: { x, y, width, height }
  z_index: number
  visible: boolean
  opacity: number         // 0-1
  position: { x, y }     // user-draggable offset
  scale: { x, y }
  rotation: number
  history: string[]       // previous png_paths for undo
  locked: boolean
  blend_mode: BlendMode   // 16 modes
  adjustments: LayerAdjustments  // 14 sliders
  group_id?: string
}

type BlendMode = 'normal' | 'multiply' | 'screen' | 'overlay' | 'darken' |
  'lighten' | 'color-dodge' | 'color-burn' | 'hard-light' | 'soft-light' |
  'difference' | 'exclusion' | 'hue' | 'saturation' | 'color' | 'luminosity'

interface LayerAdjustments {
  brightness: number   // 0-200, default 100
  contrast: number     // 0-200, default 100
  saturation: number   // 0-200, default 100
  exposure: number     // -100 to 100, default 0
  highlights: number   // -100 to 100, default 0
  shadows: number      // -100 to 100, default 0
  temperature: number  // -100 to 100, default 0
  tint: number         // -100 to 100, default 0
  hue: number          // -180 to 180, default 0
  sharpness: number    // 0-200, default 100
  clarity: number      // 0-100, default 0
  fade: number         // 0-100, default 0
  vignette: number     // 0-100, default 0
  grain: number        // 0-100, default 0
}
```

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Z` | Undo |
| `Ctrl+Y` / `Ctrl+Shift+Z` | Redo |
| `Ctrl+D` | Duplicate selected layer |
| `Delete` / `Backspace` | Delete selected layer |
| `Ctrl+Enter` | Apply edit (in prompt box) |
| `Ctrl+S` | Save project |
| `V` | Move tool |
| `B` | Brush tool |
| `E` | Eraser tool |
| `P` | Pencil tool |
| `I` | Color picker |
| `M` | Rect select |
| `W` | Magic select |
| `L` | Lasso select |
| `C` | Crop tool |
| `T` | Text tool |
| `S` | Clone stamp |
| `J` | Heal brush |
| `Esc` | Clear selection |
| Mouse wheel | Zoom canvas |
| Alt+drag / Middle mouse+drag | Pan canvas |
| Arrow keys | Nudge layer (Shift = 10px) |

---

## Prompt Routing

`services/prompt_service.py → parse_edit_prompt`:

1. **Compound detection:** Checks for separators ("and", "also", ";", ", then", "plus")
2. **LLM attempt:** Sends prompt + layer name to Qwen2.5 with structured system prompt
3. **JSON extraction:** Regex-based extraction from LLM output (single object or array)
4. **Normalization:** Maps invalid edit types, forces correct inpaint_prompt, color word normalization
5. **Heuristic fallback:** Keyword-based parsing if LLM fails
6. **Compound splitting:** If LLM returns single edit for compound prompt, splits and re-parses

The system prompt enforces:
- "make X color" → `recolor` with `edit_params.color`
- "add pattern" → `other` with `inpaint_prompt`
- Color normalization (navy→blue, teal→green, etc.)

---

## Request/Response Flow

### Upload → Detect → Segment
```
ImageUploader → processFile()
  ↓ POST /detect (multipart) or POST /upload + POST /identify + POST /detect/re
  ↓ Backend: Grounding DINO → objects list
  ↓ Frontend: setSession(session_id, image_path, blobUrl, w, h)
  ↓ POST /segment {session_id, image_path, objects}
  ↓ Backend: SAM2 → LayerData[] → session_meta.json
  ↓ Frontend: setLayers(layers) → Canvas renders
```

### Edit Layer
```
PropertiesPanel → editLayer()
  ↓ POST /edit {session_id, layer_id, prompt, image_path, ...}
  ↓ Backend: prompt_service.parse_edit_prompt() → LLM or heuristic
  ↓ Backend: editing_service dispatches to handler
  ↓ Backend: saves edited PNG → updates session_meta.json
  ↓ Frontend: updateLayer(png_path) → Canvas re-renders
```

### Export
```
ExportModal → POST /export {session_id, layers, w, h, format, upscale}
  ↓ Backend: merge_service.merge_layers() → alpha composite
  ↓ Backend: optional Real-ESRGAN upscale
  ↓ Backend: saves to outputs/ → returns FileResponse
  ↓ Frontend: blob download
```

### Session Restore (on page load)
```
App.tsx useEffect → GET /session/latest
  ↓ Backend: scan temp/ for most recent session_meta.json
  ↓ Validate all layer PNGs exist
  ↓ Frontend: fetch image blob → setSession + setLayers
  ↓ toast.info("Resumed last session")
```

---

## Storage

No database. All persistence is filesystem-based.

| Storage | Path | Format | Contents |
|---------|------|--------|----------|
| Session working files | `temp/{session_id}/` | PNG + JSON | Layer PNGs, masks, edited PNGs, session_meta.json |
| Project saves | `projects/{name}.json` | JSON | Full SaveRequest payload |
| Exported images | `outputs/export_{uuid}.{fmt}` | PNG/JPEG/WebP | Final composited image |
| Model weights | `weights/` | .pth / HF cache | AI model files |
| Frontend state | `localStorage["editor-store"]` | JSON | Zustand persisted state |

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

**Ollama (optional, for auto-detection):**
```bash
# Install Ollama, then:
ollama pull llava:7b
# The backend auto-starts Ollama if installed but not running
```

---

## Running on Google Colab / Remote

1. Set `ENV=server` (default) so paths point to Google Drive.
2. Run `start_ngrok.py` or `start_colab.py` to get a public ngrok URL.
3. Update `frontend/src/config.ts` → `baseUrl` with the ngrok URL.
4. Rebuild frontend or use the pre-built version.

---

## Known Issues / Notes

- `project.py` router: `load_project` function is defined but the `@router.post("/load")` decorator is missing — the endpoint is unreachable via the `/project` prefix (it's mounted directly).
- `config.ts` has the ngrok URL hardcoded — needs updating when the tunnel restarts.
- `vite.config.ts` proxy `baseUrl` is `null` (falls back to `localhost:8000`) — correct for local dev.
- The `models/` directory at root contains duplicate weight files separate from `weights/` — only `weights/` is used by the backend.
- `backend/.venv` is a Windows venv (Scripts/ folder) — not used on Linux; use root `.venv` or system Python.
- The `identify_service` auto-starts Ollama but waits 3s which may not be enough on slow machines.
- `setup_colab.py` has `pkill -f 'ollama serve'` commented out to avoid killing the server during setup.
