# AI Image Editor

A production-quality, fully local AI image editor that automatically separates any image into semantic layers — one layer per detected object — and lets you edit each layer independently using natural language prompts.

Everything runs **100% locally on your GPU**. No API keys. No cloud services.

---

## Features

- **Auto-segmentation** — Upload any image; Grounding DINO detects every object and SAM2 creates a precise mask for each one
- **Layer-based editing** — Every object becomes its own transparent PNG layer, just like Photoshop
- **Natural language edits** — Type `"make shirt blue"`, `"replace car with Ferrari"`, `"cartoon style"` — the LLM routes the edit to the correct layer and model
- **Non-destructive** — Full undo/redo history per layer
- **Export** — PNG, JPEG, WebP; optional Real-ESRGAN upscaling
- **Project save/load** — Full JSON project format preserves all layers, masks, and history

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Frontend (React)                     │
│  Toolbar │ Layer Panel │ Canvas (Konva) │ Properties     │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP (Vite proxy)
┌────────────────────────▼────────────────────────────────┐
│                   Backend (FastAPI)                      │
│  /detect  /segment  /edit  /merge  /export  /project    │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                   AI Pipeline                            │
│  Grounding DINO → SAM2 → LLM → FLUX/SDXL → Real-ESRGAN │
└─────────────────────────────────────────────────────────┘
```

---

## Folder Structure

```
ai-image-editor/
├── frontend/               # React + TypeScript + Vite
│   ├── src/
│   │   ├── components/     # UI components
│   │   ├── store/          # Zustand state
│   │   ├── api/            # API client
│   │   ├── hooks/          # Custom hooks
│   │   └── types/          # TypeScript types
│   └── package.json
├── backend/                # FastAPI Python backend
│   ├── main.py             # App entry point
│   ├── schemas.py          # Pydantic models
│   ├── routers/            # API route handlers
│   ├── services/           # Business logic
│   │   ├── model_manager.py
│   │   ├── detection_service.py
│   │   ├── segmentation_service.py
│   │   ├── editing_service.py
│   │   ├── merge_service.py
│   │   ├── prompt_service.py
│   │   └── progress_store.py
│   └── tests/              # Unit tests
├── weights/                # Downloaded model weights (auto)
├── projects/               # Saved project JSON files
├── outputs/                # Exported images
├── temp/                   # Session working files
├── Dockerfile
├── docker-compose.yml
├── environment.yml
├── setup.py
├── install.sh / install.bat
└── start.sh / start.bat
```

---

## Installation

### Requirements

- Python 3.11+
- Node.js 18+
- NVIDIA GPU with CUDA 11.8+ (CPU fallback available, much slower)
- 16 GB+ VRAM recommended (8 GB minimum with CPU offload)

### One-Click Install

**Linux / macOS:**
```bash
git clone <repo-url>
cd ai-image-editor
chmod +x install.sh start.sh
./install.sh
```

**Windows:**
```
install.bat
```

### Manual Install

```bash
# Backend
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt

# SAM2 (optional, falls back to SAM if unavailable)
pip install git+https://github.com/facebookresearch/segment-anything-2.git

# Frontend
cd frontend
npm install
```

### Conda

```bash
conda env create -f environment.yml
conda activate ai-image-editor
```

### Docker

```bash
docker-compose up --build
```

---

## Starting the Application

**Linux / macOS:**
```bash
./start.sh
```

**Windows:**
```
start.bat
```

Then open **http://localhost:5173** in your browser.

---

## Model Download

All models are downloaded automatically on first use and cached in `./weights/`.

| Model | Size | Purpose |
|-------|------|---------|
| Grounding DINO Base | ~700 MB | Object detection |
| SAM2 Hiera Large | ~900 MB | Precise segmentation |
| FLUX.1-dev (inpaint) | ~24 GB | AI editing (primary) |
| SDXL Inpainting | ~6 GB | AI editing (fallback) |
| Qwen2.5-7B-Instruct | ~15 GB | Prompt understanding |
| Llama 3.1-8B-Instruct | ~16 GB | Prompt understanding (fallback) |
| Real-ESRGAN x2/x4 | ~65 MB | Upscaling |
| rembg u2net | ~170 MB | Background removal |

> **Tip:** If VRAM is limited, the system automatically uses CPU offloading and falls back to lighter models.

---

## GPU Setup

### CUDA (NVIDIA)
```bash
# Verify CUDA is available
python -c "import torch; print(torch.cuda.is_available())"
```

If `False`, install the correct PyTorch version for your CUDA:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### CPU Fallback
The application automatically falls back to CPU if no GPU is detected. Expect 10–50× slower inference.

### Mixed Precision
On CUDA, `torch.float16` is used automatically for all models to halve VRAM usage.

---

## How the Editing Pipeline Works

```
1. User uploads image
        ↓
2. Grounding DINO scans the image with a broad prompt
   ("person . car . sky . tree . building …")
   → Returns bounding boxes + labels for every detected object
        ↓
3. SAM2 receives each bounding box as a prompt
   → Produces a precise pixel-level mask for each object
        ↓
4. Mask refinement:
   - Morphological close (fill small holes)
   - Flood-fill (fill large interior holes)
   - Morphological open (remove noise)
   - Gaussian blur (feather edges / anti-aliasing)
        ↓
5. Each object is saved as a transparent RGBA PNG
   (cropped to its bounding box, mask applied as alpha channel)
        ↓
6. Frontend displays layers in the sidebar
        ↓
7. User selects a layer and types an edit prompt
        ↓
8. LLM (Qwen2.5 / Llama 3.1) parses the prompt:
   { target_object, target_region, edit_type, edit_params, inpaint_prompt }
        ↓
9. Edit is routed to the correct handler:
   - recolor / blur / sharpen / brightness / contrast / saturation
     → Fast deterministic PIL/OpenCV transforms (no GPU needed)
   - cartoon / sketch / pixel_art
     → OpenCV-based style transforms
   - background_remove → rembg (u2net)
   - replace / generative_fill / anime / oil_painting
     → FLUX Kontext inpainting (or SDXL fallback)
   - upscale → Real-ESRGAN
        ↓
10. Edited layer PNG replaces the original in the session
        ↓
11. User clicks Export → all visible layers are alpha-composited
    in z-index order → final PNG/JPEG/WebP is saved
```

---

## How Masks Are Created

1. **Grounding DINO** provides a bounding box `[x1, y1, x2, y2]` for each object.
2. **SAM2** (`SAM2ImagePredictor`) receives the bounding box as a box prompt and returns 3 candidate masks. The highest-scoring mask is selected.
3. **Refinement** (`services/segmentation_service.py → refine_mask`):
   - `cv2.morphologyEx(MORPH_CLOSE)` — closes small gaps
   - Flood-fill from corner — fills interior holes
   - `cv2.morphologyEx(MORPH_OPEN)` — removes isolated noise pixels
   - `cv2.GaussianBlur` — feathers edges for smooth compositing
4. The refined mask is saved as a grayscale PNG and also applied as the alpha channel of the layer's RGBA PNG.

---

## How Merging Works

`services/merge_service.py → merge_layers`:

1. Layers are sorted by `z_index` (ascending = bottom to top).
2. Hidden layers (`visible=False`) are skipped.
3. For each visible layer:
   - Opacity is applied by scaling the alpha channel.
   - Rotation and scale transforms are applied with `Image.rotate` / `Image.resize`.
   - The layer is pasted onto a full-canvas temporary image at `(bbox.x + position.x, bbox.y + position.y)`.
   - `Image.alpha_composite` blends it onto the accumulating canvas — this correctly handles soft edges, semi-transparent pixels, and shadow blending.
4. The final RGBA canvas is saved in the requested format.

---

## How Prompt Routing Works

`services/prompt_service.py → parse_edit_prompt`:

1. The user's prompt and the layer name are sent to the LLM with a structured system prompt.
2. The LLM returns JSON:
   ```json
   {
     "target_object": "person",
     "target_region": "shirt",
     "edit_type": "recolor",
     "edit_params": { "color": "blue" },
     "inpaint_prompt": "blue shirt"
   }
   ```
3. If the LLM is unavailable or returns invalid JSON, a keyword-based heuristic fallback is used.
4. `editing_service.py` routes `edit_type` to the correct handler function.

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/detect` | POST | Upload image, run Grounding DINO |
| `/segment` | POST | Run SAM2 on detected objects |
| `/layers` | POST | Rebuild layers for a session |
| `/edit` | POST | Apply AI edit to a single layer |
| `/merge` | POST | Composite all layers into one image |
| `/project/save` | POST | Save project as JSON |
| `/project/load` | POST | Load a saved project |
| `/export` | POST | Export final image (with optional upscale) |
| `/progress/{id}` | GET | Poll task progress |
| `/health` | GET | Health check |

---

## Running Tests

```bash
cd backend
pytest tests/ -v
```

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Z` | Undo |
| `Ctrl+Y` / `Ctrl+Shift+Z` | Redo |
| `Ctrl+D` | Duplicate selected layer |
| `Delete` / `Backspace` | Delete selected layer |
| `Ctrl+Enter` (in prompt box) | Apply edit |
| Mouse wheel | Zoom canvas |
| Alt + drag | Pan canvas |
| Middle mouse + drag | Pan canvas |

---

## License

MIT License — all dependencies are open-source.
