# User Flows

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
   ┌────────────────────┐          │
   │  Ollama Vision     │          │
   │  (llava:7b)        │          │
   │  → identify objects│          │
   └────────┬───────────┘          │
            │                      │
            │  dot-separated       │
            │  labels              │
            ▼                      │
   ┌────────────────────┐          │
   │ Grounding DINO     │◄─────────┘
   │ 1.5                │   (user prompt
   └────────┬───────────┘    or auto labels)
            │
        object bbox
            │
            ▼
   ┌────────────────────┐
   │      SAM 2         │
   └────────┬───────────┘
            │
       pixel mask
            │
            ▼
   ┌────────────────────┐
   │  Layer Extraction  │
   │  → RGBA PNG layers │
   │  → grayscale masks │
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
            ┌──────────────────────┐
            │   Prompt Service     │
            │   parse_edit_prompt  │
            └──────────┬───────────┘
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
          │  into    │ │  (Qwen2.5)   │
          │  sub-    │ └──────┬───────┘
          │  prompts │        │
          └────┬─────┘        │
               │         ┌────┴────┐
               │         │ success?│
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
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐  ┌────────────────┐  ┌────────────────┐
│ Deterministic│  │   Generative   │  │  Style         │
│ Edits        │  │   (SDXL)       │  │  Transfer      │
│              │  │                │  │  (SDXL Img2Img)│
├──────────────┤  ├────────────────┤  ├────────────────┤
│ • recolor    │  │ • replace      │  │ • anime        │
│ • blur       │  │ • generative   │  │ • oil_painting │
│ • sharpen    │  │   _fill        │  │ • style_xfer   │
│ • brightness │  │ • other        │  │                │
│ • contrast   │  │                │  │                │
│ • saturation │  │                │  │                │
│ • erase      │  │                │  │                │
│ • cartoon    │  │                │  │                │
│ • sketch     │  │                │  │                │
│ • pixel_art  │  │                │  │                │
│ • text_edit  │  │                │  │                │
│ • bg_remove  │  │                │  │                │
│ • upscale    │  │                │  │                │
└──────┬───────┘  └───────┬────────┘  └───────┬────────┘
       │                  │                   │
       └──────────────────┼───────────────────┘
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
└──────────────┘  └──────────────┘         │
                                           ▼
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
                └──────────┬──────────┘
                           │
                    ┌──────┴──────┐
                    │  upscale?   │
                    └──┬───────┬──┘
                     YES       NO
                      │        │
                      ▼        │
           ┌──────────────┐    │
           │ Real-ESRGAN  │    │
           │ 2x or 4x     │    │
           └──────┬───────┘    │
                  │            │
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
         │  Page Close   │
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

         ┌──────────────┐
         │  Idle Timer   │
         └──────┬───────┘
                │
                ▼
         ┌─────────────────────┐
         │ After MODEL_IDLE_   │
         │ SECONDS (120s)      │
         │ → models unloaded   │
         │ → VRAM freed        │
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
     ┌─────────────────┐     ┌─────────────────┐
     │ LLM / heuristic │     │ LLM / heuristic │
     └────────┬────────┘     └────────┬────────┘
              │                       │
              ▼                       ▼
     ┌─────────────────┐     ┌─────────────────┐
     │ edit_type:      │     │ edit_type:      │
     │ "recolor"       │     │ "other"         │
     │ color: "blue"   │     │ inpaint_prompt: │
     └────────┬────────┘     │ "flower pattern"│
              │              └────────┬────────┘
              │                       │
              ▼                       ▼
     ┌─────────────────┐     ┌─────────────────┐
     │ HSV recolor     │     │ SDXL inpaint    │
     │ (no GPU needed) │     │ (generative)    │
     └────────┬────────┘     └────────┬────────┘
              │                       │
              └───────────┬───────────┘
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
      ┌──────────────┐   ┌──────────────┐
      │ LaMa         │   │ SAM2 mask    │
      │ inpainting   │   │ or manual    │
      │ (primary)    │   │              │
      └──────┬───────┘   └──────┬───────┘
             │                  │
        ┌────┴────┐             │
        │ success?│             │
        └──┬───┬──┘             │
         YES   NO               │
          │     │               │
          │     ▼               │
          │ ┌──────────┐       │
          │ │ cv2.inpaint      │
          │ │ (fallback)│       │
          │ └─────┬────┘       │
          │       │            │
          └───────┴────────────┘
                    │
                    ▼
           ┌────────────────┐
           │ Reconstructed  │
           │ background PNG │
           └────────────────┘
```
