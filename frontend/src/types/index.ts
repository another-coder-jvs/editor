export interface BoundingBox {
  x: number
  y: number
  width: number
  height: number
}

export interface LayerAdjustments {
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

export interface LayerData {
  id: string
  name: string
  mask_path: string
  png_path: string
  bbox: BoundingBox
  z_index: number
  visible: boolean
  opacity: number
  position: { x: number; y: number }
  scale: { x: number; y: number }
  rotation: number
  history: string[]
  locked: boolean
  blend_mode: BlendMode
  adjustments: LayerAdjustments
  group_id?: string
}

export type BlendMode =
  | 'normal' | 'multiply' | 'screen' | 'overlay'
  | 'darken' | 'lighten' | 'color-dodge' | 'color-burn'
  | 'hard-light' | 'soft-light' | 'difference' | 'exclusion'
  | 'hue' | 'saturation' | 'color' | 'luminosity'

export interface DetectedObject {
  label: string
  score: number
  bbox: BoundingBox
}

export type Tool =
  | 'move'
  | 'crop'
  | 'brush'
  | 'eraser'
  | 'magic_select'
  | 'object_select'
  | 'text_prompt'
  | 'rect_select'
  | 'ellipse_select'
  | 'lasso_select'
  | 'free_select'
  | 'text_add'
  | 'shape_rect'
  | 'shape_ellipse'
  | 'shape_line'
  | 'shape_arrow'
  | 'shape_triangle'
  | 'shape_star'
  | 'clone'
  | 'heal'
  | 'pencil'
  | 'marker'
  | 'color_picker'

export type ExportFormat = 'png' | 'jpeg' | 'webp'

export interface ProgressInfo {
  session_id: string
  task: string
  progress: number
  message: string
  done: boolean
}

export interface ProjectData {
  session_id: string
  project_name: string
  original_image_path: string
  layers: LayerData[]
  canvas_width: number
  canvas_height: number
  canvas_position: { x: number; y: number }
  settings: Record<string, unknown>
  prompts: string[]
}

export const DEFAULT_ADJUSTMENTS: LayerAdjustments = {
  brightness: 100, contrast: 100, saturation: 100,
  exposure: 0, highlights: 0, shadows: 0,
  temperature: 0, tint: 0, hue: 0,
  sharpness: 100, clarity: 0, fade: 0,
  vignette: 0, grain: 0,
}
