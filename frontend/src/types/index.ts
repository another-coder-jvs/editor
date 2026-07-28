export interface BoundingBox {
  x: number
  y: number
  width: number
  height: number
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
}

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
