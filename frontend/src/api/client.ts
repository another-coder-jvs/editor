import axios from 'axios'
import { LayerData, ProgressInfo } from '../types'
import {baseUrl} from "@/config"
const api = axios.create({ baseURL: baseUrl || "http://localhost:8000"})

export async function detectObjects(
  file: File,
  prompt?: string
): Promise<{ session_id: string; objects: Array<{ label: string; score: number; bbox: { x: number; y: number; width: number; height: number } }> }> {
  const form = new FormData()
  form.append('file', file)
  if (prompt) form.append('prompt', prompt)
  const { data } = await api.post('/detect', form)
  return data
}

export async function segmentObjects(
  session_id: string,
  image_path: string,
  objects: unknown[]
): Promise<{ session_id: string; layers: LayerData[] }> {
  const { data } = await api.post('/segment', { session_id, image_path, objects })
  return data
}

export async function editLayer(payload: {
  session_id: string
  layer_id: string
  prompt: string
  image_path: string
  strength?: number
  guidance_scale?: number
  steps?: number
}): Promise<{ layer_id: string; edited_png_path: string; session_id: string }> {
  const { data } = await api.post('/edit', payload)
  return data
}

export async function mergeLayers(payload: {
  session_id: string
  layers: LayerData[]
  canvas_width: number
  canvas_height: number
  output_format?: string
}): Promise<{ output_path: string; session_id: string }> {
  const { data } = await api.post('/merge', payload)
  return data
}

export async function saveProject(payload: {
  session_id: string
  project_name: string
  original_image_path: string
  layers: LayerData[]
  canvas_width: number
  canvas_height: number
  canvas_position: { x: number; y: number }
  settings: Record<string, unknown>
  prompts: string[]
}): Promise<{ status: string; path: string }> {
  const { data } = await api.post('/project/save', payload)
  return data
}

export async function loadProject(project_name: string): Promise<unknown> {
  const { data } = await api.post('/project/load', { project_name })
  return data
}

export async function exportImage(payload: {
  session_id: string
  layers: LayerData[]
  canvas_width: number
  canvas_height: number
  format?: string
  upscale?: boolean
  upscale_factor?: number
}): Promise<Blob> {
  const { data } = await api.post('/export', payload, { responseType: 'blob' })
  return data
}

export async function getProgress(session_id: string): Promise<ProgressInfo> {
  const { data } = await api.get(`/progress/${session_id}`)
  return data
}

export async function getLatestSession(): Promise<{
  session: {
    session_id: string
    image_path: string
    layers: LayerData[]
  } | null
}> {
  const { data } = await api.get('/session/latest')
  return data
}
