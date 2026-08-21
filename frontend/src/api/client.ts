import axios from 'axios'
import { LayerData, ProgressInfo } from '../types'
import {baseUrl} from "@/config"

// ── Global loading counter (module-level, reactive via listeners) ──
let _pending = 0
const _listeners = new Set<() => void>()
export const apiLoading = {
  get active() { return _pending > 0 },
  subscribe(fn: () => void) { _listeners.add(fn); return () => { _listeners.delete(fn) } },
  _inc() { _pending++; _listeners.forEach(f => f()) },
  _dec() { _pending = Math.max(0, _pending - 1); _listeners.forEach(f => f()) },
}

/** Drop-in replacement for fetch() that triggers the API spinner */
export async function trackedFetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  apiLoading._inc()
  try {
    return await fetch(input, init)
  } finally {
    apiLoading._dec()
  }
}

const api = axios.create({
  baseURL: baseUrl || "http://localhost:8000",
  headers: { 'ngrok-skip-browser-warning': '1' },
})

api.interceptors.request.use(cfg  => { apiLoading._inc(); return cfg })
api.interceptors.response.use(
  res => { apiLoading._dec(); return res },
  err => { apiLoading._dec(); return Promise.reject(err) },
)

export async function uploadFile(
  file: File,
): Promise<{ session_id: string; image_path: string }> {
  const form = new FormData()
  form.append('file', file)
  const { data } = await api.post('/upload', form)
  return data
}

export async function detectObjects(
  file: File,
  prompt?: string
): Promise<{ session_id: string; image_path: string; objects: Array<{ label: string; score: number; bbox: { x: number; y: number; width: number; height: number } }> }> {
  const form = new FormData()
  form.append('file', file)
  if (prompt) form.append('prompt', prompt)
  const { data } = await api.post('/detect', form)
  return data
}

export async function detectObjectsWithPrompt(
  session_id: string,
  image_path: string,
  prompt: string,
): Promise<{ session_id: string; image_path: string; objects: Array<{ label: string; score: number; bbox: { x: number; y: number; width: number; height: number } }> }> {
  const form = new FormData()
  // We need to send a file for the detect endpoint, but we already have the session.
  // So we send a re-detect request with the prompt.
  const { data } = await api.post('/detect', form, {
    params: { session_id, image_path, prompt },
  })
  return data
}

export async function identifyObjects(
  image_path: string,
): Promise<{ objects: string }> {
  const { data } = await api.post('/identify', { image_path })
  return data
}

export async function redetectObjects(
  session_id: string,
  image_path: string,
  prompt?: string,
): Promise<{ session_id: string; image_path: string; objects: Array<{ label: string; score: number; bbox: { x: number; y: number; width: number; height: number } }> }> {
  const { data } = await api.post('/detect/re', { session_id, image_path, prompt: prompt || null })
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
  edit_type?: string
  edit_params?: Record<string, string>
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

export async function listSessions(): Promise<{ sessions: { session_id: string; image_path: string; layer_count: number; mtime: number }[] }> {
  const { data } = await api.get('/session/list')
  return data
}

export async function loadSession(session_id: string): Promise<{ session: { session_id: string; image_path: string; layers: LayerData[] } }> {
  const { data } = await api.get(`/session/${session_id}`)
  return data
}

export async function deleteSession(session_id: string): Promise<void> {
  await api.delete(`/session/${session_id}`)
}

export async function listProjects(): Promise<{ projects: string[] }> {
  const { data } = await api.get('/project/list')
  return data
}

export async function deleteProject(name: string): Promise<void> {
  await api.delete(`/project/${name}`)
}

export async function reconstructBackground(payload: {
  session_id: string
  layer_id: string
  image_path: string
  mask_path: string
}): Promise<{ path: string }> {
  const { data } = await api.post('/inpaint-bg', payload)
  return data
}
