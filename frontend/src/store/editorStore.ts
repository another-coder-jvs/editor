import { create } from 'zustand'
import { LayerData, Tool, ProgressInfo } from '../types'

interface EditorState {
  // Session
  sessionId: string | null
  originalImagePath: string | null
  originalImageUrl: string | null
  canvasWidth: number
  canvasHeight: number

  // Layers
  layers: LayerData[]
  selectedLayerIds: string[]

  // Tool
  activeTool: Tool

  // Progress
  progress: ProgressInfo | null

  // History (undo/redo per layer)
  undoStack: LayerData[][]
  redoStack: LayerData[][]

  // Canvas transform
  canvasScale: number
  canvasOffset: { x: number; y: number }

  // Detected text regions per layer id
  detectedTextRegions: Record<string, any[]>
  setDetectedTextRegions: (layerId: string, regions: any[]) => void

  // Live text overlays for preview (no API, pure CSS)
  textOverlays: Record<string, { text: string; color: string; font_size: number; shadow: boolean; shadow_color: string; rotation: number; bbox: number[] }>
  setTextOverlay: (key: string, overlay: any) => void
  clearTextOverlays: (layerId: string) => void

  // Actions
  setSession: (id: string, imagePath: string, imageUrl: string, w: number, h: number) => void
  setLayers: (layers: LayerData[]) => void
  updateLayer: (id: string, patch: Partial<LayerData>) => void
  selectLayer: (id: string, multi?: boolean) => void
  clearSelection: () => void
  setActiveTool: (tool: Tool) => void
  setProgress: (p: ProgressInfo | null) => void
  pushHistory: () => void
  undo: () => void
  redo: () => void
  reorderLayer: (id: string, direction: 'up' | 'down') => void
  addLayer: (layer: LayerData) => void
  duplicateLayer: (id: string) => void
  deleteLayer: (id: string) => void
  setCanvasScale: (s: number) => void
  setCanvasOffset: (o: { x: number; y: number }) => void
  reset: () => void
}

export const useEditorStore = create<EditorState>((set, get) => ({
  sessionId: null,
  originalImagePath: null,
  originalImageUrl: null,
  canvasWidth: 800,
  canvasHeight: 600,
  layers: [],
  selectedLayerIds: [],
  activeTool: 'move',
  progress: null,
  undoStack: [],
  redoStack: [],
  canvasScale: 1,
  canvasOffset: { x: 0, y: 0 },
  detectedTextRegions: {},
  setDetectedTextRegions: (layerId, regions) =>
    set((s) => ({ detectedTextRegions: { ...s.detectedTextRegions, [layerId]: regions } })),

  textOverlays: {},
  setTextOverlay: (key, overlay) =>
    set((s) => ({ textOverlays: { ...s.textOverlays, [key]: overlay } })),
  clearTextOverlays: (layerId) =>
    set((s) => ({
      textOverlays: Object.fromEntries(Object.entries(s.textOverlays).filter(([k]) => !k.startsWith(layerId + '_')))
    })),

  textOverlays: {},
  setTextOverlay: (key, overlay) =>
    set((s) => ({ textOverlays: { ...s.textOverlays, [key]: overlay } })),
  clearTextOverlays: (layerId) =>
    set((s) => ({
      textOverlays: Object.fromEntries(Object.entries(s.textOverlays).filter(([k]) => !k.startsWith(layerId + '_')))
    })),

  setSession: (id, imagePath, imageUrl, w, h) =>
    set({ sessionId: id, originalImagePath: imagePath, originalImageUrl: imageUrl, canvasWidth: w, canvasHeight: h }),

  setLayers: (layers) => set({ layers }),

  updateLayer: (id, patch) =>
    set((s) => ({
      layers: s.layers.map((l) => (l.id === id ? { ...l, ...patch } : l)),
    })),

  selectLayer: (id, multi = false) =>
    set((s) => ({
      selectedLayerIds: multi
        ? s.selectedLayerIds.includes(id)
          ? s.selectedLayerIds.filter((x) => x !== id)
          : [...s.selectedLayerIds, id]
        : [id],
    })),

  clearSelection: () => set({ selectedLayerIds: [] }),

  setActiveTool: (tool) => set({ activeTool: tool }),

  setProgress: (p) => set({ progress: p }),

  pushHistory: () =>
    set((s) => ({
      undoStack: [...s.undoStack, s.layers],
      redoStack: [],
    })),

  undo: () =>
    set((s) => {
      if (s.undoStack.length === 0) return s
      const prev = s.undoStack[s.undoStack.length - 1]
      return {
        layers: prev,
        undoStack: s.undoStack.slice(0, -1),
        redoStack: [s.layers, ...s.redoStack],
      }
    }),

  redo: () =>
    set((s) => {
      if (s.redoStack.length === 0) return s
      const next = s.redoStack[0]
      return {
        layers: next,
        undoStack: [...s.undoStack, s.layers],
        redoStack: s.redoStack.slice(1),
      }
    }),

  reorderLayer: (id, direction) =>
    set((s) => {
      const arr = [...s.layers]
      const idx = arr.findIndex((l) => l.id === id)
      if (idx === -1) return s
      const swapIdx = direction === 'up' ? idx - 1 : idx + 1
      if (swapIdx < 0 || swapIdx >= arr.length) return s
      ;[arr[idx], arr[swapIdx]] = [arr[swapIdx], arr[idx]]
      return { layers: arr.map((l, i) => ({ ...l, z_index: arr.length - i })) }
    }),

  addLayer: (layer) => set((s) => ({ layers: [...s.layers, layer] })),

  duplicateLayer: (id) =>
    set((s) => {
      const layer = s.layers.find((l) => l.id === id)
      if (!layer) return s
      const clone: LayerData = {
        ...layer,
        id: `${layer.id}_copy_${Date.now()}`,
        name: `${layer.name} copy`,
        z_index: layer.z_index + 1,
        position: { x: layer.position.x + 10, y: layer.position.y + 10 },
      }
      return { layers: [...s.layers, clone] }
    }),

  deleteLayer: (id) =>
    set((s) => ({
      layers: s.layers.filter((l) => l.id !== id),
      selectedLayerIds: s.selectedLayerIds.filter((x) => x !== id),
    })),

  setCanvasScale: (s) => set({ canvasScale: s }),
  setCanvasOffset: (o) => set({ canvasOffset: o }),

  reset: () =>
    set({
      sessionId: null,
      originalImagePath: null,
      originalImageUrl: null,
      layers: [],
      selectedLayerIds: [],
      undoStack: [],
      redoStack: [],
      progress: null,
      detectedTextRegions: {},
      textOverlays: {},
    }),
}))
