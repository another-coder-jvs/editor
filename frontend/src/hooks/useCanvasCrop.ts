/**
 * useCanvasCrop – interactive crop tool with draggable handles.
 */
import { useRef, useEffect, useCallback, useState } from 'react'
import { useEditorStore } from '../store/editorStore'
import { baseImagesUrl } from '../config'

const API_BASE = baseImagesUrl || 'http://localhost:8000'

type CropRect = { x: number; y: number; w: number; h: number }
type Handle = 'tl' | 'tr' | 'bl' | 'br' | 'move' | null

export function useCanvasCrop(
  cropRef: React.RefObject<HTMLCanvasElement>,
  canvasWidth: number,
  canvasHeight: number,
  canvasScale: number,
) {
  const { activeTool, layers, selectedLayerIds, updateLayer, pushHistory } = useEditorStore()
  const [cropRect, setCropRect] = useState<CropRect | null>(null)
  const dragging = useRef<Handle>(null)
  const dragStart = useRef<{ mx: number; my: number; rect: CropRect } | null>(null)
  const drawing = useRef(false)
  const drawStart = useRef<{ x: number; y: number } | null>(null)

  const getPos = useCallback((e: MouseEvent) => {
    const canvas = cropRef.current
    if (!canvas) return { x: 0, y: 0 }
    const rect = canvas.getBoundingClientRect()
    return { x: (e.clientX - rect.left) / canvasScale, y: (e.clientY - rect.top) / canvasScale }
  }, [canvasScale, cropRef])

  const drawCrop = useCallback((rect: CropRect | null) => {
    const canvas = cropRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.clearRect(0, 0, canvasWidth, canvasHeight)
    if (!rect) return

    // Darken outside
    ctx.fillStyle = 'rgba(0,0,0,0.5)'
    ctx.fillRect(0, 0, canvasWidth, canvasHeight)
    ctx.clearRect(rect.x, rect.y, rect.w, rect.h)

    // Border
    ctx.strokeStyle = '#fff'
    ctx.lineWidth = 1.5 / canvasScale
    ctx.setLineDash([])
    ctx.strokeRect(rect.x, rect.y, rect.w, rect.h)

    // Rule of thirds grid
    ctx.strokeStyle = 'rgba(255,255,255,0.3)'
    ctx.lineWidth = 0.5 / canvasScale
    for (let i = 1; i < 3; i++) {
      ctx.beginPath()
      ctx.moveTo(rect.x + (rect.w * i) / 3, rect.y)
      ctx.lineTo(rect.x + (rect.w * i) / 3, rect.y + rect.h)
      ctx.stroke()
      ctx.beginPath()
      ctx.moveTo(rect.x, rect.y + (rect.h * i) / 3)
      ctx.lineTo(rect.x + rect.w, rect.y + (rect.h * i) / 3)
      ctx.stroke()
    }

    // Handles
    const hs = 8 / canvasScale
    ctx.fillStyle = '#fff'
    ;[[rect.x, rect.y], [rect.x + rect.w, rect.y], [rect.x, rect.y + rect.h], [rect.x + rect.w, rect.y + rect.h]].forEach(([hx, hy]) => {
      ctx.fillRect(hx - hs / 2, hy - hs / 2, hs, hs)
    })
  }, [canvasWidth, canvasHeight, canvasScale, cropRef])

  const hitHandle = useCallback((pos: { x: number; y: number }, rect: CropRect): Handle => {
    const hs = 12 / canvasScale
    if (Math.abs(pos.x - rect.x) < hs && Math.abs(pos.y - rect.y) < hs) return 'tl'
    if (Math.abs(pos.x - (rect.x + rect.w)) < hs && Math.abs(pos.y - rect.y) < hs) return 'tr'
    if (Math.abs(pos.x - rect.x) < hs && Math.abs(pos.y - (rect.y + rect.h)) < hs) return 'bl'
    if (Math.abs(pos.x - (rect.x + rect.w)) < hs && Math.abs(pos.y - (rect.y + rect.h)) < hs) return 'br'
    if (pos.x > rect.x && pos.x < rect.x + rect.w && pos.y > rect.y && pos.y < rect.y + rect.h) return 'move'
    return null
  }, [canvasScale])

  const onMouseDown = useCallback((e: MouseEvent) => {
    if (activeTool !== 'crop') return
    if (e.button !== 0) return
    const pos = getPos(e)

    if (cropRect) {
      const handle = hitHandle(pos, cropRect)
      if (handle) {
        dragging.current = handle
        dragStart.current = { mx: pos.x, my: pos.y, rect: { ...cropRect } }
        return
      }
    }

    // Start new crop rect
    drawing.current = true
    drawStart.current = pos
    setCropRect(null)
  }, [activeTool, cropRect, getPos, hitHandle])

  const onMouseMove = useCallback((e: MouseEvent) => {
    const pos = getPos(e)

    if (drawing.current && drawStart.current) {
      const rect: CropRect = {
        x: Math.min(drawStart.current.x, pos.x),
        y: Math.min(drawStart.current.y, pos.y),
        w: Math.abs(pos.x - drawStart.current.x),
        h: Math.abs(pos.y - drawStart.current.y),
      }
      setCropRect(rect)
      drawCrop(rect)
      return
    }

    if (dragging.current && dragStart.current) {
      const dx = pos.x - dragStart.current.mx
      const dy = pos.y - dragStart.current.my
      const r = { ...dragStart.current.rect }
      if (dragging.current === 'move') {
        r.x += dx; r.y += dy
      } else if (dragging.current === 'tl') {
        r.x += dx; r.y += dy; r.w -= dx; r.h -= dy
      } else if (dragging.current === 'tr') {
        r.y += dy; r.w += dx; r.h -= dy
      } else if (dragging.current === 'bl') {
        r.x += dx; r.w -= dx; r.h += dy
      } else if (dragging.current === 'br') {
        r.w += dx; r.h += dy
      }
      if (r.w > 5 && r.h > 5) {
        setCropRect(r)
        drawCrop(r)
      }
    }
  }, [getPos, drawCrop])

  const onMouseUp = useCallback(() => {
    drawing.current = false
    dragging.current = null
    dragStart.current = null
  }, [])

  // Apply crop on Enter
  const applyCrop = useCallback(async () => {
    if (!cropRect) return
    const selectedLayer = layers.find(l => selectedLayerIds.includes(l.id))
    if (!selectedLayer) return

    const offscreen = document.createElement('canvas')
    offscreen.width = cropRect.w
    offscreen.height = cropRect.h
    const ctx = offscreen.getContext('2d')!

    const layerUrl = selectedLayer.png_path.startsWith('blob:') || selectedLayer.png_path.startsWith('data:')
      ? selectedLayer.png_path
      : `${API_BASE}${selectedLayer.png_path}`

    await new Promise<void>(resolve => {
      const img = new Image()
      img.crossOrigin = 'anonymous'
      img.onload = () => {
        // Map crop rect (canvas coords) to layer-local coords
        const lx = cropRect.x - selectedLayer.bbox.x - selectedLayer.position.x
        const ly = cropRect.y - selectedLayer.bbox.y - selectedLayer.position.y
        ctx.drawImage(img, lx, ly, cropRect.w, cropRect.h, 0, 0, cropRect.w, cropRect.h)
        resolve()
      }
      img.onerror = () => resolve()
      img.src = layerUrl
    })

    pushHistory()
    updateLayer(selectedLayer.id, {
      png_path: offscreen.toDataURL('image/png'),
      bbox: {
        x: cropRect.x,
        y: cropRect.y,
        width: cropRect.w,
        height: cropRect.h,
      },
      position: { x: 0, y: 0 },
    })

    setCropRect(null)
    const canvas = cropRef.current
    if (canvas) canvas.getContext('2d')?.clearRect(0, 0, canvasWidth, canvasHeight)
  }, [cropRect, layers, selectedLayerIds, updateLayer, pushHistory, canvasWidth, canvasHeight, cropRef])

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (activeTool !== 'crop') return
      if (e.key === 'Enter') applyCrop()
      if (e.key === 'Escape') {
        setCropRect(null)
        cropRef.current?.getContext('2d')?.clearRect(0, 0, canvasWidth, canvasHeight)
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [activeTool, applyCrop, canvasWidth, canvasHeight, cropRef])

  useEffect(() => {
    const canvas = cropRef.current
    if (!canvas) return
    canvas.addEventListener('mousedown', onMouseDown)
    window.addEventListener('mousemove', onMouseMove)
    window.addEventListener('mouseup', onMouseUp)
    return () => {
      canvas.removeEventListener('mousedown', onMouseDown)
      window.removeEventListener('mousemove', onMouseMove)
      window.removeEventListener('mouseup', onMouseUp)
    }
  }, [onMouseDown, onMouseMove, onMouseUp, cropRef])

  // Redraw when tool changes away from crop
  useEffect(() => {
    if (activeTool !== 'crop') {
      setCropRect(null)
      cropRef.current?.getContext('2d')?.clearRect(0, 0, canvasWidth, canvasHeight)
    }
  }, [activeTool, canvasWidth, canvasHeight, cropRef])

  return { cropRect, applyCrop }
}
