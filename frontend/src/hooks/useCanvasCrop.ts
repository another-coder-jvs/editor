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
  const { activeTool } = useEditorStore()
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
      // FIX M2a: clamp to image bounds
      const rect: CropRect = {
        x: Math.max(0, Math.min(canvasWidth,  Math.min(drawStart.current.x, pos.x))),
        y: Math.max(0, Math.min(canvasHeight, Math.min(drawStart.current.y, pos.y))),
        w: 0, h: 0,
      }
      rect.w = Math.max(0, Math.min(canvasWidth  - rect.x, Math.abs(pos.x - drawStart.current.x)))
      rect.h = Math.max(0, Math.min(canvasHeight - rect.y, Math.abs(pos.y - drawStart.current.y)))
      setCropRect(rect)
      drawCrop(rect)
      return
    }

    if (dragging.current && dragStart.current) {
      const dx = pos.x - dragStart.current.mx
      const dy = pos.y - dragStart.current.my
      let r = { ...dragStart.current.rect }
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
      // FIX M2a: clamp after handle drag too
      r.x = Math.max(0, Math.min(canvasWidth,  r.x))
      r.y = Math.max(0, Math.min(canvasHeight, r.y))
      r.w = Math.max(5, Math.min(canvasWidth  - r.x, r.w))
      r.h = Math.max(5, Math.min(canvasHeight - r.y, r.h))
      if (r.w > 5 && r.h > 5) {
        setCropRect(r)
        drawCrop(r)
      }
    }
  }, [getPos, drawCrop, canvasWidth, canvasHeight])

  const onMouseUp = useCallback(() => {
    drawing.current = false
    dragging.current = null
    dragStart.current = null
  }, [])

  // Apply crop on Enter
  const applyCrop = useCallback(async () => {
    if (!cropRect || cropRect.w < 2 || cropRect.h < 2) return

    const { layers, addLayer, pushHistory, setSession, sessionId, originalImagePath, originalImageUrl } = useEditorStore.getState()

    const cropW = Math.round(cropRect.w)
    const cropH = Math.round(cropRect.h)
    const cropX = Math.round(cropRect.x)
    const cropY = Math.round(cropRect.y)

    const offscreen = document.createElement('canvas')
    offscreen.width  = cropW
    offscreen.height = cropH
    const ctx = offscreen.getContext('2d')!

    const visible = [...layers].filter(l => l.visible && l.png_path).sort((a, b) => a.z_index - b.z_index)
    for (const layer of visible) {
      const src = layer.png_path.startsWith('blob:') || layer.png_path.startsWith('data:')
        ? layer.png_path : `${API_BASE}${layer.png_path}`

      let imgSrc = src
      if (src.startsWith('http://') || src.startsWith('https://')) {
        try {
          const res = await fetch(src, { headers: { 'ngrok-skip-browser-warning': '1' } })
          if (res.ok) imgSrc = URL.createObjectURL(await res.blob())
        } catch { continue }
      }

      await new Promise<void>(resolve => {
        const img = new Image()
        img.onload = () => {
          ctx.save()
          ctx.globalAlpha = layer.opacity
          ctx.globalCompositeOperation = (layer.blend_mode || 'normal') as GlobalCompositeOperation
          ctx.drawImage(img,
            layer.bbox.x + layer.position.x - cropX,
            layer.bbox.y + layer.position.y - cropY,
            layer.bbox.width, layer.bbox.height,
          )
          ctx.restore()
          resolve()
        }
        img.onerror = () => resolve()
        img.src = imgSrc
      })
    }

    pushHistory()
    const dataUrl = offscreen.toDataURL('image/png')

    // FIX M2b: update the store's canvas dimensions so the viewport matches the crop
    setSession(
      sessionId || '',
      originalImagePath || '',
      originalImageUrl || '',
      cropW,
      cropH,
    )

    const maxZ = layers.length > 0 ? Math.max(...layers.map(l => l.z_index)) : 0
    addLayer({
      id: `crop_${Date.now()}`,
      name: 'Crop',
      mask_path: '',
      png_path: dataUrl,
      bbox: { x: 0, y: 0, width: cropW, height: cropH },
      z_index: maxZ + 1,
      visible: true,
      opacity: 1,
      position: { x: 0, y: 0 },
      scale: { x: 1, y: 1 },
      rotation: 0,
      history: ['crop'],
      locked: false,
      blend_mode: 'normal',
      adjustments: {
        brightness: 100, contrast: 100, saturation: 100, exposure: 0,
        highlights: 0, shadows: 0, temperature: 0, tint: 0, hue: 0,
        sharpness: 100, clarity: 0, fade: 0, vignette: 0, grain: 0,
      },
    })

    setCropRect(null)
    cropRef.current?.getContext('2d')?.clearRect(0, 0, canvasWidth, canvasHeight)
  }, [cropRect, canvasWidth, canvasHeight, cropRef])

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

  // FIX M2c: aspect-ratio preset handler — called from ToolOptionsPanel buttons
  const setAspectRatio = useCallback((ratio: string) => {
    if (!cropRect) {
      // No crop box yet — create one in the center of the canvas
      const cw = canvasWidth, ch = canvasHeight
      const defaultSize = Math.min(cw, ch) * 0.6
      let w = defaultSize, h = defaultSize
      if (ratio !== 'Free') {
        const [rw, rh] = ratio.split(':').map(Number)
        if (rw && rh) {
          w = defaultSize
          h = defaultSize * rh / rw
          if (h > ch * 0.9) { h = ch * 0.9; w = h * rw / rh }
        }
      }
      const r: CropRect = { x: (cw - w) / 2, y: (ch - h) / 2, w, h }
      setCropRect(r)
      drawCrop(r)
      return
    }
    if (ratio === 'Free') return
    const [rw, rh] = ratio.split(':').map(Number)
    if (!rw || !rh) return
    // Adjust height to match ratio, keep top-left corner
    const newH = cropRect.w * rh / rw
    const r: CropRect = { ...cropRect, h: Math.min(newH, canvasHeight - cropRect.y) }
    setCropRect(r)
    drawCrop(r)
  }, [cropRect, canvasWidth, canvasHeight, drawCrop])

  // Listen for aspect-ratio preset events dispatched by ToolOptionsPanel buttons
  useEffect(() => {
    const handler = (e: Event) => {
      if (activeTool !== 'crop') return
      setAspectRatio((e as CustomEvent).detail as string)
    }
    window.addEventListener('crop-set-ratio', handler)
    return () => window.removeEventListener('crop-set-ratio', handler)
  }, [activeTool, setAspectRatio])

  return { cropRect, applyCrop, setAspectRatio }
}
