/**
 * useCanvasDraw – handles all pixel-level drawing tools on an overlay canvas.
 * Tools: brush, pencil, marker, eraser, clone, heal, color_picker,
 *        shape_rect, shape_ellipse, shape_line, shape_arrow, shape_triangle, shape_star
 */
import { useRef, useEffect, useCallback } from 'react'
import { useEditorStore } from '../store/editorStore'
import { baseImagesUrl } from '../config'

const API_BASE = baseImagesUrl || 'http://localhost:8000'

function hexToRgba(hex: string, alpha: number) {
  const r = parseInt(hex.slice(1, 3), 16)
  const g = parseInt(hex.slice(3, 5), 16)
  const b = parseInt(hex.slice(5, 7), 16)
  return `rgba(${r},${g},${b},${alpha})`
}

export function useCanvasDraw(
  overlayRef: React.RefObject<HTMLCanvasElement>,
  canvasWidth: number,
  canvasHeight: number,
  canvasScale: number,
) {
  const {
    activeTool, toolOptions, layers, selectedLayerIds,
    updateLayer, pushHistory, addLayer, sessionId,
  } = useEditorStore()

  const drawing = useRef(false)
  const lastPos = useRef<{ x: number; y: number } | null>(null)
  const shapeStart = useRef<{ x: number; y: number } | null>(null)
  const snapshotRef = useRef<ImageData | null>(null)
  const cloneSource = useRef<{ x: number; y: number } | null>(null)
  const cloneOffset = useRef<{ x: number; y: number } | null>(null)

  const getPos = useCallback((e: MouseEvent | React.MouseEvent): { x: number; y: number } => {
    const canvas = overlayRef.current
    if (!canvas) return { x: 0, y: 0 }
    const rect = canvas.getBoundingClientRect()
    return {
      x: (e.clientX - rect.left) / canvasScale,
      y: (e.clientY - rect.top) / canvasScale,
    }
  }, [canvasScale, overlayRef])

  const getCtx = useCallback(() => {
    const canvas = overlayRef.current
    return canvas ? canvas.getContext('2d') : null
  }, [overlayRef])

  // Commit overlay canvas pixels onto the selected layer's image
  const commitToLayer = useCallback(async () => {
    const canvas = overlayRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const selectedLayer = layers.find(l => selectedLayerIds.includes(l.id))
    if (!selectedLayer) {
      // No layer selected: create a new draw layer
      const dataUrl = canvas.toDataURL('image/png')
      const newId = `draw_${Date.now()}`
      addLayer({
        id: newId,
        name: 'Drawing',
        mask_path: '',
        png_path: dataUrl,
        bbox: { x: 0, y: 0, width: canvasWidth, height: canvasHeight },
        z_index: (layers.length > 0 ? Math.max(...layers.map(l => l.z_index)) : 0) + 1,
        visible: true,
        opacity: 1,
        position: { x: 0, y: 0 },
        scale: { x: 1, y: 1 },
        rotation: 0,
        history: ['draw'],
        locked: false,
        blend_mode: 'normal',
        adjustments: {
          brightness: 100, contrast: 100, saturation: 100,
          exposure: 0, highlights: 0, shadows: 0,
          temperature: 0, tint: 0, hue: 0,
          sharpness: 100, clarity: 0, fade: 0, vignette: 0, grain: 0,
        },
      })
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      return
    }

    // Composite overlay onto the layer image
    const offscreen = document.createElement('canvas')
    offscreen.width = canvasWidth
    offscreen.height = canvasHeight
    const offCtx = offscreen.getContext('2d')!

    // Draw existing layer image
    const layerUrl = selectedLayer.png_path.startsWith('blob:') || selectedLayer.png_path.startsWith('data:')
      ? selectedLayer.png_path
      : `${API_BASE}${selectedLayer.png_path}`

    await new Promise<void>(resolve => {
      const img = new Image()
      img.crossOrigin = 'anonymous'
      img.onload = () => {
        offCtx.drawImage(img,
          selectedLayer.bbox.x + selectedLayer.position.x,
          selectedLayer.bbox.y + selectedLayer.position.y,
          selectedLayer.bbox.width, selectedLayer.bbox.height)
        resolve()
      }
      img.onerror = () => resolve()
      img.src = layerUrl
    })

    // Composite overlay
    offCtx.drawImage(canvas, 0, 0)

    // Crop back to layer bbox
    const cropCanvas = document.createElement('canvas')
    cropCanvas.width = selectedLayer.bbox.width
    cropCanvas.height = selectedLayer.bbox.height
    const cropCtx = cropCanvas.getContext('2d')!
    cropCtx.drawImage(offscreen,
      selectedLayer.bbox.x + selectedLayer.position.x,
      selectedLayer.bbox.y + selectedLayer.position.y,
      selectedLayer.bbox.width, selectedLayer.bbox.height,
      0, 0, selectedLayer.bbox.width, selectedLayer.bbox.height)

    const dataUrl = cropCanvas.toDataURL('image/png')
    pushHistory()
    updateLayer(selectedLayer.id, { png_path: dataUrl })
    ctx.clearRect(0, 0, canvas.width, canvas.height)
  }, [layers, selectedLayerIds, canvasWidth, canvasHeight, updateLayer, pushHistory, addLayer])

  const drawBrushStroke = useCallback((ctx: CanvasRenderingContext2D, from: { x: number; y: number }, to: { x: number; y: number }) => {
    const { brushSize, brushOpacity, brushColor, brushHardness, eraserSize } = toolOptions
    const isEraser = activeTool === 'eraser'
    const size = isEraser ? eraserSize : brushSize

    ctx.save()
    if (isEraser) {
      ctx.globalCompositeOperation = 'destination-out'
      ctx.strokeStyle = 'rgba(0,0,0,1)'
    } else if (activeTool === 'marker') {
      ctx.globalCompositeOperation = 'multiply'
      ctx.strokeStyle = hexToRgba(brushColor, brushOpacity * 0.6)
    } else {
      ctx.globalCompositeOperation = 'source-over'
      ctx.strokeStyle = hexToRgba(brushColor, activeTool === 'pencil' ? brushOpacity * 0.9 : brushOpacity)
    }

    ctx.lineWidth = size
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'

    if (activeTool === 'pencil') {
      ctx.lineWidth = Math.max(1, size * 0.3)
    }

    // Soft brush via shadow
    if (activeTool === 'brush' && brushHardness < 1) {
      const blur = size * (1 - brushHardness) * 0.5
      ctx.shadowBlur = blur
      ctx.shadowColor = isEraser ? 'rgba(0,0,0,1)' : hexToRgba(brushColor, brushOpacity)
    }

    ctx.beginPath()
    ctx.moveTo(from.x, from.y)
    ctx.lineTo(to.x, to.y)
    ctx.stroke()
    ctx.restore()
  }, [activeTool, toolOptions])

  const drawShape = useCallback((ctx: CanvasRenderingContext2D, start: { x: number; y: number }, end: { x: number; y: number }, preview = false) => {
    const { shapeStroke, shapeFill, shapeStrokeWidth, shapeOpacity } = toolOptions
    ctx.save()
    ctx.globalAlpha = shapeOpacity
    ctx.strokeStyle = shapeStroke
    ctx.fillStyle = shapeFill === 'transparent' ? 'rgba(0,0,0,0)' : shapeFill
    ctx.lineWidth = shapeStrokeWidth
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'

    const x = Math.min(start.x, end.x)
    const y = Math.min(start.y, end.y)
    const w = Math.abs(end.x - start.x)
    const h = Math.abs(end.y - start.y)

    ctx.beginPath()
    if (activeTool === 'shape_rect') {
      ctx.rect(x, y, w, h)
    } else if (activeTool === 'shape_ellipse') {
      ctx.ellipse(x + w / 2, y + h / 2, w / 2, h / 2, 0, 0, Math.PI * 2)
    } else if (activeTool === 'shape_line') {
      ctx.moveTo(start.x, start.y)
      ctx.lineTo(end.x, end.y)
    } else if (activeTool === 'shape_arrow') {
      const angle = Math.atan2(end.y - start.y, end.x - start.x)
      const headLen = Math.max(10, Math.hypot(end.x - start.x, end.y - start.y) * 0.2)
      ctx.moveTo(start.x, start.y)
      ctx.lineTo(end.x, end.y)
      ctx.moveTo(end.x, end.y)
      ctx.lineTo(end.x - headLen * Math.cos(angle - Math.PI / 6), end.y - headLen * Math.sin(angle - Math.PI / 6))
      ctx.moveTo(end.x, end.y)
      ctx.lineTo(end.x - headLen * Math.cos(angle + Math.PI / 6), end.y - headLen * Math.sin(angle + Math.PI / 6))
    } else if (activeTool === 'shape_triangle') {
      ctx.moveTo(x + w / 2, y)
      ctx.lineTo(x + w, y + h)
      ctx.lineTo(x, y + h)
      ctx.closePath()
    } else if (activeTool === 'shape_star') {
      const cx = x + w / 2, cy = y + h / 2
      const outerR = Math.min(w, h) / 2, innerR = outerR * 0.4
      for (let i = 0; i < 10; i++) {
        const angle = (i * Math.PI) / 5 - Math.PI / 2
        const r = i % 2 === 0 ? outerR : innerR
        if (i === 0) ctx.moveTo(cx + r * Math.cos(angle), cy + r * Math.sin(angle))
        else ctx.lineTo(cx + r * Math.cos(angle), cy + r * Math.sin(angle))
      }
      ctx.closePath()
    }

    if (shapeFill !== 'transparent') ctx.fill()
    ctx.stroke()
    ctx.restore()
  }, [activeTool, toolOptions])

  const onMouseDown = useCallback((e: MouseEvent) => {
    const DRAW_TOOLS = ['brush', 'pencil', 'marker', 'eraser', 'clone', 'heal',
      'shape_rect', 'shape_ellipse', 'shape_line', 'shape_arrow', 'shape_triangle', 'shape_star']
    if (!DRAW_TOOLS.includes(activeTool)) return
    if (e.button !== 0) return

    const pos = getPos(e)
    drawing.current = true
    lastPos.current = pos

    const ctx = getCtx()
    if (!ctx) return

    if (['shape_rect', 'shape_ellipse', 'shape_line', 'shape_arrow', 'shape_triangle', 'shape_star'].includes(activeTool)) {
      shapeStart.current = pos
      snapshotRef.current = ctx.getImageData(0, 0, canvasWidth, canvasHeight)
      return
    }

    if (activeTool === 'clone') {
      if (e.altKey) {
        cloneSource.current = pos
        return
      }
      if (cloneSource.current) {
        cloneOffset.current = { x: pos.x - cloneSource.current.x, y: pos.y - cloneSource.current.y }
      }
    }

    drawBrushStroke(ctx, pos, pos)
  }, [activeTool, getPos, getCtx, drawBrushStroke, canvasWidth, canvasHeight])

  const onMouseMove = useCallback((e: MouseEvent) => {
    if (!drawing.current) return
    const pos = getPos(e)
    const ctx = getCtx()
    if (!ctx) return

    if (['shape_rect', 'shape_ellipse', 'shape_line', 'shape_arrow', 'shape_triangle', 'shape_star'].includes(activeTool)) {
      if (!shapeStart.current || !snapshotRef.current) return
      ctx.putImageData(snapshotRef.current, 0, 0)
      drawShape(ctx, shapeStart.current, pos, true)
      return
    }

    if (activeTool === 'clone' && cloneSource.current && cloneOffset.current) {
      const srcX = pos.x - cloneOffset.current.x
      const srcY = pos.y - cloneOffset.current.y
      const size = toolOptions.brushSize
      ctx.save()
      ctx.globalAlpha = toolOptions.brushOpacity
      ctx.drawImage(overlayRef.current!, srcX - size / 2, srcY - size / 2, size, size, pos.x - size / 2, pos.y - size / 2, size, size)
      ctx.restore()
    } else if (lastPos.current) {
      drawBrushStroke(ctx, lastPos.current, pos)
    }

    lastPos.current = pos
  }, [activeTool, getPos, getCtx, drawBrushStroke, drawShape, toolOptions, overlayRef])

  const onMouseUp = useCallback(async (e: MouseEvent) => {
    if (!drawing.current) return
    drawing.current = false

    const pos = getPos(e)
    const ctx = getCtx()

    if (ctx && ['shape_rect', 'shape_ellipse', 'shape_line', 'shape_arrow', 'shape_triangle', 'shape_star'].includes(activeTool)) {
      if (shapeStart.current && snapshotRef.current) {
        ctx.putImageData(snapshotRef.current, 0, 0)
        drawShape(ctx, shapeStart.current, pos)
        shapeStart.current = null
        snapshotRef.current = null
      }
    }

    lastPos.current = null
    await commitToLayer()
  }, [activeTool, getPos, getCtx, drawShape, commitToLayer])

  useEffect(() => {
    const canvas = overlayRef.current
    if (!canvas) return
    canvas.addEventListener('mousedown', onMouseDown)
    window.addEventListener('mousemove', onMouseMove)
    window.addEventListener('mouseup', onMouseUp)
    return () => {
      canvas.removeEventListener('mousedown', onMouseDown)
      window.removeEventListener('mousemove', onMouseMove)
      window.removeEventListener('mouseup', onMouseUp)
    }
  }, [onMouseDown, onMouseMove, onMouseUp, overlayRef])

  // Color picker
  const pickColor = useCallback((e: MouseEvent) => {
    if (activeTool !== 'color_picker') return
    const canvas = overlayRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const pos = getPos(e)
    const pixel = ctx.getImageData(Math.round(pos.x), Math.round(pos.y), 1, 1).data
    const hex = '#' + [pixel[0], pixel[1], pixel[2]].map(v => v.toString(16).padStart(2, '0')).join('')
    useEditorStore.getState().setToolOption('brushColor', hex)
  }, [activeTool, getPos, overlayRef])

  useEffect(() => {
    const canvas = overlayRef.current
    if (!canvas) return
    canvas.addEventListener('click', pickColor)
    return () => canvas.removeEventListener('click', pickColor)
  }, [pickColor, overlayRef])
}
