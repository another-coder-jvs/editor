/**
 * useCanvasDraw – handles all pixel-level drawing tools on an overlay canvas.
 * Tools: brush, pencil, marker, eraser, clone, heal, color_picker,
 *        shape_rect, shape_ellipse, shape_line, shape_arrow, shape_triangle, shape_star
 */
import { useRef, useEffect, useCallback } from 'react'
import { useEditorStore } from '../store/editorStore'
import { baseImagesUrl } from '../config'

const API_BASE = baseImagesUrl || 'http://localhost:8000'

/** Load an image, fetching via proxy for http URLs to avoid CORS/ngrok issues */
async function loadImageSafe(src: string): Promise<HTMLImageElement | null> {
  try {
    let url = src
    if (src.startsWith('http://') || src.startsWith('https://')) {
      const res = await fetch(src, { headers: { 'ngrok-skip-browser-warning': '1' } })
      if (!res.ok) return null
      url = URL.createObjectURL(await res.blob())
    }
    return await new Promise<HTMLImageElement>(resolve => {
      const img = new Image()
      img.onload = () => resolve(img)
      img.onerror = () => resolve(null as any)
      img.src = url
    })
  } catch { return null }
}

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
  guideRef?: React.RefObject<HTMLCanvasElement>,
) {
  const {
    activeTool, toolOptions, layers,
    updateLayer, pushHistory, addLayer, sessionId,
  } = useEditorStore()

  const drawing = useRef(false)
  // Composite snapshot of all visible layers — source pixels for clone/heal
  const compositeRef = useRef<HTMLCanvasElement | null>(null)
  const hoverPos = useRef<{ x: number; y: number } | null>(null)
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

  // Step 1: snapshot all visible layers into an offscreen canvas so clone/heal
  // can copy REAL image pixels (the overlay canvas itself is transparent).
  const rebuildComposite = useCallback(async () => {
    if (activeTool !== 'clone' && activeTool !== 'heal') return
    const c = document.createElement('canvas')
    c.width = canvasWidth
    c.height = canvasHeight
    const ctx = c.getContext('2d')
    if (!ctx) return

    const sorted = [...layers].filter(l => l.visible && l.png_path).sort((a, b) => a.z_index - b.z_index)
    for (const l of sorted) {
      const url = l.png_path.startsWith('blob:') || l.png_path.startsWith('data:')
        ? l.png_path
        : `${API_BASE}${l.png_path}`
      const img = await loadImageSafe(url)
      if (!img) continue
      ctx.save()
      ctx.globalAlpha = l.opacity ?? 1
      ctx.translate(
        l.bbox.x + l.position.x + l.bbox.width / 2,
        l.bbox.y + l.position.y + l.bbox.height / 2,
      )
      ctx.rotate(((l.rotation || 0) * Math.PI) / 180)
      ctx.scale(l.scale?.x ?? 1, l.scale?.y ?? 1)
      ctx.drawImage(img, -l.bbox.width / 2, -l.bbox.height / 2, l.bbox.width, l.bbox.height)
      ctx.restore()
    }
    compositeRef.current = c
  }, [activeTool, layers, canvasWidth, canvasHeight])

  useEffect(() => { rebuildComposite() }, [rebuildComposite])

  const isFullCanvasDrawLayer = useCallback((layer: typeof layers[number]) =>
    layer.bbox.x === 0 &&
    layer.bbox.y === 0 &&
    layer.bbox.width === canvasWidth &&
    layer.bbox.height === canvasHeight &&
    layer.position.x === 0 &&
    layer.position.y === 0
  , [canvasWidth, canvasHeight])

  const createDrawLayerDefaults = useCallback((id: string, pngPath: string) => ({
    id,
    name: 'Drawing',
    mask_path: '',
    png_path: pngPath,
    bbox: { x: 0, y: 0, width: canvasWidth, height: canvasHeight },
    z_index: (layers.length > 0 ? Math.max(...layers.map(l => l.z_index)) : 0) + 1,
    visible: true,
    opacity: 1,
    position: { x: 0, y: 0 },
    scale: { x: 1, y: 1 },
    rotation: 0,
    history: ['draw'],
    locked: false,
    blend_mode: 'normal' as const,
    adjustments: {
      brightness: 100, contrast: 100, saturation: 100,
      exposure: 0, highlights: 0, shadows: 0,
      temperature: 0, tint: 0, hue: 0,
      sharpness: 100, clarity: 0, fade: 0, vignette: 0, grain: 0,
    },
  }), [canvasWidth, canvasHeight, layers])

  // Commit overlay canvas pixels onto a full-canvas drawing layer
  const commitToLayer = useCallback(async () => {
    const canvas = overlayRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const drawLayer = layers.find(l => l.id.startsWith('draw_') && isFullCanvasDrawLayer(l))

    if (!drawLayer) {
      const dataUrl = canvas.toDataURL('image/png')
      addLayer(createDrawLayerDefaults(`draw_${Date.now()}`, dataUrl))
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      return
    }

    const offscreen = document.createElement('canvas')
    offscreen.width = canvasWidth
    offscreen.height = canvasHeight
    const offCtx = offscreen.getContext('2d')!

    const layerUrl = drawLayer.png_path.startsWith('blob:') || drawLayer.png_path.startsWith('data:')
      ? drawLayer.png_path
      : `${API_BASE}${drawLayer.png_path}`

    const img = await loadImageSafe(layerUrl)
    if (img) {
      offCtx.drawImage(img, 0, 0, canvasWidth, canvasHeight)
    }

    offCtx.drawImage(canvas, 0, 0)
    const dataUrl = offscreen.toDataURL('image/png')
    pushHistory()
    updateLayer(drawLayer.id, { png_path: dataUrl })
    ctx.clearRect(0, 0, canvas.width, canvas.height)
  }, [layers, canvasWidth, canvasHeight, updateLayer, pushHistory, addLayer, isFullCanvasDrawLayer, createDrawLayerDefaults])

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

  // Step 2: copy a soft-edged circular patch from the composite (source)
  // onto the overlay at the target position.
  const stampClone = useCallback((ctx: CanvasRenderingContext2D, target: { x: number; y: number }) => {
    const comp = compositeRef.current
    if (!comp || !cloneOffset.current) return

    const isHeal = activeTool === 'heal'
    const size = Math.max(4, toolOptions.brushSize)
    const r = size / 2
    const srcX = Math.round(target.x - cloneOffset.current.x)
    const srcY = Math.round(target.y - cloneOffset.current.y)

    // Draw sampled patch into a temp canvas
    const tmp = document.createElement('canvas')
    tmp.width = size
    tmp.height = size
    const tctx = tmp.getContext('2d')!
    if (isHeal) tctx.filter = 'blur(1.5px)' // heal: soften texture so it blends
    tctx.drawImage(comp, srcX - r, srcY - r, size, size, 0, 0, size, size)
    tctx.filter = 'none'

    // Radial feather mask (heal gets a softer edge to blend with surroundings)
    const feather = isHeal ? 0.35 : 0.7
    const grad = tctx.createRadialGradient(r, r, size * feather * 0.5, r, r, r)
    grad.addColorStop(0, 'rgba(0,0,0,1)')
    grad.addColorStop(1, 'rgba(0,0,0,0)')
    tctx.globalCompositeOperation = 'destination-in'
    tctx.fillStyle = grad
    tctx.fillRect(0, 0, size, size)
    tctx.globalCompositeOperation = 'source-over'

    ctx.save()
    ctx.globalAlpha = isHeal ? toolOptions.brushOpacity * 0.85 : toolOptions.brushOpacity
    ctx.drawImage(tmp, target.x - r, target.y - r)
    ctx.restore()
  }, [activeTool, toolOptions])

  // Step 3b: live guide — shows the source pointer & brush circle so the user
  // can see exactly what will be copied where BEFORE committing.
  const drawGuide = useCallback(() => {
    const g = guideRef?.current
    if (!g || (activeTool !== 'clone' && activeTool !== 'heal')) return
    const gctx = g.getContext('2d')
    if (!gctx) return
    gctx.clearRect(0, 0, g.width, g.height)

    const p = hoverPos.current
    if (!p) return

    const lw = 1.5 / canvasScale // keep screen-space thickness constant
    const size = Math.max(4, toolOptions.brushSize)
    const r = size / 2

    // Live sampled source follows the target once offset is locked
    const srcNow = cloneOffset.current && cloneSource.current
      ? { x: p.x - cloneOffset.current.x, y: p.y - cloneOffset.current.y }
      : cloneSource.current

    gctx.save()
    gctx.lineWidth = lw

    // Connector line between source and target
    if (srcNow) {
      gctx.strokeStyle = 'rgba(34,211,238,0.5)'
      gctx.setLineDash([5 / canvasScale, 5 / canvasScale])
      gctx.beginPath()
      gctx.moveTo(srcNow.x, srcNow.y)
      gctx.lineTo(p.x, p.y)
      gctx.stroke()
      gctx.setLineDash([])
    }

    // Source marker: crosshair + circle (what is being copied FROM)
    if (srcNow) {
      gctx.strokeStyle = 'rgba(250,204,21,0.95)' // yellow
      gctx.beginPath()
      gctx.arc(srcNow.x, srcNow.y, r, 0, Math.PI * 2)
      gctx.stroke()
      gctx.beginPath()
      gctx.moveTo(srcNow.x - r * 1.6, srcNow.y); gctx.lineTo(srcNow.x - r * 0.6, srcNow.y)
      gctx.moveTo(srcNow.x + r * 0.6, srcNow.y); gctx.lineTo(srcNow.x + r * 1.6, srcNow.y)
      gctx.moveTo(srcNow.x, srcNow.y - r * 1.6); gctx.lineTo(srcNow.x, srcNow.y - r * 0.6)
      gctx.moveTo(srcNow.x, srcNow.y + r * 0.6); gctx.lineTo(srcNow.x, srcNow.y + r * 1.6)
      gctx.stroke()
    }

    // Target marker: cyan brush circle (where content will be painted)
    gctx.strokeStyle = srcNow ? 'rgba(34,211,238,0.95)' : 'rgba(156,163,175,0.7)'
    gctx.beginPath()
    gctx.arc(p.x, p.y, r, 0, Math.PI * 2)
    gctx.stroke()
    gctx.restore()
  }, [activeTool, canvasScale, toolOptions.brushSize, guideRef])

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

    if (activeTool === 'clone' || activeTool === 'heal') {
      if (e.altKey || e.shiftKey) {
        // Set the source pointer — pixels will be copied FROM here
        cloneSource.current = pos
        cloneOffset.current = null
        e.stopPropagation()
        e.preventDefault()
        return
      }
      if (!cloneSource.current) {
        // No source set yet — do nothing (don't paint a random brush stroke)
        return
      }
      // Lock offset: target - source. Both pointers move together while dragging.
      cloneOffset.current = { x: pos.x - cloneSource.current.x, y: pos.y - cloneSource.current.y }
      stampClone(ctx, pos)
      return
    }

    drawBrushStroke(ctx, pos, pos)
  }, [activeTool, getPos, getCtx, drawBrushStroke, stampClone, canvasWidth, canvasHeight])

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

    if ((activeTool === 'clone' || activeTool === 'heal') && cloneSource.current && cloneOffset.current) {
      // Both pointers move together: source follows target at fixed offset
      stampClone(ctx, pos)
    } else if (lastPos.current) {
      drawBrushStroke(ctx, lastPos.current, pos)
    }

    lastPos.current = pos
  }, [activeTool, getPos, getCtx, drawBrushStroke, drawShape, stampClone, overlayRef])

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

  // Guide overlay: track hover position for clone/heal and redraw markers
  useEffect(() => {
    const onHoverMove = (e: MouseEvent) => {
      if (activeTool !== 'clone' && activeTool !== 'heal') return
      const canvas = overlayRef.current
      if (!canvas) return
      const rect = canvas.getBoundingClientRect()
      hoverPos.current = {
        x: (e.clientX - rect.left) / canvasScale,
        y: (e.clientY - rect.top) / canvasScale,
      }
      drawGuide()
    }
    const clearGuide = () => {
      hoverPos.current = null
      const g = guideRef?.current
      const gctx = g?.getContext('2d')
      if (g && gctx) gctx.clearRect(0, 0, g.width, g.height)
    }
    const onLeave = () => { clearGuide() }
    window.addEventListener('mousemove', onHoverMove)
    const container = overlayRef.current?.parentElement ?? null
    container?.addEventListener('mouseleave', onLeave)
    return () => {
      window.removeEventListener('mousemove', onHoverMove)
      container?.removeEventListener('mouseleave', onLeave)
      clearGuide()
    }
  }, [activeTool, canvasScale, overlayRef, guideRef, drawGuide])

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
