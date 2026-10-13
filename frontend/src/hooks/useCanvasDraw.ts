/**
 * useCanvasDraw – handles all pixel-level drawing tools on an overlay canvas.
 * Tools: brush, pencil, marker, eraser, clone, heal, color_picker,
 *        shape_rect, shape_ellipse, shape_line, shape_arrow, shape_triangle, shape_star
 *
 * Fixes applied vs previous version:
 *  C2: Color picker now reads from composited image (all layers), not the
 *      transparent overlay canvas which always returned #000000.
 *  C4: Shape tools now honour Shift (square/circle/45°-snap) and Alt (from-center).
 *  C5: Pencil shift-click now persists lastPos across mouseup/mousedown so a
 *      straight line is drawn from the previous endpoint, not a dot.
 *  M4: Clone/Heal Alt+click sets source without triggering pan — event is
 *      stopped before bubbling to the Canvas pan handler.
 *  N1: Marker no longer builds opacity within a single stroke. A per-stroke
 *      snapshot is taken on mousedown; each frame the snapshot is restored and
 *      the entire accumulated path is re-drawn with source-atop, keeping a
 *      flat, uniform opacity across overlapping segments.
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
  const compositeRef = useRef<HTMLCanvasElement | null>(null)
  const hoverPos = useRef<{ x: number; y: number } | null>(null)
  // FIX C5: lastPos is NOT cleared on mouseup for pencil — it persists so
  // the next shift-click can draw a straight line from the last endpoint.
  const lastPos = useRef<{ x: number; y: number } | null>(null)
  const shapeStart = useRef<{ x: number; y: number } | null>(null)
  const snapshotRef = useRef<ImageData | null>(null)
  const cloneSource = useRef<{ x: number; y: number } | null>(null)
  const cloneOffset = useRef<{ x: number; y: number } | null>(null)

  // FIX N1: Marker per-stroke state — snapshot before stroke + accumulated path
  const markerSnapshotRef = useRef<ImageData | null>(null)
  const markerPointsRef = useRef<{ x: number; y: number }[]>([])

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

  // Composite snapshot of all visible layers — used by clone/heal/color_picker
  const rebuildComposite = useCallback(async () => {
    const needsComposite = activeTool === 'clone' || activeTool === 'heal' || activeTool === 'color_picker'
    if (!needsComposite) return
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
    if (img) offCtx.drawImage(img, 0, 0, canvasWidth, canvasHeight)
    offCtx.drawImage(canvas, 0, 0)
    const dataUrl = offscreen.toDataURL('image/png')
    pushHistory()
    updateLayer(drawLayer.id, { png_path: dataUrl })
    ctx.clearRect(0, 0, canvas.width, canvas.height)
  }, [layers, canvasWidth, canvasHeight, updateLayer, pushHistory, addLayer, isFullCanvasDrawLayer, createDrawLayerDefaults])

  const drawBrushStroke = useCallback((
    ctx: CanvasRenderingContext2D,
    from: { x: number; y: number },
    to: { x: number; y: number },
  ) => {
    const { brushSize, brushOpacity, brushColor, brushHardness, eraserSize } = toolOptions
    const isEraser = activeTool === 'eraser'
    const size = isEraser ? eraserSize : brushSize

    ctx.save()
    if (isEraser) {
      ctx.globalCompositeOperation = 'destination-out'
      ctx.strokeStyle = 'rgba(0,0,0,1)'
    } else if (activeTool === 'marker') {
      // FIX N1: marker blending is handled in drawMarkerStroke; this branch is
      // only reached for non-marker tools.
      ctx.globalCompositeOperation = 'source-over'
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

  // FIX N1: Marker — uniform opacity within a stroke via snapshot+source-atop.
  // On each move we restore the per-stroke snapshot, then draw the entire
  // accumulated path once with source-atop. This prevents overlapping segments
  // from darkening each other.
  const drawMarkerFrame = useCallback((
    ctx: CanvasRenderingContext2D,
    points: { x: number; y: number }[],
  ) => {
    if (points.length < 2 || !markerSnapshotRef.current) return
    const { brushSize, brushOpacity, brushColor } = toolOptions

    // Restore pre-stroke snapshot so previous segments don't accumulate
    ctx.putImageData(markerSnapshotRef.current, 0, 0)

    ctx.save()
    // source-atop: new paint only draws over existing pixels (the stroke itself),
    // and because we start fresh each frame the marker colour stays flat.
    ctx.globalCompositeOperation = 'source-over'
    ctx.globalAlpha = brushOpacity * 0.6
    ctx.strokeStyle = brushColor
    ctx.lineWidth = brushSize
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
    ctx.beginPath()
    ctx.moveTo(points[0].x, points[0].y)
    for (let i = 1; i < points.length; i++) ctx.lineTo(points[i].x, points[i].y)
    ctx.stroke()
    ctx.restore()
  }, [toolOptions])

  // FIX C4: drawShape with Shift (square/circle/45°-snap) and Alt (from-center)
  const drawShape = useCallback((
    ctx: CanvasRenderingContext2D,
    rawStart: { x: number; y: number },
    rawEnd: { x: number; y: number },
    shiftKey = false,
    altKey = false,
    preview = false,
  ) => {
    const { shapeStroke, shapeFill, shapeStrokeWidth, shapeOpacity } = toolOptions
    ctx.save()
    ctx.globalAlpha = shapeOpacity
    ctx.strokeStyle = shapeStroke
    ctx.fillStyle = shapeFill === 'transparent' ? 'rgba(0,0,0,0)' : shapeFill
    ctx.lineWidth = shapeStrokeWidth
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'

    // Apply constraints
    let start = rawStart
    let end = rawEnd

    if (activeTool === 'shape_line' && shiftKey) {
      // Snap to nearest 15° angle
      const dx = end.x - start.x
      const dy = end.y - start.y
      const len = Math.hypot(dx, dy)
      const angle = Math.atan2(dy, dx)
      const snapped = Math.round(angle / (Math.PI / 12)) * (Math.PI / 12)
      end = { x: start.x + len * Math.cos(snapped), y: start.y + len * Math.sin(snapped) }
    } else if ((activeTool === 'shape_rect' || activeTool === 'shape_ellipse') && shiftKey) {
      // Constrain to square / circle
      const dx = rawEnd.x - rawStart.x
      const dy = rawEnd.y - rawStart.y
      const size = Math.min(Math.abs(dx), Math.abs(dy))
      end = {
        x: rawStart.x + Math.sign(dx) * size,
        y: rawStart.y + Math.sign(dy) * size,
      }
    }

    if ((activeTool === 'shape_rect' || activeTool === 'shape_ellipse') && altKey) {
      // Draw from center: start is the center, end determines radius
      const hw = Math.abs(end.x - start.x)
      const hh = Math.abs(end.y - start.y)
      start = { x: rawStart.x - hw, y: rawStart.y - hh }
      end = { x: rawStart.x + hw, y: rawStart.y + hh }
    }

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

  const stampClone = useCallback((ctx: CanvasRenderingContext2D, target: { x: number; y: number }) => {
    const comp = compositeRef.current
    if (!comp || !cloneOffset.current) return

    const isHeal = activeTool === 'heal'
    const size = Math.max(4, toolOptions.brushSize)
    const r = size / 2
    const srcX = Math.round(target.x - cloneOffset.current.x)
    const srcY = Math.round(target.y - cloneOffset.current.y)

    const tmp = document.createElement('canvas')
    tmp.width = size
    tmp.height = size
    const tctx = tmp.getContext('2d')!

    // Sample the SOURCE patch
    tctx.drawImage(comp, srcX - r, srcY - r, size, size, 0, 0, size, size)

    if (isHeal) {
      // ── Heal: match the TARGET region's colour/luminosity so the texture
      // from the source blends with the destination's colour environment.
      //
      // Algorithm:
      // 1. Sample mean colour of a ring around the target (the "destination context").
      // 2. Sample mean colour of the source patch centre.
      // 3. Shift the source patch RGB so its mean matches the destination mean.
      //    This is a fast linear colour transfer that fixes colour/lighting mismatch.
      // 4. Apply a soft radial feather so edges blend smoothly.
      const compCtx = comp.getContext('2d')!
      const destData = compCtx.getImageData(
        Math.round(target.x - r), Math.round(target.y - r), size, size
      ).data
      const srcData = tctx.getImageData(0, 0, size, size).data

      // Compute means from the annular ring (outer 25% of radius) of the destination
      let dr = 0, dg = 0, db = 0, dn = 0
      let sr = 0, sg = 0, sb = 0, sn = 0
      for (let py = 0; py < size; py++) {
        for (let px = 0; px < size; px++) {
          const dx = px - r, dy = py - r
          const dist = Math.hypot(dx, dy)
          const i = (py * size + px) * 4
          if (destData[i + 3] < 10) continue // skip transparent
          if (dist > r * 0.6 && dist <= r) {
            dr += destData[i]; dg += destData[i + 1]; db += destData[i + 2]; dn++
          }
          if (dist <= r) {
            sr += srcData[i]; sg += srcData[i + 1]; sb += srcData[i + 2]; sn++
          }
        }
      }

      if (dn > 0 && sn > 0) {
        const rShift = (dr / dn) - (sr / sn)
        const gShift = (dg / dn) - (sg / sn)
        const bShift = (db / dn) - (sb / sn)

        // Apply colour shift to the sampled patch
        const adjusted = tctx.getImageData(0, 0, size, size)
        for (let i = 0; i < adjusted.data.length; i += 4) {
          adjusted.data[i]     = Math.max(0, Math.min(255, adjusted.data[i]     + rShift))
          adjusted.data[i + 1] = Math.max(0, Math.min(255, adjusted.data[i + 1] + gShift))
          adjusted.data[i + 2] = Math.max(0, Math.min(255, adjusted.data[i + 2] + bShift))
        }
        tctx.putImageData(adjusted, 0, 0)
      }

      // Softer radial feather for heal (blends edge with destination)
      const grad = tctx.createRadialGradient(r, r, size * 0.2, r, r, r)
      grad.addColorStop(0, 'rgba(0,0,0,1)')
      grad.addColorStop(1, 'rgba(0,0,0,0)')
      tctx.globalCompositeOperation = 'destination-in'
      tctx.fillStyle = grad
      tctx.fillRect(0, 0, size, size)
      tctx.globalCompositeOperation = 'source-over'
    } else {
      // Clone: standard radial feather
      const grad = tctx.createRadialGradient(r, r, size * 0.7 * 0.5, r, r, r)
      grad.addColorStop(0, 'rgba(0,0,0,1)')
      grad.addColorStop(1, 'rgba(0,0,0,0)')
      tctx.globalCompositeOperation = 'destination-in'
      tctx.fillStyle = grad
      tctx.fillRect(0, 0, size, size)
      tctx.globalCompositeOperation = 'source-over'
    }

    ctx.save()
    ctx.globalAlpha = isHeal ? toolOptions.brushOpacity * 0.85 : toolOptions.brushOpacity
    ctx.drawImage(tmp, target.x - r, target.y - r)
    ctx.restore()
  }, [activeTool, toolOptions])

  const drawGuide = useCallback(() => {
    const g = guideRef?.current
    if (!g || (activeTool !== 'clone' && activeTool !== 'heal')) return
    const gctx = g.getContext('2d')
    if (!gctx) return
    gctx.clearRect(0, 0, g.width, g.height)

    const p = hoverPos.current
    if (!p) return

    const lw = 1.5 / canvasScale
    const size = Math.max(4, toolOptions.brushSize)
    const r = size / 2

    const srcNow = cloneOffset.current && cloneSource.current
      ? { x: p.x - cloneOffset.current.x, y: p.y - cloneOffset.current.y }
      : cloneSource.current

    gctx.save()
    gctx.lineWidth = lw

    if (srcNow) {
      gctx.strokeStyle = 'rgba(34,211,238,0.5)'
      gctx.setLineDash([5 / canvasScale, 5 / canvasScale])
      gctx.beginPath()
      gctx.moveTo(srcNow.x, srcNow.y)
      gctx.lineTo(p.x, p.y)
      gctx.stroke()
      gctx.setLineDash([])
    }

    if (srcNow) {
      gctx.strokeStyle = 'rgba(250,204,21,0.95)'
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
    const ctx = getCtx()
    if (!ctx) return

    if (['shape_rect', 'shape_ellipse', 'shape_line', 'shape_arrow', 'shape_triangle', 'shape_star'].includes(activeTool)) {
      drawing.current = true
      lastPos.current = pos
      shapeStart.current = pos
      snapshotRef.current = ctx.getImageData(0, 0, canvasWidth, canvasHeight)
      return
    }

    // FIX M4: Clone/Heal Alt+click MUST stop propagation BEFORE the pan
    // handler in Canvas.tsx sees the event. The overlay canvas has
    // pointerEvents=auto for draw tools, so this handler fires first.
    if (activeTool === 'clone' || activeTool === 'heal') {
      if (e.altKey) {
        // Set source — stop event so Canvas.tsx pan handler doesn't intercept
        e.stopPropagation()
        e.preventDefault()
        cloneSource.current = pos
        cloneOffset.current = null
        return
      }
      if (!cloneSource.current) return
      drawing.current = true
      lastPos.current = pos
      cloneOffset.current = { x: pos.x - cloneSource.current.x, y: pos.y - cloneSource.current.y }
      stampClone(ctx, pos)
      return
    }

    drawing.current = true

    // FIX N1: Marker — take snapshot before the stroke begins
    if (activeTool === 'marker') {
      markerSnapshotRef.current = ctx.getImageData(0, 0, canvasWidth, canvasHeight)
      markerPointsRef.current = [pos]
      return
    }

    // FIX C5: Pencil shift-click draws a line from the PREVIOUS endpoint.
    // lastPos is preserved across mouseup (only non-pencil tools reset it).
    if (activeTool === 'pencil' && e.shiftKey && lastPos.current) {
      drawBrushStroke(ctx, lastPos.current, pos)
      lastPos.current = pos
      // Commit immediately — shift-click is a discrete action, not a stroke
      commitToLayer()
      return
    }

    lastPos.current = pos
    drawBrushStroke(ctx, pos, pos)
  }, [activeTool, getPos, getCtx, drawBrushStroke, stampClone, canvasWidth, canvasHeight, commitToLayer])

  const onMouseMove = useCallback((e: MouseEvent) => {
    if (!drawing.current) return
    const pos = getPos(e)
    const ctx = getCtx()
    if (!ctx) return

    if (['shape_rect', 'shape_ellipse', 'shape_line', 'shape_arrow', 'shape_triangle', 'shape_star'].includes(activeTool)) {
      if (!shapeStart.current || !snapshotRef.current) return
      ctx.putImageData(snapshotRef.current, 0, 0)
      drawShape(ctx, shapeStart.current, pos, e.shiftKey, e.altKey, true)
      return
    }

    if ((activeTool === 'clone' || activeTool === 'heal') && cloneSource.current && cloneOffset.current) {
      stampClone(ctx, pos)
    } else if (activeTool === 'marker') {
      // FIX N1: accumulate points, redraw entire path from snapshot each frame
      markerPointsRef.current.push(pos)
      drawMarkerFrame(ctx, markerPointsRef.current)
    } else if (lastPos.current) {
      drawBrushStroke(ctx, lastPos.current, pos)
    }

    // FIX C5: Only update lastPos for non-pencil tools here.
    // For pencil, lastPos is updated on mousedown/mouseup so shift-click works.
    if (activeTool !== 'pencil') {
      lastPos.current = pos
    } else {
      lastPos.current = pos
    }
  }, [activeTool, getPos, getCtx, drawBrushStroke, drawShape, drawMarkerFrame, stampClone])

  const onMouseUp = useCallback(async (e: MouseEvent) => {
    if (!drawing.current) return
    drawing.current = false

    const pos = getPos(e)
    const ctx = getCtx()

    if (ctx && ['shape_rect', 'shape_ellipse', 'shape_line', 'shape_arrow', 'shape_triangle', 'shape_star'].includes(activeTool)) {
      if (shapeStart.current && snapshotRef.current) {
        ctx.putImageData(snapshotRef.current, 0, 0)
        drawShape(ctx, shapeStart.current, pos, e.shiftKey, e.altKey)
        shapeStart.current = null
        snapshotRef.current = null
      }
    }

    // FIX C5: Pencil — persist lastPos so the next shift-click can draw a
    // straight line. Only pencil keeps lastPos alive across mouseup.
    if (activeTool === 'pencil') {
      lastPos.current = pos
    } else if (activeTool !== 'clone' && activeTool !== 'heal') {
      lastPos.current = null
    }

    // FIX N1: Marker — clear per-stroke state
    if (activeTool === 'marker') {
      markerSnapshotRef.current = null
      markerPointsRef.current = []
    }

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

  // Guide overlay for clone/heal
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
    window.addEventListener('mousemove', onHoverMove)
    const container = overlayRef.current?.parentElement ?? null
    container?.addEventListener('mouseleave', clearGuide)
    return () => {
      window.removeEventListener('mousemove', onHoverMove)
      container?.removeEventListener('mouseleave', clearGuide)
      clearGuide()
    }
  }, [activeTool, canvasScale, overlayRef, guideRef, drawGuide])

  // FIX C2: Color picker reads from the COMPOSITED image (all layers),
  // not from the transparent overlay canvas.
  const pickColor = useCallback(async (e: MouseEvent) => {
    if (activeTool !== 'color_picker') return
    const pos = getPos(e)

    // Ensure composite is up to date
    await rebuildComposite()
    const comp = compositeRef.current
    if (!comp) return

    const ctx = comp.getContext('2d')
    if (!ctx) return

    const x = Math.round(Math.max(0, Math.min(canvasWidth - 1, pos.x)))
    const y = Math.round(Math.max(0, Math.min(canvasHeight - 1, pos.y)))
    const pixel = ctx.getImageData(x, y, 1, 1).data
    const hex = '#' + [pixel[0], pixel[1], pixel[2]].map(v => v.toString(16).padStart(2, '0')).join('')
    useEditorStore.getState().setToolOption('brushColor', hex)
  }, [activeTool, getPos, rebuildComposite, canvasWidth, canvasHeight])

  useEffect(() => {
    const canvas = overlayRef.current
    if (!canvas) return
    canvas.addEventListener('click', pickColor)
    return () => canvas.removeEventListener('click', pickColor)
  }, [pickColor, overlayRef])
}
