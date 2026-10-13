/**
 * useCanvasSelect – selection tools: rect, ellipse, lasso, free, magic_select, object_select
 *
 * Fixes applied vs previous version:
 *   C3: free_select now works as a proper polygon tool:
 *       - click adds a vertex
 *       - double-click OR clicking within 10px of the first vertex closes the polygon
 *       - Backspace removes the last vertex
 *       - Esc cancels the in-progress polygon
 *       Previously it accumulated points on every mousemove (= lasso behaviour).
 *   C4: rect_select and ellipse_select now honour:
 *       - Shift = constrain to square / circle
 *       - Alt  = draw from center
 */
import { useRef, useEffect, useCallback, useState } from 'react'
import { useEditorStore } from '../store/editorStore'

export type SelectionMask = {
  type: 'rect' | 'ellipse' | 'lasso' | 'free'
  points?: { x: number; y: number }[]
  rect?: { x: number; y: number; w: number; h: number }
}

export function useCanvasSelect(
  selectionRef: React.RefObject<HTMLCanvasElement>,
  canvasWidth: number,
  canvasHeight: number,
  canvasScale: number,
  onSelectionChange: (mask: SelectionMask | null) => void,
) {
  const { activeTool, toolOptions, aiSelections } = useEditorStore()
  const drawing = useRef(false)
  const startPos = useRef<{ x: number; y: number } | null>(null)
  const lassoPoints = useRef<{ x: number; y: number }[]>([])
  const [selection, setSelection] = useState<SelectionMask | null>(null)
  const marchOffset = useRef(0)
  const animRef = useRef<number>(0)

  // FIX C3: polygon state for free_select
  const polyVertices = useRef<{ x: number; y: number }[]>([])
  const polyActive = useRef(false)    // true while a polygon is being built
  const polyPreview = useRef<{ x: number; y: number } | null>(null) // cursor for preview line

  const SELECT_TOOLS = ['rect_select', 'ellipse_select', 'lasso_select', 'free_select']

  const getPos = useCallback((e: MouseEvent): { x: number; y: number } => {
    const canvas = selectionRef.current
    if (!canvas) return { x: 0, y: 0 }
    const rect = canvas.getBoundingClientRect()
    return {
      x: (e.clientX - rect.left) / canvasScale,
      y: (e.clientY - rect.top) / canvasScale,
    }
  }, [canvasScale, selectionRef])

  // ── Draw helpers ──────────────────────────────────────────────────────────

  const drawMask = useCallback((ctx: CanvasRenderingContext2D, mask: SelectionMask) => {
    ctx.save()
    ctx.setLineDash([6, 3])
    ctx.lineDashOffset = -marchOffset.current
    ctx.strokeStyle = '#fff'
    ctx.lineWidth = 1.5 / canvasScale
    ctx.shadowColor = '#000'
    ctx.shadowBlur = 1

    ctx.beginPath()
    if (mask.type === 'rect' && mask.rect) {
      ctx.rect(mask.rect.x, mask.rect.y, mask.rect.w, mask.rect.h)
    } else if (mask.type === 'ellipse' && mask.rect) {
      const { x, y, w, h } = mask.rect
      ctx.ellipse(x + w / 2, y + h / 2, Math.abs(w / 2), Math.abs(h / 2), 0, 0, Math.PI * 2)
    } else if ((mask.type === 'lasso' || mask.type === 'free') && mask.points?.length) {
      ctx.moveTo(mask.points[0].x, mask.points[0].y)
      mask.points.forEach(p => ctx.lineTo(p.x, p.y))
      ctx.closePath()
    }
    ctx.stroke()
    ctx.setLineDash([])
    ctx.fillStyle = 'rgba(79,142,247,0.12)'
    ctx.fill()
    ctx.restore()
  }, [canvasScale])

  const drawAiSelections = useCallback((ctx: CanvasRenderingContext2D) => {
    if (!aiSelections.length) return
    aiSelections.forEach((obj, i) => {
      const { x, y, width, height } = obj.bbox
      ctx.save()
      ctx.setLineDash([6, 3])
      ctx.lineDashOffset = -marchOffset.current - i * 4
      ctx.strokeStyle = '#4f8ef7'
      ctx.lineWidth = 2 / canvasScale
      ctx.shadowColor = '#000'
      ctx.shadowBlur = 2
      ctx.strokeRect(x, y, width, height)
      ctx.setLineDash([])
      ctx.fillStyle = 'rgba(79,142,247,0.1)'
      ctx.fillRect(x, y, width, height)

      const label = `${obj.label} ${Math.round(obj.score * 100)}%`
      ctx.font = `bold ${12 / canvasScale}px sans-serif`
      const textW = ctx.measureText(label).width
      const labelH = 16 / canvasScale
      ctx.fillStyle = 'rgba(79,142,247,0.9)'
      ctx.fillRect(x, y - labelH - 2, textW + 8 / canvasScale, labelH + 4 / canvasScale)
      ctx.fillStyle = '#fff'
      ctx.fillText(label, x + 4 / canvasScale, y - 4 / canvasScale)
      ctx.restore()
    })
  }, [aiSelections, canvasScale])

  // FIX C3: draw the in-progress polygon with vertex dots and a preview line
  const drawPolyInProgress = useCallback((ctx: CanvasRenderingContext2D) => {
    const verts = polyVertices.current
    if (verts.length === 0) return

    ctx.save()
    ctx.strokeStyle = '#fff'
    ctx.setLineDash([4, 3])
    ctx.lineDashOffset = -marchOffset.current
    ctx.lineWidth = 1.5 / canvasScale
    ctx.shadowColor = '#000'
    ctx.shadowBlur = 1

    ctx.beginPath()
    ctx.moveTo(verts[0].x, verts[0].y)
    for (let i = 1; i < verts.length; i++) ctx.lineTo(verts[i].x, verts[i].y)
    // Preview line to cursor
    if (polyPreview.current) ctx.lineTo(polyPreview.current.x, polyPreview.current.y)
    ctx.stroke()
    ctx.setLineDash([])

    // Vertex dots
    verts.forEach((v, i) => {
      ctx.beginPath()
      ctx.arc(v.x, v.y, i === 0 ? 5 / canvasScale : 3 / canvasScale, 0, Math.PI * 2)
      ctx.fillStyle = i === 0 ? '#4f8ef7' : '#fff'
      ctx.fill()
      ctx.strokeStyle = '#000'
      ctx.lineWidth = 1 / canvasScale
      ctx.stroke()
    })
    ctx.restore()
  }, [canvasScale])

  const drawSelection = useCallback((mask: SelectionMask | null) => {
    const canvas = selectionRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.clearRect(0, 0, canvasWidth, canvasHeight)
    drawAiSelections(ctx)
    if (mask) drawMask(ctx, mask)
    // If polygon tool active, overlay the in-progress polygon
    if (activeTool === 'free_select' && polyActive.current) {
      drawPolyInProgress(ctx)
    }
  }, [canvasWidth, canvasHeight, selectionRef, drawAiSelections, drawMask, drawPolyInProgress, activeTool])

  // Animation loop
  useEffect(() => {
    const animate = () => {
      marchOffset.current = (marchOffset.current + 0.3) % 9
      drawSelection(selection)
      animRef.current = requestAnimationFrame(animate)
    }
    animRef.current = requestAnimationFrame(animate)
    return () => cancelAnimationFrame(animRef.current)
  }, [selection, drawSelection, aiSelections])

  // ── Mouse handlers ────────────────────────────────────────────────────────

  const onMouseDown = useCallback((e: MouseEvent) => {
    if (!SELECT_TOOLS.includes(activeTool)) return
    if (e.button !== 0) return

    const pos = getPos(e)

    // FIX C3: free_select is click-based, not drag-based
    if (activeTool === 'free_select') {
      // Handled in onClick instead — return early here
      return
    }

    drawing.current = true
    startPos.current = pos
    lassoPoints.current = [pos]
  }, [activeTool, getPos])

  const onMouseMove = useCallback((e: MouseEvent) => {
    const pos = getPos(e)

    // FIX C3: update preview line for in-progress polygon
    if (activeTool === 'free_select' && polyActive.current) {
      polyPreview.current = pos
      // Re-draw without changing committed selection
      drawSelection(selection)
      return
    }

    if (!drawing.current || !startPos.current) return

    let mask: SelectionMask | null = null
    if (activeTool === 'rect_select') {
      // FIX C4: Shift = square, Alt = from-center
      let ex = pos.x, ey = pos.y
      const dx = pos.x - startPos.current.x
      const dy = pos.y - startPos.current.y
      if (e.shiftKey) {
        const size = Math.min(Math.abs(dx), Math.abs(dy))
        ex = startPos.current.x + Math.sign(dx) * size
        ey = startPos.current.y + Math.sign(dy) * size
      }
      let x = Math.min(startPos.current.x, ex)
      let y = Math.min(startPos.current.y, ey)
      let w = Math.abs(ex - startPos.current.x)
      let h = Math.abs(ey - startPos.current.y)
      if (e.altKey) {
        x = startPos.current.x - w
        y = startPos.current.y - h
        w *= 2
        h *= 2
      }
      mask = { type: 'rect', rect: { x, y, w, h } }
    } else if (activeTool === 'ellipse_select') {
      // FIX C4: Shift = circle, Alt = from-center
      let ex = pos.x, ey = pos.y
      const dx = pos.x - startPos.current.x
      const dy = pos.y - startPos.current.y
      if (e.shiftKey) {
        const size = Math.min(Math.abs(dx), Math.abs(dy))
        ex = startPos.current.x + Math.sign(dx) * size
        ey = startPos.current.y + Math.sign(dy) * size
      }
      let x = Math.min(startPos.current.x, ex)
      let y = Math.min(startPos.current.y, ey)
      let w = Math.abs(ex - startPos.current.x)
      let h = Math.abs(ey - startPos.current.y)
      if (e.altKey) {
        x = startPos.current.x - w
        y = startPos.current.y - h
        w *= 2
        h *= 2
      }
      mask = { type: 'ellipse', rect: { x, y, w, h } }
    } else if (activeTool === 'lasso_select') {
      lassoPoints.current.push(pos)
      mask = { type: 'lasso', points: [...lassoPoints.current] }
    }

    setSelection(mask)
    drawSelection(mask)
  }, [activeTool, getPos, drawSelection, selection])

  const onMouseUp = useCallback((e: MouseEvent) => {
    if (activeTool === 'free_select') return // handled by click
    if (!drawing.current) return
    drawing.current = false
    const pos = getPos(e)

    let finalMask: SelectionMask | null = selection

    if (activeTool === 'lasso_select') {
      lassoPoints.current.push(pos)
      finalMask = { type: 'lasso', points: [...lassoPoints.current] }
    }

    const feather = toolOptions.selectionFeather
    if (finalMask && feather > 0) {
      ;(finalMask as any).feather = feather
    }

    setSelection(finalMask)
    onSelectionChange(finalMask)
  }, [activeTool, getPos, selection, toolOptions.selectionFeather, onSelectionChange])

  // FIX C3: polygon tool click handler
  const onClickForPolygon = useCallback((e: MouseEvent) => {
    if (activeTool !== 'free_select') return
    const pos = getPos(e)
    const verts = polyVertices.current

    // Double-click OR click near first vertex closes the polygon
    const isClose = verts.length >= 3 &&
      Math.hypot(pos.x - verts[0].x, pos.y - verts[0].y) < 10 / canvasScale
    const isDbl = e.detail >= 2

    if (!polyActive.current) {
      // Start a new polygon
      polyVertices.current = [pos]
      polyActive.current = true
      polyPreview.current = pos
      drawSelection(null)
      return
    }

    if (isClose || isDbl) {
      // Close and commit
      const finalMask: SelectionMask = { type: 'free', points: [...verts] }
      polyVertices.current = []
      polyActive.current = false
      polyPreview.current = null
      setSelection(finalMask)
      onSelectionChange(finalMask)
      drawSelection(finalMask)
      return
    }

    // Add vertex
    polyVertices.current = [...verts, pos]
    drawSelection(selection)
  }, [activeTool, canvasScale, getPos, drawSelection, onSelectionChange, selection])

  // ── Keyboard handlers ─────────────────────────────────────────────────────

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      // Esc: cancel polygon OR clear any selection
      if (e.key === 'Escape') {
        if (activeTool === 'free_select' && polyActive.current) {
          polyVertices.current = []
          polyActive.current = false
          polyPreview.current = null
          drawSelection(null)
          return
        }
        setSelection(null)
        onSelectionChange(null)
        const canvas = selectionRef.current
        if (canvas) canvas.getContext('2d')?.clearRect(0, 0, canvasWidth, canvasHeight)
        return
      }

      // FIX C3: Backspace removes the last vertex from an in-progress polygon
      if (e.key === 'Backspace' && activeTool === 'free_select' && polyActive.current) {
        e.preventDefault()
        if (polyVertices.current.length > 1) {
          polyVertices.current = polyVertices.current.slice(0, -1)
          drawSelection(selection)
        } else {
          // Only one vertex left — cancel the polygon
          polyVertices.current = []
          polyActive.current = false
          polyPreview.current = null
          drawSelection(null)
        }
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [activeTool, onSelectionChange, canvasWidth, canvasHeight, selectionRef, drawSelection, selection])

  // Cancel polygon when tool switches away
  useEffect(() => {
    if (activeTool !== 'free_select' && polyActive.current) {
      polyVertices.current = []
      polyActive.current = false
      polyPreview.current = null
    }
  }, [activeTool])

  // ── Event listener wiring ─────────────────────────────────────────────────

  useEffect(() => {
    const canvas = selectionRef.current
    if (!canvas) return
    canvas.addEventListener('mousedown', onMouseDown)
    canvas.addEventListener('click', onClickForPolygon)
    window.addEventListener('mousemove', onMouseMove)
    window.addEventListener('mouseup', onMouseUp)
    return () => {
      canvas.removeEventListener('mousedown', onMouseDown)
      canvas.removeEventListener('click', onClickForPolygon)
      window.removeEventListener('mousemove', onMouseMove)
      window.removeEventListener('mouseup', onMouseUp)
    }
  }, [onMouseDown, onMouseMove, onMouseUp, onClickForPolygon, selectionRef])

  const clearSelectionFn = useCallback(() => {
    setSelection(null)
    onSelectionChange(null)
    polyVertices.current = []
    polyActive.current = false
    polyPreview.current = null
  }, [onSelectionChange])

  return { selection, clearSelection: clearSelectionFn }
}
