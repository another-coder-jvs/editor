/**
 * useCanvasSelect – handles selection tools: rect, ellipse, lasso, free, magic_select, object_select
 * Draws marching-ants selection overlay and exposes selection mask.
 * Also renders AI-detected bbox overlays (magic_select / object_select).
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

  // Draw marching ants for a single selection mask
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

    // Fill with semi-transparent blue
    ctx.setLineDash([])
    ctx.fillStyle = 'rgba(79,142,247,0.12)'
    ctx.fill()
    ctx.restore()
  }, [canvasScale])

  // Draw AI detection bbox overlays with marching ants
  const drawAiSelections = useCallback((ctx: CanvasRenderingContext2D) => {
    if (!aiSelections.length) return
    aiSelections.forEach((obj, i) => {
      const { x, y, width, height } = obj.bbox
      ctx.save()
      ctx.setLineDash([6, 3])
      ctx.lineDashOffset = -marchOffset.current - i * 4 // offset per box
      ctx.strokeStyle = '#4f8ef7'
      ctx.lineWidth = 2 / canvasScale
      ctx.shadowColor = '#000'
      ctx.shadowBlur = 2

      // Draw bbox
      ctx.strokeRect(x, y, width, height)

      // Fill
      ctx.setLineDash([])
      ctx.fillStyle = 'rgba(79,142,247,0.1)'
      ctx.fillRect(x, y, width, height)

      // Label
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

  // Combined draw: selection mask + AI bboxes
  const drawSelection = useCallback((mask: SelectionMask | null) => {
    const canvas = selectionRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.clearRect(0, 0, canvasWidth, canvasHeight)

    // Draw AI selection bboxes
    drawAiSelections(ctx)

    // Draw geometric selection mask
    if (mask) drawMask(ctx, mask)
  }, [canvasWidth, canvasScale, selectionRef, drawAiSelections, drawMask])

  // Animate marching ants (selection mask + AI bboxes)
  useEffect(() => {
    const animate = () => {
      marchOffset.current = (marchOffset.current + 0.3) % 9
      drawSelection(selection)
      animRef.current = requestAnimationFrame(animate)
    }
    animRef.current = requestAnimationFrame(animate)
    return () => cancelAnimationFrame(animRef.current)
  }, [selection, drawSelection, aiSelections])

  const onMouseDown = useCallback((e: MouseEvent) => {
    if (!SELECT_TOOLS.includes(activeTool)) return
    if (e.button !== 0) return
    drawing.current = true
    const pos = getPos(e)
    startPos.current = pos
    lassoPoints.current = [pos]
  }, [activeTool, getPos])

  const onMouseMove = useCallback((e: MouseEvent) => {
    if (!drawing.current || !startPos.current) return
    const pos = getPos(e)

    let mask: SelectionMask | null = null
    if (activeTool === 'rect_select') {
      mask = {
        type: 'rect',
        rect: {
          x: Math.min(startPos.current.x, pos.x),
          y: Math.min(startPos.current.y, pos.y),
          w: Math.abs(pos.x - startPos.current.x),
          h: Math.abs(pos.y - startPos.current.y),
        },
      }
    } else if (activeTool === 'ellipse_select') {
      mask = {
        type: 'ellipse',
        rect: {
          x: Math.min(startPos.current.x, pos.x),
          y: Math.min(startPos.current.y, pos.y),
          w: Math.abs(pos.x - startPos.current.x),
          h: Math.abs(pos.y - startPos.current.y),
        },
      }
    } else if (activeTool === 'lasso_select' || activeTool === 'free_select') {
      lassoPoints.current.push(pos)
      mask = {
        type: activeTool === 'lasso_select' ? 'lasso' : 'free',
        points: [...lassoPoints.current],
      }
    }

    setSelection(mask)
    drawSelection(mask)
  }, [activeTool, getPos, drawSelection])

  const onMouseUp = useCallback((e: MouseEvent) => {
    if (!drawing.current) return
    drawing.current = false
    const pos = getPos(e)

    let finalMask: SelectionMask | null = selection

    if (activeTool === 'lasso_select' || activeTool === 'free_select') {
      lassoPoints.current.push(pos)
      finalMask = {
        type: activeTool === 'lasso_select' ? 'lasso' : 'free',
        points: [...lassoPoints.current],
      }
    }

    // Apply feather if set
    const feather = toolOptions.selectionFeather
    if (finalMask && feather > 0) {
      // Store feather in mask for downstream use
      ;(finalMask as any).feather = feather
    }

    setSelection(finalMask)
    onSelectionChange(finalMask)
  }, [activeTool, getPos, selection, toolOptions.selectionFeather, onSelectionChange])

  // Clear selection on Escape
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setSelection(null)
        onSelectionChange(null)
        const canvas = selectionRef.current
        if (canvas) canvas.getContext('2d')?.clearRect(0, 0, canvasWidth, canvasHeight)
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onSelectionChange, canvasWidth, canvasHeight, selectionRef])

  useEffect(() => {
    const canvas = selectionRef.current
    if (!canvas) return
    canvas.addEventListener('mousedown', onMouseDown)
    window.addEventListener('mousemove', onMouseMove)
    window.addEventListener('mouseup', onMouseUp)
    return () => {
      canvas.removeEventListener('mousedown', onMouseDown)
      window.removeEventListener('mousemove', onMouseMove)
      window.removeEventListener('mouseup', onMouseUp)
    }
  }, [onMouseDown, onMouseMove, onMouseUp, selectionRef])

  return { selection, clearSelection: () => { setSelection(null); onSelectionChange(null) } }
}
