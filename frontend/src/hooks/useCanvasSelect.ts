/**
 * useCanvasSelect – handles selection tools: rect, ellipse, lasso, free, magic_select
 * Draws marching-ants selection overlay and exposes selection mask.
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
  const { activeTool, toolOptions } = useEditorStore()
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

  // Draw marching ants
  const drawSelection = useCallback((mask: SelectionMask | null) => {
    const canvas = selectionRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.clearRect(0, 0, canvasWidth, canvasHeight)
    if (!mask) return

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
  }, [canvasWidth, canvasHeight, canvasScale, selectionRef])

  // Animate marching ants
  useEffect(() => {
    const animate = () => {
      marchOffset.current = (marchOffset.current + 0.3) % 9
      if (selection) drawSelection(selection)
      animRef.current = requestAnimationFrame(animate)
    }
    animRef.current = requestAnimationFrame(animate)
    return () => cancelAnimationFrame(animRef.current)
  }, [selection, drawSelection])

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
