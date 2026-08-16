import React, { useRef, useCallback, useEffect, useState } from 'react'

type ResizeEdge = 'right' | 'left' | 'bottom' | 'top'

interface Props {
  children: React.ReactNode
  edge: ResizeEdge
  defaultSize?: number
  minSize?: number
  maxSize?: number
  title?: string
  className?: string
  style?: React.CSSProperties
  onResize?: (size: number) => void
}

// Collapsed sizes: horizontal panels show a 28px wide tab; vertical show a 28px tall bar
const COLLAPSED_H = 28  // px — width when a left/right panel is closed
const COLLAPSED_V = 28  // px — height when a top/bottom panel is closed

export const ResizablePanel: React.FC<Props> = ({
  children, edge, defaultSize, minSize = 80, maxSize = 800,
  title, className = '', style, onResize,
}) => {
  const [size, setSize] = useState(defaultSize)
  const [open, setOpen] = useState(true)
  const dragging = useRef(false)
  const startPos = useRef(0)
  const startSize = useRef(0)
  const panelRef = useRef<HTMLDivElement>(null)

  const isHorizontal = edge === 'right' || edge === 'left'
  const fill = defaultSize === undefined

  const onMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    dragging.current = true
    startPos.current = isHorizontal ? e.clientX : e.clientY
    const el = panelRef.current
    startSize.current = el ? (isHorizontal ? el.offsetWidth : el.offsetHeight) : (size ?? 200)
  }, [isHorizontal, size])

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (!dragging.current) return
      const delta = isHorizontal
        ? (edge === 'right' ? e.clientX - startPos.current : startPos.current - e.clientX)
        : (edge === 'bottom' ? e.clientY - startPos.current : startPos.current - e.clientY)
      const next = Math.min(maxSize, Math.max(minSize, startSize.current + delta))
      setSize(next)
      onResize?.(next)
    }
    const onUp = () => { dragging.current = false }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
    return () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }
  }, [edge, isHorizontal, maxSize, minSize, onResize])

  const sizeStyle: React.CSSProperties = (() => {
    if (!open) {
      return isHorizontal
        ? { width: COLLAPSED_H, minWidth: COLLAPSED_H, maxWidth: COLLAPSED_H, flex: 'none' }
        : { height: COLLAPSED_V, minHeight: COLLAPSED_V, maxHeight: COLLAPSED_V, flex: 'none' }
    }
    if (fill) return { flex: 1, minWidth: 0, minHeight: 0 }
    return isHorizontal
      ? { width: size, minWidth: size, maxWidth: size, flex: 'none' }
      : { height: size, minHeight: size, maxHeight: size, flex: 'none' }
  })()

  const handleStyle: React.CSSProperties = {
    position: 'absolute', zIndex: 10,
    ...(edge === 'right'  && { top: 0, right: -3, width: 6, height: '100%', cursor: 'col-resize' }),
    ...(edge === 'left'   && { top: 0, left: -3,  width: 6, height: '100%', cursor: 'col-resize' }),
    ...(edge === 'bottom' && { bottom: -3, left: 0, height: 6, width: '100%', cursor: 'row-resize' }),
    ...(edge === 'top'    && { top: -3, left: 0,   height: 6, width: '100%', cursor: 'row-resize' }),
  }

  // ── Collapsed state: always show a clickable strip to re-open ──
  if (!open) {
    if (isHorizontal) {
      // Vertical tab strip — rotated title, click to reopen
      return (
        <div
          ref={panelRef}
          className={`relative flex-shrink-0 flex items-center justify-center bg-dark-800 border-dark-600 cursor-pointer hover:bg-dark-700 transition-colors ${className}`}
          style={{ ...sizeStyle, ...style, borderRight: edge === 'right' ? '1px solid #2e2e2e' : undefined, borderLeft: edge === 'left' ? '1px solid #2e2e2e' : undefined }}
          onClick={() => setOpen(true)}
          title={`Expand ${title}`}
        >
          <span style={{
            writingMode: 'vertical-rl',
            transform: edge === 'left' ? 'rotate(180deg)' : 'none',
            fontSize: 10,
            fontWeight: 600,
            letterSpacing: '0.08em',
            textTransform: 'uppercase',
            color: '#6b7280',
            userSelect: 'none',
            whiteSpace: 'nowrap',
          }}>
            {title ?? ''}
          </span>
        </div>
      )
    } else {
      // Horizontal bar — title + chevron, click to reopen
      return (
        <div
          ref={panelRef}
          className={`relative flex-shrink-0 flex items-center gap-1.5 px-3 bg-dark-800 border-dark-600 cursor-pointer hover:bg-dark-700 transition-colors ${className}`}
          style={{ ...sizeStyle, ...style, borderBottom: edge === 'bottom' ? '1px solid #2e2e2e' : undefined, borderTop: edge === 'top' ? '1px solid #2e2e2e' : undefined }}
          onClick={() => setOpen(true)}
          title={`Expand ${title}`}
        >
          <span className="flex-1 text-xs font-semibold text-gray-500 uppercase tracking-wider select-none">{title}</span>
          <span className="text-gray-600 text-xs" style={{ transform: 'rotate(-90deg)', display: 'inline-block' }}>▾</span>
        </div>
      )
    }
  }

  // ── Open state ──
  return (
    <div
      ref={panelRef}
      className={`relative flex flex-col overflow-hidden ${className}`}
      style={{ ...sizeStyle, ...style }}
    >
      {/* Header with collapse toggle */}
      {title && (
        <button
          onClick={() => setOpen(false)}
          className="flex items-center gap-1.5 w-full px-3 py-1.5 text-xs font-semibold text-gray-400 uppercase tracking-wider border-b border-dark-600 hover:text-white hover:bg-dark-700 transition-colors flex-shrink-0 text-left"
          title={`Collapse ${title}`}
        >
          <span className="flex-1">{title}</span>
          <span style={{ display: 'inline-block' }}>{isHorizontal ? '›' : '▾'}</span>
        </button>
      )}

      {/* Content */}
      <div className="flex-1 overflow-hidden flex flex-col min-h-0">
        {children}
      </div>

      {/* Drag handle */}
      {!fill && (
        <div style={handleStyle} onMouseDown={onMouseDown} className="group">
          <div style={{
            position: 'absolute',
            ...(isHorizontal
              ? { top: '50%', left: '50%', transform: 'translate(-50%,-50%)', width: 2, height: 32 }
              : { left: '50%', top: '50%', transform: 'translate(-50%,-50%)', height: 2, width: 32 }),
            background: '#4f8ef7', borderRadius: 2, opacity: 0, transition: 'opacity 0.15s',
          }} className="group-hover:opacity-100" />
        </div>
      )}
    </div>
  )
}
