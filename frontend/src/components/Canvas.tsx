import React, { useRef, useEffect, useCallback, useState } from 'react'
import { useEditorStore } from '../store/editorStore'

const CHECKERBOARD = `
  repeating-conic-gradient(#2a2a2a 0% 25%, #1a1a1a 0% 50%)
  0 0 / 20px 20px
`
const API_BASE = 'http://localhost:5000'
export const Canvas: React.FC = () => {
  const {
    layers, originalImageUrl, canvasWidth, canvasHeight,
    canvasScale, canvasOffset, setCanvasScale, setCanvasOffset,
    selectedLayerIds, selectLayer, clearSelection,
  } = useEditorStore()

  const containerRef = useRef<HTMLDivElement>(null)
  const [isPanning, setIsPanning] = useState(false)
  const panStart = useRef({ x: 0, y: 0, ox: 0, oy: 0 })

  // Zoom with wheel
  const onWheel = useCallback(
    (e: WheelEvent) => {
      e.preventDefault()
      const delta = e.deltaY > 0 ? 0.9 : 1.1
      setCanvasScale(Math.min(8, Math.max(0.1, canvasScale * delta)))
    },
    [canvasScale, setCanvasScale]
  )

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [onWheel])

  // Pan with middle mouse or space+drag
  const onMouseDown = (e: React.MouseEvent) => {
    if (e.button === 1 || e.altKey) {
      setIsPanning(true)
      panStart.current = { x: e.clientX, y: e.clientY, ox: canvasOffset.x, oy: canvasOffset.y }
      e.preventDefault()
    }
  }
  const onMouseMove = (e: React.MouseEvent) => {
    if (!isPanning) return
    setCanvasOffset({
      x: panStart.current.ox + (e.clientX - panStart.current.x),
      y: panStart.current.oy + (e.clientY - panStart.current.y),
    })
  }
  const onMouseUp = () => setIsPanning(false)

  const sortedLayers = [...layers]
    .filter((l) => l.visible)
    .sort((a, b) => a.z_index - b.z_index)

  return (
    <div
      ref={containerRef}
      className="flex-1 overflow-hidden relative"
      style={{ background: CHECKERBOARD, cursor: isPanning ? 'grabbing' : 'default' }}
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
      onMouseUp={onMouseUp}
      onMouseLeave={onMouseUp}
      onClick={(e) => {
        if ((e.target as HTMLElement).dataset.canvas === 'bg') clearSelection()
      }}
    >
      {/* Zoom indicator */}
      <div className="absolute top-3 right-3 z-10 bg-dark-800 text-xs text-gray-400 px-2 py-1 rounded">
        {Math.round(canvasScale * 100)}%
      </div>

      {/* Zoom controls */}
      <div className="absolute bottom-3 right-3 z-10 flex gap-1">
        <button
          className="bg-dark-700 text-gray-300 px-2 py-1 rounded text-sm hover:bg-dark-600"
          onClick={() => setCanvasScale(Math.min(8, canvasScale * 1.2))}
        >+</button>
        <button
          className="bg-dark-700 text-gray-300 px-2 py-1 rounded text-sm hover:bg-dark-600"
          onClick={() => setCanvasScale(1)}
        >1:1</button>
        <button
          className="bg-dark-700 text-gray-300 px-2 py-1 rounded text-sm hover:bg-dark-600"
          onClick={() => setCanvasScale(Math.max(0.1, canvasScale * 0.8))}
        >−</button>
      </div>

      {/* Canvas area */}
      <div
        style={{
          position: 'absolute',
          left: '50%',
          top: '50%',
          transform: `translate(calc(-50% + ${canvasOffset.x}px), calc(-50% + ${canvasOffset.y}px)) scale(${canvasScale})`,
          transformOrigin: 'center center',
          width: canvasWidth,
          height: canvasHeight,
          boxShadow: '0 0 40px rgba(0,0,0,0.8)',
        }}
      >
        {/* Background image */}
        {originalImageUrl && (
          <img
            src={originalImageUrl}
            alt="original"
            data-canvas="bg"
            style={{
              position: 'absolute', inset: 0,
              width: '100%', height: '100%',
              objectFit: 'fill', userSelect: 'none',
            }}
            draggable={false}
          />
        )}

        {/* Layers */}
        {sortedLayers.map((layer) => {
          const isSelected = selectedLayerIds.includes(layer.id)
          // const thumbUrl = layer.png_path
          //   ? `/temp/${layer.png_path.split('/temp/')[1] ?? layer.png_path}`
          //   : null
          const thumbUrl = layer.png_path ? `${API_BASE}${layer.png_path}` : null
          console.log("GOT : " , thumbUrl)
          return (
            <div
              key={layer.id}
              onClick={(e) => { e.stopPropagation(); selectLayer(layer.id, e.ctrlKey || e.metaKey) }}
              style={{
                position: 'absolute',
                left: layer.bbox.x + layer.position.x,
                top: layer.bbox.y + layer.position.y,
                width: layer.bbox.width,
                height: layer.bbox.height,
                opacity: layer.opacity,
                transform: `rotate(${layer.rotation}deg) scale(${layer.scale.x}, ${layer.scale.y})`,
                transformOrigin: 'center center',
                outline: isSelected ? '2px solid #4f8ef7' : 'none',
                cursor: layer.locked ? 'not-allowed' : 'pointer',
                zIndex: layer.z_index,
              }}
            >
              {thumbUrl && (
                <img
                  src={thumbUrl}
                  alt={layer.name}
                  style={{ width: '100%', height: '100%', objectFit: 'fill' }}
                  draggable={false}
                />
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
