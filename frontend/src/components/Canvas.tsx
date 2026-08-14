import React, { useRef, useEffect, useCallback, useState } from 'react'
import { useEditorStore } from '../store/editorStore'
import {baseImagesUrl} from "@/config"
import { useBlobUrl } from '../hooks/useBlobUrl'
const CHECKERBOARD = `
  repeating-conic-gradient(#2a2a2a 0% 25%, #1a1a1a 0% 50%)
  0 0 / 20px 20px
`
const API_BASE = baseImagesUrl || 'http://localhost:5000'

interface LayerImageProps {
  layer: any
  rawUrl: string | null
  isSelected: boolean
  selectLayer: (id: string, multi: boolean) => void
  updateLayer: (id: string, patch: any) => void
  canvasScale: number
}
const LayerImage: React.FC<LayerImageProps> = ({ layer, rawUrl, isSelected, selectLayer, updateLayer, canvasScale }) => {
  const thumbUrl = useBlobUrl(rawUrl)
  const dragStart = useRef<{ mx: number; my: number; px: number; py: number } | null>(null)
  const [inlineEditing, setInlineEditing] = useState(false)
  const editRef = useRef<HTMLDivElement>(null)

  const isTextPreview = layer.id.includes('_textpreview_')
  const isTxtLayer = layer.id.includes('_txt_')
  const currentText = isTextPreview && layer.name.startsWith('textval:')
    ? layer.name.slice('textval:'.length)
    : isTxtLayer && layer.name.startsWith('text:')
    ? layer.name.slice(5) : ''

  const onMouseDown = (e: React.MouseEvent) => {
    if (inlineEditing) return
    if (layer.locked || e.button !== 0) return
    e.stopPropagation()
    selectLayer(layer.id, e.ctrlKey || e.metaKey)
    dragStart.current = {
      mx: e.clientX, my: e.clientY,
      px: layer.position.x, py: layer.position.y,
    }
    e.preventDefault()
  }

  const onDoubleClick = (e: React.MouseEvent) => {
    if (!isTextPreview && !isTxtLayer) return
    e.stopPropagation()
    setInlineEditing(true)
    setTimeout(() => {
      if (editRef.current) {
        editRef.current.innerText = currentText
        editRef.current.focus()
        const range = document.createRange()
        range.selectNodeContents(editRef.current)
        range.collapse(false)
        window.getSelection()?.removeAllRanges()
        window.getSelection()?.addRange(range)
      }
    }, 0)
  }

  const commitInlineEdit = () => {
    if (!editRef.current) return
    const newText = editRef.current.innerText.trim()
    setInlineEditing(false)
    if (newText) {
      window.dispatchEvent(new CustomEvent('canvas-text-edit', {
        detail: { layerId: layer.id, newText }
      }))
    }
  }

  const onMouseMove = useCallback((e: MouseEvent) => {
    if (!dragStart.current) return
    const dx = (e.clientX - dragStart.current.mx) / canvasScale
    const dy = (e.clientY - dragStart.current.my) / canvasScale
    updateLayer(layer.id, { position: { x: dragStart.current.px + dx, y: dragStart.current.py + dy } })
  }, [canvasScale, layer.id, updateLayer])

  const onMouseUp = useCallback(() => { dragStart.current = null }, [])

  useEffect(() => {
    window.addEventListener('mousemove', onMouseMove)
    window.addEventListener('mouseup', onMouseUp)
    return () => {
      window.removeEventListener('mousemove', onMouseMove)
      window.removeEventListener('mouseup', onMouseUp)
    }
  }, [onMouseMove, onMouseUp])

  return (
    <div
      onMouseDown={onMouseDown}
      onDoubleClick={onDoubleClick}
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
        cursor: inlineEditing ? 'text' : (layer.locked ? 'not-allowed' : (dragStart.current ? 'grabbing' : 'grab')),
        zIndex: layer.z_index,
        userSelect: 'none',
      }}
    >
      {thumbUrl && !inlineEditing && (
        <img
          src={thumbUrl}
          alt={layer.name}
          style={{ width: '100%', height: '100%', objectFit: 'fill' }}
          draggable={false}
        />
      )}
      {inlineEditing && (
        <div
          ref={editRef}
          contentEditable
          suppressContentEditableWarning
          onBlur={commitInlineEdit}
          onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); commitInlineEdit() } if (e.key === 'Escape') setInlineEditing(false) }}
          style={{
            width: '100%', height: '100%',
            lineHeight: `${layer.bbox.height}px`,
            fontSize: layer.bbox.height, fontWeight: 'bold',
            color: 'white', outline: '2px dashed #4f8ef7',
            background: 'rgba(0,0,0,0.15)', userSelect: 'text',
            cursor: 'text', whiteSpace: 'nowrap', overflow: 'hidden',
            textAlign: 'center',
          }}
        />
      )}
    </div>
  )
}

export const Canvas: React.FC = () => {
  const {
    layers, originalImageUrl, canvasWidth, canvasHeight,
    canvasScale, canvasOffset, setCanvasScale, setCanvasOffset,
    selectedLayerIds, selectLayer, clearSelection, updateLayer,
    detectedTextRegions, textOverlays,
  } = useEditorStore()

  const containerRef = useRef<HTMLDivElement>(null)
  const [isPanning, setIsPanning] = useState(false)
  const [spaceHeld, setSpaceHeld] = useState(false)
  const panStart = useRef({ x: 0, y: 0, ox: 0, oy: 0 })

  // Track space key
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.code === 'Space' && e.target === document.body) {
        e.preventDefault()
        setSpaceHeld(true)
      }
    }
    const onKeyUp = (e: KeyboardEvent) => {
      if (e.code === 'Space') {
        setSpaceHeld(false)
        setIsPanning(false)
      }
    }
    window.addEventListener('keydown', onKeyDown)
    window.addEventListener('keyup', onKeyUp)
    return () => { window.removeEventListener('keydown', onKeyDown); window.removeEventListener('keyup', onKeyUp) }
  }, [])

  // Inline text editor spawned directly on canvas from double-click on original image
  const [inlineEditor, setInlineEditor] = useState<{
    regionIdx: number; layerId: string; x: number; y: number; w: number; h: number; text: string
  } | null>(null)
  const inlineRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (inlineEditor && inlineRef.current) {
      inlineRef.current.innerText = inlineEditor.text
      inlineRef.current.focus()
      const range = document.createRange()
      range.selectNodeContents(inlineRef.current)
      range.collapse(false)
      window.getSelection()?.removeAllRanges()
      window.getSelection()?.addRange(range)
    }
  }, [inlineEditor?.regionIdx, inlineEditor?.layerId])

  const commitInlineCanvasEdit = () => {
    if (!inlineEditor || !inlineRef.current) return
    const newText = inlineRef.current.innerText.trim()
    setInlineEditor(null)
    if (newText) {
      window.dispatchEvent(new CustomEvent('canvas-text-edit', {
        detail: { layerId: `${inlineEditor.layerId}_textpreview_${inlineEditor.regionIdx}`, newText }
      }))
    }
  }

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

  // Pan with middle mouse, alt+drag, or space+drag
  const onMouseDown = (e: React.MouseEvent) => {
    if (e.button === 1 || e.altKey || (spaceHeld && e.button === 0)) {
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
      style={{ background: CHECKERBOARD, cursor: isPanning ? 'grabbing' : spaceHeld ? 'grab' : 'default' }}
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
        onDoubleClick={(e) => {
          // Hit-test against detected text regions of the selected layer
          const selectedLayer = layers.find(l => selectedLayerIds.includes(l.id) && !l.id.includes('_textpreview_'))
          if (!selectedLayer) return
          const regions = detectedTextRegions[selectedLayer.id]
          if (!regions?.length) return
          // Convert click coords to canvas-local coords
          const rect = (e.currentTarget as HTMLElement).getBoundingClientRect()
          const cx = (e.clientX - rect.left) / canvasScale
          const cy = (e.clientY - rect.top) / canvasScale
          for (let i = 0; i < regions.length; i++) {
            const r = regions[i]
            const [rx1, ry1, rx2, ry2] = r.bbox
            // bbox is in full-image coords; layer offset
            const lx1 = rx1 - selectedLayer.bbox.x, ly1 = ry1 - selectedLayer.bbox.y
            const lx2 = rx2 - selectedLayer.bbox.x, ly2 = ry2 - selectedLayer.bbox.y
            const ax = selectedLayer.bbox.x + selectedLayer.position.x + lx1
            const ay = selectedLayer.bbox.y + selectedLayer.position.y + ly1
            if (cx >= ax && cx <= ax + (lx2 - lx1) && cy >= ay && cy <= ay + (ly2 - ly1)) {
              // Check if preview layer already has a value
              const pid = `${selectedLayer.id}_textpreview_${i}`
              const previewLayer = layers.find(l => l.id === pid)
              const currentText = previewLayer?.name.startsWith('textval:')
                ? previewLayer.name.slice('textval:'.length) : r.text
              setInlineEditor({
                regionIdx: i, layerId: selectedLayer.id,
                x: ax, y: ay, w: lx2 - lx1, h: ly2 - ly1,
                text: currentText,
              })
              e.stopPropagation()
              return
            }
          }
        }}
      >
        {/* Layers */}
        {sortedLayers.map((layer) => {
          const isSelected = selectedLayerIds.includes(layer.id)
          const rawUrl = layer.png_path ? (layer.png_path.startsWith("blob:") || layer.png_path.startsWith("data:") ? layer.png_path : `${API_BASE}${layer.png_path}`) : null
          return <LayerImage key={layer.id} layer={layer} rawUrl={rawUrl} isSelected={isSelected} selectLayer={selectLayer} updateLayer={updateLayer} canvasScale={canvasScale} />
        })}

        {/* Live text overlays (no API, pure CSS preview) */}
        {Object.entries(textOverlays).map(([key, ov]) => {
          if (!ov) return null
          const [x1, y1, x2, y2] = ov.bbox
          const w = x2 - x1, h = y2 - y1
          return (
            <div key={key} style={{
              position: 'absolute', left: x1, top: y1, width: w, height: h,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: ov.font_size, fontWeight: 'bold',
              color: ov.color,
              transform: `rotate(${ov.rotation}deg)`,
              transformOrigin: 'center center',
              textShadow: ov.shadow ? `2px 2px 0 ${ov.shadow_color}` : 'none',
              pointerEvents: 'none',
              whiteSpace: 'nowrap',
              overflow: 'visible',
              zIndex: 9999,
            }}>
              {ov.text}
            </div>
          )
        })}

        {/* Inline text editor overlay */}
        {inlineEditor && (
          <div
            ref={inlineRef}
            contentEditable
            suppressContentEditableWarning
            onBlur={commitInlineCanvasEdit}
            onKeyDown={e => {
              if (e.key === 'Enter') { e.preventDefault(); commitInlineCanvasEdit() }
              if (e.key === 'Escape') setInlineEditor(null)
            }}
            style={{
              position: 'absolute',
              left: inlineEditor.x, top: inlineEditor.y,
              width: inlineEditor.w, height: inlineEditor.h,
              fontSize: inlineEditor.h, fontWeight: 'bold',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              outline: '2px dashed #4f8ef7',
              background: 'rgba(0,0,0,0.15)',
              color: 'white', cursor: 'text',
              userSelect: 'text', whiteSpace: 'nowrap', overflow: 'hidden',
              zIndex: 9999,
            }}
          />
        )}
      </div>
    </div>
  )
}
