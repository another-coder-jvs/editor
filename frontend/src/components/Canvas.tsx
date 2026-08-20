import React, { useRef, useEffect, useCallback, useState } from 'react'
import { useEditorStore } from '../store/editorStore'
import { baseImagesUrl } from '../config'
import { useBlobUrl } from '../hooks/useBlobUrl'
import { useCanvasDraw } from '../hooks/useCanvasDraw'
import { useCanvasSelect, SelectionMask } from '../hooks/useCanvasSelect'
import { useCanvasCrop } from '../hooks/useCanvasCrop'
import { LayerAdjustments } from '../types'
import { detectObjects, segmentObjects } from '../api/client'
import { toast } from 'react-toastify'
import { Scissors, Copy, Trash2, Move } from 'lucide-react'

const CHECKERBOARD = `repeating-conic-gradient(#2a2a2a 0% 25%, #1a1a1a 0% 50%) 0 0 / 20px 20px`
const API_BASE = baseImagesUrl || 'http://localhost:8000'

// Build CSS filter string from layer adjustments
function buildFilter(adj: LayerAdjustments): string {
  const brightness = adj.brightness / 100
  const contrast = adj.contrast / 100
  const saturation = adj.saturation / 100
  const hueRotate = adj.hue
  const sharpness = adj.sharpness > 100 ? (adj.sharpness - 100) / 100 : 0
  // exposure: treat as additional brightness multiplier
  const exposure = Math.pow(2, adj.exposure / 100)
  return [
    `brightness(${(brightness * exposure).toFixed(3)})`,
    `contrast(${contrast.toFixed(3)})`,
    `saturate(${saturation.toFixed(3)})`,
    `hue-rotate(${hueRotate}deg)`,
    sharpness > 0 ? `contrast(${(1 + sharpness * 0.3).toFixed(3)})` : '',
  ].filter(Boolean).join(' ')
}

// Build CSS for vignette/grain/fade overlays
function buildOverlayStyle(adj: LayerAdjustments, w: number, h: number) {
  const styles: React.CSSProperties = {}
  return styles
}

interface LayerImageProps {
  layer: any
  rawUrl: string | null
  isSelected: boolean
  selectLayer: (id: string, multi: boolean) => void
  updateLayer: (id: string, patch: any) => void
  canvasScale: number
  pushHistory: () => void
  activeTool: string
}

const LayerImage: React.FC<LayerImageProps> = ({
  layer, rawUrl, isSelected, selectLayer, updateLayer, canvasScale, pushHistory, activeTool,
}) => {
  const thumbUrl = useBlobUrl(rawUrl)
  const dragStart = useRef<{ mx: number; my: number; px: number; py: number } | null>(null)
  const didPushHistory = useRef(false)
  const [inlineEditing, setInlineEditing] = useState(false)
  const editRef = useRef<HTMLDivElement>(null)

  const isTextPreview = layer.id.includes('_textpreview_')
  const isTxtLayer = layer.id.includes('_txt_')
  const currentText = isTextPreview && layer.name.startsWith('textval:')
    ? layer.name.slice('textval:'.length)
    : isTxtLayer && layer.name.startsWith('text:')
    ? layer.name.slice(5) : ''

  const canDrag = activeTool === 'move' && !layer.locked

  const onMouseDown = (e: React.MouseEvent) => {
    if (inlineEditing) return
    if (!canDrag || e.button !== 0) return
    e.stopPropagation()
    selectLayer(layer.id, e.ctrlKey || e.metaKey)
    dragStart.current = { mx: e.clientX, my: e.clientY, px: layer.position.x, py: layer.position.y }
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
      window.dispatchEvent(new CustomEvent('canvas-text-edit', { detail: { layerId: layer.id, newText } }))
    }
  }

  const onMouseMove = useCallback((e: MouseEvent) => {
    if (!dragStart.current) return
    const dx = (e.clientX - dragStart.current.mx) / canvasScale
    const dy = (e.clientY - dragStart.current.my) / canvasScale
    if (!didPushHistory.current && (Math.abs(dx) > 2 || Math.abs(dy) > 2)) {
      pushHistory()
      didPushHistory.current = true
    }
    updateLayer(layer.id, { position: { x: dragStart.current.px + dx, y: dragStart.current.py + dy } })
  }, [canvasScale, layer.id, updateLayer, pushHistory])

  const onMouseUp = useCallback(() => { dragStart.current = null; didPushHistory.current = false }, [])

  useEffect(() => {
    window.addEventListener('mousemove', onMouseMove)
    window.addEventListener('mouseup', onMouseUp)
    return () => { window.removeEventListener('mousemove', onMouseMove); window.removeEventListener('mouseup', onMouseUp) }
  }, [onMouseMove, onMouseUp])

  const adj: LayerAdjustments = layer.adjustments || {
    brightness: 100, contrast: 100, saturation: 100, exposure: 0,
    highlights: 0, shadows: 0, temperature: 0, tint: 0, hue: 0,
    sharpness: 100, clarity: 0, fade: 0, vignette: 0, grain: 0,
  }
  const cssFilter = buildFilter(adj)
  const blendMode = layer.blend_mode || 'normal'

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
        cursor: inlineEditing ? 'text' : (layer.locked ? 'not-allowed' : canDrag ? (dragStart.current ? 'grabbing' : 'grab') : 'default'),
        zIndex: layer.z_index,
        userSelect: 'none',
        mixBlendMode: blendMode as any,
        filter: cssFilter,
      }}
    >
      {thumbUrl && !inlineEditing && (
        <img src={thumbUrl} alt={layer.name} style={{ width: '100%', height: '100%', objectFit: 'fill' }} draggable={false} />
      )}
      {/* Vignette overlay */}
      {adj.vignette > 0 && (
        <div style={{
          position: 'absolute', inset: 0, pointerEvents: 'none',
          background: `radial-gradient(ellipse at center, transparent ${100 - adj.vignette}%, rgba(0,0,0,${adj.vignette / 100}) 100%)`,
        }} />
      )}
      {/* Grain overlay */}
      {adj.grain > 0 && (
        <div style={{
          position: 'absolute', inset: 0, pointerEvents: 'none',
          opacity: adj.grain / 200,
          backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E")`,
          backgroundSize: '150px 150px',
          mixBlendMode: 'overlay',
        }} />
      )}
      {/* Fade overlay */}
      {adj.fade > 0 && (
        <div style={{
          position: 'absolute', inset: 0, pointerEvents: 'none',
          background: `rgba(255,255,255,${adj.fade / 100})`,
        }} />
      )}
      {inlineEditing && (
        <div
          ref={editRef}
          contentEditable
          suppressContentEditableWarning
          onBlur={commitInlineEdit}
          onKeyDown={e => {
            if (e.key === 'Enter') { e.preventDefault(); commitInlineEdit() }
            if (e.key === 'Escape') setInlineEditing(false)
          }}
          style={{
            width: '100%', height: '100%', lineHeight: `${layer.bbox.height}px`,
            fontSize: layer.bbox.height, fontWeight: 'bold', color: 'white',
            outline: '2px dashed #4f8ef7', background: 'rgba(0,0,0,0.15)',
            userSelect: 'text', cursor: 'text', whiteSpace: 'nowrap', overflow: 'hidden', textAlign: 'center',
          }}
        />
      )}
    </div>
  )
}

export const Canvas: React.FC = () => {
  const {
    layers, originalImageUrl, canvasWidth, canvasHeight, canvasBg,
    canvasScale, canvasOffset, setCanvasScale, setCanvasOffset,
    selectedLayerIds, selectLayer, clearSelection, updateLayer,
    detectedTextRegions, textOverlays, pushHistory, activeTool,
    sessionId, originalImagePath, addLayer, deleteLayer,
    aiSelections, aiSelectionLayerId, setAiSelections, clearAiSelections,
  } = useEditorStore()

  const containerRef = useRef<HTMLDivElement>(null)
  const overlayRef = useRef<HTMLCanvasElement>(null)
  const selectionRef = useRef<HTMLCanvasElement>(null)
  const cropRef = useRef<HTMLCanvasElement>(null)

  const [isPanning, setIsPanning] = useState(false)
  const [spaceHeld, setSpaceHeld] = useState(false)
  const panStart = useRef({ x: 0, y: 0, ox: 0, oy: 0 })

  // Activate drawing hooks
  useCanvasDraw(overlayRef, canvasWidth, canvasHeight, canvasScale)
  const { cropRect, applyCrop } = useCanvasCrop(cropRef, canvasWidth, canvasHeight, canvasScale)
  const { selection } = useCanvasSelect(selectionRef, canvasWidth, canvasHeight, canvasScale, () => {})

  // AI Detection state
  const [aiDetecting, setAiDetecting] = useState(false)

  // ── Magic Select: click → detect objects → show bbox overlays ──
  const handleMagicSelectClick = useCallback(async (e: React.MouseEvent) => {
    if (aiDetecting) return
    const selectedLayer = layers.find(l => selectedLayerIds.includes(l.id) && !l.id.includes('_textpreview_') && !l.id.includes('_txt_'))
    if (!selectedLayer) { toast.info('Select a layer first'); return }
    if (!selectedLayer.png_path) { toast.warn('Layer has no image'); return }

    setAiDetecting(true)
    try {
      // Fetch the layer image as a File
      const imgUrl = selectedLayer.png_path.startsWith('blob:') || selectedLayer.png_path.startsWith('data:')
        ? selectedLayer.png_path : `${API_BASE}${selectedLayer.png_path}`
      const res = await fetch(imgUrl, { headers: { 'ngrok-skip-browser-warning': '1' } })
      const blob = await res.blob()
      const file = new File([blob], 'layer.png', { type: 'image/png' })

      const result = await detectObjects(file)
      if (!result.objects.length) {
        toast.info('No objects detected in this layer')
        return
      }

      // Offset bboxes by layer position so they align on the canvas
      const offsetObjects = result.objects.map(obj => ({
        ...obj,
        bbox: {
          x: selectedLayer.bbox.x + selectedLayer.position.x + obj.bbox.x,
          y: selectedLayer.bbox.y + selectedLayer.position.y + obj.bbox.y,
          width: obj.bbox.width,
          height: obj.bbox.height,
        },
      }))

      setAiSelections(offsetObjects, selectedLayer.id)
      toast.success(`Detected ${result.objects.length} object${result.objects.length > 1 ? 's' : ''}`)
    } catch (err: any) {
      toast.error('Detection failed: ' + (err?.message || err))
    } finally {
      setAiDetecting(false)
    }
  }, [layers, selectedLayerIds, aiDetecting, setAiSelections])

  // ── Object Select: click → detect + SAM2 segment → extract as layers ──
  const handleObjectSelectClick = useCallback(async (e: React.MouseEvent) => {
    if (aiDetecting) return
    if (!sessionId || !originalImagePath) { toast.warn('No session active'); return }
    const selectedLayer = layers.find(l => selectedLayerIds.includes(l.id) && !l.id.includes('_textpreview_') && !l.id.includes('_txt_'))
    if (!selectedLayer) { toast.info('Select a layer first'); return }
    if (!selectedLayer.png_path) { toast.warn('Layer has no image'); return }

    setAiDetecting(true)
    try {
      // Step 1: Detect objects
      const imgUrl = selectedLayer.png_path.startsWith('blob:') || selectedLayer.png_path.startsWith('data:')
        ? selectedLayer.png_path : `${API_BASE}${selectedLayer.png_path}`
      const res = await fetch(imgUrl, { headers: { 'ngrok-skip-browser-warning': '1' } })
      const blob = await res.blob()
      const file = new File([blob], 'layer.png', { type: 'image/png' })

      const detectResult = await detectObjects(file)
      if (!detectResult.objects.length) {
        toast.info('No objects detected')
        return
      }

      // Step 2: Run SAM2 segmentation on detected objects
      toast.info(`Segmenting ${detectResult.objects.length} object(s)…`)
      const segResult = await segmentObjects(
        detectResult.session_id,
        detectResult.image_path,
        detectResult.objects,
      )

      // Step 3: Add segmented objects as new layers
      if (segResult.layers?.length) {
        segResult.layers.forEach(layer => {
          addLayer({
            ...layer,
            position: { x: 0, y: 0 },
            scale: { x: 1, y: 1 },
            rotation: 0,
            history: ['object-select'],
            locked: false,
            blend_mode: 'normal',
            adjustments: {
              brightness: 100, contrast: 100, saturation: 100,
              exposure: 0, highlights: 0, shadows: 0,
              temperature: 0, tint: 0, hue: 0,
              sharpness: 100, clarity: 0, fade: 0, vignette: 0, grain: 0,
            },
          })
        })
        toast.success(`Extracted ${segResult.layers.length} object(s) as layers`)
      }

      clearAiSelections()
    } catch (err: any) {
      toast.error('Object select failed: ' + (err?.message || err))
    } finally {
      setAiDetecting(false)
    }
  }, [layers, selectedLayerIds, sessionId, originalImagePath, addLayer, clearAiSelections, aiDetecting])

  // Space key for pan
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.code === 'Space' && e.target === document.body) { e.preventDefault(); setSpaceHeld(true) }
      if (e.key === 'Escape') { clearAiSelections(); clearSelection() }
    }
    const onKeyUp = (e: KeyboardEvent) => {
      if (e.code === 'Space') { setSpaceHeld(false); setIsPanning(false) }
    }
    window.addEventListener('keydown', onKeyDown)
    window.addEventListener('keyup', onKeyUp)
    return () => { window.removeEventListener('keydown', onKeyDown); window.removeEventListener('keyup', onKeyUp) }
  }, [clearAiSelections, clearSelection])

  // Inline text editor
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

  // Zoom
  const onWheel = useCallback((e: WheelEvent) => {
    e.preventDefault()
    const delta = e.deltaY > 0 ? 0.9 : 1.1
    setCanvasScale(Math.min(8, Math.max(0.1, canvasScale * delta)))
  }, [canvasScale, setCanvasScale])

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [onWheel])

  // Pan
  const onMouseDown = (e: React.MouseEvent) => {
    if (e.button === 1 || e.altKey || (spaceHeld && e.button === 0)) {
      setIsPanning(true)
      panStart.current = { x: e.clientX, y: e.clientY, ox: canvasOffset.x, oy: canvasOffset.y }
      e.preventDefault()
    }
  }
  const onMouseMove = (e: React.MouseEvent) => {
    if (!isPanning) return
    setCanvasOffset({ x: panStart.current.ox + (e.clientX - panStart.current.x), y: panStart.current.oy + (e.clientY - panStart.current.y) })
  }
  const onMouseUp = () => setIsPanning(false)

  // Determine cursor
  const DRAW_TOOLS = ['brush', 'pencil', 'marker', 'eraser', 'clone', 'heal',
    'shape_rect', 'shape_ellipse', 'shape_line', 'shape_arrow', 'shape_triangle', 'shape_star']
  const SELECT_TOOLS = ['rect_select', 'ellipse_select', 'lasso_select', 'free_select', 'magic_select', 'object_select']
  let cursor = 'default'
  if (isPanning || spaceHeld) cursor = isPanning ? 'grabbing' : 'grab'
  else if (activeTool === 'move') cursor = 'default'
  else if (activeTool === 'crop') cursor = 'crosshair'
  else if (activeTool === 'color_picker') cursor = 'crosshair'
  else if (DRAW_TOOLS.includes(activeTool)) cursor = 'crosshair'
  else if (SELECT_TOOLS.includes(activeTool)) cursor = 'crosshair'
  else if (activeTool === 'text_add') cursor = 'text'

  const sortedLayers = [...layers].filter(l => l.visible).sort((a, b) => a.z_index - b.z_index)

  // Canvas background
  const bgStyle: React.CSSProperties = canvasBg === 'transparent'
    ? { background: CHECKERBOARD }
    : { background: canvasBg }

  // Overlay canvas pointer events: only active for draw/select/crop tools
  const overlayPointerEvents = DRAW_TOOLS.includes(activeTool) || activeTool === 'color_picker' ? 'auto' : 'none'
  const selectionPointerEvents = SELECT_TOOLS.includes(activeTool) ? 'auto' : 'none'
  const cropPointerEvents = activeTool === 'crop' ? 'auto' : 'none'

  return (
    <div
      ref={containerRef}
      className="flex-1 overflow-hidden relative"
      style={{ background: CHECKERBOARD, cursor }}
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
      onMouseUp={onMouseUp}
      onMouseLeave={onMouseUp}
      onClick={(e) => {
        if ((e.target as HTMLElement).dataset.canvas === 'bg') clearSelection()
      }}
    >
      {/* Zoom indicator */}
      <div className="absolute top-3 right-3 z-20 bg-dark-800 text-xs text-gray-400 px-2 py-1 rounded pointer-events-none">
        {Math.round(canvasScale * 100)}%
      </div>

      {/* Crop apply hint */}
      {activeTool === 'crop' && cropRect && (
        <div className="absolute top-3 left-1/2 -translate-x-1/2 z-20 bg-dark-700 text-xs text-white px-3 py-1 rounded shadow">
          Press Enter to apply crop · Esc to cancel
        </div>
      )}

      {/* Zoom controls */}
      <div className="absolute bottom-3 right-3 z-20 flex gap-1">
        <button className="bg-dark-700 text-gray-300 px-2 py-1 rounded text-sm hover:bg-dark-600"
          onClick={() => setCanvasScale(Math.min(8, canvasScale * 1.2))}>+</button>
        <button className="bg-dark-700 text-gray-300 px-2 py-1 rounded text-sm hover:bg-dark-600"
          onClick={() => setCanvasScale(1)}>1:1</button>
        <button className="bg-dark-700 text-gray-300 px-2 py-1 rounded text-sm hover:bg-dark-600"
          onClick={() => setCanvasScale(Math.max(0.1, canvasScale * 0.8))}>−</button>
      </div>

      {/* Canvas area */}
      <div
        data-canvas="bg"
        style={{
          position: 'absolute', left: '50%', top: '50%',
          transform: `translate(calc(-50% + ${canvasOffset.x}px), calc(-50% + ${canvasOffset.y}px)) scale(${canvasScale})`,
          transformOrigin: 'center center',
          width: canvasWidth, height: canvasHeight,
          boxShadow: '0 0 40px rgba(0,0,0,0.8)',
          ...bgStyle,
        }}
        onClick={(e) => {
          // If AI selections are showing, don't trigger detection — just dismiss on bg click
          if (aiSelections.length > 0) {
            clearAiSelections()
            return
          }
          // Magic Select / Object Select
          if (activeTool === 'magic_select') {
            e.stopPropagation()
            handleMagicSelectClick(e)
            return
          }
          if (activeTool === 'object_select') {
            e.stopPropagation()
            handleObjectSelectClick(e)
            return
          }
        }}
        onDoubleClick={(e) => {
          const selectedLayer = layers.find(l => selectedLayerIds.includes(l.id) && !l.id.includes('_textpreview_'))
          if (!selectedLayer) return
          const regions = detectedTextRegions[selectedLayer.id]
          if (!regions?.length) return
          const rect = (e.currentTarget as HTMLElement).getBoundingClientRect()
          const cx = (e.clientX - rect.left) / canvasScale
          const cy = (e.clientY - rect.top) / canvasScale
          for (let i = 0; i < regions.length; i++) {
            const r = regions[i]
            const [rx1, ry1, rx2, ry2] = r.bbox
            const ax = selectedLayer.bbox.x + selectedLayer.position.x + rx1
            const ay = selectedLayer.bbox.y + selectedLayer.position.y + ry1
            if (cx >= ax && cx <= ax + (rx2 - rx1) && cy >= ay && cy <= ay + (ry2 - ry1)) {
              const pid = `${selectedLayer.id}_textpreview_${i}`
              const previewLayer = layers.find(l => l.id === pid)
              const currentText = previewLayer?.name.startsWith('textval:') ? previewLayer.name.slice('textval:'.length) : r.text
              setInlineEditor({ regionIdx: i, layerId: selectedLayer.id, x: ax, y: ay, w: rx2 - rx1, h: ry2 - ry1, text: currentText })
              e.stopPropagation()
              return
            }
          }
        }}
      >
        {/* Layer images */}
        {sortedLayers.map((layer) => {
          const isSelected = selectedLayerIds.includes(layer.id)
          const rawUrl = layer.png_path
            ? (layer.png_path.startsWith('blob:') || layer.png_path.startsWith('data:') ? layer.png_path : `${API_BASE}${layer.png_path}`)
            : null
          return (
            <LayerImage
              key={layer.id} layer={layer} rawUrl={rawUrl} isSelected={isSelected}
              selectLayer={selectLayer} updateLayer={updateLayer}
              canvasScale={canvasScale} pushHistory={pushHistory} activeTool={activeTool}
            />
          )
        })}

        {/* Live text overlays */}
        {Object.entries(textOverlays).map(([key, ov]) => {
          if (!ov) return null
          const [x1, y1, x2, y2] = ov.bbox
          return (
            <div key={key} style={{
              position: 'absolute', left: x1, top: y1, width: x2 - x1, height: y2 - y1,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: ov.font_size, fontWeight: 'bold', color: ov.color,
              transform: `rotate(${ov.rotation}deg)`, transformOrigin: 'center center',
              textShadow: ov.shadow ? `2px 2px 0 ${ov.shadow_color}` : 'none',
              pointerEvents: 'none', whiteSpace: 'nowrap', overflow: 'visible', zIndex: 9999,
            }}>
              {ov.text}
            </div>
          )
        })}

        {/* Inline text editor */}
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
              position: 'absolute', left: inlineEditor.x, top: inlineEditor.y,
              width: inlineEditor.w, height: inlineEditor.h,
              fontSize: inlineEditor.h, fontWeight: 'bold',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              outline: '2px dashed #4f8ef7', background: 'rgba(0,0,0,0.15)',
              color: 'white', cursor: 'text', userSelect: 'text',
              whiteSpace: 'nowrap', overflow: 'hidden', zIndex: 9999,
            }}
          />
        )}

        {/* Draw overlay canvas */}
        <canvas
          ref={overlayRef}
          width={canvasWidth}
          height={canvasHeight}
          style={{ position: 'absolute', inset: 0, pointerEvents: overlayPointerEvents, zIndex: 10000 }}
        />

        {/* Selection overlay canvas */}
        <canvas
          ref={selectionRef}
          width={canvasWidth}
          height={canvasHeight}
          style={{ position: 'absolute', inset: 0, pointerEvents: selectionPointerEvents, zIndex: 10001 }}
        />

        {/* Crop overlay canvas */}
        <canvas
          ref={cropRef}
          width={canvasWidth}
          height={canvasHeight}
          style={{ position: 'absolute', inset: 0, pointerEvents: cropPointerEvents, zIndex: 10002 }}
        />
      </div>

      {/* Selection Action Toolbar — OUTSIDE canvas div so clicks don't trigger detection */}
      <SelectionToolbar
        selection={selection}
        aiSelections={aiSelections}
        canvasOffset={canvasOffset}
        canvasScale={canvasScale}
        layers={layers}
        selectedLayerIds={selectedLayerIds}
        pushHistory={pushHistory}
        updateLayer={updateLayer}
        addLayer={addLayer}
        clearAiSelections={clearAiSelections}
        clearSelection={clearSelection}
      />

      {/* AI detecting spinner */}
      {aiDetecting && (
        <div className="absolute top-14 left-1/2 -translate-x-1/2 z-30 bg-dark-700 border border-dark-500 text-white text-xs px-4 py-2 rounded-full shadow-lg flex items-center gap-2">
          <div className="w-3 h-3 border-2 border-accent border-t-transparent rounded-full animate-spin" />
          Detecting objects…
        </div>
      )}

    </div>
  )
}

// ── Selection Action Toolbar ───────────────────────────────────────────────
// Rendered OUTSIDE the canvas div — positioned using screen-space coordinates
// so clicks on buttons never bubble into the canvas onClick handler.
interface SelectionToolbarProps {
  selection: SelectionMask | null
  aiSelections: Array<{ label: string; score: number; bbox: { x: number; y: number; width: number; height: number } }>
  canvasOffset: { x: number; y: number }
  canvasScale: number
  layers: any[]
  selectedLayerIds: string[]
  pushHistory: () => void
  updateLayer: (id: string, patch: any) => void
  addLayer: (layer: any) => void
  clearAiSelections: () => void
  clearSelection: () => void
}

function loadImage(url: string): Promise<HTMLImageElement | null> {
  return new Promise(resolve => {
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => resolve(img)
    img.onerror = () => resolve(null)
    img.src = url
  })
}

function getLayerUrl(pngPath: string): string {
  if (pngPath.startsWith('blob:') || pngPath.startsWith('data:')) return pngPath
  return `${API_BASE}${pngPath}`
}

const SelectionToolbar: React.FC<SelectionToolbarProps> = ({
  selection, aiSelections, canvasOffset, canvasScale,
  layers, selectedLayerIds, pushHistory, updateLayer, addLayer,
  clearAiSelections, clearSelection,
}) => {
  const hasAiSelections = aiSelections.length > 0
  const hasGeometricSelection = !!selection
  if (!hasAiSelections && !hasGeometricSelection) return null

  const selectedLayer = layers.find(l =>
    selectedLayerIds.includes(l.id) &&
    !l.id.includes('_textpreview_') && !l.id.includes('_txt_') && !l.id.startsWith('draw_')
  )

  // Convert canvas-space bbox center to screen-space (relative to container)
  // Container is the parent: flex-1 overflow-hidden relative
  // Canvas div is centered with: translate(calc(-50% + offsetX), calc(-50% + offsetY)) scale(scale)
  const containerW = typeof window !== 'undefined' ? window.innerWidth : 1200
  const containerH = typeof window !== 'undefined' ? window.innerHeight : 800
  const canvasCenterX = containerW / 2 + canvasOffset.x
  const canvasCenterY = containerH / 2 + canvasOffset.y

  let screenX = canvasCenterX
  let screenY = canvasCenterY - 20
  if (hasAiSelections && aiSelections.length > 0) {
    const first = aiSelections[0].bbox
    screenX = canvasCenterX + (first.x + first.width / 2) * canvasScale
    screenY = canvasCenterY + first.y * canvasScale - 20
  } else if (selection?.rect) {
    screenX = canvasCenterX + (selection.rect.x + selection.rect.w / 2) * canvasScale
    screenY = canvasCenterY + selection.rect.y * canvasScale - 20
  }
  // Clamp to viewport
  screenX = Math.max(120, Math.min(screenX, containerW - 120))
  screenY = Math.max(60, screenY)

  const handleCut = async () => {
    if (!selectedLayer?.png_path) { toast.warn('Select a layer first'); return }
    pushHistory()
    const img = await loadImage(getLayerUrl(selectedLayer.png_path))
    if (!img) { toast.error('Failed to load layer image'); return }
    const offscreen = document.createElement('canvas')
    offscreen.width = selectedLayer.bbox.width
    offscreen.height = selectedLayer.bbox.height
    const ctx = offscreen.getContext('2d')!
    ctx.drawImage(img, 0, 0)
    const imageData = ctx.getImageData(0, 0, offscreen.width, offscreen.height)
    const data = imageData.data
    if (hasAiSelections) {
      aiSelections.forEach(obj => {
        const lx = Math.max(0, Math.floor(obj.bbox.x - selectedLayer.bbox.x))
        const ly = Math.max(0, Math.floor(obj.bbox.y - selectedLayer.bbox.y))
        const rx = Math.min(offscreen.width, Math.ceil(obj.bbox.x - selectedLayer.bbox.x + obj.bbox.width))
        const ry = Math.min(offscreen.height, Math.ceil(obj.bbox.y - selectedLayer.bbox.y + obj.bbox.height))
        for (let y = ly; y < ry; y++) for (let x = lx; x < rx; x++) data[(y * offscreen.width + x) * 4 + 3] = 0
      })
      ctx.putImageData(imageData, 0, 0)
      updateLayer(selectedLayer.id, { png_path: offscreen.toDataURL('image/png') })
      clearAiSelections()
      toast.success('Cut selection')
    } else if (selection?.rect) {
      const { x, y, w, h } = selection.rect
      const lx = Math.max(0, Math.floor(x - selectedLayer.bbox.x))
      const ly = Math.max(0, Math.floor(y - selectedLayer.bbox.y))
      const lw = Math.min(offscreen.width - lx, Math.ceil(w))
      const lh = Math.min(offscreen.height - ly, Math.ceil(h))
      const area = ctx.getImageData(lx, ly, lw, lh)
      for (let i = 3; i < area.data.length; i += 4) area.data[i] = 0
      ctx.putImageData(area, lx, ly)
      updateLayer(selectedLayer.id, { png_path: offscreen.toDataURL('image/png') })
      clearSelection()
      toast.success('Cut selection')
    }
  }

  const handleCopy = async () => {
    if (!selectedLayer?.png_path) { toast.warn('Select a layer first'); return }
    pushHistory()
    const img = await loadImage(getLayerUrl(selectedLayer.png_path))
    if (!img) { toast.error('Failed to load layer image'); return }
    const offscreen = document.createElement('canvas')
    offscreen.width = selectedLayer.bbox.width
    offscreen.height = selectedLayer.bbox.height
    const ctx = offscreen.getContext('2d')!
    ctx.drawImage(img, 0, 0)
    const clip = document.createElement('canvas')
    clip.width = selectedLayer.bbox.width
    clip.height = selectedLayer.bbox.height
    const clipCtx = clip.getContext('2d')!
    if (hasAiSelections) {
      aiSelections.forEach(obj => {
        const lx = Math.floor(obj.bbox.x - selectedLayer.bbox.x)
        const ly = Math.floor(obj.bbox.y - selectedLayer.bbox.y)
        clipCtx.drawImage(offscreen, lx, ly, Math.ceil(obj.bbox.width), Math.ceil(obj.bbox.height), lx, ly, Math.ceil(obj.bbox.width), Math.ceil(obj.bbox.height))
      })
    } else if (selection?.rect) {
      const { x, y, w, h } = selection.rect
      clipCtx.drawImage(offscreen, Math.floor(x - selectedLayer.bbox.x), Math.floor(y - selectedLayer.bbox.y), Math.ceil(w), Math.ceil(h), 0, 0, Math.ceil(w), Math.ceil(h))
    }
    addLayer({ ...selectedLayer, id: `copy_${selectedLayer.id}_${Date.now()}`, name: `copy:${selectedLayer.name}`, png_path: clip.toDataURL('image/png'), z_index: selectedLayer.z_index + 1 })
    toast.success('Copied to new layer')
  }

  const handleMove = async () => {
    if (!selectedLayer?.png_path) { toast.warn('Select a layer first'); return }
    pushHistory()
    const img = await loadImage(getLayerUrl(selectedLayer.png_path))
    if (!img) { toast.error('Failed to load layer image'); return }
    const offscreen = document.createElement('canvas')
    offscreen.width = selectedLayer.bbox.width
    offscreen.height = selectedLayer.bbox.height
    const ctx = offscreen.getContext('2d')!
    ctx.drawImage(img, 0, 0)
    const clip = document.createElement('canvas')
    clip.width = selectedLayer.bbox.width
    clip.height = selectedLayer.bbox.height
    const clipCtx = clip.getContext('2d')!
    if (hasAiSelections) {
      aiSelections.forEach(obj => {
        const lx = Math.floor(obj.bbox.x - selectedLayer.bbox.x)
        const ly = Math.floor(obj.bbox.y - selectedLayer.bbox.y)
        clipCtx.drawImage(offscreen, lx, ly, Math.ceil(obj.bbox.width), Math.ceil(obj.bbox.height), lx, ly, Math.ceil(obj.bbox.width), Math.ceil(obj.bbox.height))
      })
      const imageData = ctx.getImageData(0, 0, offscreen.width, offscreen.height)
      const data = imageData.data
      aiSelections.forEach(obj => {
        const lx = Math.max(0, Math.floor(obj.bbox.x - selectedLayer.bbox.x))
        const ly = Math.max(0, Math.floor(obj.bbox.y - selectedLayer.bbox.y))
        const rx = Math.min(offscreen.width, Math.ceil(obj.bbox.x - selectedLayer.bbox.x + obj.bbox.width))
        const ry = Math.min(offscreen.height, Math.ceil(obj.bbox.y - selectedLayer.bbox.y + obj.bbox.height))
        for (let y = ly; y < ry; y++) for (let x = lx; x < rx; x++) data[(y * offscreen.width + x) * 4 + 3] = 0
      })
      ctx.putImageData(imageData, 0, 0)
      updateLayer(selectedLayer.id, { png_path: offscreen.toDataURL('image/png') })
      addLayer({ ...selectedLayer, id: `moved_${selectedLayer.id}_${Date.now()}`, name: `moved:${selectedLayer.name}`, png_path: clip.toDataURL('image/png'), z_index: selectedLayer.z_index + 1 })
      clearAiSelections()
      toast.success('Moved to new layer')
    } else if (selection?.rect) {
      const { x, y, w, h } = selection.rect
      const lx = Math.floor(x - selectedLayer.bbox.x)
      const ly = Math.floor(y - selectedLayer.bbox.y)
      clipCtx.drawImage(offscreen, lx, ly, Math.ceil(w), Math.ceil(h), 0, 0, Math.ceil(w), Math.ceil(h))
      const area = ctx.getImageData(lx, ly, Math.ceil(w), Math.ceil(h))
      for (let i = 3; i < area.data.length; i += 4) area.data[i] = 0
      ctx.putImageData(area, lx, ly)
      updateLayer(selectedLayer.id, { png_path: offscreen.toDataURL('image/png') })
      addLayer({ ...selectedLayer, id: `moved_${selectedLayer.id}_${Date.now()}`, name: `moved:${selectedLayer.name}`, png_path: clip.toDataURL('image/png'), bbox: { x, y, width: Math.ceil(w), height: Math.ceil(h) }, z_index: selectedLayer.z_index + 1 })
      clearSelection()
      toast.success('Moved to new layer')
    }
  }

  const btnClass = 'flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-semibold transition-colors cursor-pointer whitespace-nowrap'

  return (
    <div
      className="fixed z-[99999] bg-dark-800 border border-dark-500 rounded-xl shadow-2xl flex items-center gap-1.5 px-3 py-2"
      style={{ left: screenX, top: screenY, transform: 'translate(-50%, -100%)' }}
    >
      <button onClick={handleCut} className={`${btnClass} hover:bg-red-500/20 text-red-400 hover:text-red-300`}>
        <Scissors size={16} /> Cut
      </button>
      <button onClick={handleCopy} className={`${btnClass} hover:bg-blue-500/20 text-blue-400 hover:text-blue-300`}>
        <Copy size={16} /> Copy
      </button>
      <button onClick={handleMove} className={`${btnClass} hover:bg-green-500/20 text-green-400 hover:text-green-300`}>
        <Move size={16} /> Move
      </button>
      <div className="w-px h-6 bg-dark-500 mx-1" />
      <button onClick={() => { clearAiSelections(); clearSelection() }} className={`${btnClass} hover:bg-dark-500 text-gray-400 hover:text-white`}>
        ✕
      </button>
    </div>
  )
}
