import React, { useState, useRef } from 'react'
import { useEditorStore } from '../store/editorStore'
import { LayerData, BlendMode } from '../types'
import { Eye, EyeOff, Lock, Unlock, Trash2, Copy, ChevronUp, ChevronDown, Plus } from 'lucide-react'
import { baseImagesUrl } from '../config'
import { useBlobUrl } from '../hooks/useBlobUrl'

const API_BASE = baseImagesUrl || 'http://localhost:8000'

const BLEND_MODES: BlendMode[] = [
  'normal', 'multiply', 'screen', 'overlay',
  'darken', 'lighten', 'color-dodge', 'color-burn',
  'hard-light', 'soft-light', 'difference', 'exclusion',
  'hue', 'saturation', 'color', 'luminosity',
]

export const LayerPanel: React.FC = () => {
  const {
    layers, selectedLayerIds, selectLayer,
    updateLayer, deleteLayer, duplicateLayer, reorderLayer, addLayer,
    canvasWidth, canvasHeight, sessionId,
  } = useEditorStore()

  const sorted = [...layers].sort((a, b) => b.z_index - a.z_index)

  const handleAddLayer = () => {
    const maxZ = layers.length > 0 ? Math.max(...layers.map(l => l.z_index)) : 0
    addLayer({
      id: `layer_${Date.now()}`,
      name: 'New Layer',
      mask_path: '',
      png_path: '',
      bbox: { x: 0, y: 0, width: canvasWidth, height: canvasHeight },
      z_index: maxZ + 1,
      visible: true,
      opacity: 1,
      position: { x: 0, y: 0 },
      scale: { x: 1, y: 1 },
      rotation: 0,
      history: [],
      locked: false,
      blend_mode: 'normal',
      adjustments: {
        brightness: 100, contrast: 100, saturation: 100, exposure: 0,
        highlights: 0, shadows: 0, temperature: 0, tint: 0, hue: 0,
        sharpness: 100, clarity: 0, fade: 0, vignette: 0, grain: 0,
      },
    })
  }

  return (
    <div className="flex flex-col h-full w-full">
      <div className="px-3 py-2 text-xs font-semibold text-gray-400 uppercase tracking-wider border-b border-dark-600 flex items-center justify-between">
        <span>Layers</span>
        <button title="Add Layer" onClick={handleAddLayer} className="text-gray-400 hover:text-white p-0.5 rounded hover:bg-dark-600">
          <Plus size={14} />
        </button>
      </div>
      <div className="flex-1 overflow-y-auto py-1">
        {sorted.length === 0 && (
          <p className="text-xs text-gray-600 text-center mt-8 px-3">Upload an image to generate layers</p>
        )}
        {sorted.map((layer) => (
          <LayerItem
            key={layer.id}
            layer={layer}
            selected={selectedLayerIds.includes(layer.id)}
            onSelect={(e) => selectLayer(layer.id, e.ctrlKey || e.metaKey)}
            onToggleVisible={() => updateLayer(layer.id, { visible: !layer.visible })}
            onToggleLock={() => updateLayer(layer.id, { locked: !layer.locked })}
            onDelete={() => deleteLayer(layer.id)}
            onDuplicate={() => duplicateLayer(layer.id)}
            onMoveUp={() => reorderLayer(layer.id, 'up')}
            onMoveDown={() => reorderLayer(layer.id, 'down')}
            onRename={(name) => updateLayer(layer.id, { name })}
            onBlendMode={(bm) => updateLayer(layer.id, { blend_mode: bm })}
            onOpacity={(op) => updateLayer(layer.id, { opacity: op })}
          />
        ))}
      </div>
    </div>
  )
}

interface LayerItemProps {
  layer: LayerData
  selected: boolean
  onSelect: (e: React.MouseEvent) => void
  onToggleVisible: () => void
  onToggleLock: () => void
  onDelete: () => void
  onDuplicate: () => void
  onMoveUp: () => void
  onMoveDown: () => void
  onRename: (name: string) => void
  onBlendMode: (bm: BlendMode) => void
  onOpacity: (op: number) => void
}

const LayerItem: React.FC<LayerItemProps> = ({
  layer, selected, onSelect, onToggleVisible, onToggleLock,
  onDelete, onDuplicate, onMoveUp, onMoveDown, onRename, onBlendMode, onOpacity,
}) => {
  const thumbUrl = useBlobUrl(
    layer.png_path
      ? (layer.png_path.startsWith('blob:') || layer.png_path.startsWith('data:') ? layer.png_path : `${API_BASE}${layer.png_path}`)
      : null
  )
  const [editing, setEditing] = useState(false)
  const [nameVal, setNameVal] = useState(layer.name)
  const [showBlend, setShowBlend] = useState(false)
  const [blendRect, setBlendRect] = useState<DOMRect | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const blendBtnRef = useRef<HTMLButtonElement>(null)

  const commitRename = () => {
    setEditing(false)
    if (nameVal.trim()) onRename(nameVal.trim())
    else setNameVal(layer.name)
  }

  return (
    <div className={`layer-item group flex-col gap-0 px-2 py-1.5 ${selected ? 'selected' : ''}`} onClick={onSelect}>
      <div className="flex items-center gap-1.5 w-full">
        {/* Thumbnail */}
        <div className="w-7 h-7 rounded bg-dark-600 overflow-hidden flex-shrink-0 border border-dark-500"
          style={{ backgroundImage: 'repeating-conic-gradient(#333 0% 25%, #222 0% 50%) 0 0 / 8px 8px' }}>
          {thumbUrl && (
            <img src={thumbUrl} alt={layer.name} className="w-full h-full object-cover" style={{ opacity: layer.visible ? 1 : 0.3 }} />
          )}
        </div>

        {/* Name */}
        {editing ? (
          <input
            ref={inputRef}
            className="flex-1 text-xs bg-dark-600 text-white rounded px-1 py-0.5 border border-accent outline-none min-w-0"
            value={nameVal}
            onChange={e => setNameVal(e.target.value)}
            onBlur={commitRename}
            onKeyDown={e => { if (e.key === 'Enter') commitRename(); if (e.key === 'Escape') { setEditing(false); setNameVal(layer.name) } }}
            onClick={e => e.stopPropagation()}
            autoFocus
          />
        ) : (
          <span
            className="flex-1 text-xs truncate min-w-0"
            style={{ opacity: layer.visible ? 1 : 0.4 }}
            onDoubleClick={e => { e.stopPropagation(); setEditing(true); setNameVal(layer.name) }}
            title="Double-click to rename"
          >
            {layer.name}
          </span>
        )}

        {/* Controls */}
        <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0">
          <IconBtn title="Move up" onClick={(e) => { e.stopPropagation(); onMoveUp() }}><ChevronUp size={11} /></IconBtn>
          <IconBtn title="Move down" onClick={(e) => { e.stopPropagation(); onMoveDown() }}><ChevronDown size={11} /></IconBtn>
          <IconBtn title={layer.visible ? 'Hide' : 'Show'} onClick={(e) => { e.stopPropagation(); onToggleVisible() }}>
            {layer.visible ? <Eye size={11} /> : <EyeOff size={11} />}
          </IconBtn>
          <IconBtn title={layer.locked ? 'Unlock' : 'Lock'} onClick={(e) => { e.stopPropagation(); onToggleLock() }}>
            {layer.locked ? <Lock size={11} /> : <Unlock size={11} />}
          </IconBtn>
          <IconBtn title="Duplicate" onClick={(e) => { e.stopPropagation(); onDuplicate() }}><Copy size={11} /></IconBtn>
          <IconBtn title="Delete" onClick={(e) => { e.stopPropagation(); onDelete() }} className="text-red-400"><Trash2 size={11} /></IconBtn>
        </div>
      </div>

      {/* Opacity + Blend mode row (visible when selected) */}
      {selected && (
        <div className="flex items-center gap-1.5 mt-1 w-full" onClick={e => e.stopPropagation()}>
          {/* Opacity mini slider */}
          <input
            type="range" min={0} max={1} step={0.01} value={layer.opacity}
            onChange={e => onOpacity(parseFloat(e.target.value))}
            className="flex-1 h-1"
            title={`Opacity: ${Math.round(layer.opacity * 100)}%`}
          />
          <span className="text-xs text-gray-500 w-7 text-right">{Math.round(layer.opacity * 100)}%</span>

          {/* Blend mode */}
          <div className="relative">
            <button
              ref={blendBtnRef}
              className="text-xs bg-dark-600 hover:bg-dark-500 text-gray-300 px-1.5 py-0.5 rounded"
              onClick={e => {
                e.stopPropagation()
                const rect = blendBtnRef.current?.getBoundingClientRect() ?? null
                setBlendRect(rect)
                setShowBlend(!showBlend)
              }}
              title="Blend mode"
            >
              {(layer.blend_mode || 'normal').slice(0, 4)}
            </button>
            {showBlend && blendRect && (
              <div
                className="fixed bg-dark-700 border border-dark-500 rounded shadow-xl z-[9999] py-1 max-h-48 overflow-y-auto min-w-max"
                style={{ bottom: window.innerHeight - blendRect.top + 4, right: window.innerWidth - blendRect.right }}
              >
                {BLEND_MODES.map(bm => (
                  <button
                    key={bm}
                    onClick={e => { e.stopPropagation(); onBlendMode(bm); setShowBlend(false) }}
                    className={`block w-full text-left px-3 py-1 text-xs hover:bg-dark-600 ${layer.blend_mode === bm ? 'text-accent' : 'text-gray-300'}`}
                  >
                    {bm}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

const IconBtn: React.FC<{
  title: string; onClick: (e: React.MouseEvent) => void; children: React.ReactNode; className?: string
}> = ({ title, onClick, children, className = 'text-gray-400' }) => (
  <button title={title} onClick={onClick} className={`p-0.5 rounded hover:bg-dark-500 ${className}`}>
    {children}
  </button>
)
