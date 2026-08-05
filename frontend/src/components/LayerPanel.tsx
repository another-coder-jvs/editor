import React from 'react'
import { useEditorStore } from '../store/editorStore'
import { LayerData } from '../types'
import {
  Eye, EyeOff, Lock, Unlock, Trash2, Copy,
  ChevronUp, ChevronDown,
} from 'lucide-react'

import { baseImagesUrl } from '@/config'
import { useBlobUrl } from '../hooks/useBlobUrl'
const API_BASE = baseImagesUrl || 'http://localhost:5000'

export const LayerPanel: React.FC = () => {
  const {
    layers, selectedLayerIds, selectLayer,
    updateLayer, deleteLayer, duplicateLayer, reorderLayer,
  } = useEditorStore()

  const sorted = [...layers].sort((a, b) => b.z_index - a.z_index)

  return (
    <div className="panel w-56 flex flex-col h-full">
      <div className="px-3 py-2 text-xs font-semibold text-gray-400 uppercase tracking-wider border-b border-dark-600">
        Layers
      </div>
      <div className="flex-1 overflow-y-auto py-1">
        {sorted.length === 0 && (
          <p className="text-xs text-gray-600 text-center mt-8 px-3">
            Upload an image to generate layers
          </p>
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
}

const LayerItem: React.FC<LayerItemProps> = ({
  layer, selected, onSelect, onToggleVisible,
  onToggleLock, onDelete, onDuplicate, onMoveUp, onMoveDown,
}) => {
  // const thumbUrl = layer.png_path
  //   ? `/temp/${layer.png_path.split('/temp/')[1] ?? layer.png_path}`
  //   : null
  const thumbUrl = useBlobUrl(layer.png_path ? (layer.png_path.startsWith("blob:") ? layer.png_path : `${API_BASE}${layer.png_path}`) : null)

  return (
    <div
      className={`layer-item ${selected ? 'selected' : ''}`}
      onClick={onSelect}
    >
      {/* Thumbnail */}
      <div className="w-8 h-8 rounded bg-dark-600 overflow-hidden flex-shrink-0 border border-dark-500">
        {thumbUrl && (
          <img
            src={thumbUrl}
            alt={layer.name}
            className="w-full h-full object-cover"
            style={{ opacity: layer.visible ? 1 : 0.3 }}
          />
        )}
      </div>

      {/* Name */}
      <span
        className="flex-1 text-xs truncate"
        style={{ opacity: layer.visible ? 1 : 0.4 }}
      >
        {layer.name}
      </span>

      {/* Controls */}
      <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
        <IconBtn title="Move up" onClick={(e) => { e.stopPropagation(); onMoveUp() }}>
          <ChevronUp size={12} />
        </IconBtn>
        <IconBtn title="Move down" onClick={(e) => { e.stopPropagation(); onMoveDown() }}>
          <ChevronDown size={12} />
        </IconBtn>
        <IconBtn title={layer.visible ? 'Hide' : 'Show'} onClick={(e) => { e.stopPropagation(); onToggleVisible() }}>
          {layer.visible ? <Eye size={12} /> : <EyeOff size={12} />}
        </IconBtn>
        <IconBtn title={layer.locked ? 'Unlock' : 'Lock'} onClick={(e) => { e.stopPropagation(); onToggleLock() }}>
          {layer.locked ? <Lock size={12} /> : <Unlock size={12} />}
        </IconBtn>
        <IconBtn title="Duplicate" onClick={(e) => { e.stopPropagation(); onDuplicate() }}>
          <Copy size={12} />
        </IconBtn>
        <IconBtn title="Delete" onClick={(e) => { e.stopPropagation(); onDelete() }} className="text-red-400">
          <Trash2 size={12} />
        </IconBtn>
      </div>
    </div>
  )
}

const IconBtn: React.FC<{
  title: string
  onClick: (e: React.MouseEvent) => void
  children: React.ReactNode
  className?: string
}> = ({ title, onClick, children, className = 'text-gray-400' }) => (
  <button
    title={title}
    onClick={onClick}
    className={`p-0.5 rounded hover:bg-dark-500 ${className}`}
  >
    {children}
  </button>
)
