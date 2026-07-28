import React from 'react'
import { useEditorStore } from '../store/editorStore'
import { Tool } from '../types'
import {
  Move, Crop, Paintbrush, Eraser, Wand2, MousePointer2,
  MessageSquare, Undo2, Redo2, Download,
} from 'lucide-react'

const TOOLS: { id: Tool; icon: React.ReactNode; label: string }[] = [
  { id: 'move', icon: <Move size={18} />, label: 'Move' },
  { id: 'crop', icon: <Crop size={18} />, label: 'Crop' },
  { id: 'brush', icon: <Paintbrush size={18} />, label: 'Brush' },
  { id: 'eraser', icon: <Eraser size={18} />, label: 'Eraser' },
  { id: 'magic_select', icon: <Wand2 size={18} />, label: 'Magic Select' },
  { id: 'object_select', icon: <MousePointer2 size={18} />, label: 'Object Select' },
  { id: 'text_prompt', icon: <MessageSquare size={18} />, label: 'Prompt' },
]

interface Props {
  onExport: () => void
}

export const Toolbar: React.FC<Props> = ({ onExport }) => {
  const { activeTool, setActiveTool, undo, redo, undoStack, redoStack } = useEditorStore()

  return (
    <div className="flex items-center gap-1 px-3 py-1 bg-dark-800 border-b border-dark-600 h-12">
      {TOOLS.map((t) => (
        <button
          key={t.id}
          title={t.label}
          onClick={() => setActiveTool(t.id)}
          className={`toolbar-btn ${activeTool === t.id ? 'active' : 'text-gray-400'}`}
        >
          {t.icon}
        </button>
      ))}

      <div className="w-px h-6 bg-dark-500 mx-1" />

      <button
        title="Undo (Ctrl+Z)"
        onClick={undo}
        disabled={undoStack.length === 0}
        className="toolbar-btn text-gray-400 disabled:opacity-30"
      >
        <Undo2 size={18} />
      </button>
      <button
        title="Redo (Ctrl+Y)"
        onClick={redo}
        disabled={redoStack.length === 0}
        className="toolbar-btn text-gray-400 disabled:opacity-30"
      >
        <Redo2 size={18} />
      </button>

      <div className="flex-1" />

      <button
        title="Export"
        onClick={onExport}
        className="toolbar-btn text-accent hover:text-white"
      >
        <Download size={18} />
        <span className="text-xs">Export</span>
      </button>
    </div>
  )
}
