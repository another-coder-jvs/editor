import React from 'react'
import { useEditorStore } from '../store/editorStore'
import { saveProject } from '../api/client'
import { toast } from 'react-toastify'
import { Undo2, Redo2, Download, FolderOpen, Save, RotateCcw, ImageIcon } from 'lucide-react'
import { ApiSpinner } from './ApiSpinner'

interface Props {
  onExport: () => void
  onOpenProjects: () => void
}

export const Toolbar: React.FC<Props> = ({ onExport, onOpenProjects }) => {
  const {
    undo, redo, undoStack, redoStack,
    sessionId, originalImagePath, layers, canvasWidth, canvasHeight, canvasOffset,
    currentProjectName, setCurrentProjectName, reset, activeTool,
  } = useEditorStore()

  const handleSave = async () => {
    if (!sessionId || !originalImagePath) { toast.warn('Nothing to save'); return }
    const name = currentProjectName || `project_${sessionId.slice(0, 8)}`
    try {
      await saveProject({
        session_id: sessionId,
        project_name: name,
        original_image_path: originalImagePath,
        layers,
        canvas_width: canvasWidth,
        canvas_height: canvasHeight,
        canvas_position: canvasOffset,
        settings: {},
        prompts: layers.flatMap(l => l.history),
      })
      setCurrentProjectName(name)
      toast.success(`Saved "${name}"`)
    } catch { toast.error('Save failed') }
  }

  const TOOL_LABELS: Partial<Record<string, string>> = {
    move: 'Move', crop: 'Crop', brush: 'Brush', pencil: 'Pencil', marker: 'Marker',
    eraser: 'Eraser', color_picker: 'Color Picker', magic_select: 'Magic Select',
    object_select: 'Object Select', text_prompt: 'AI Prompt', rect_select: 'Rect Select',
    ellipse_select: 'Ellipse Select', lasso_select: 'Lasso', free_select: 'Free Select',
    text_add: 'Text', shape_rect: 'Rectangle', shape_ellipse: 'Ellipse', shape_line: 'Line',
    shape_arrow: 'Arrow', shape_triangle: 'Triangle', shape_star: 'Star',
    clone: 'Clone Stamp', heal: 'Heal',
  }

  return (
    <header className="relative flex items-center gap-1 px-3 h-11 bg-dark-800 border-b border-dark-600 flex-shrink-0">
      {/* Logo / brand */}
      <div className="flex items-center gap-1.5 mr-2">
        <ImageIcon size={18} className="text-accent" />
        <span className="text-sm font-semibold text-white hidden sm:block">AI Editor</span>
      </div>

      {/* API loading spinner — appears next to logo during any request */}
      <ApiSpinner />

      <div className="w-px h-5 bg-dark-500 mx-1" />

      {/* Undo / Redo */}
      <button title="Undo (Ctrl+Z)" onClick={undo} disabled={undoStack.length === 0}
        className="toolbar-btn text-gray-400 disabled:opacity-30">
        <Undo2 size={15} />
      </button>
      <button title="Redo (Ctrl+Y)" onClick={redo} disabled={redoStack.length === 0}
        className="toolbar-btn text-gray-400 disabled:opacity-30">
        <Redo2 size={15} />
      </button>

      <div className="w-px h-5 bg-dark-500 mx-1" />

      {/* New project */}
      <button
        title="New Project"
        onClick={() => { if (window.confirm('Start new project? Unsaved changes will be lost.')) reset() }}
        className="toolbar-btn text-gray-400"
      >
        <RotateCcw size={15} />
      </button>

      {/* Active tool indicator */}
      {activeTool && (
        <span className="text-xs text-gray-500 ml-2 hidden md:block">
          {TOOL_LABELS[activeTool] ?? activeTool}
        </span>
      )}

      <div className="flex-1" />

      {/* Project name */}
      {currentProjectName && (
        <span className="text-xs text-gray-600 mr-2 hidden lg:block truncate max-w-32" title={currentProjectName}>
          {currentProjectName}
        </span>
      )}

      {/* File actions */}
      <button title="Save Project (Ctrl+S)" onClick={handleSave}
        className="toolbar-btn text-gray-400 hover:text-white flex-row gap-1">
        <Save size={15} /><span className="text-xs hidden sm:block">Save</span>
      </button>
      <button title="Open Projects" onClick={onOpenProjects}
        className="toolbar-btn text-gray-400 hover:text-white flex-row gap-1">
        <FolderOpen size={15} /><span className="text-xs hidden sm:block">Projects</span>
      </button>
      <button title="Export Image" onClick={onExport}
        className="toolbar-btn bg-accent hover:bg-accent-hover text-white flex-row gap-1 px-3">
        <Download size={15} /><span className="text-xs hidden sm:block">Export</span>
      </button>
    </header>
  )
}
