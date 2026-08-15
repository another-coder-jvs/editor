import React, { useState } from 'react'
import { useEditorStore } from '../store/editorStore'
import { Tool } from '../types'
import { saveProject } from '../api/client'
import { toast } from 'react-toastify'
import {
  Move, Crop, Paintbrush, Eraser, Wand2, MousePointer2,
  MessageSquare, Undo2, Redo2, Download, FolderOpen, Save,
  Square, Circle, Minus, ArrowRight, Triangle, Star,
  Pencil, Highlighter, Pipette, Lasso, RectangleHorizontal,
  Scissors, Type, ChevronDown, RotateCcw,
} from 'lucide-react'

// Custom icons not in lucide
const CopyIcon = ({ size }: { size: number }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <rect x="9" y="9" width="13" height="13" rx="2"/>
    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
  </svg>
)
const HealIcon = ({ size }: { size: number }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="12" cy="12" r="10"/>
    <path d="M12 8v8M8 12h8"/>
  </svg>
)

interface ToolGroup {
  label: string
  tools: { id: Tool; icon: React.ReactNode; label: string }[]
}

const TOOL_GROUPS: ToolGroup[] = [
  {
    label: 'Select',
    tools: [
      { id: 'move',          icon: <Move size={15} />,              label: 'Move (V)' },
      { id: 'rect_select',   icon: <RectangleHorizontal size={15}/>, label: 'Rect Select (M)' },
      { id: 'ellipse_select',icon: <Circle size={15} />,            label: 'Ellipse Select' },
      { id: 'lasso_select',  icon: <Lasso size={15} />,             label: 'Lasso Select (L)' },
      { id: 'free_select',   icon: <Scissors size={15} />,          label: 'Free Select' },
      { id: 'magic_select',  icon: <Wand2 size={15} />,             label: 'Magic Select (W)' },
      { id: 'object_select', icon: <MousePointer2 size={15} />,     label: 'Object Select' },
    ],
  },
  {
    label: 'Draw',
    tools: [
      { id: 'brush',        icon: <Paintbrush size={15} />,  label: 'Brush (B)' },
      { id: 'pencil',       icon: <Pencil size={15} />,      label: 'Pencil' },
      { id: 'marker',       icon: <Highlighter size={15} />, label: 'Marker' },
      { id: 'eraser',       icon: <Eraser size={15} />,      label: 'Eraser (E)' },
      { id: 'color_picker', icon: <Pipette size={15} />,     label: 'Color Picker (I)' },
    ],
  },
  {
    label: 'Shape',
    tools: [
      { id: 'shape_rect',     icon: <Square size={15} />,      label: 'Rectangle' },
      { id: 'shape_ellipse',  icon: <Circle size={15} />,      label: 'Ellipse' },
      { id: 'shape_line',     icon: <Minus size={15} />,       label: 'Line' },
      { id: 'shape_arrow',    icon: <ArrowRight size={15} />,  label: 'Arrow' },
      { id: 'shape_triangle', icon: <Triangle size={15} />,    label: 'Triangle' },
      { id: 'shape_star',     icon: <Star size={15} />,        label: 'Star' },
    ],
  },
  {
    label: 'Edit',
    tools: [
      { id: 'crop',        icon: <Crop size={15} />,           label: 'Crop (C)' },
      { id: 'text_add',    icon: <Type size={15} />,           label: 'Text (T)' },
      { id: 'clone',       icon: <CopyIcon size={15} />,       label: 'Clone Stamp (S)' },
      { id: 'heal',        icon: <HealIcon size={15} />,       label: 'Heal (J)' },
      { id: 'text_prompt', icon: <MessageSquare size={15} />,  label: 'AI Prompt' },
    ],
  },
]

interface Props {
  onExport: () => void
  onOpenProjects: () => void
}

export const Toolbar: React.FC<Props> = ({ onExport, onOpenProjects }) => {
  const {
    activeTool, setActiveTool, undo, redo, undoStack, redoStack,
    sessionId, originalImagePath, layers, canvasWidth, canvasHeight, canvasOffset,
    currentProjectName, setCurrentProjectName, reset,
  } = useEditorStore()

  const [openGroup, setOpenGroup] = useState<string | null>(null)

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

  return (
    <div className="flex items-center gap-0.5 px-2 py-1 bg-dark-800 border-b border-dark-600 h-11">
      {TOOL_GROUPS.map((group) => {
        const activeInGroup = group.tools.find(t => t.id === activeTool)
        const displayTool = activeInGroup || group.tools[0]
        const isGroupActive = !!activeInGroup

        return (
          <div key={group.label} className="relative">
            <div className="flex items-center">
              <button
                title={displayTool.label}
                onClick={() => { setActiveTool(displayTool.id); setOpenGroup(null) }}
                className={`toolbar-btn px-2 py-1 ${isGroupActive ? 'active' : 'text-gray-400'}`}
              >
                {displayTool.icon}
              </button>
              {group.tools.length > 1 && (
                <button
                  onClick={(e) => { e.stopPropagation(); setOpenGroup(openGroup === group.label ? null : group.label) }}
                  className={`px-0.5 py-1 rounded hover:bg-dark-600 ${isGroupActive ? 'text-white' : 'text-gray-500'}`}
                >
                  <ChevronDown size={10} />
                </button>
              )}
            </div>

            {openGroup === group.label && (
              <div
                className="absolute top-full left-0 mt-1 bg-dark-700 border border-dark-500 rounded shadow-xl z-50 py-1 min-w-max"
                onMouseLeave={() => setOpenGroup(null)}
              >
                {group.tools.map(t => (
                  <button
                    key={t.id}
                    onClick={() => { setActiveTool(t.id); setOpenGroup(null) }}
                    className={`flex items-center gap-2 w-full px-3 py-1.5 text-xs hover:bg-dark-600 ${activeTool === t.id ? 'text-accent' : 'text-gray-300'}`}
                  >
                    {t.icon} {t.label}
                  </button>
                ))}
              </div>
            )}
          </div>
        )
      })}

      <div className="w-px h-6 bg-dark-500 mx-1" />

      <button title="Undo (Ctrl+Z)" onClick={undo} disabled={undoStack.length === 0}
        className="toolbar-btn text-gray-400 disabled:opacity-30">
        <Undo2 size={15} />
      </button>
      <button title="Redo (Ctrl+Y)" onClick={redo} disabled={redoStack.length === 0}
        className="toolbar-btn text-gray-400 disabled:opacity-30">
        <Redo2 size={15} />
      </button>
      <button
        title="New Project"
        onClick={() => { if (window.confirm('Start new project? Unsaved changes will be lost.')) reset() }}
        className="toolbar-btn text-gray-400"
      >
        <RotateCcw size={15} />
      </button>

      <div className="flex-1" />

      <span className="text-xs text-gray-600 mr-2 hidden md:block">
        {TOOL_GROUPS.flatMap(g => g.tools).find(t => t.id === activeTool)?.label}
      </span>

      <button title="Save Project" onClick={handleSave} className="toolbar-btn text-gray-400 hover:text-white flex-row gap-1">
        <Save size={15} /><span className="text-xs hidden sm:block">Save</span>
      </button>
      <button title="Projects" onClick={onOpenProjects} className="toolbar-btn text-gray-400 hover:text-white flex-row gap-1">
        <FolderOpen size={15} /><span className="text-xs hidden sm:block">Projects</span>
      </button>
      <button title="Export" onClick={onExport} className="toolbar-btn text-accent hover:text-white flex-row gap-1">
        <Download size={15} /><span className="text-xs hidden sm:block">Export</span>
      </button>
    </div>
  )
}
