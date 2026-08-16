import React, { useState, useRef, useEffect } from 'react'
import { useEditorStore } from '../store/editorStore'
import { Tool } from '../types'
import {
  Move, Crop, Paintbrush, Eraser, Wand2, MousePointer2,
  MessageSquare, Square, Circle, Minus, ArrowRight, Triangle, Star,
  Pencil, Highlighter, Pipette, Lasso, RectangleHorizontal,
  Scissors, Type, ChevronRight,
} from 'lucide-react'

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

interface ToolDef { id: Tool; icon: React.ReactNode; label: string }
interface ToolGroup { label: string; tools: ToolDef[] }

const TOOL_GROUPS: ToolGroup[] = [
  {
    label: 'Select & Move',
    tools: [
      { id: 'move',           icon: <Move size={16} />,               label: 'Move (V)' },
      { id: 'rect_select',    icon: <RectangleHorizontal size={16} />, label: 'Rect Select (M)' },
      { id: 'ellipse_select', icon: <Circle size={16} />,             label: 'Ellipse Select' },
      { id: 'lasso_select',   icon: <Lasso size={16} />,              label: 'Lasso Select (L)' },
      { id: 'free_select',    icon: <Scissors size={16} />,           label: 'Free Select' },
      { id: 'magic_select',   icon: <Wand2 size={16} />,              label: 'Magic Select (W)' },
      { id: 'object_select',  icon: <MousePointer2 size={16} />,      label: 'Object Select' },
    ],
  },
  {
    label: 'Draw',
    tools: [
      { id: 'brush',        icon: <Paintbrush size={16} />, label: 'Brush (B)' },
      { id: 'pencil',       icon: <Pencil size={16} />,     label: 'Pencil' },
      { id: 'marker',       icon: <Highlighter size={16} />, label: 'Marker' },
      { id: 'eraser',       icon: <Eraser size={16} />,     label: 'Eraser (E)' },
      { id: 'color_picker', icon: <Pipette size={16} />,    label: 'Color Picker (I)' },
    ],
  },
  {
    label: 'Shapes',
    tools: [
      { id: 'shape_rect',     icon: <Square size={16} />,     label: 'Rectangle' },
      { id: 'shape_ellipse',  icon: <Circle size={16} />,     label: 'Ellipse' },
      { id: 'shape_line',     icon: <Minus size={16} />,      label: 'Line' },
      { id: 'shape_arrow',    icon: <ArrowRight size={16} />, label: 'Arrow' },
      { id: 'shape_triangle', icon: <Triangle size={16} />,   label: 'Triangle' },
      { id: 'shape_star',     icon: <Star size={16} />,       label: 'Star' },
    ],
  },
  {
    label: 'Edit',
    tools: [
      { id: 'crop',        icon: <Crop size={16} />,          label: 'Crop (C)' },
      { id: 'text_add',    icon: <Type size={16} />,          label: 'Text (T)' },
      { id: 'clone',       icon: <CopyIcon size={16} />,      label: 'Clone Stamp (S)' },
      { id: 'heal',        icon: <HealIcon size={16} />,      label: 'Heal (J)' },
      { id: 'text_prompt', icon: <MessageSquare size={16} />, label: 'AI Prompt' },
    ],
  },
]

export const ToolSidebar: React.FC = () => {
  const { activeTool, setActiveTool } = useEditorStore()
  const [flyout, setFlyout] = useState<string | null>(null)
  const [flyoutY, setFlyoutY] = useState(0)
  const sidebarRef = useRef<HTMLDivElement>(null)

  // Close flyout on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (sidebarRef.current && !sidebarRef.current.contains(e.target as Node)) {
        setFlyout(null)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  return (
    <div ref={sidebarRef} className="tool-sidebar flex flex-col items-center py-2 gap-0.5 relative">
      {TOOL_GROUPS.map((group, gi) => {
        const activeInGroup = group.tools.find(t => t.id === activeTool)
        const displayTool = activeInGroup || group.tools[0]
        const isActive = !!activeInGroup

        return (
          <React.Fragment key={group.label}>
            {gi > 0 && <div className="w-6 h-px bg-dark-600 my-1" />}
            <div className="relative">
              <button
                title={displayTool.label}
                className={`tool-sidebar-btn ${isActive ? 'active' : ''}`}
                onClick={(e) => {
                  if (group.tools.length === 1) {
                    setActiveTool(displayTool.id)
                    setFlyout(null)
                  } else {
                    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect()
                    setFlyoutY(rect.top)
                    setFlyout(flyout === group.label ? null : group.label)
                  }
                }}
                onDoubleClick={() => {
                  setActiveTool(displayTool.id)
                  setFlyout(null)
                }}
              >
                {displayTool.icon}
                {group.tools.length > 1 && (
                  <ChevronRight size={7} className="absolute bottom-0.5 right-0.5 opacity-50" />
                )}
              </button>

              {flyout === group.label && (
                <div
                  className="fixed left-12 bg-dark-700 border border-dark-500 rounded shadow-xl z-[9999] py-1 min-w-max"
                  style={{ top: flyoutY }}
                >
                  <div className="px-3 py-1 text-xs text-gray-500 uppercase tracking-wider border-b border-dark-600 mb-1">
                    {group.label}
                  </div>
                  {group.tools.map(t => (
                    <button
                      key={t.id}
                      onClick={() => { setActiveTool(t.id); setFlyout(null) }}
                      className={`flex items-center gap-2.5 w-full px-3 py-1.5 text-xs hover:bg-dark-600 ${activeTool === t.id ? 'text-accent' : 'text-gray-300'}`}
                    >
                      <span className="w-4 flex justify-center">{t.icon}</span>
                      {t.label}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </React.Fragment>
        )
      })}
    </div>
  )
}
