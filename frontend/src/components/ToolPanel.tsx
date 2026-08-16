import React, { useRef, useState } from 'react'
import { useEditorStore } from '../store/editorStore'
import { Tool, BlendMode } from '../types'
import {
  Move, Crop, Paintbrush, Eraser, Wand2, MousePointer2,
  MessageSquare, Square, Circle, Minus, ArrowRight, Triangle, Star,
  Pencil, Highlighter, Pipette, Lasso, RectangleHorizontal,
  Scissors, Type,
} from 'lucide-react'

const CopyIcon = () => (
  <svg width={14} height={14} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
  </svg>
)
const HealIcon = () => (
  <svg width={14} height={14} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="12" cy="12" r="10"/><path d="M12 8v8M8 12h8"/>
  </svg>
)

interface ToolDef { id: Tool; icon: React.ReactNode; label: string; shortcut?: string }
interface ToolGroup { label: string; tools: ToolDef[] }

const TOOL_GROUPS: ToolGroup[] = [
  {
    label: 'Select & Move',
    tools: [
      { id: 'move',           icon: <Move size={14} />,               label: 'Move',         shortcut: 'V' },
      { id: 'rect_select',    icon: <RectangleHorizontal size={14} />, label: 'Rect',         shortcut: 'M' },
      { id: 'ellipse_select', icon: <Circle size={14} />,             label: 'Ellipse' },
      { id: 'lasso_select',   icon: <Lasso size={14} />,              label: 'Lasso',        shortcut: 'L' },
      { id: 'free_select',    icon: <Scissors size={14} />,           label: 'Free' },
      { id: 'magic_select',   icon: <Wand2 size={14} />,              label: 'Magic',        shortcut: 'W' },
      { id: 'object_select',  icon: <MousePointer2 size={14} />,      label: 'Object' },
    ],
  },
  {
    label: 'Draw',
    tools: [
      { id: 'brush',        icon: <Paintbrush size={14} />, label: 'Brush',   shortcut: 'B' },
      { id: 'pencil',       icon: <Pencil size={14} />,     label: 'Pencil',  shortcut: 'P' },
      { id: 'marker',       icon: <Highlighter size={14} />, label: 'Marker' },
      { id: 'eraser',       icon: <Eraser size={14} />,     label: 'Eraser',  shortcut: 'E' },
      { id: 'color_picker', icon: <Pipette size={14} />,    label: 'Picker',  shortcut: 'I' },
      { id: 'clone',        icon: <CopyIcon />,             label: 'Clone',   shortcut: 'S' },
      { id: 'heal',         icon: <HealIcon />,             label: 'Heal',    shortcut: 'J' },
    ],
  },
  {
    label: 'Shapes',
    tools: [
      { id: 'shape_rect',     icon: <Square size={14} />,     label: 'Rect' },
      { id: 'shape_ellipse',  icon: <Circle size={14} />,     label: 'Ellipse' },
      { id: 'shape_line',     icon: <Minus size={14} />,      label: 'Line' },
      { id: 'shape_arrow',    icon: <ArrowRight size={14} />, label: 'Arrow' },
      { id: 'shape_triangle', icon: <Triangle size={14} />,   label: 'Triangle' },
      { id: 'shape_star',     icon: <Star size={14} />,       label: 'Star' },
    ],
  },
  {
    label: 'Edit',
    tools: [
      { id: 'crop',        icon: <Crop size={14} />,          label: 'Crop',    shortcut: 'C' },
      { id: 'text_add',    icon: <Type size={14} />,          label: 'Text',    shortcut: 'T' },
      { id: 'text_prompt', icon: <MessageSquare size={14} />, label: 'AI Prompt' },
    ],
  },
]

// ── Tool Options ──────────────────────────────────────────────
const Row: React.FC<{ label: string; children: React.ReactNode }> = ({ label, children }) => (
  <div className="flex items-center gap-2">
    <span className="text-xs text-gray-500 w-14 flex-shrink-0">{label}</span>
    {children}
  </div>
)

const ToolOptions: React.FC = () => {
  const { activeTool, toolOptions, setToolOption } = useEditorStore()

  const DRAW = ['brush', 'pencil', 'marker', 'eraser', 'clone', 'heal']
  const SHAPES = ['shape_rect', 'shape_ellipse', 'shape_line', 'shape_arrow', 'shape_triangle', 'shape_star']
  const SELECT = ['rect_select', 'ellipse_select', 'lasso_select', 'free_select', 'magic_select']

  if (DRAW.includes(activeTool)) return (
    <div className="px-3 py-2 space-y-2 border-t border-dark-600">
      {activeTool !== 'eraser' && (
        <Row label="Color">
          <input type="color" value={toolOptions.brushColor}
            onChange={e => setToolOption('brushColor', e.target.value)}
            className="w-7 h-5 rounded cursor-pointer border-0 bg-transparent flex-shrink-0" />
          <span className="text-xs text-gray-400 font-mono">{toolOptions.brushColor}</span>
        </Row>
      )}
      <Row label="Size">
        <input type="range" min={1} max={200}
          value={activeTool === 'eraser' ? toolOptions.eraserSize : toolOptions.brushSize}
          onChange={e => setToolOption(activeTool === 'eraser' ? 'eraserSize' : 'brushSize', parseInt(e.target.value))}
          className="flex-1 h-1" />
        <span className="text-xs text-white w-8 text-right">
          {activeTool === 'eraser' ? toolOptions.eraserSize : toolOptions.brushSize}
        </span>
      </Row>
      {activeTool !== 'eraser' && (
        <Row label="Opacity">
          <input type="range" min={0.01} max={1} step={0.01} value={toolOptions.brushOpacity}
            onChange={e => setToolOption('brushOpacity', parseFloat(e.target.value))}
            className="flex-1 h-1" />
          <span className="text-xs text-white w-8 text-right">{Math.round(toolOptions.brushOpacity * 100)}%</span>
        </Row>
      )}
      {activeTool === 'brush' && (
        <Row label="Hardness">
          <input type="range" min={0} max={1} step={0.05} value={toolOptions.brushHardness}
            onChange={e => setToolOption('brushHardness', parseFloat(e.target.value))}
            className="flex-1 h-1" />
          <span className="text-xs text-white w-8 text-right">{Math.round(toolOptions.brushHardness * 100)}%</span>
        </Row>
      )}
      {activeTool === 'clone' && <p className="text-xs text-gray-600">Alt+Click to set source</p>}
    </div>
  )

  if (SHAPES.includes(activeTool)) return (
    <div className="px-3 py-2 space-y-2 border-t border-dark-600">
      <Row label="Stroke">
        <input type="color" value={toolOptions.shapeStroke}
          onChange={e => setToolOption('shapeStroke', e.target.value)}
          className="w-7 h-5 rounded cursor-pointer border-0 bg-transparent flex-shrink-0" />
      </Row>
      <Row label="Fill">
        <input type="color"
          value={toolOptions.shapeFill === 'transparent' ? '#000000' : toolOptions.shapeFill}
          onChange={e => setToolOption('shapeFill', e.target.value)}
          className="w-7 h-5 rounded cursor-pointer border-0 bg-transparent flex-shrink-0"
          disabled={toolOptions.shapeFill === 'transparent'} />
        <label className="flex items-center gap-1 text-xs text-gray-400 cursor-pointer">
          <input type="checkbox" checked={toolOptions.shapeFill === 'transparent'}
            onChange={e => setToolOption('shapeFill', e.target.checked ? 'transparent' : '#ffffff')} />
          None
        </label>
      </Row>
      <Row label="Width">
        <input type="range" min={1} max={50} value={toolOptions.shapeStrokeWidth}
          onChange={e => setToolOption('shapeStrokeWidth', parseInt(e.target.value))}
          className="flex-1 h-1" />
        <span className="text-xs text-white w-8 text-right">{toolOptions.shapeStrokeWidth}px</span>
      </Row>
      <Row label="Opacity">
        <input type="range" min={0.01} max={1} step={0.01} value={toolOptions.shapeOpacity}
          onChange={e => setToolOption('shapeOpacity', parseFloat(e.target.value))}
          className="flex-1 h-1" />
        <span className="text-xs text-white w-8 text-right">{Math.round(toolOptions.shapeOpacity * 100)}%</span>
      </Row>
    </div>
  )

  if (SELECT.includes(activeTool)) return (
    <div className="px-3 py-2 space-y-2 border-t border-dark-600">
      <Row label="Feather">
        <input type="range" min={0} max={50} value={toolOptions.selectionFeather}
          onChange={e => setToolOption('selectionFeather', parseInt(e.target.value))}
          className="flex-1 h-1" />
        <span className="text-xs text-white w-8 text-right">{toolOptions.selectionFeather}px</span>
      </Row>
      <p className="text-xs text-gray-600">Esc to clear selection</p>
    </div>
  )

  if (activeTool === 'color_picker') return (
    <div className="px-3 py-2 border-t border-dark-600 flex items-center gap-2">
      <div className="w-7 h-7 rounded border border-dark-500 flex-shrink-0" style={{ background: toolOptions.brushColor }} />
      <span className="text-xs text-white font-mono">{toolOptions.brushColor}</span>
      <span className="text-xs text-gray-600 ml-1">Click canvas to pick</span>
    </div>
  )

  if (activeTool === 'crop') return (
    <div className="px-3 py-2 border-t border-dark-600">
      <p className="text-xs text-gray-600 mb-2">Drag to crop · Enter to apply · Esc to cancel</p>
      <div className="flex flex-wrap gap-1">
        {['Free', '1:1', '4:3', '16:9', '3:2'].map(r => (
          <button key={r} className="text-xs bg-dark-600 hover:bg-dark-500 text-gray-300 px-2 py-0.5 rounded">{r}</button>
        ))}
      </div>
    </div>
  )

  if (activeTool === 'move') return (
    <div className="px-3 py-2 border-t border-dark-600">
      <p className="text-xs text-gray-600">Drag to move · Arrow keys to nudge · Shift+Arrow = 10px</p>
    </div>
  )

  return null
}

// ── Layer Opacity + Blend (shown when a layer is selected) ───
const BLEND_MODES: BlendMode[] = [
  'normal','multiply','screen','overlay','darken','lighten',
  'color-dodge','color-burn','hard-light','soft-light',
  'difference','exclusion','hue','saturation','color','luminosity',
]

const LayerOptions: React.FC = () => {
  const { layers, selectedLayerIds, updateLayer } = useEditorStore()
  const layer = layers.find(l => l.id === selectedLayerIds[0])
  const [showBlend, setShowBlend] = useState(false)
  const btnRef = useRef<HTMLButtonElement>(null)
  const [blendRect, setBlendRect] = useState<DOMRect | null>(null)

  if (!layer) return null

  return (
    <div className="px-3 py-2 border-t border-dark-600 space-y-2">
      <div className="text-xs text-gray-500 uppercase tracking-wider">Layer — {layer.name}</div>
      <Row label="Opacity">
        <input type="range" min={0} max={1} step={0.01} value={layer.opacity}
          onChange={e => updateLayer(layer.id, { opacity: parseFloat(e.target.value) })}
          className="flex-1 h-1" />
        <span className="text-xs text-white w-8 text-right">{Math.round(layer.opacity * 100)}%</span>
      </Row>
      <Row label="Blend">
        <div className="relative flex-1">
          <button
            ref={btnRef}
            onClick={e => {
              const rect = btnRef.current?.getBoundingClientRect() ?? null
              setBlendRect(rect); setShowBlend(v => !v)
            }}
            className="w-full text-left text-xs bg-dark-600 hover:bg-dark-500 text-gray-300 px-2 py-1 rounded"
          >
            {layer.blend_mode || 'normal'}
          </button>
          {showBlend && blendRect && (
            <div
              className="fixed bg-dark-700 border border-dark-500 rounded shadow-xl z-[9999] py-1 max-h-48 overflow-y-auto min-w-max"
              style={{ bottom: window.innerHeight - blendRect.top + 4, left: blendRect.left }}
            >
              {BLEND_MODES.map(bm => (
                <button key={bm}
                  onClick={() => { updateLayer(layer.id, { blend_mode: bm }); setShowBlend(false) }}
                  className={`block w-full text-left px-3 py-1 text-xs hover:bg-dark-600 ${layer.blend_mode === bm ? 'text-accent' : 'text-gray-300'}`}
                >
                  {bm}
                </button>
              ))}
            </div>
          )}
        </div>
      </Row>
    </div>
  )
}

// ── Main ToolPanel ────────────────────────────────────────────
export const ToolPanel: React.FC = () => {
  const { activeTool, setActiveTool } = useEditorStore()

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <div className="px-3 py-2 text-xs font-semibold text-gray-400 uppercase tracking-wider border-b border-dark-600 flex-shrink-0">
        Tools
      </div>

      <div className="flex-1 overflow-y-auto">
        {TOOL_GROUPS.map(group => (
          <div key={group.label} className="px-2 pt-2 pb-1">
            <div className="text-xs text-gray-600 uppercase tracking-wider mb-1 px-1">{group.label}</div>
            <div className="grid grid-cols-4 gap-1">
              {group.tools.map(t => {
                const isActive = activeTool === t.id
                return (
                  <button
                    key={t.id}
                    title={t.shortcut ? `${t.label} (${t.shortcut})` : t.label}
                    onClick={() => setActiveTool(t.id)}
                    className={`flex flex-col items-center justify-center gap-0.5 py-1.5 rounded text-xs transition-colors ${
                      isActive
                        ? 'bg-accent text-white'
                        : 'bg-dark-700 hover:bg-dark-600 text-gray-400 hover:text-white'
                    }`}
                  >
                    {t.icon}
                    <span className="text-[9px] leading-none truncate w-full text-center px-0.5">{t.label}</span>
                  </button>
                )
              })}
            </div>
          </div>
        ))}

        {/* Inline options for active tool */}
        <ToolOptions />

        {/* Layer opacity + blend — shown when a layer is selected */}
        <LayerOptions />
      </div>
    </div>
  )
}
