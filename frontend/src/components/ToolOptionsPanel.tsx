import React from 'react'
import { useEditorStore } from '../store/editorStore'

export const ToolOptionsPanel: React.FC = () => {
  const { activeTool, toolOptions, setToolOption } = useEditorStore()

  const DRAW_TOOLS = ['brush', 'pencil', 'marker', 'eraser', 'clone', 'heal']
  const SHAPE_TOOLS = ['shape_rect', 'shape_ellipse', 'shape_line', 'shape_arrow', 'shape_triangle', 'shape_star']
  const SELECT_TOOLS = ['rect_select', 'ellipse_select', 'lasso_select', 'free_select', 'magic_select']

  const Row: React.FC<{ label: string; children: React.ReactNode }> = ({ label, children }) => (
    <div className="flex items-center gap-2">
      <span className="text-xs text-gray-400 w-16 flex-shrink-0">{label}</span>
      {children}
    </div>
  )

  if (DRAW_TOOLS.includes(activeTool)) {
    return (
      <div className="p-3 space-y-2">
        <div className="text-xs font-semibold text-gray-300 mb-2 capitalize">{activeTool} Options</div>
        {activeTool !== 'eraser' && (
          <Row label="Color">
            <input type="color" value={toolOptions.brushColor}
              onChange={e => setToolOption('brushColor', e.target.value)}
              className="w-8 h-6 rounded cursor-pointer border-0 bg-transparent" />
            <span className="text-xs text-gray-400">{toolOptions.brushColor}</span>
          </Row>
        )}
        <Row label={activeTool === 'eraser' ? 'Size' : 'Size'}>
          <input type="range" min={1} max={200}
            value={activeTool === 'eraser' ? toolOptions.eraserSize : toolOptions.brushSize}
            onChange={e => setToolOption(activeTool === 'eraser' ? 'eraserSize' : 'brushSize', parseInt(e.target.value))}
            className="flex-1 h-1" />
          <span className="text-xs text-white w-10 text-right">
            {activeTool === 'eraser' ? toolOptions.eraserSize : toolOptions.brushSize}px
          </span>
        </Row>
        {activeTool !== 'eraser' && (
          <Row label="Opacity">
            <input type="range" min={0.01} max={1} step={0.01} value={toolOptions.brushOpacity}
              onChange={e => setToolOption('brushOpacity', parseFloat(e.target.value))}
              className="flex-1 h-1" />
            <span className="text-xs text-white w-10 text-right">{Math.round(toolOptions.brushOpacity * 100)}%</span>
          </Row>
        )}
        {activeTool === 'brush' && (
          <Row label="Hardness">
            <input type="range" min={0} max={1} step={0.05} value={toolOptions.brushHardness}
              onChange={e => setToolOption('brushHardness', parseFloat(e.target.value))}
              className="flex-1 h-1" />
            <span className="text-xs text-white w-10 text-right">{Math.round(toolOptions.brushHardness * 100)}%</span>
          </Row>
        )}
        {activeTool === 'clone' && (
          <p className="text-xs text-gray-500 mt-2">Alt+Click to set clone source, then paint to clone.</p>
        )}
        {activeTool === 'heal' && (
          <p className="text-xs text-gray-500 mt-2">Paint over blemishes to heal them.</p>
        )}
      </div>
    )
  }

  if (SHAPE_TOOLS.includes(activeTool)) {
    return (
      <div className="p-3 space-y-2">
        <div className="text-xs font-semibold text-gray-300 mb-2">Shape Options</div>
        <Row label="Stroke">
          <input type="color" value={toolOptions.shapeStroke}
            onChange={e => setToolOption('shapeStroke', e.target.value)}
            className="w-8 h-6 rounded cursor-pointer border-0 bg-transparent" />
        </Row>
        <Row label="Fill">
          <input type="color"
            value={toolOptions.shapeFill === 'transparent' ? '#000000' : toolOptions.shapeFill}
            onChange={e => setToolOption('shapeFill', e.target.value)}
            className="w-8 h-6 rounded cursor-pointer border-0 bg-transparent" />
          <label className="flex items-center gap-1 text-xs text-gray-400">
            <input type="checkbox" checked={toolOptions.shapeFill === 'transparent'}
              onChange={e => setToolOption('shapeFill', e.target.checked ? 'transparent' : '#000000')} />
            No fill
          </label>
        </Row>
        <Row label="Width">
          <input type="range" min={1} max={50} value={toolOptions.shapeStrokeWidth}
            onChange={e => setToolOption('shapeStrokeWidth', parseInt(e.target.value))}
            className="flex-1 h-1" />
          <span className="text-xs text-white w-10 text-right">{toolOptions.shapeStrokeWidth}px</span>
        </Row>
        <Row label="Opacity">
          <input type="range" min={0.01} max={1} step={0.01} value={toolOptions.shapeOpacity}
            onChange={e => setToolOption('shapeOpacity', parseFloat(e.target.value))}
            className="flex-1 h-1" />
          <span className="text-xs text-white w-10 text-right">{Math.round(toolOptions.shapeOpacity * 100)}%</span>
        </Row>
      </div>
    )
  }

  if (SELECT_TOOLS.includes(activeTool)) {
    return (
      <div className="p-3 space-y-2">
        <div className="text-xs font-semibold text-gray-300 mb-2">Selection Options</div>
        <Row label="Feather">
          <input type="range" min={0} max={50} value={toolOptions.selectionFeather}
            onChange={e => setToolOption('selectionFeather', parseInt(e.target.value))}
            className="flex-1 h-1" />
          <span className="text-xs text-white w-10 text-right">{toolOptions.selectionFeather}px</span>
        </Row>
        <p className="text-xs text-gray-500 mt-1">Esc to clear selection.</p>
      </div>
    )
  }

  if (activeTool === 'crop') {
    return (
      <div className="p-3 space-y-2">
        <div className="text-xs font-semibold text-gray-300 mb-2">Crop</div>
        <p className="text-xs text-gray-500">Drag on canvas to set crop area.<br />Enter to apply · Esc to cancel.</p>
        <div className="mt-3 space-y-1">
          {['Free', '1:1', '4:3', '16:9', '3:2', '9:16'].map(label => (
            <button key={label} className="w-full text-left text-xs bg-dark-600 hover:bg-dark-500 text-gray-300 px-2 py-1 rounded">
              {label}
            </button>
          ))}
        </div>
      </div>
    )
  }

  if (activeTool === 'color_picker') {
    return (
      <div className="p-3 space-y-2">
        <div className="text-xs font-semibold text-gray-300 mb-2">Color Picker</div>
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded border border-dark-500" style={{ background: toolOptions.brushColor }} />
          <span className="text-xs text-white font-mono">{toolOptions.brushColor}</span>
        </div>
        <p className="text-xs text-gray-500 mt-1">Click on canvas to pick a color.</p>
      </div>
    )
  }

  if (activeTool === 'text_add') {
    return (
      <div className="p-3 space-y-2">
        <div className="text-xs font-semibold text-gray-300 mb-2">Text Tool</div>
        <p className="text-xs text-gray-500">Click on canvas to place a text layer.<br />Use the AI Edit tab to detect &amp; edit existing text.</p>
      </div>
    )
  }

  if (activeTool === 'move') {
    return (
      <div className="p-3 space-y-2">
        <div className="text-xs font-semibold text-gray-300 mb-2">Move Tool</div>
        <p className="text-xs text-gray-500">Drag layers to reposition.<br />Alt+drag or middle-mouse to pan canvas.<br />Scroll to zoom.</p>
      </div>
    )
  }

  return (
    <div className="p-3">
      <p className="text-xs text-gray-600">Select a tool to see options.</p>
    </div>
  )
}
