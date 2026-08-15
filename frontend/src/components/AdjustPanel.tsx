// AdjustPanel.tsx – Brightness, Contrast, Saturation, Exposure, Highlights, Shadows,
// Temperature, Tint, Hue, Sharpness, Clarity, Fade, Vignette, Grain
// All edits are non-destructive (stored in layer.adjustments, applied via CSS filter in Canvas)
import React from 'react'
import { useEditorStore } from '../store/editorStore'
import { LayerAdjustments, DEFAULT_ADJUSTMENTS } from '../types'

const Slider: React.FC<{
  label: string; value: number; min: number; max: number; step?: number
  onChange: (v: number) => void; defaultVal?: number
}> = ({ label, value, min, max, step = 1, onChange, defaultVal }) => (
  <div className="flex items-center gap-2">
    <span className="text-xs text-gray-400 w-20 flex-shrink-0">{label}</span>
    <input type="range" min={min} max={max} step={step} value={value}
      onChange={e => onChange(parseFloat(e.target.value))} className="flex-1 h-1" />
    <span className="text-xs text-white w-8 text-right">{value}</span>
    {defaultVal !== undefined && value !== defaultVal && (
      <button onClick={() => onChange(defaultVal)} className="text-xs text-gray-500 hover:text-accent" title="Reset">↺</button>
    )}
  </div>
)

export const AdjustPanel: React.FC = () => {
  const { layers, selectedLayerIds, updateLayer, pushHistory } = useEditorStore()
  const layer = layers.find(l => selectedLayerIds[0] === l.id)
  if (!layer) return <p className="text-xs text-gray-600 text-center mt-4 px-3">Select a layer</p>

  const adj: LayerAdjustments = { ...DEFAULT_ADJUSTMENTS, ...layer.adjustments }

  const update = (key: keyof LayerAdjustments, value: number) => {
    updateLayer(layer.id, { adjustments: { ...adj, [key]: value } })
  }

  const resetAll = () => {
    pushHistory()
    updateLayer(layer.id, { adjustments: { ...DEFAULT_ADJUSTMENTS } })
  }

  return (
    <div className="p-3 space-y-2">
      <div className="flex justify-between items-center mb-2">
        <span className="text-xs font-semibold text-gray-300">Adjustments</span>
        <button onClick={resetAll} className="text-xs text-gray-500 hover:text-accent">Reset All</button>
      </div>

      <div className="text-xs text-gray-500 uppercase tracking-wider mb-1">Light</div>
      <Slider label="Brightness" value={adj.brightness} min={0} max={200} onChange={v => update('brightness', v)} defaultVal={100} />
      <Slider label="Contrast" value={adj.contrast} min={0} max={200} onChange={v => update('contrast', v)} defaultVal={100} />
      <Slider label="Exposure" value={adj.exposure} min={-100} max={100} onChange={v => update('exposure', v)} defaultVal={0} />
      <Slider label="Highlights" value={adj.highlights} min={-100} max={100} onChange={v => update('highlights', v)} defaultVal={0} />
      <Slider label="Shadows" value={adj.shadows} min={-100} max={100} onChange={v => update('shadows', v)} defaultVal={0} />

      <div className="text-xs text-gray-500 uppercase tracking-wider mt-3 mb-1">Color</div>
      <Slider label="Saturation" value={adj.saturation} min={0} max={200} onChange={v => update('saturation', v)} defaultVal={100} />
      <Slider label="Temperature" value={adj.temperature} min={-100} max={100} onChange={v => update('temperature', v)} defaultVal={0} />
      <Slider label="Tint" value={adj.tint} min={-100} max={100} onChange={v => update('tint', v)} defaultVal={0} />
      <Slider label="Hue" value={adj.hue} min={-180} max={180} onChange={v => update('hue', v)} defaultVal={0} />

      <div className="text-xs text-gray-500 uppercase tracking-wider mt-3 mb-1">Detail</div>
      <Slider label="Sharpness" value={adj.sharpness} min={0} max={200} onChange={v => update('sharpness', v)} defaultVal={100} />
      <Slider label="Clarity" value={adj.clarity} min={0} max={100} onChange={v => update('clarity', v)} defaultVal={0} />

      <div className="text-xs text-gray-500 uppercase tracking-wider mt-3 mb-1">Effects</div>
      <Slider label="Fade" value={adj.fade} min={0} max={100} onChange={v => update('fade', v)} defaultVal={0} />
      <Slider label="Vignette" value={adj.vignette} min={0} max={100} onChange={v => update('vignette', v)} defaultVal={0} />
      <Slider label="Grain" value={adj.grain} min={0} max={100} onChange={v => update('grain', v)} defaultVal={0} />
    </div>
  )
}
