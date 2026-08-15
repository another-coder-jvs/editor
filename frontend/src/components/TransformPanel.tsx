// TransformPanel.tsx – Resize, Rotate, Flip, Scale, Position controls
import React, { useState, useEffect } from 'react'
import { useEditorStore } from '../store/editorStore'
import { baseImagesUrl } from '../config'

const API_BASE = baseImagesUrl || 'http://localhost:8000'

const Slider: React.FC<{
  label: string; value: number; min: number; max: number; step?: number
  onChange: (v: number) => void; display?: string
}> = ({ label, value, min, max, step = 1, onChange, display }) => (
  <div>
    <div className="flex justify-between text-xs text-gray-400 mb-0.5">
      <span>{label}</span><span className="text-white">{display ?? value}</span>
    </div>
    <input type="range" min={min} max={max} step={step} value={value}
      onChange={e => onChange(parseFloat(e.target.value))} className="w-full h-1" />
  </div>
)

export const TransformPanel: React.FC = () => {
  const { layers, selectedLayerIds, updateLayer, pushHistory } = useEditorStore()
  const layer = layers.find(l => selectedLayerIds[0] === l.id)
  if (!layer) return <p className="text-xs text-gray-600 text-center mt-4 px-3">Select a layer</p>

  const [resizeW, setResizeW] = useState(layer.bbox.width)
  const [resizeH, setResizeH] = useState(layer.bbox.height)
  const [keepAspect, setKeepAspect] = useState(true)
  const aspect = layer.bbox.width / (layer.bbox.height || 1)

  // Sync resize inputs when selected layer changes
  useEffect(() => {
    setResizeW(layer.bbox.width)
    setResizeH(layer.bbox.height)
  }, [layer.id, layer.bbox.width, layer.bbox.height])

  const applyResize = async () => {
    const offscreen = document.createElement('canvas')
    offscreen.width = resizeW
    offscreen.height = resizeH
    const ctx = offscreen.getContext('2d')!
    const url = layer.png_path.startsWith('blob:') || layer.png_path.startsWith('data:')
      ? layer.png_path : `${API_BASE}${layer.png_path}`
    await new Promise<void>(resolve => {
      const img = new Image(); img.crossOrigin = 'anonymous'
      img.onload = () => { ctx.drawImage(img, 0, 0, resizeW, resizeH); resolve() }
      img.onerror = () => resolve()
      img.src = url
    })
    pushHistory()
    updateLayer(layer.id, {
      png_path: offscreen.toDataURL('image/png'),
      bbox: { ...layer.bbox, width: resizeW, height: resizeH },
    })
  }

  const flipH = async () => {
    const offscreen = document.createElement('canvas')
    offscreen.width = layer.bbox.width; offscreen.height = layer.bbox.height
    const ctx = offscreen.getContext('2d')!
    const url = layer.png_path.startsWith('blob:') || layer.png_path.startsWith('data:')
      ? layer.png_path : `${API_BASE}${layer.png_path}`
    await new Promise<void>(resolve => {
      const img = new Image(); img.crossOrigin = 'anonymous'
      img.onload = () => {
        ctx.translate(layer.bbox.width, 0); ctx.scale(-1, 1)
        ctx.drawImage(img, 0, 0); resolve()
      }
      img.onerror = () => resolve(); img.src = url
    })
    pushHistory()
    updateLayer(layer.id, { png_path: offscreen.toDataURL('image/png') })
  }

  const flipV = async () => {
    const offscreen = document.createElement('canvas')
    offscreen.width = layer.bbox.width; offscreen.height = layer.bbox.height
    const ctx = offscreen.getContext('2d')!
    const url = layer.png_path.startsWith('blob:') || layer.png_path.startsWith('data:')
      ? layer.png_path : `${API_BASE}${layer.png_path}`
    await new Promise<void>(resolve => {
      const img = new Image(); img.crossOrigin = 'anonymous'
      img.onload = () => {
        ctx.translate(0, layer.bbox.height); ctx.scale(1, -1)
        ctx.drawImage(img, 0, 0); resolve()
      }
      img.onerror = () => resolve(); img.src = url
    })
    pushHistory()
    updateLayer(layer.id, { png_path: offscreen.toDataURL('image/png') })
  }

  const rotate90 = async (deg: 90 | -90) => {
    const w = layer.bbox.height, h = layer.bbox.width
    const offscreen = document.createElement('canvas')
    offscreen.width = w; offscreen.height = h
    const ctx = offscreen.getContext('2d')!
    const url = layer.png_path.startsWith('blob:') || layer.png_path.startsWith('data:')
      ? layer.png_path : `${API_BASE}${layer.png_path}`
    await new Promise<void>(resolve => {
      const img = new Image(); img.crossOrigin = 'anonymous'
      img.onload = () => {
        ctx.translate(w / 2, h / 2)
        ctx.rotate((deg * Math.PI) / 180)
        ctx.drawImage(img, -layer.bbox.width / 2, -layer.bbox.height / 2)
        resolve()
      }
      img.onerror = () => resolve(); img.src = url
    })
    pushHistory()
    updateLayer(layer.id, {
      png_path: offscreen.toDataURL('image/png'),
      bbox: { ...layer.bbox, width: w, height: h },
    })
  }

  return (
    <div className="p-3 space-y-3">
      <div className="text-xs font-semibold text-gray-300">Transform</div>

      {/* Position */}
      <div className="space-y-1">
        <div className="text-xs text-gray-500 uppercase tracking-wider">Position</div>
        <div className="grid grid-cols-2 gap-1">
          <div className="flex items-center gap-1">
            <span className="text-xs text-gray-400 w-4">X</span>
            <input type="number" value={Math.round(layer.position.x)}
              onChange={e => updateLayer(layer.id, { position: { ...layer.position, x: parseInt(e.target.value) || 0 } })}
              className="flex-1 bg-dark-600 text-xs text-white rounded px-1 py-0.5 border border-dark-500 outline-none" />
          </div>
          <div className="flex items-center gap-1">
            <span className="text-xs text-gray-400 w-4">Y</span>
            <input type="number" value={Math.round(layer.position.y)}
              onChange={e => updateLayer(layer.id, { position: { ...layer.position, y: parseInt(e.target.value) || 0 } })}
              className="flex-1 bg-dark-600 text-xs text-white rounded px-1 py-0.5 border border-dark-500 outline-none" />
          </div>
        </div>
      </div>

      {/* Rotation */}
      <Slider label="Rotation" value={layer.rotation} min={-180} max={180}
        onChange={v => updateLayer(layer.id, { rotation: v })} display={`${layer.rotation}°`} />

      {/* Scale */}
      <div className="space-y-1">
        <div className="text-xs text-gray-500 uppercase tracking-wider">Scale</div>
        <Slider label="Scale X" value={layer.scale.x} min={0.1} max={5} step={0.01}
          onChange={v => updateLayer(layer.id, { scale: { ...layer.scale, x: v } })} display={layer.scale.x.toFixed(2)} />
        <Slider label="Scale Y" value={layer.scale.y} min={0.1} max={5} step={0.01}
          onChange={v => updateLayer(layer.id, { scale: { ...layer.scale, y: v } })} display={layer.scale.y.toFixed(2)} />
      </div>

      {/* Resize */}
      <div className="space-y-1">
        <div className="text-xs text-gray-500 uppercase tracking-wider">Resize (px)</div>
        <div className="grid grid-cols-2 gap-1">
          <div className="flex items-center gap-1">
            <span className="text-xs text-gray-400 w-4">W</span>
            <input type="number" min={1} value={resizeW}
              onChange={e => {
                const v = parseInt(e.target.value) || 1
                setResizeW(v)
                if (keepAspect) setResizeH(Math.round(v / aspect))
              }}
              className="flex-1 bg-dark-600 text-xs text-white rounded px-1 py-0.5 border border-dark-500 outline-none" />
          </div>
          <div className="flex items-center gap-1">
            <span className="text-xs text-gray-400 w-4">H</span>
            <input type="number" min={1} value={resizeH}
              onChange={e => {
                const v = parseInt(e.target.value) || 1
                setResizeH(v)
                if (keepAspect) setResizeW(Math.round(v * aspect))
              }}
              className="flex-1 bg-dark-600 text-xs text-white rounded px-1 py-0.5 border border-dark-500 outline-none" />
          </div>
        </div>
        <label className="flex items-center gap-1 text-xs text-gray-400">
          <input type="checkbox" checked={keepAspect} onChange={e => setKeepAspect(e.target.checked)} />
          Keep aspect ratio
        </label>
        <button onClick={applyResize} className="w-full bg-dark-600 hover:bg-dark-500 text-gray-300 text-xs py-1 rounded">
          Apply Resize
        </button>
      </div>

      {/* Flip & Rotate */}
      <div className="space-y-1">
        <div className="text-xs text-gray-500 uppercase tracking-wider">Flip & Rotate</div>
        <div className="grid grid-cols-2 gap-1">
          <button onClick={flipH} className="bg-dark-600 hover:bg-dark-500 text-gray-300 text-xs py-1 rounded">↔ Flip H</button>
          <button onClick={flipV} className="bg-dark-600 hover:bg-dark-500 text-gray-300 text-xs py-1 rounded">↕ Flip V</button>
          <button onClick={() => rotate90(-90)} className="bg-dark-600 hover:bg-dark-500 text-gray-300 text-xs py-1 rounded">↺ 90° CCW</button>
          <button onClick={() => rotate90(90)} className="bg-dark-600 hover:bg-dark-500 text-gray-300 text-xs py-1 rounded">↻ 90° CW</button>
        </div>
      </div>
    </div>
  )
}
