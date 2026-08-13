import React, { useState, useEffect, useRef } from 'react'
import { useEditorStore } from '../store/editorStore'
import { editLayer } from '../api/client'
import { toast } from 'react-toastify'
import { baseUrl } from '../config'
import { Type, Pipette } from 'lucide-react'

const hexToRgb = (hex: string): number[] => {
  const r = parseInt(hex.slice(1, 3), 16)
  const g = parseInt(hex.slice(3, 5), 16)
  const b = parseInt(hex.slice(5, 7), 16)
  return [r, g, b]
}

const STYLE_PRESETS = [
  { label: 'Cartoon', prompt: 'cartoon style' },
  { label: 'Anime', prompt: 'anime style' },
  { label: 'Oil Paint', prompt: 'oil painting style' },
  { label: 'Sketch', prompt: 'pencil sketch' },
  { label: 'Pixel Art', prompt: 'pixel art style' },
]

interface TextRegion { bbox: number[]; text: string; color: number[]; font_size: number }
interface TextStyle { color: string; font_size: number; shadow: boolean; shadow_color: string; shadow_offset: [number,number]; rotation: number }

export const PropertiesPanel: React.FC = () => {
  const {
    layers, selectedLayerIds, updateLayer, addLayer,
    sessionId, originalImagePath, pushHistory, setProgress,
    setDetectedTextRegions, setTextOverlay, clearTextOverlays,
  } = useEditorStore()

  const [prompt, setPrompt] = useState('')
  const [strength, setStrength] = useState(0.75)
  const [steps, setSteps] = useState(20)
  const [isEditing, setIsEditing] = useState(false)

  // Text editing state
  const [textRegionsByLayer, setTextRegionsByLayer] = useState<Record<string, TextRegion[]>>({})
  const [textEditsByLayer, setTextEditsByLayer] = useState<Record<string, Record<number, string>>>({})
  const [textStylesByLayer, setTextStylesByLayer] = useState<Record<string, Record<number, TextStyle>>>({})
  const [detectingText, setDetectingText] = useState(false)
  const [eyedropper, setEyedropper] = useState<{ idx: number; field: 'color' | 'shadow_color' } | null>(null)
  // const [previewLayerIds, setPreviewLayerIds] = useState<Record<number, string>>({})

  const selectedLayer = layers.find((l) => selectedLayerIds[0] === l.id)

  const textRegions = selectedLayer ? (textRegionsByLayer[selectedLayer.id] ?? []) : []
  const textEdits   = selectedLayer ? (textEditsByLayer[selectedLayer.id]   ?? {}) : {}
  const textStyles  = selectedLayer ? (textStylesByLayer[selectedLayer.id]  ?? {}) : {}

  const setTextRegions = (regions: TextRegion[]) => {
    if (!selectedLayer) return
    setTextRegionsByLayer(prev => ({ ...prev, [selectedLayer.id]: regions }))
  }
  const setTextEdits = (edits: Record<number, string>) => {
    if (!selectedLayer) return
    setTextEditsByLayer(prev => ({ ...prev, [selectedLayer.id]: edits }))
  }
  const setTextStyles = (styles: Record<number, TextStyle>) => {
    if (!selectedLayer) return
    setTextStylesByLayer(prev => ({ ...prev, [selectedLayer.id]: styles }))
  }
  const updateTextStyle = (i: number, patch: Partial<TextStyle>) => {
    setTextStyles({ ...textStyles, [i]: { ...defaultStyle(textRegions[i]), ...textStyles[i], ...patch } })
  }
  const defaultStyle = (r?: TextRegion): TextStyle => ({
    color: r ? `rgb(${r.color[0]},${r.color[1]},${r.color[2]})` : '#ffffff',
    font_size: r?.font_size ?? 24,
    shadow: false,
    shadow_color: '#000000',
    shadow_offset: [2, 2],
    rotation: 0,
  })

  // Track created text layer IDs per region so re-apply updates instead of creates
  const txtLayerIds = React.useRef<Record<string, string>>({})

  // Reset text state when layer changes
  const originalLayerPngByLayer = React.useRef<Record<string, string>>({})

  // ── Text layer direct editor (when a txt_ layer is selected) ──────────────
  const isTxtLayer = selectedLayer?.id.includes('_txt_')
  const [txtLayerText, setTxtLayerText] = useState('')
  const [txtLayerColor, setTxtLayerColor] = useState('#ffffff')
  const [txtLayerFontSize, setTxtLayerFontSize] = useState(24)
  const [txtLayerWidth, setTxtLayerWidth] = useState(0)
  const [txtLayerHeight, setTxtLayerHeight] = useState(0)
  const [txtLayerRegion, setTxtLayerRegion] = useState<TextRegion | null>(null)

  // Populate controls when a txt_ layer is selected
  useEffect(() => {
    if (!selectedLayer || !isTxtLayer) return
    const name = selectedLayer.name.startsWith('text:') ? selectedLayer.name.slice(5) : selectedLayer.name
    setTxtLayerText(name)
    setTxtLayerWidth(selectedLayer.bbox.width)
    setTxtLayerHeight(selectedLayer.bbox.height)
    // Reconstruct a minimal region from the layer's bbox (full-image coords)
    setTxtLayerRegion({
      bbox: [selectedLayer.bbox.x, selectedLayer.bbox.y, selectedLayer.bbox.x + selectedLayer.bbox.width, selectedLayer.bbox.y + selectedLayer.bbox.height],
      text: name,
      color: [255, 255, 255],
      font_size: selectedLayer.bbox.height,
    })
    setTxtLayerFontSize(selectedLayer.bbox.height)
  }, [selectedLayer?.id])

  const handleUpdateTxtLayer = async () => {
    if (!selectedLayer || !sessionId || !txtLayerRegion) return
    setIsEditing(true)
    try {
      const region = { ...txtLayerRegion, bbox: [selectedLayer.bbox.x, selectedLayer.bbox.y, selectedLayer.bbox.x + txtLayerWidth, selectedLayer.bbox.y + txtLayerHeight], font_size: txtLayerFontSize }
      const overrides = { color: hexToRgb(txtLayerColor), font_size: txtLayerFontSize, shadow_color: null, shadow_offset: [2,2], rotation: selectedLayer.rotation }
      const res = await fetch(`${baseUrl}/text/render`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'ngrok-skip-browser-warning': '1' },
        body: JSON.stringify({ session_id: sessionId, region, new_text: txtLayerText, overrides }),
      })
      const data = await res.json()
      updateLayer(selectedLayer.id, {
        png_path: data.path,
        name: `text:${txtLayerText}`,
        bbox: { ...selectedLayer.bbox, width: txtLayerWidth, height: txtLayerHeight },
      })
      setTextOverlay(`txtlayer_${selectedLayer.id}`, null)
      toast.success('Text layer updated!')
    } catch (err: any) {
      toast.error('Update failed: ' + (err?.message || err))
    } finally { setIsEditing(false) }
  }

  // Live preview for txt_ layer editor
  useEffect(() => {
    if (!selectedLayer || !isTxtLayer) return
    const key = `txtlayer_${selectedLayer.id}`
    if (txtLayerText.trim()) {
      setTextOverlay(key, {
        text: txtLayerText,
        color: txtLayerColor,
        font_size: txtLayerFontSize,
        shadow: false,
        shadow_color: '#000000',
        rotation: selectedLayer.rotation,
        bbox: [selectedLayer.bbox.x, selectedLayer.bbox.y, selectedLayer.bbox.x + txtLayerWidth, selectedLayer.bbox.y + txtLayerHeight],
      })
    } else {
      setTextOverlay(key, null)
    }
  }, [txtLayerText, txtLayerColor, txtLayerFontSize, txtLayerWidth, txtLayerHeight])

  const handleTextChangeRef = React.useRef<(i: number, value: string) => void>(() => {})

  // Reset preview layer IDs when switching layers
  // useEffect(() => {
  //   setPreviewLayerIds({})
  // }, [selectedLayer?.id])

  // Listen for inline canvas text edits (double-click on text layer)
  useEffect(() => {
    const handler = (e: Event) => {
      const { layerId, newText } = (e as CustomEvent).detail
      // txt_ layer double-click edit
      if (layerId.includes('_txt_')) {
        setTxtLayerText(newText)
        return
      }
      const match = layerId.match(/_textpreview_(\d+)$/)
      if (!match) return
      handleTextChangeRef.current(parseInt(match[1]), newText)
    }
    window.addEventListener('canvas-text-edit', handler)
    return () => window.removeEventListener('canvas-text-edit', handler)
  }, [])

  const handleDetectText = async () => {
    if (!selectedLayer || !sessionId) return
    setDetectingText(true)
    try {
      const res = await fetch(`${baseUrl}/text/detect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'ngrok-skip-browser-warning': '1' },
        body: JSON.stringify({ session_id: sessionId, layer_id: selectedLayer.id, image_path: originalImagePath || '' }),
      })
      const data = await res.json()
      setTextRegions(data.regions || [])
      if (selectedLayer) setDetectedTextRegions(selectedLayer.id, data.regions || [])
      if (!data.regions?.length) toast.info('No text detected in this layer')
    } catch { toast.error('Text detection failed') }
    finally { setDetectingText(false) }
  }

  // ── helpers: call BE for bg erase and text render ──────────────────────────

  const _getOriginalLayerPath = (): string => {
    if (!selectedLayer) return ''
    const lid = selectedLayer.id
    if (!originalLayerPngByLayer.current[lid]) {
      originalLayerPngByLayer.current[lid] = selectedLayer.png_path
    }
    return originalLayerPngByLayer.current[lid]
  }

  const _callEraseBg = async (editedIndices: number[]): Promise<string> => {
    const regions = editedIndices.map(i => textRegions[i]).filter(Boolean)
    const res = await fetch(`${baseUrl}/text/erase-bg`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'ngrok-skip-browser-warning': '1' },
      body: JSON.stringify({
        session_id: sessionId,
        layer_id: selectedLayer!.id,
        image_path: _getOriginalLayerPath(),
        regions,
      }),
    })
    const data = await res.json()
    return data.path as string  // /temp/session/file.png
  }

  const _callRenderText = async (r: TextRegion, newText: string, style?: TextStyle): Promise<string> => {
    const overrides = {
      color: r.color,
      font_size: style?.font_size ?? r.font_size,
      shadow_color: style?.shadow ? hexToRgb(style.shadow_color) : null,
      shadow_offset: style?.shadow_offset ?? [2, 2],
      rotation: style?.rotation ?? 0,
    }
    const res = await fetch(`${baseUrl}/text/render`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'ngrok-skip-browser-warning': '1' },
      body: JSON.stringify({ session_id: sessionId, region: r, new_text: newText, overrides }),
    })
    const data = await res.json()
    return data.path as string
  }

  const handleTextChange = (i: number, value: string) => {
    setTextEdits({ ...textEdits, [i]: value })
    _pushOverlay(i, value, textStyles[i])
  }

  const _pushOverlay = (i: number, text: string, style?: Partial<TextStyle>) => {
    if (!selectedLayer || !textRegions[i]) return
    const r = textRegions[i]
    const s: TextStyle = { ...defaultStyle(r), ...textStyles[i], ...style }
    const key = `${selectedLayer.id}_${i}`
    if (text.trim()) {
      setTextOverlay(key, { text, color: s.color, font_size: s.font_size, shadow: s.shadow, shadow_color: s.shadow_color, rotation: s.rotation, bbox: r.bbox })
    } else {
      // clear overlay if text emptied
      setTextOverlay(key, null)
    }
  }

  // Re-render preview when style changes for any already-typed text
  useEffect(() => {
    Object.entries(textStyles).forEach(([idxStr, style]) => {
      const idx = +idxStr
      const text = textEdits[idx]
      if (text?.trim()) _pushOverlay(idx, text, style)
    })
  }, [textStyles])

  // canvas double-click → dispatches this
  handleTextChangeRef.current = (i: number, value: string) => {
    const next = { ...textEdits, [i]: value }
    setTextEdits(next)
    _pushOverlay(i, value)
  }

  const handleApplyTextEdits = async () => {
    if (!selectedLayer || !sessionId) return
    const edits = Object.entries(textEdits).filter(([, v]) => v.trim())
    if (!edits.length) return
    setIsEditing(true)
    pushHistory()
    try {
      const editedIndices = edits.map(([k]) => +k)
      const bgPath = await _callEraseBg(editedIndices)
      const textPrompt = edits.map(([i, v]) => `"${textRegions[+i]?.text}" → "${v}"`).join(', ')
      updateLayer(selectedLayer.id, { png_path: bgPath, history: [...selectedLayer.history, textPrompt] })

      for (const [i, newText] of edits) {
        const r = textRegions[+i]; if (!r) continue
        const txtPath = await _callRenderText(r, newText, { ...defaultStyle(r), ...textStyles[+i] })
        const offX = selectedLayer.bbox.x, offY = selectedLayer.bbox.y
        const lx1 = r.bbox[0] - offX, ly1 = r.bbox[1] - offY
        const w = r.bbox[2] - r.bbox[0], h = r.bbox[3] - r.bbox[1]
        const regionKey = `${selectedLayer.id}_${i}`
        const existingTxtId = txtLayerIds.current[regionKey]
        if (existingTxtId && layers.find(l => l.id === existingTxtId)) {
          updateLayer(existingTxtId, { png_path: txtPath, name: `text:${newText}` })
        } else {
          const newId = `${selectedLayer.id}_txt_${i}_${Date.now()}`
          txtLayerIds.current[regionKey] = newId
          addLayer({
            ...selectedLayer,
            id: newId,
            name: `text:${newText}`,
            png_path: txtPath,
            bbox: { x: selectedLayer.bbox.x + lx1, y: selectedLayer.bbox.y + ly1, width: w, height: h },
            position: { x: 0, y: 0 }, scale: { x: 1, y: 1 }, rotation: 0,
            z_index: selectedLayer.z_index + 1 + +i,
            history: [], locked: false,
          })
        }
      }
      clearTextOverlays(selectedLayer.id)
      toast.success('Text applied!')
      setTextEdits({})
      setTextStyles({})
      // update the "original" reference so next edit erases from the new result
      originalLayerPngByLayer.current[selectedLayer.id] = bgPath
    } catch (err: any) {
      toast.error('Text edit failed: ' + (err?.message || err))
    } finally { setIsEditing(false) }
  }

  const handleEdit = async () => {
    if (!selectedLayer || !sessionId || !originalImagePath || !prompt.trim()) return
    if (selectedLayer.locked) { toast.warn('Layer is locked'); return }

    setIsEditing(true)
    pushHistory()
    setProgress({ session_id: sessionId, task: 'edit', progress: 0, message: 'Editing…', done: false })

    try {
      const result = await editLayer({
        session_id: sessionId,
        layer_id: selectedLayer.id,
        prompt,
        image_path: originalImagePath,
        strength,
        steps,
      })
      updateLayer(selectedLayer.id, {
        png_path: result.edited_png_path,
        history: [...selectedLayer.history, prompt],
      })
      toast.success('Layer edited!')
      setPrompt('')
    } catch (err: unknown) {
      const detail = (err as any)?.response?.data?.detail
      const msg = detail || (err instanceof Error ? err.message : 'Edit failed')
      toast.error(msg, { autoClose: 6000 })
    } finally {
      setIsEditing(false)
      setProgress(null)
    }
  }

  return (
    <div className="w-60 bg-dark-800 border-l border-dark-600 flex flex-col overflow-y-auto">
      <div className="px-3 py-2 text-xs font-semibold text-gray-400 uppercase tracking-wider border-b border-dark-600">
        Properties
      </div>

      {!selectedLayer ? (
        <p className="text-xs text-gray-600 text-center mt-8 px-3">Select a layer to edit</p>
      ) : (
        <div className="p-3 flex flex-col gap-4">
          <div>
            <p className="text-xs text-gray-400 mb-1">Layer: <span className="text-white">{selectedLayer.name}</span></p>
          </div>

          <Slider label="Opacity" value={selectedLayer.opacity} min={0} max={1} step={0.01}
            onChange={(v) => updateLayer(selectedLayer.id, { opacity: v })}
            display={`${Math.round(selectedLayer.opacity * 100)}%`} />
          <Slider label="Rotation" value={selectedLayer.rotation} min={-180} max={180} step={1}
            onChange={(v) => updateLayer(selectedLayer.id, { rotation: v })}
            display={`${selectedLayer.rotation}°`} />
          <Slider label="Scale X" value={selectedLayer.scale.x} min={0.1} max={3} step={0.01}
            onChange={(v) => updateLayer(selectedLayer.id, { scale: { ...selectedLayer.scale, x: v } })}
            display={selectedLayer.scale.x.toFixed(2)} />
          <Slider label="Scale Y" value={selectedLayer.scale.y} min={0.1} max={3} step={0.01}
            onChange={(v) => updateLayer(selectedLayer.id, { scale: { ...selectedLayer.scale, y: v } })}
            display={selectedLayer.scale.y.toFixed(2)} />

          {/* ── Text Editor ── */}
          <div className="border-t border-dark-600 pt-3">
            <div className="flex items-center justify-between mb-2">
              <p className="text-xs text-gray-400 flex items-center gap-1"><Type size={12} /> Text Editor</p>
              <button onClick={handleDetectText} disabled={detectingText}
                className="text-xs bg-dark-600 hover:bg-dark-500 text-gray-300 px-2 py-0.5 rounded disabled:opacity-40">
                {detectingText ? 'Scanning…' : 'Detect Text'}
              </button>
            </div>

            {textRegions.length > 0 && (
              <div className="space-y-2">
                {textRegions.map((r, i) => {
                  const style: TextStyle = { ...defaultStyle(r), ...textStyles[i] }
                  return (
                  <div key={i} className="bg-dark-700 rounded p-2 space-y-1">
                    <p className="text-xs text-gray-500 truncate">Detected: "{r.text}"</p>
                    <input
                      className="w-full bg-dark-600 text-sm text-white rounded px-2 py-1 border border-dark-500 focus:border-accent outline-none"
                      placeholder={r.text}
                      value={textEdits[i] ?? ''}
                      onChange={e => handleTextChange(i, e.target.value)}
                    />
                    {/* Style controls */}
                    <div className="grid grid-cols-2 gap-1">
                      {/* Color */}
                      <div className="flex items-center gap-1">
                        <span className="text-xs text-gray-400 w-10">Color</span>
                        <input type="color" value={style.color} onChange={e => updateTextStyle(i, { color: e.target.value })}
                          className="w-6 h-6 rounded cursor-pointer border-0 bg-transparent" />
                        <button title="Pick from image" onClick={() => setEyedropper(eyedropper?.idx === i && eyedropper.field === 'color' ? null : { idx: i, field: 'color' })}
                          className={`p-0.5 rounded ${eyedropper?.idx === i && eyedropper.field === 'color' ? 'text-accent' : 'text-gray-400 hover:text-white'}`}>
                          <Pipette size={12} />
                        </button>
                      </div>
                      {/* Font size */}
                      <div className="flex items-center gap-1">
                        <span className="text-xs text-gray-400 w-10">Size</span>
                        <input type="number" min={6} max={300} value={style.font_size}
                          onChange={e => updateTextStyle(i, { font_size: parseInt(e.target.value) || style.font_size })}
                          className="w-14 bg-dark-600 text-xs text-white rounded px-1 py-0.5 border border-dark-500 outline-none" />
                      </div>
                      {/* Rotation */}
                      <div className="flex items-center gap-1">
                        <span className="text-xs text-gray-400 w-10">Rotate</span>
                        <input type="number" min={-180} max={180} value={style.rotation}
                          onChange={e => updateTextStyle(i, { rotation: parseInt(e.target.value) || 0 })}
                          className="w-14 bg-dark-600 text-xs text-white rounded px-1 py-0.5 border border-dark-500 outline-none" />
                      </div>
                      {/* Shadow toggle */}
                      <div className="flex items-center gap-1">
                        <span className="text-xs text-gray-400 w-10">Shadow</span>
                        <input type="checkbox" checked={style.shadow} onChange={e => updateTextStyle(i, { shadow: e.target.checked })} />
                        {style.shadow && (
                          <>
                            <input type="color" value={style.shadow_color} onChange={e => updateTextStyle(i, { shadow_color: e.target.value })}
                              className="w-5 h-5 rounded cursor-pointer border-0 bg-transparent" />
                            <button title="Pick shadow color from image" onClick={() => setEyedropper(eyedropper?.idx === i && eyedropper.field === 'shadow_color' ? null : { idx: i, field: 'shadow_color' })}
                              className={`p-0.5 rounded ${eyedropper?.idx === i && eyedropper.field === 'shadow_color' ? 'text-accent' : 'text-gray-400 hover:text-white'}`}>
                              <Pipette size={12} />
                            </button>
                          </>
                        )}
                      </div>
                    </div>
                    {/* Eyedropper canvas overlay */}
                    {eyedropper?.idx === i && (
                      <EyedropperCanvas onPick={(hex) => { updateTextStyle(i, { [eyedropper.field]: hex }); setEyedropper(null) }} />
                    )}
                  </div>
                  )
                })}
                <button onClick={handleApplyTextEdits} disabled={isEditing || !Object.values(textEdits).some(v => v.trim())}
                  className="w-full bg-accent hover:bg-accent-hover text-white text-sm py-1.5 rounded disabled:opacity-40">
                  {isEditing ? 'Applying…' : 'Apply Text Changes'}
                </button>
              </div>
            )}
          </div>

          {/* ── Text Layer Editor (when a rendered text layer is selected) ── */}
          {isTxtLayer && (
            <div className="border-t border-dark-600 pt-3 space-y-2">
              <p className="text-xs text-gray-400 flex items-center gap-1"><Type size={12} /> Edit Text Layer</p>
              <input
                className="w-full bg-dark-600 text-sm text-white rounded px-2 py-1 border border-dark-500 focus:border-accent outline-none"
                value={txtLayerText}
                onChange={e => setTxtLayerText(e.target.value)}
                placeholder="Text content"
              />
              <div className="grid grid-cols-2 gap-1">
                <div className="flex items-center gap-1">
                  <span className="text-xs text-gray-400 w-10">Color</span>
                  <input type="color" value={txtLayerColor} onChange={e => setTxtLayerColor(e.target.value)}
                    className="w-6 h-6 rounded cursor-pointer border-0 bg-transparent" />
                </div>
                <div className="flex items-center gap-1">
                  <span className="text-xs text-gray-400 w-10">Size</span>
                  <input type="number" min={6} max={300} value={txtLayerFontSize}
                    onChange={e => setTxtLayerFontSize(parseInt(e.target.value) || txtLayerFontSize)}
                    className="w-14 bg-dark-600 text-xs text-white rounded px-1 py-0.5 border border-dark-500 outline-none" />
                </div>
                <div className="flex items-center gap-1">
                  <span className="text-xs text-gray-400 w-10">Width</span>
                  <input type="number" min={1} value={txtLayerWidth}
                    onChange={e => setTxtLayerWidth(parseInt(e.target.value) || txtLayerWidth)}
                    className="w-14 bg-dark-600 text-xs text-white rounded px-1 py-0.5 border border-dark-500 outline-none" />
                </div>
                <div className="flex items-center gap-1">
                  <span className="text-xs text-gray-400 w-10">Height</span>
                  <input type="number" min={1} value={txtLayerHeight}
                    onChange={e => setTxtLayerHeight(parseInt(e.target.value) || txtLayerHeight)}
                    className="w-14 bg-dark-600 text-xs text-white rounded px-1 py-0.5 border border-dark-500 outline-none" />
                </div>
              </div>
              <button onClick={handleUpdateTxtLayer} disabled={isEditing}
                className="w-full bg-accent hover:bg-accent-hover text-white text-sm py-1.5 rounded disabled:opacity-40">
                {isEditing ? 'Updating…' : 'Update Text Layer'}
              </button>
            </div>
          )}

          {/* ── AI Edit ── */}
          <div className="border-t border-dark-600 pt-3">
            <p className="text-xs text-gray-400 mb-2">AI Edit Prompt</p>
            <textarea
              className="w-full bg-dark-700 text-sm text-white rounded p-2 resize-none border border-dark-500 focus:border-accent outline-none"
              rows={3}
              placeholder='e.g. "make shirt blue", "replace car with Ferrari"'
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter' && e.ctrlKey) handleEdit() }}
            />
            <div className="flex flex-wrap gap-1 mt-2">
              {STYLE_PRESETS.map((p) => (
                <button key={p.label} onClick={() => setPrompt(p.prompt)}
                  className="text-xs bg-dark-600 hover:bg-dark-500 text-gray-300 px-2 py-0.5 rounded">
                  {p.label}
                </button>
              ))}
            </div>
            <Slider label="Strength" value={strength} min={0.1} max={1} step={0.05}
              onChange={setStrength} display={strength.toFixed(2)} />
            <Slider label="Steps" value={steps} min={5} max={50} step={1}
              onChange={setSteps} display={String(steps)} />
            <button onClick={handleEdit} disabled={isEditing || !prompt.trim()}
              className="w-full mt-2 bg-accent hover:bg-accent-hover text-white text-sm py-2 rounded disabled:opacity-40 transition-colors">
              {isEditing ? 'Editing…' : 'Apply Edit (Ctrl+Enter)'}
            </button>
          </div>

          {selectedLayer.history.length > 0 && (
            <div className="border-t border-dark-600 pt-3">
              <p className="text-xs text-gray-400 mb-1">Edit History</p>
              <ul className="text-xs text-gray-500 space-y-0.5 max-h-24 overflow-y-auto">
                {selectedLayer.history.map((h, i) => <li key={i} className="truncate">• {h}</li>)}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

const Slider: React.FC<{
  label: string; value: number; min: number; max: number; step: number
  onChange: (v: number) => void; display: string
}> = ({ label, value, min, max, step, onChange, display }) => (
  <div>
    <div className="flex justify-between text-xs text-gray-400 mb-1">
      <span>{label}</span><span className="text-white">{display}</span>
    </div>
    <input type="range" min={min} max={max} step={step} value={value}
      onChange={(e) => onChange(parseFloat(e.target.value))} className="w-full" />
  </div>
)

const EyedropperCanvas: React.FC<{ onPick: (hex: string) => void }> = ({ onPick }) => {
  useEffect(() => {
    const pick = async () => {
      try {
        // @ts-ignore
        const dropper = new (window as any).EyeDropper()
        const result = await dropper.open()
        onPick(result.sRGBHex)
      } catch {}
    }
    pick()
  }, [])
  return null
}

