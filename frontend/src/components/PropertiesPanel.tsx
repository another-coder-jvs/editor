import React, { useState, useEffect } from 'react'
import { useEditorStore } from '../store/editorStore'
import { editLayer } from '../api/client'
import { toast } from 'react-toastify'
import { baseUrl } from '../config'
import { Type } from 'lucide-react'

const STYLE_PRESETS = [
  { label: 'Cartoon', prompt: 'cartoon style' },
  { label: 'Anime', prompt: 'anime style' },
  { label: 'Oil Paint', prompt: 'oil painting style' },
  { label: 'Sketch', prompt: 'pencil sketch' },
  { label: 'Pixel Art', prompt: 'pixel art style' },
]

interface TextRegion { bbox: number[]; text: string; color: number[]; font_size: number }

export const PropertiesPanel: React.FC = () => {
  const {
    layers, selectedLayerIds, updateLayer, addLayer,
    sessionId, originalImagePath, pushHistory, setProgress,
  } = useEditorStore()

  const [prompt, setPrompt] = useState('')
  const [strength, setStrength] = useState(0.75)
  const [steps, setSteps] = useState(20)
  const [isEditing, setIsEditing] = useState(false)

  // Text editing state
  const [textRegionsByLayer, setTextRegionsByLayer] = useState<Record<string, TextRegion[]>>({})
  const [textEditsByLayer, setTextEditsByLayer] = useState<Record<string, Record<number, string>>>({})
  const [previewLayerIds, setPreviewLayerIds] = useState<Record<number, string>>({})
  const [detectingText, setDetectingText] = useState(false)

  const selectedLayer = layers.find((l) => selectedLayerIds[0] === l.id)

  const textRegions = selectedLayer ? (textRegionsByLayer[selectedLayer.id] ?? []) : []
  const textEdits   = selectedLayer ? (textEditsByLayer[selectedLayer.id]   ?? {}) : {}

  const setTextRegions = (regions: TextRegion[]) => {
    if (!selectedLayer) return
    setTextRegionsByLayer(prev => ({ ...prev, [selectedLayer.id]: regions }))
  }
  const setTextEdits = (edits: Record<number, string>) => {
    if (!selectedLayer) return
    setTextEditsByLayer(prev => ({ ...prev, [selectedLayer.id]: edits }))
  }

  // Reset text state when layer changes
  const originalLayerPngByLayer = React.useRef<Record<string, string>>({})

  const handleTextChangeRef = React.useRef<(i: number, value: string) => void>(() => {})

  // Reset preview layer IDs when switching layers
  useEffect(() => {
    setPreviewLayerIds({})
  }, [selectedLayer?.id])

  // Listen for inline canvas text edits (double-click on text layer)
  useEffect(() => {
    const handler = (e: Event) => {
      const { layerId, newText } = (e as CustomEvent).detail
      const match = layerId.match(/_textpreview_(\d+)$/)
      if (!match) return
      handleTextChangeRef.current(parseInt(match[1]), newText)
    }
    window.addEventListener('canvas-text-edit', handler)
    return () => window.removeEventListener('canvas-text-edit', handler)
  }, [])

  // Listen for inline canvas text edits (double-click on text layer)
  useEffect(() => {
    const handler = (e: Event) => {
      const { layerId, newText } = (e as CustomEvent).detail
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
      if (!data.regions?.length) toast.info('No text detected in this layer')
    } catch { toast.error('Text detection failed') }
    finally { setDetectingText(false) }
  }

  const _loadLayerImg = async () => {
    if (!selectedLayer) return null
    const lid = selectedLayer.id
    const srcUrl = originalLayerPngByLayer.current[lid]
      ?? (selectedLayer.png_path.startsWith('blob:') || selectedLayer.png_path.startsWith('data:')
          ? selectedLayer.png_path : `${baseUrl}${selectedLayer.png_path}`)
    const blob = await fetch(srcUrl, { headers: { 'ngrok-skip-browser-warning': '1' } }).then(r => r.blob())
    const objectUrl = URL.createObjectURL(blob)
    const img = await new Promise<HTMLImageElement>((res, rej) => {
      const el = new Image(); el.onload = () => res(el); el.onerror = rej; el.src = objectUrl
    })
    URL.revokeObjectURL(objectUrl)
    return img
  }

  // Inpaint background for all edited regions, return bgUrl blob
  const _renderBg = async (img: HTMLImageElement, currentEdits: Record<number, string>) => {
    const W = img.naturalWidth, H = img.naturalHeight
    const offX = selectedLayer!.bbox.x, offY = selectedLayer!.bbox.y
    const bgCanvas = document.createElement('canvas')
    bgCanvas.width = W; bgCanvas.height = H
    const bgCtx = bgCanvas.getContext('2d')!
    bgCtx.drawImage(img, 0, 0)
    for (const [i] of Object.entries(currentEdits).filter(([, v]) => v.trim())) {
      const r = textRegions[+i]; if (!r) continue
      const lx1 = r.bbox[0] - offX, ly1 = r.bbox[1] - offY
      const lx2 = r.bbox[2] - offX, ly2 = r.bbox[3] - offY
      const w = lx2 - lx1, h = ly2 - ly1
      if (w < 1 || h < 1) continue
      const pad = Math.max(4, Math.round(Math.min(w, h) * 0.15))
      const sx1 = Math.max(0, lx1 - pad), sy1 = Math.max(0, ly1 - pad)
      const sx2 = Math.min(W, lx2 + pad), sy2 = Math.min(H, ly2 + pad)
      const borderData = bgCtx.getImageData(sx1, sy1, sx2 - sx1, sy2 - sy1)
      const bd = borderData.data; const bw = sx2 - sx1, bh = sy2 - sy1
      let rSum = 0, gSum = 0, bSum = 0, count = 0
      for (let py = 0; py < bh; py++) for (let px = 0; px < bw; px++) {
        const absX = sx1 + px, absY = sy1 + py
        if (absX >= lx1 && absX < lx2 && absY >= ly1 && absY < ly2) continue
        const idx = (py * bw + px) * 4
        rSum += bd[idx]; gSum += bd[idx + 1]; bSum += bd[idx + 2]; count++
      }
      if (count === 0) { bgCtx.clearRect(lx1, ly1, w, h); continue }
      const avgR = rSum / count, avgG = gSum / count, avgB = bSum / count
      const textArea = bgCtx.getImageData(lx1, ly1, w, h); const td = textArea.data
      for (let py = 0; py < h; py++) for (let px = 0; px < w; px++) {
        const nearTop = py, nearBot = h - 1 - py, nearLeft = px, nearRight = w - 1 - px
        const minDist = Math.min(nearTop, nearBot, nearLeft, nearRight)
        const blendT = Math.min(1, minDist / (pad + 1))
        let bpx = px + lx1 - sx1, bpy = py + ly1 - sy1
        if (nearTop <= minDist) bpy = 0
        else if (nearBot <= minDist) bpy = bh - 1
        else if (nearLeft <= minDist) bpx = 0
        else bpx = bw - 1
        bpx = Math.max(0, Math.min(bw - 1, bpx)); bpy = Math.max(0, Math.min(bh - 1, bpy))
        const bidx = (bpy * bw + bpx) * 4
        const idx = (py * w + px) * 4
        td[idx]   = Math.round(bd[bidx]   * (1 - blendT) + avgR * blendT)
        td[idx+1] = Math.round(bd[bidx+1] * (1 - blendT) + avgG * blendT)
        td[idx+2] = Math.round(bd[bidx+2] * (1 - blendT) + avgB * blendT)
      }
      bgCtx.putImageData(textArea, lx1, ly1)
    }
    return new Promise<string>(res => bgCanvas.toBlob(b => res(URL.createObjectURL(b!)), 'image/png'))
  }

  // Render a single text region as a tight-cropped transparent PNG blob URL
  const _renderTextPng = (r: TextRegion, newText: string, offX: number, offY: number): string => {
    const lx1 = r.bbox[0] - offX, ly1 = r.bbox[1] - offY
    const lx2 = r.bbox[2] - offX, ly2 = r.bbox[3] - offY
    const w = Math.max(1, lx2 - lx1), h = Math.max(1, ly2 - ly1)
    const c = document.createElement('canvas'); c.width = w; c.height = h
    const ctx = c.getContext('2d')!
    const [tr, tg, tb] = r.color
    ctx.fillStyle = `rgb(${tr},${tg},${tb})`
    ctx.font = `bold ${r.font_size}px sans-serif`
    ctx.textBaseline = 'middle'; ctx.textAlign = 'center'
    ctx.fillText(newText, w / 2, h / 2, w)
    return c.toDataURL('image/png')  // sync, no async needed for preview
  }

  // Live preview on every keystroke
  const handleTextChange = async (i: number, value: string) => {
    const next = { ...textEdits, [i]: value }
    setTextEdits(next)
    if (!selectedLayer) return
    if (!originalLayerPngByLayer.current[selectedLayer.id]) {
      originalLayerPngByLayer.current[selectedLayer.id] = selectedLayer.png_path.startsWith('blob:') || selectedLayer.png_path.startsWith('data:')
        ? selectedLayer.png_path : `${baseUrl}${selectedLayer.png_path}`
    }
    const img = await _loadLayerImg(); if (!img) return
    const offX = selectedLayer.bbox.x, offY = selectedLayer.bbox.y
    const bgUrl = await _renderBg(img, next)
    updateLayer(selectedLayer.id, { png_path: bgUrl })

    if (!value.trim()) return
    const r = textRegions[i]; if (!r) return
    const txtDataUrl = _renderTextPng(r, value, offX, offY)
    const lx1 = r.bbox[0] - offX, ly1 = r.bbox[1] - offY
    const w = r.bbox[2] - r.bbox[0], h = r.bbox[3] - r.bbox[1]

    const existingPid = previewLayerIds[i]
    if (existingPid && layers.find(l => l.id === existingPid)) {
      updateLayer(existingPid, { png_path: txtDataUrl })
    } else {
      const pid = `${selectedLayer.id}_textpreview_${i}`
      setPreviewLayerIds(prev => ({ ...prev, [i]: pid }))
      addLayer({
        ...selectedLayer,
        id: pid,
        name: `text: ${r.text}`,
        png_path: txtDataUrl,
        bbox: { x: selectedLayer.bbox.x + lx1, y: selectedLayer.bbox.y + ly1, width: w, height: h },
        position: { x: 0, y: 0 },
        scale: { x: 1, y: 1 },
        rotation: 0,
        z_index: selectedLayer.z_index + 1 + i,
        history: [],
        locked: false,
      })
    }
  }
  handleTextChangeRef.current = handleTextChange

  const handleApplyTextEdits = async () => {
    if (!selectedLayer || !sessionId) return
    const edits = Object.entries(textEdits).filter(([, v]) => v.trim())
    if (!edits.length) return
    setIsEditing(true)
    pushHistory()
    try {
      const img = await _loadLayerImg(); if (!img) return
      const offX = selectedLayer.bbox.x, offY = selectedLayer.bbox.y
      const bgUrl = await _renderBg(img, textEdits)
      const textPrompt = edits.map(([i, v]) => `"${textRegions[+i]?.text}" → "${v}"`).join(', ')
      updateLayer(selectedLayer.id, { png_path: bgUrl, history: [...selectedLayer.history, textPrompt] })

      // Finalize each preview layer (already positioned correctly)
      for (const [i, newText] of edits) {
        const r = textRegions[+i]; if (!r) continue
        const txtDataUrl = _renderTextPng(r, newText, offX, offY)
        const pid = previewLayerIds[+i]
        if (pid && layers.find(l => l.id === pid)) {
          updateLayer(pid, { png_path: txtDataUrl, history: [textPrompt] })
        }
      }
      toast.success('Text updated!')
      setTextRegions([])
      setTextEdits({})
      setPreviewLayerIds({})
      if (selectedLayer) delete originalLayerPngByLayer.current[selectedLayer.id]
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
                {textRegions.map((r, i) => (
                  <div key={i} className="bg-dark-700 rounded p-2">
                    <p className="text-xs text-gray-500 mb-1 truncate">Detected: "{r.text}"</p>
                    <input
                      className="w-full bg-dark-600 text-sm text-white rounded px-2 py-1 border border-dark-500 focus:border-accent outline-none"
                      placeholder={r.text}
                      value={textEdits[i] ?? ''}
                      onChange={e => handleTextChange(i, e.target.value)}
                    />
                    <div className="flex items-center gap-1 mt-1">
                      <span className="text-xs text-gray-500">Color:</span>
                      <div className="w-4 h-4 rounded border border-dark-500"
                        style={{ background: `rgb(${r.color[0]},${r.color[1]},${r.color[2]})` }} />
                      <span className="text-xs text-gray-500">Size: {r.font_size}px</span>
                    </div>
                  </div>
                ))}
                <button onClick={handleApplyTextEdits} disabled={isEditing || !Object.values(textEdits).some(v => v.trim())}
                  className="w-full bg-accent hover:bg-accent-hover text-white text-sm py-1.5 rounded disabled:opacity-40">
                  {isEditing ? 'Applying…' : 'Apply Text Changes'}
                </button>
              </div>
            )}
          </div>

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
