import React, { useState } from 'react'
import { useEditorStore } from '../store/editorStore'
import { exportImage, saveProject, loadProject } from '../api/client'
import { toast } from 'react-toastify'
import { X } from 'lucide-react'

interface Props {
  onClose: () => void
}

export const ExportModal: React.FC<Props> = ({ onClose }) => {
  const { sessionId, layers, canvasWidth, canvasHeight, canvasOffset, originalImagePath } = useEditorStore()
  const [format, setFormat] = useState<'png' | 'jpeg' | 'webp'>('png')
  const [upscale, setUpscale] = useState(false)
  const [upscaleFactor, setUpscaleFactor] = useState(2)
  const [projectName, setProjectName] = useState('my_project')
  const [loading, setLoading] = useState(false)

  const handleExport = async () => {
    if (!sessionId) return
    setLoading(true)
    try {
      const blob = await exportImage({
        session_id: sessionId,
        layers,
        canvas_width: canvasWidth,
        canvas_height: canvasHeight,
        format,
        upscale,
        upscale_factor: upscaleFactor,
      })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `export.${format}`
      a.click()
      URL.revokeObjectURL(url)
      toast.success('Exported!')
      onClose()
    } catch (err: unknown) {
      toast.error(err instanceof Error ? err.message : 'Export failed')
    } finally {
      setLoading(false)
    }
  }

  const handleSave = async () => {
    if (!sessionId || !originalImagePath) return
    setLoading(true)
    try {
      await saveProject({
        session_id: sessionId,
        project_name: projectName,
        original_image_path: originalImagePath,
        layers,
        canvas_width: canvasWidth,
        canvas_height: canvasHeight,
        canvas_position: canvasOffset,
        settings: {},
        prompts: layers.flatMap((l) => l.history),
      })
      toast.success('Project saved!')
    } catch (err: unknown) {
      toast.error(err instanceof Error ? err.message : 'Save failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50">
      <div className="bg-dark-800 border border-dark-500 rounded-xl p-6 w-96 shadow-2xl">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-white font-semibold">Export / Save</h2>
          <button onClick={onClose} className="text-gray-400 hover:text-white">
            <X size={18} />
          </button>
        </div>

        <div className="space-y-4">
          {/* Format */}
          <div>
            <label className="text-xs text-gray-400 block mb-1">Export Format</label>
            <div className="flex gap-2">
              {(['png', 'jpeg', 'webp'] as const).map((f) => (
                <button
                  key={f}
                  onClick={() => setFormat(f)}
                  className={`flex-1 py-1.5 rounded text-sm ${
                    format === f ? 'bg-accent text-white' : 'bg-dark-600 text-gray-300'
                  }`}
                >
                  {f.toUpperCase()}
                </button>
              ))}
            </div>
          </div>

          {/* Upscale */}
          <div className="flex items-center gap-3">
            <input
              type="checkbox"
              id="upscale"
              checked={upscale}
              onChange={(e) => setUpscale(e.target.checked)}
              className="accent-accent"
            />
            <label htmlFor="upscale" className="text-sm text-gray-300">
              Upscale with Real-ESRGAN
            </label>
            {upscale && (
              <select
                value={upscaleFactor}
                onChange={(e) => setUpscaleFactor(Number(e.target.value))}
                className="bg-dark-600 text-white text-sm rounded px-2 py-1"
              >
                <option value={2}>2×</option>
                <option value={4}>4×</option>
              </select>
            )}
          </div>

          <button
            onClick={handleExport}
            disabled={loading}
            className="w-full bg-accent hover:bg-accent-hover text-white py-2 rounded disabled:opacity-40"
          >
            {loading ? 'Exporting…' : 'Export Image'}
          </button>

          <div className="border-t border-dark-600 pt-4">
            <label className="text-xs text-gray-400 block mb-1">Project Name</label>
            <input
              type="text"
              value={projectName}
              onChange={(e) => setProjectName(e.target.value)}
              className="w-full bg-dark-700 text-white text-sm rounded px-3 py-1.5 border border-dark-500 outline-none focus:border-accent"
            />
            <button
              onClick={handleSave}
              disabled={loading}
              className="w-full mt-2 bg-dark-600 hover:bg-dark-500 text-white py-2 rounded disabled:opacity-40"
            >
              Save Project (.json)
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
