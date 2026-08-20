import React, { useCallback, useState } from 'react'
import { useEditorStore } from '../store/editorStore'
import { detectObjects, segmentObjects } from '../api/client'
import { toast } from 'react-toastify'
import { Upload, Loader2, Wand2, Search } from 'lucide-react'

export const ImageUploader: React.FC = () => {
  const { setSession, setLayers, setProgress } = useEditorStore()
  const [loading, setLoading] = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const [userPrompt, setUserPrompt] = useState('')
  const [selectedFile, setSelectedFile] = useState<File | null>(null)

  const processFile = useCallback(
    async (file: File, prompt?: string) => {
      if (!file.type.startsWith('image/')) {
        toast.error('Please upload an image file')
        return
      }
      setLoading(true)
      setProgress({ session_id: '', task: 'detect', progress: 0.05, message: 'Detecting objects…', done: false })

      try {
        // 1. Detect
        const detectResult = await detectObjects(file, prompt)
        const { session_id, objects } = detectResult

        // Get image dimensions
        const url = URL.createObjectURL(file)
        const dims = await getImageDimensions(url)

        setSession(
          session_id,
          `/temp/${session_id}/${file.name}`,
          url,
          dims.width,
          dims.height,
        )

        setProgress({ session_id, task: 'segment', progress: 0.4, message: 'Segmenting objects…', done: false })

        // 2. Segment
        const segResult = await segmentObjects(
          session_id,
          `/temp/${session_id}/${file.name}`,
          objects,
        )

        setLayers(segResult.layers)
        setProgress({ session_id, task: 'done', progress: 1, message: 'Done!', done: true })
        toast.success(`Created ${segResult.layers.length} layers`)
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : 'Processing failed'
        toast.error(msg)
        setProgress(null)
      } finally {
        setLoading(false)
      }
    },
    [setSession, setLayers, setProgress],
  )

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) {
      setSelectedFile(file)
      // Auto-detect on drop
      processFile(file)
    }
  }

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setSelectedFile(file)
    }
  }

  const handleAutoDetect = () => {
    if (!selectedFile) {
      toast.error('Please select an image first')
      return
    }
    processFile(selectedFile)
  }

  const handleManualDetect = () => {
    if (!selectedFile) {
      toast.error('Please select an image first')
      return
    }
    if (!userPrompt.trim()) {
      toast.error('Please enter what objects are in the image')
      return
    }
    processFile(selectedFile, userPrompt.trim())
  }

  return (
    <div
      className={`flex flex-col items-center justify-center h-full gap-4 transition-colors ${
        dragOver ? 'bg-dark-700' : ''
      }`}
      onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
      onDragLeave={() => setDragOver(false)}
      onDrop={onDrop}
    >
      <div className="border-2 border-dashed border-dark-500 rounded-xl p-12 flex flex-col items-center gap-4 hover:border-accent transition-colors">
        {loading ? (
          <Loader2 size={40} className="text-accent animate-spin" />
        ) : (
          <Upload size={40} className="text-gray-500" />
        )}
        <p className="text-gray-400 text-sm">
          {loading ? 'Processing image…' : 'Drop an image here or click to upload'}
        </p>
        {!loading && (
          <label className="bg-accent hover:bg-accent-hover text-white text-sm px-4 py-2 rounded cursor-pointer transition-colors">
            Choose Image
            <input
              type="file"
              accept="image/*"
              className="hidden"
              onChange={onFileChange}
            />
          </label>
        )}
      </div>

      {/* Prompt input + buttons — shown after file selected */}
      {selectedFile && !loading && (
        <div className="w-full max-w-md flex flex-col gap-3 px-4">
          <div className="flex items-center gap-2">
            <div className="relative flex-1">
              <input
                type="text"
                value={userPrompt}
                onChange={(e) => setUserPrompt(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter') handleManualDetect() }}
                placeholder="Describe what's in the image (e.g. sneaker, logo, text, phone icon)"
                className="w-full bg-dark-800 border border-dark-500 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-accent transition-colors"
              />
            </div>
          </div>
          <div className="flex gap-2">
            <button
              onClick={handleAutoDetect}
              className="flex-1 flex items-center justify-center gap-2 bg-purple-600 hover:bg-purple-700 text-white text-sm px-4 py-2 rounded-lg transition-colors"
              title="Auto-detect objects using AI models"
            >
              <Wand2 size={14} />
              Auto Detect
            </button>
            <button
              onClick={handleManualDetect}
              className="flex-1 flex items-center justify-center gap-2 bg-accent hover:bg-accent-hover text-white text-sm px-4 py-2 rounded-lg transition-colors"
              title="Detect using your description"
            >
              <Search size={14} />
              Detect
            </button>
          </div>
          <p className="text-xs text-gray-500 text-center">
            <span className="text-purple-400">Auto Detect</span> — AI finds objects automatically
            {' · '}
            <span className="text-accent">Detect</span> — use your description
          </p>
        </div>
      )}
    </div>
  )
}

function getImageDimensions(url: string): Promise<{ width: number; height: number }> {
  return new Promise((resolve) => {
    const img = new Image()
    img.onload = () => resolve({ width: img.naturalWidth, height: img.naturalHeight })
    img.src = url
  })
}
