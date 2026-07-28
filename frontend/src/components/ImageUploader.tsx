import React, { useCallback, useState } from 'react'
import { useEditorStore } from '../store/editorStore'
import { detectObjects, segmentObjects } from '../api/client'
import { toast } from 'react-toastify'
import { Upload, Loader2 } from 'lucide-react'

export const ImageUploader: React.FC = () => {
  const { setSession, setLayers, setProgress } = useEditorStore()
  const [loading, setLoading] = useState(false)
  const [dragOver, setDragOver] = useState(false)

  const processFile = useCallback(
    async (file: File) => {
      if (!file.type.startsWith('image/')) {
        toast.error('Please upload an image file')
        return
      }
      setLoading(true)
      setProgress({ session_id: '', task: 'detect', progress: 0.05, message: 'Detecting objects…', done: false })

      try {
        // 1. Detect
        const detectResult = await detectObjects(file)
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
    [setSession, setLayers, setProgress]
  )

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) processFile(file)
  }

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) processFile(file)
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
