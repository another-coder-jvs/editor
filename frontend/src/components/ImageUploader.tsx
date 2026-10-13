import React, { useCallback, useState } from 'react'
import { useEditorStore } from '../store/editorStore'
import { detectObjects, segmentObjects } from '../api/client'
import { toast } from 'react-toastify'
import { Upload, Loader2 } from 'lucide-react'
import { DetectionModeSelector } from './DetectionModeSelector'

export const ImageUploader: React.FC = () => {
  const { setSession, setLayers, setProgress } = useEditorStore()
  const [loading, setLoading] = useState(false)
  const [dragOver, setDragOver] = useState(false)

  // Hold the file until user picks a detection mode
  const [pendingFile, setPendingFile] = useState<File | null>(null)
  const [pendingImageUrl, setPendingImageUrl] = useState<string | null>(null)

  const processWithPrompt = useCallback(
    async (prompt: string) => {
      if (!pendingFile) return
      setLoading(true)
      setProgress({ session_id: '', task: 'detect', progress: 0.1, message: 'Detecting objects…', done: false })

      try {
        // Send file + prompt to /detect (original endpoint)
        const detectResult = await detectObjects(pendingFile, prompt || undefined)
        const { session_id, objects, image_path } = detectResult

        // Get image dimensions
        const dims = await getImageDimensions(pendingImageUrl!)

        setSession(
          session_id,
          image_path,
          pendingImageUrl!,
          dims.width,
          dims.height,
        )

        setProgress({ session_id, task: 'segment', progress: 0.4, message: 'Segmenting objects…', done: false })

        // Segment
        const segResult = await segmentObjects(
          session_id,
          image_path,
          objects,
        )

        setLayers(segResult.layers)
        setProgress({ session_id, task: 'done', progress: 1, message: 'Done!', done: true })
        toast.success(`Created ${segResult.layers.length} layers`)

        // Clear pending state
        setPendingFile(null)
        setPendingImageUrl(null)
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : 'Processing failed'
        toast.error(msg)
        setProgress(null)
      } finally {
        setLoading(false)
      }
    },
    [pendingFile, pendingImageUrl, setSession, setLayers, setProgress],
  )

  const handleFileSelect = useCallback(
    (file: File) => {
      if (!file.type.startsWith('image/')) {
        toast.error('Please upload an image file')
        return
      }
      const url = URL.createObjectURL(file)
      setPendingFile(file)
      setPendingImageUrl(url)
    },
    [],
  )

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFileSelect(file)
  }

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) handleFileSelect(file)
  }

  // Show detection mode selector after file is selected
  if (pendingImageUrl) {
    return (
      <DetectionModeSelector
        imageUrl={pendingImageUrl}
        fileName={pendingFile?.name || ''}
        onDetect={processWithPrompt}
        onCancel={() => {
          setPendingFile(null)
          setPendingImageUrl(null)
        }}
      />
    )
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
          {loading ? 'Processing…' : 'Drop an image here or click to upload'}
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
