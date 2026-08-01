import React, { useState, useEffect } from 'react'
import { ToastContainer, toast } from 'react-toastify'
import 'react-toastify/dist/ReactToastify.css'

import { useEditorStore } from './store/editorStore'
import { useKeyboardShortcuts } from './hooks/useKeyboardShortcuts'
import { Toolbar } from './components/Toolbar'
import { LayerPanel } from './components/LayerPanel'
import { Canvas } from './components/Canvas'
import { PropertiesPanel } from './components/PropertiesPanel'
import { HistoryPanel } from './components/HistoryPanel'
import { ProgressBar } from './components/ProgressBar'
import { ImageUploader } from './components/ImageUploader'
import { ExportModal } from './components/ExportModal'
import { getLatestSession } from './api/client'

export default function App() {
  useKeyboardShortcuts()
  const { originalImageUrl, setSession, setLayers } = useEditorStore()
  const [showExport, setShowExport] = useState(false)

  useEffect(() => {
    if (originalImageUrl) return
    getLatestSession().then(({ session }) => {
      if (!session) return
      const { session_id, image_path, layers } = session
      const imageUrl = `http://localhost:8000/${image_path.replace(/^\//, '')}`
      const img = new Image()
      img.onload = () => {
        setSession(session_id, image_path, imageUrl, img.naturalWidth, img.naturalHeight)
        setLayers(layers)
        toast.info('Resumed last session')
      }
      img.onerror = () => { /* no-op: fall through to upload screen */ }
      img.src = imageUrl
    }).catch(() => { /* no cache, show uploader */ })
  }, [])

  return (
    <div className="flex flex-col h-screen bg-dark-900 text-white select-none">
      <Toolbar onExport={() => setShowExport(true)} />
      <div className="flex flex-1 overflow-hidden">
        <LayerPanel />
        <div className="flex-1 flex flex-col overflow-hidden">
          {originalImageUrl ? <Canvas /> : <ImageUploader />}
          <HistoryPanel />
        </div>
        <PropertiesPanel />
      </div>
      <ProgressBar />
      {showExport && <ExportModal onClose={() => setShowExport(false)} />}
      <ToastContainer
        position="bottom-right"
        theme="dark"
        autoClose={3000}
        hideProgressBar={false}
        newestOnTop
      />
    </div>
  )
}
