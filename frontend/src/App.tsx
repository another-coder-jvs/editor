import React, { useState, useEffect } from 'react'
import { ToastContainer, toast } from 'react-toastify'
// @ts-ignore: side-effect import of CSS without type declarations
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
import { baseUrl } from './config'
export default function App() {
  useKeyboardShortcuts()
  const { originalImageUrl, setSession, setLayers } = useEditorStore()
  const [showExport, setShowExport] = useState(false)

  useEffect(() => {
    if (originalImageUrl) return
    getLatestSession().then(({ session }) => {
      if (!session) return
      const { session_id, image_path, layers } = session
      const imageUrl = `${baseUrl}/temp/${image_path.replace(/^\/temp\//, '').replace(/^\//, '')}`
      console.log(`OPENING : ${imageUrl}`)
      const img = new Image()
      const restore = (w = 1920, h = 1080) => {
        setSession(session_id, image_path, imageUrl, w, h)
        setLayers(layers)
        toast.info('Resumed last session')
      }
      img.onload = () => restore(img.naturalWidth, img.naturalHeight)
      img.onerror = () => restore()
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
