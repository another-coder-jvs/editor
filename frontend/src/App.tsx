import React, { useState, useEffect } from 'react'
import { ToastContainer, toast } from 'react-toastify'
// @ts-ignore: side-effect import of CSS without type declarations
import 'react-toastify/dist/ReactToastify.css'

import { useEditorStore } from './store/editorStore'
import { useKeyboardShortcuts } from './hooks/useKeyboardShortcuts'
import { Toolbar } from './components/Toolbar'
import { LeftPanel } from './components/LeftPanel'
import { Canvas } from './components/Canvas'
import { PropertiesPanel } from './components/PropertiesPanel'
import { ResizablePanel } from './components/ResizablePanel'
import { ProgressBar } from './components/ProgressBar'
import { ImageUploader } from './components/ImageUploader'
import { ExportModal } from './components/ExportModal'
import { getLatestSession } from './api/client'
import { baseUrl } from './config'
import { ProjectManager } from './components/ProjectManager'

export default function App() {
  useKeyboardShortcuts()
  const { originalImageUrl, setSession, setLayers, reset } = useEditorStore()
  const [showExport, setShowExport] = useState(false)
  const [showProjects, setShowProjects] = useState(false)

  useEffect(() => {
    if (originalImageUrl) return
    getLatestSession().then(({ session }) => {
      if (!session) return
      const { session_id, image_path, layers } = session
      const imageUrl = `${baseUrl}/temp/${image_path.replace(/^\/temp\//, '').replace(/^\//, '')}`
      fetch(imageUrl, { headers: { 'ngrok-skip-browser-warning': '1' } })
        .then(r => r.blob())
        .then(blob => {
          const blobUrl = URL.createObjectURL(blob)
          const img = new Image()
          img.onload = () => {
            setSession(session_id, image_path, blobUrl, img.naturalWidth, img.naturalHeight)
            setLayers(layers)
            toast.info('Resumed last session')
          }
          img.src = blobUrl
        })
        .catch(() => {
          setSession(session_id, image_path, imageUrl, 1920, 1080)
          setLayers(layers)
          toast.info('Resumed last session')
        })
    }).catch(() => { /* no cache, show uploader */ })
  }, [])

  return (
    <div className="flex flex-col h-screen bg-dark-900 text-white select-none overflow-hidden">
      {/* ── Top header bar ── */}
      <Toolbar onExport={() => setShowExport(true)} onOpenProjects={() => setShowProjects(true)} />

      {/* ── Main workspace ── */}
      <div className="flex flex-1 overflow-hidden">

        {/* Left: tools (icon strip + options) + layers */}
        <ResizablePanel
          edge="right"
          defaultSize={260}
          minSize={160}
          maxSize={480}
          title="Tools & Layers"
          className="bg-dark-800 border-r border-dark-600 flex-shrink-0"
        >
          <LeftPanel />
        </ResizablePanel>

        {/* Center: canvas */}
        <main className="flex-1 flex flex-col overflow-hidden min-w-0">
          {originalImageUrl ? <Canvas /> : <ImageUploader />}
        </main>

        {/* Right: Inspector (AI Edit / Adjust / Transform) */}
        <ResizablePanel
          edge="left"
          defaultSize={260}
          minSize={180}
          maxSize={480}
          title="Inspector"
          className="bg-dark-800 border-l border-dark-600 flex-shrink-0"
        >
          <PropertiesPanel />
        </ResizablePanel>

      </div>

      <ProgressBar />
      {showExport && <ExportModal onClose={() => setShowExport(false)} />}
      {showProjects && (
        <ProjectManager
          onClose={() => setShowProjects(false)}
          onNew={() => { reset(); setShowProjects(false) }}
        />
      )}
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
