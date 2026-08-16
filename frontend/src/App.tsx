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
import { ToolPanel } from './components/ToolPanel'
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

        {/* Left: Layers only */}
        <ResizablePanel
          edge="right"
          defaultSize={220}
          minSize={160}
          maxSize={400}
          title="Layers"
          className="bg-dark-800 border-r border-dark-600 flex-shrink-0"
        >
          <LayerPanel />
        </ResizablePanel>

        {/* Center: canvas */}
        <main className="flex-1 flex flex-col overflow-hidden min-w-0">
          {originalImageUrl ? <Canvas /> : <ImageUploader />}
        </main>

        {/* Right: Tools (top) + Inspector (bottom) */}
        <ResizablePanel
          edge="left"
          defaultSize={280}
          minSize={200}
          maxSize={480}
          title="Tools & Inspector"
          className="bg-dark-800 border-l border-dark-600 flex-shrink-0"
        >
          {/* Tools — resizable height */}
          <ResizablePanel
            edge="bottom"
            defaultSize={320}
            minSize={120}
            maxSize={600}
            title="Tools"
            className="border-b border-dark-600 w-full"
          >
            <ToolPanel />
          </ResizablePanel>

          {/* Inspector — fills remaining */}
          <ResizablePanel
            title="Inspector"
            edge="bottom"
            className="w-full"
          >
            <PropertiesPanel />
          </ResizablePanel>
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
