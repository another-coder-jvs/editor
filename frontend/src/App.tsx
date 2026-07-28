import React, { useState } from 'react'
import { ToastContainer } from 'react-toastify'
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

export default function App() {
  useKeyboardShortcuts()
  const { originalImageUrl } = useEditorStore()
  const [showExport, setShowExport] = useState(false)

  return (
    <div className="flex flex-col h-screen bg-dark-900 text-white select-none">
      {/* Top toolbar */}
      <Toolbar onExport={() => setShowExport(true)} />

      {/* Main area */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left: Layers */}
        <LayerPanel />

        {/* Center: Canvas */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {originalImageUrl ? <Canvas /> : <ImageUploader />}
          <HistoryPanel />
        </div>

        {/* Right: Properties */}
        <PropertiesPanel />
      </div>

      {/* Progress overlay */}
      <ProgressBar />

      {/* Export modal */}
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
