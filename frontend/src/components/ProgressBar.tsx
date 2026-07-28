import React from 'react'
import { useEditorStore } from '../store/editorStore'

export const ProgressBar: React.FC = () => {
  const progress = useEditorStore((s) => s.progress)
  if (!progress || progress.done) return null

  return (
    <div className="fixed bottom-32 left-1/2 -translate-x-1/2 z-50 bg-dark-800 border border-dark-500 rounded-lg px-4 py-3 shadow-xl min-w-64">
      <div className="flex justify-between text-xs text-gray-400 mb-2">
        <span className="capitalize">{progress.task}</span>
        <span>{Math.round(progress.progress * 100)}%</span>
      </div>
      <div className="w-full bg-dark-600 rounded-full h-1.5">
        <div
          className="bg-accent h-1.5 rounded-full transition-all duration-300"
          style={{ width: `${progress.progress * 100}%` }}
        />
      </div>
      {progress.message && (
        <p className="text-xs text-gray-500 mt-1">{progress.message}</p>
      )}
    </div>
  )
}
