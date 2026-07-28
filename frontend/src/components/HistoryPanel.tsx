import React from 'react'
import { useEditorStore } from '../store/editorStore'

export const HistoryPanel: React.FC = () => {
  const { undoStack, layers, undo, redo, redoStack } = useEditorStore()

  return (
    <div className="h-28 bg-dark-800 border-t border-dark-600 flex flex-col">
      <div className="px-3 py-1 text-xs font-semibold text-gray-400 uppercase tracking-wider border-b border-dark-600 flex items-center justify-between">
        <span>History</span>
        <div className="flex gap-2">
          <button
            onClick={undo}
            disabled={undoStack.length === 0}
            className="text-xs text-accent disabled:opacity-30 hover:underline"
          >
            Undo
          </button>
          <button
            onClick={redo}
            disabled={redoStack.length === 0}
            className="text-xs text-accent disabled:opacity-30 hover:underline"
          >
            Redo
          </button>
        </div>
      </div>
      <div className="flex-1 overflow-x-auto flex items-center gap-2 px-3">
        {undoStack.length === 0 && (
          <span className="text-xs text-gray-600">No history yet</span>
        )}
        {undoStack.map((_, i) => (
          <div
            key={i}
            className="flex-shrink-0 text-xs bg-dark-600 text-gray-400 px-2 py-1 rounded"
          >
            State {i + 1}
          </div>
        ))}
        <div className="flex-shrink-0 text-xs bg-accent text-white px-2 py-1 rounded">
          Current
        </div>
      </div>
    </div>
  )
}
