import React from 'react'
import { useEditorStore } from '../store/editorStore'

export const HistoryPanel: React.FC = () => {
  const { undoStack, undo, redo, redoStack, reset } = useEditorStore()

  return (
    <div className="h-24 bg-dark-800 border-t border-dark-600 flex flex-col flex-shrink-0">
      <div className="px-3 py-1 text-xs font-semibold text-gray-400 uppercase tracking-wider border-b border-dark-600 flex items-center justify-between">
        <span>History ({undoStack.length})</span>
        <div className="flex gap-2">
          <button onClick={undo} disabled={undoStack.length === 0}
            className="text-xs text-accent disabled:opacity-30 hover:underline">Undo</button>
          <button onClick={redo} disabled={redoStack.length === 0}
            className="text-xs text-accent disabled:opacity-30 hover:underline">Redo</button>
          <button
            onClick={() => { if (window.confirm('Reset all history and start fresh?')) reset() }}
            className="text-xs text-red-400 hover:underline"
          >Reset</button>
        </div>
      </div>
      <div className="flex-1 overflow-x-auto flex items-center gap-2 px-3">
        {undoStack.length === 0 && (
          <span className="text-xs text-gray-600">No history yet</span>
        )}
        {undoStack.map((_, i) => (
          <div key={i} className="flex-shrink-0 text-xs bg-dark-600 text-gray-400 px-2 py-1 rounded">
            State {i + 1}
          </div>
        ))}
        {undoStack.length > 0 && (
          <div className="flex-shrink-0 text-xs bg-accent text-white px-2 py-1 rounded">Current</div>
        )}
      </div>
    </div>
  )
}
