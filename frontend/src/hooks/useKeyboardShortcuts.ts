import { useEffect } from 'react'
import { useEditorStore } from '../store/editorStore'

export function useKeyboardShortcuts() {
  const { undo, redo, deleteLayer, selectedLayerIds, duplicateLayer } = useEditorStore()

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const ctrl = e.ctrlKey || e.metaKey
      if (ctrl && e.key === 'z' && !e.shiftKey) { e.preventDefault(); undo() }
      if (ctrl && (e.key === 'y' || (e.key === 'z' && e.shiftKey))) { e.preventDefault(); redo() }
      if (ctrl && e.key === 'd') {
        e.preventDefault()
        selectedLayerIds.forEach((id) => duplicateLayer(id))
      }
      if (e.key === 'Delete' || e.key === 'Backspace') {
        const target = e.target as HTMLElement
        const tag = target.tagName
        if (tag !== 'INPUT' && tag !== 'TEXTAREA' && !target.isContentEditable) {
          selectedLayerIds.forEach((id) => deleteLayer(id))
        }
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [undo, redo, deleteLayer, duplicateLayer, selectedLayerIds])
}
