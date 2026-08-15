import { useEffect } from 'react'
import { useEditorStore } from '../store/editorStore'

export function useKeyboardShortcuts() {
  const { undo, redo, deleteLayer, selectedLayerIds, duplicateLayer, setActiveTool } = useEditorStore()

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const ctrl = e.ctrlKey || e.metaKey
      const target = e.target as HTMLElement
      const inInput = target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable

      // Ctrl shortcuts (always active)
      if (ctrl && e.key === 'z' && !e.shiftKey) { e.preventDefault(); undo(); return }
      if (ctrl && (e.key === 'y' || (e.key === 'z' && e.shiftKey))) { e.preventDefault(); redo(); return }
      if (ctrl && e.key === 'd') { e.preventDefault(); selectedLayerIds.forEach(id => duplicateLayer(id)); return }

      // Skip tool shortcuts when typing in inputs
      if (inInput) return

      if (e.key === 'Delete' || e.key === 'Backspace') {
        selectedLayerIds.forEach(id => deleteLayer(id))
        return
      }

      // Tool shortcuts
      switch (e.key.toLowerCase()) {
        case 'v': setActiveTool('move'); break
        case 'b': setActiveTool('brush'); break
        case 'e': setActiveTool('eraser'); break
        case 'c': if (!ctrl) setActiveTool('crop'); break
        case 't': setActiveTool('text_add'); break
        case 'm': setActiveTool('rect_select'); break
        case 'l': setActiveTool('lasso_select'); break
        case 'w': setActiveTool('magic_select'); break
        case 's': if (!ctrl) setActiveTool('clone'); break
        case 'i': setActiveTool('color_picker'); break
        case 'p': setActiveTool('pencil'); break
        case 'u': setActiveTool('shape_rect'); break
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [undo, redo, deleteLayer, duplicateLayer, selectedLayerIds, setActiveTool])
}
