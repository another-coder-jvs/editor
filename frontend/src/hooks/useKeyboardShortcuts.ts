import { useEffect } from 'react'
import { useEditorStore } from '../store/editorStore'

export function useKeyboardShortcuts() {
  const { undo, redo, deleteLayer, selectedLayerIds, duplicateLayer, setActiveTool, reorderLayer, layers, updateLayer } = useEditorStore()

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const ctrl = e.ctrlKey || e.metaKey
      const target = e.target as HTMLElement
      const inInput = target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable

      if (ctrl && e.key === 'z' && !e.shiftKey) { e.preventDefault(); undo(); return }
      if (ctrl && (e.key === 'y' || (e.key === 'z' && e.shiftKey))) { e.preventDefault(); redo(); return }
      if (ctrl && e.key === 'd') { e.preventDefault(); selectedLayerIds.forEach(id => duplicateLayer(id)); return }

      if (inInput) return

      // Ctrl+ArrowUp/Down → reorder layer in stack
      if (ctrl && e.key === 'ArrowUp' && selectedLayerIds.length === 1) {
        e.preventDefault(); reorderLayer(selectedLayerIds[0], 'up'); return
      }
      if (ctrl && e.key === 'ArrowDown' && selectedLayerIds.length === 1) {
        e.preventDefault(); reorderLayer(selectedLayerIds[0], 'down'); return
      }

      // Arrow keys → move selected layer position (Shift = 10px, bare = 1px)
      const ARROW_KEYS = ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight']
      if (ARROW_KEYS.includes(e.key) && selectedLayerIds.length > 0) {
        e.preventDefault()
        const step = e.shiftKey ? 100 : 5
        selectedLayerIds.forEach(id => {
          const layer = layers.find(l => l.id === id)
          if (!layer) return
          const { x, y } = layer.position
          const next =
            e.key === 'ArrowUp'    ? { x, y: y - step } :
            e.key === 'ArrowDown'  ? { x, y: y + step } :
            e.key === 'ArrowLeft'  ? { x: x - step, y } :
                                     { x: x + step, y }
          updateLayer(id, { position: next })
        })
        return
      }

      if (e.key === 'Delete' || e.key === 'Backspace') {
        selectedLayerIds.forEach(id => deleteLayer(id)); return
      }

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
  }, [undo, redo, deleteLayer, duplicateLayer, selectedLayerIds, setActiveTool, reorderLayer, layers, updateLayer])
}
