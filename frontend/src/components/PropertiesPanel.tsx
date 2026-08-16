import React, { useState } from 'react'
import { useEditorStore } from '../store/editorStore'
import { AdjustPanel } from './AdjustPanel'
import { TransformPanel } from './TransformPanel'
import { AIEditPanel } from './AIEditPanel'

const TABS = ['AI Edit', 'Adjust', 'Transform'] as const
type Tab = typeof TABS[number]

export const PropertiesPanel: React.FC = () => {
  const { selectedLayerIds, layers } = useEditorStore()
  const [tab, setTab] = useState<Tab>('AI Edit')
  const selectedLayer = layers.find(l => selectedLayerIds[0] === l.id)

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Context label */}
      <div className="px-3 py-1.5 border-b border-dark-600 flex-shrink-0">
        {selectedLayer ? (
          <div className="flex items-center gap-1.5">
            <span className="text-xs font-medium text-white truncate">{selectedLayer.name}</span>
            <span className="text-xs text-gray-600 flex-shrink-0">
              {selectedLayer.bbox.width}×{selectedLayer.bbox.height}
            </span>
          </div>
        ) : (
          <span className="text-xs text-gray-600">No layer selected</span>
        )}
      </div>

      {/* Tab bar */}
      <div className="flex border-b border-dark-600 flex-shrink-0">
        {TABS.map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`flex-1 py-1.5 text-xs transition-colors ${
              tab === t
                ? 'text-white border-b-2 border-accent bg-dark-700'
                : 'text-gray-500 hover:text-gray-300'
            }`}
          >
            {t}
          </button>
        ))}
      </div>

      {/* Panel content */}
      <div className="flex-1 overflow-y-auto">
        {tab === 'AI Edit'   && <AIEditPanel />}
        {tab === 'Adjust'    && <AdjustPanel />}
        {tab === 'Transform' && <TransformPanel />}
      </div>
    </div>
  )
}
