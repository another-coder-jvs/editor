import React, { useState } from 'react'
import { useEditorStore } from '../store/editorStore'
import { AdjustPanel } from './AdjustPanel'
import { TransformPanel } from './TransformPanel'
import { AIEditPanel } from './AIEditPanel'
import { ToolOptionsPanel } from './ToolOptionsPanel'

const TABS = ['Tool', 'Adjust', 'Transform', 'AI Edit'] as const
type Tab = typeof TABS[number]

export const PropertiesPanel: React.FC = () => {
  const { selectedLayerIds, layers } = useEditorStore()
  const [tab, setTab] = useState<Tab>('Tool')
  const selectedLayer = layers.find(l => selectedLayerIds[0] === l.id)

  return (
    <div className="w-60 bg-dark-800 border-l border-dark-600 flex flex-col overflow-hidden">
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
        {tab === 'Tool' && <ToolOptionsPanel />}
        {tab === 'Adjust' && <AdjustPanel />}
        {tab === 'Transform' && <TransformPanel />}
        {tab === 'AI Edit' && <AIEditPanel />}
      </div>
    </div>
  )
}
