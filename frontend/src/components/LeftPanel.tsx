import React from 'react'
import { ToolSidebar } from './ToolSidebar'
import { ToolOptionsPanel } from './ToolOptionsPanel'
import { LayerPanel } from './LayerPanel'
import { ResizablePanel } from './ResizablePanel'

export const LeftPanel: React.FC = () => (
  <div className="flex flex-col h-full w-full overflow-hidden">

    {/* ── Tools: icon strip + options side by side ── */}
    <ResizablePanel edge="bottom" defaultSize={220} minSize={80} maxSize={500} className="w-full border-b border-dark-600 flex-shrink-0">
      <div className="flex h-full overflow-hidden">
        {/* Icon strip */}
        <div className="flex-shrink-0 border-r border-dark-600 overflow-y-auto">
          <ToolSidebar />
        </div>
        {/* Tool options */}
        <div className="flex-1 overflow-y-auto">
          <ToolOptionsPanel />
        </div>
      </div>
    </ResizablePanel>

    {/* ── Layers ── */}
    <div className="flex-1 overflow-hidden flex flex-col min-h-0">
      <LayerPanel />
    </div>

  </div>
)
