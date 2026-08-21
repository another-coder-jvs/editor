import React, { useState, useCallback } from 'react'
import { Eye, Pencil, Loader2, Sparkles, X } from 'lucide-react'

type DetectionMode = 'auto' | 'manual'

interface Props {
  imageUrl: string
  fileName: string
  onDetect: (prompt: string) => void
  onCancel: () => void
}

export const DetectionModeSelector: React.FC<Props> = ({
  imageUrl,
  fileName,
  onDetect,
  onCancel,
}) => {
  const [mode, setMode] = useState<DetectionMode | null>(null)
  const [prompt, setPrompt] = useState('')
  const [scanning, setScanning] = useState(false)

  const handleAutoDetect = useCallback(() => {
    setMode('auto')
    setScanning(true)
    // Empty prompt = backend uses Ollama vision → fallback prompt
    onDetect('')
  }, [onDetect])

  const handleManualDetect = useCallback(() => {
    if (prompt.trim()) {
      setMode('manual')
      setScanning(true)
      onDetect(prompt.trim())
    }
  }, [prompt, onDetect])

  return (
    <div className="flex flex-col h-full">
      {/* Image Preview with scanning overlay */}
      <div className="flex-1 flex items-center justify-center p-4 bg-dark-900 relative overflow-hidden">
        <div className="relative max-w-full max-h-full">
          <img
            src={imageUrl}
            alt={fileName}
            className="max-w-full max-h-[70vh] object-contain rounded-lg shadow-2xl"
          />

          {/* Scanning animation overlay */}
          {scanning && (
            <div className="absolute inset-0 rounded-lg overflow-hidden pointer-events-none">
              {/* Dark overlay */}
              <div className="absolute inset-0 bg-dark-900/60" />

              {/* Scanning line */}
              <div className="scan-line absolute inset-x-0 h-0.5 bg-gradient-to-r from-transparent via-accent to-transparent shadow-[0_0_15px_3px_rgba(99,102,241,0.6)]" />

              {/* Corner brackets */}
              <div className="absolute top-3 left-3 w-8 h-8 border-t-2 border-l-2 border-accent rounded-tl-lg" />
              <div className="absolute top-3 right-3 w-8 h-8 border-t-2 border-r-2 border-accent rounded-tr-lg" />
              <div className="absolute bottom-3 left-3 w-8 h-8 border-b-2 border-l-2 border-accent rounded-bl-lg" />
              <div className="absolute bottom-3 right-3 w-8 h-8 border-b-2 border-r-2 border-accent rounded-br-lg" />

              {/* Status text */}
              <div className="absolute bottom-6 left-1/2 -translate-x-1/2 flex items-center gap-2 bg-dark-800/90 px-4 py-2 rounded-full border border-dark-600">
                <Loader2 size={14} className="animate-spin text-accent" />
                <span className="text-xs text-gray-300">
                  {mode === 'auto' ? 'AI is analyzing the image...' : 'Detecting objects...'}
                </span>
              </div>
            </div>
          )}

          {/* Close button */}
          {!scanning && (
            <button
              onClick={onCancel}
              className="absolute top-3 right-3 p-2 bg-dark-700/80 hover:bg-dark-600 rounded-full text-gray-400 hover:text-white transition-colors"
            >
              <X size={18} />
            </button>
          )}
        </div>
      </div>

      {/* Detection Mode Panel */}
      {!scanning && (
        <div className="border-t border-dark-600 bg-dark-800 p-4">
          <div className="max-w-lg mx-auto">
            <div className="flex items-center gap-2 mb-3">
              <Sparkles size={16} className="text-accent" />
              <h3 className="text-sm font-medium text-white">Choose detection mode</h3>
            </div>

            {/* Mode Cards */}
            <div className="grid grid-cols-2 gap-3 mb-3">
              {/* Auto Detect */}
              <button
                onClick={handleAutoDetect}
                className={`flex flex-col items-center gap-2 p-5 rounded-xl border-2 transition-all hover:scale-[1.02] ${
                  mode === 'auto'
                    ? 'border-accent bg-accent/10 text-accent'
                    : 'border-dark-600 bg-dark-700 hover:border-dark-500 text-gray-400 hover:text-gray-300'
                }`}
              >
                <Eye size={28} />
                <span className="text-sm font-medium">Auto Detect</span>
                <span className="text-[11px] text-gray-500">AI identifies all objects</span>
              </button>

              {/* Manual Detect */}
              <button
                onClick={() => setMode(mode === 'manual' ? null : 'manual')}
                className={`flex flex-col items-center gap-2 p-5 rounded-xl border-2 transition-all hover:scale-[1.02] ${
                  mode === 'manual'
                    ? 'border-accent bg-accent/10 text-accent'
                    : 'border-dark-600 bg-dark-700 hover:border-dark-500 text-gray-400 hover:text-gray-300'
                }`}
              >
                <Pencil size={28} />
                <span className="text-sm font-medium">Manual Detect</span>
                <span className="text-[11px] text-gray-500">Describe what to find</span>
              </button>
            </div>

            {/* Manual prompt input */}
            {mode === 'manual' && (
              <div className="flex gap-2 animate-in fade-in slide-in-from-bottom-2 duration-200">
                <input
                  type="text"
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleManualDetect()}
                  placeholder='e.g. "person, car, shoe, pillow"'
                  className="flex-1 bg-dark-700 text-sm text-white rounded-lg px-3 py-2.5 border border-dark-500 focus:border-accent outline-none placeholder-gray-500"
                  autoFocus
                />
                <button
                  onClick={handleManualDetect}
                  disabled={!prompt.trim()}
                  className="px-5 py-2.5 bg-accent hover:bg-accent-hover text-white text-sm rounded-lg disabled:opacity-40 transition-colors font-medium"
                >
                  Detect
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* CSS for scan line animation */}
      <style>{`
        .scan-line {
          animation: scan 2s ease-in-out infinite;
        }
        @keyframes scan {
          0% { top: 0; }
          50% { top: 100%; }
          100% { top: 0; }
        }
      `}</style>
    </div>
  )
}
