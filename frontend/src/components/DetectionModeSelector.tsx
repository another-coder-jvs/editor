import React, { useState, useCallback } from 'react'
import { Eye, Pencil, Sparkles, X } from 'lucide-react'

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

          {/* Scanning overlay */}
          {scanning && (
            <div className="absolute inset-0 rounded-lg overflow-hidden pointer-events-none">
              {/* Dark overlay */}
              <div className="absolute inset-0 bg-dark-900/50 backdrop-blur-[1px]" />

              {/* Animated grid */}
              <div className="scan-grid absolute inset-0 opacity-20" />

              {/* Main sweep line */}
              <div className="scan-sweep absolute inset-x-0 h-[2px] bg-gradient-to-r from-transparent via-cyan-400 to-transparent shadow-[0_0_30px_8px_rgba(34,211,238,0.5)]" />

              {/* Trailing glow */}
              <div className="scan-trail absolute inset-x-0 h-16 bg-gradient-to-b from-cyan-400/20 via-cyan-400/5 to-transparent" />

              {/* Corner brackets - animated */}
              <div className="scan-corners">
                <div className="corner-tl" />
                <div className="corner-tr" />
                <div className="corner-bl" />
                <div className="corner-br" />
              </div>

              {/* Pulse rings */}
              <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2">
                <div className="pulse-ring pulse-ring-1" />
                <div className="pulse-ring pulse-ring-2" />
                <div className="pulse-ring pulse-ring-3" />
              </div>

              {/* Status badge */}
              <div className="absolute bottom-6 left-1/2 -translate-x-1/2 flex items-center gap-3 bg-dark-800/95 px-5 py-2.5 rounded-full border border-cyan-500/30 shadow-[0_0_20px_rgba(34,211,238,0.2)]">
                <div className="relative flex h-2.5 w-2.5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75" />
                  <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-cyan-400" />
                </div>
                <span className="text-xs text-cyan-100 font-medium tracking-wide">
                  {mode === 'auto' ? 'AI VISION ANALYZING' : 'DETECTING OBJECTS'}
                </span>
                <div className="flex gap-0.5">
                  <span className="loading-dot loading-dot-1 w-1 h-1 rounded-full bg-cyan-400" />
                  <span className="loading-dot loading-dot-2 w-1 h-1 rounded-full bg-cyan-400" />
                  <span className="loading-dot loading-dot-3 w-1 h-1 rounded-full bg-cyan-400" />
                </div>
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

      {/* Scan Animation Styles */}
      <style>{`
        /* Grid background */
        .scan-grid {
          background-image:
            linear-gradient(rgba(34, 211, 238, 0.3) 1px, transparent 1px),
            linear-gradient(90deg, rgba(34, 211, 238, 0.3) 1px, transparent 1px);
          background-size: 40px 40px;
          animation: gridMove 3s linear infinite;
        }

        @keyframes gridMove {
          0% { background-position: 0 0; }
          100% { background-position: 40px 40px; }
        }

        /* Main sweep line */
        .scan-sweep {
          animation: sweep 2.5s ease-in-out infinite;
        }

        @keyframes sweep {
          0% { top: 0%; opacity: 0; }
          10% { opacity: 1; }
          90% { opacity: 1; }
          100% { top: 100%; opacity: 0; }
        }

        /* Trailing glow */
        .scan-trail {
          animation: sweep 2.5s ease-in-out infinite;
        }

        /* Corner brackets */
        .scan-corners .corner-tl,
        .scan-corners .corner-tr,
        .scan-corners .corner-bl,
        .scan-corners .corner-br {
          position: absolute;
          width: 40px;
          height: 40px;
          border-color: rgb(34, 211, 238);
          animation: cornerPulse 2s ease-in-out infinite;
        }

        .corner-tl {
          top: 16px; left: 16px;
          border-top: 2px solid;
          border-left: 2px solid;
          border-radius: 8px 0 0 0;
        }
        .corner-tr {
          top: 16px; right: 16px;
          border-top: 2px solid;
          border-right: 2px solid;
          border-radius: 0 8px 0 0;
        }
        .corner-bl {
          bottom: 16px; left: 16px;
          border-bottom: 2px solid;
          border-left: 2px solid;
          border-radius: 0 0 0 8px;
        }
        .corner-br {
          bottom: 16px; right: 16px;
          border-bottom: 2px solid;
          border-right: 2px solid;
          border-radius: 0 0 8px 0;
        }

        @keyframes cornerPulse {
          0%, 100% { opacity: 0.5; transform: scale(1); }
          50% { opacity: 1; transform: scale(1.05); }
        }

        /* Pulse rings */
        .pulse-ring {
          position: absolute;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          border: 1px solid rgba(34, 211, 238, 0.4);
          border-radius: 50%;
          animation: pulseExpand 3s ease-out infinite;
        }

        .pulse-ring-1 { animation-delay: 0s; }
        .pulse-ring-2 { animation-delay: 1s; }
        .pulse-ring-3 { animation-delay: 2s; }

        @keyframes pulseExpand {
          0% { width: 0; height: 0; opacity: 0.8; }
          100% { width: 300px; height: 300px; opacity: 0; }
        }

        /* Loading dots */
        .loading-dot {
          animation: dotBounce 1.4s ease-in-out infinite;
        }
        .loading-dot-1 { animation-delay: 0s; }
        .loading-dot-2 { animation-delay: 0.2s; }
        .loading-dot-3 { animation-delay: 0.4s; }

        @keyframes dotBounce {
          0%, 80%, 100% { transform: translateY(0); opacity: 0.4; }
          40% { transform: translateY(-4px); opacity: 1; }
        }
      `}</style>
    </div>
  )
}
