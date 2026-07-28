import { useEffect, useRef } from 'react'
import { getProgress } from '../api/client'
import { useEditorStore } from '../store/editorStore'

export function useProgressPoller(active: boolean) {
  const sessionId = useEditorStore((s) => s.sessionId)
  const setProgress = useEditorStore((s) => s.setProgress)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    if (!active || !sessionId) return
    timerRef.current = setInterval(async () => {
      try {
        const p = await getProgress(sessionId)
        setProgress(p)
        if (p.done) {
          clearInterval(timerRef.current!)
          timerRef.current = null
        }
      } catch {
        // ignore polling errors
      }
    }, 800)
    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [active, sessionId, setProgress])
}
