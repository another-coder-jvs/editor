import React, { useEffect, useRef, useState } from 'react'
import { apiLoading } from '../api/client'

export const TopBarLoader: React.FC = () => {
  const [active, setActive] = useState(false)
  const [visible, setVisible] = useState(false)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    const unsub = apiLoading.subscribe(() => {
      const isActive = apiLoading.active
      setActive(isActive)
      if (isActive) {
        setVisible(true)
        if (timerRef.current) clearTimeout(timerRef.current)
      } else {
        timerRef.current = setTimeout(() => setVisible(false), 600)
      }
    })
    return () => { unsub() }
  }, [])

  if (!visible) return null

  return (
    <div style={{
      position: 'absolute', top: 0, left: 0, right: 0, height: 2,
      zIndex: 99999, overflow: 'hidden', pointerEvents: 'none',
    }}>
      <div style={{ position: 'absolute', inset: 0, background: 'rgba(79,142,247,0.12)' }} />
      <div style={{
        position: 'absolute', top: 0, left: 0, height: '100%', width: '100%',
        background: 'linear-gradient(90deg, #4f8ef7 0%, #a78bfa 50%, #4f8ef7 100%)',
        backgroundSize: '200% 100%',
        animation: active ? 'topbar-shimmer 1.4s linear infinite' : 'topbar-done 0.45s ease-out forwards',
        boxShadow: '0 0 10px 2px rgba(79,142,247,0.6)',
      }} />
    </div>
  )
}
