import React, { useEffect, useRef, useState } from 'react'
import { apiLoading } from '../api/client'

export const ApiSpinner: React.FC = () => {
  const [visible, setVisible] = useState(false)
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    const unsub = apiLoading.subscribe(() => {
      if (apiLoading.active) {
        if (timer.current) clearTimeout(timer.current)
        setVisible(true)
      } else {
        timer.current = setTimeout(() => setVisible(false), 400)
      }
    })
    return () => { unsub() }
  }, [])

  if (!visible) return null

  return (
    <div style={{
      width: 18, height: 18, flexShrink: 0,
      borderRadius: '50%',
      border: '2px solid rgba(79,142,247,0.2)',
      borderTopColor: '#4f8ef7',
      animation: 'api-spin 0.7s linear infinite',
      boxShadow: '0 0 6px rgba(79,142,247,0.5)',
    }} />
  )
}
