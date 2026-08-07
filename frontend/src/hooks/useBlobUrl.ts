import { useState, useEffect } from 'react'

export function useBlobUrl(url: string | null): string | null {
  const [blobUrl, setBlobUrl] = useState<string | null>(null)

  useEffect(() => {
    if (!url) return
    if (url.startsWith("blob:") || url.startsWith("data:")) { setBlobUrl(url); return }
    let objectUrl: string
    fetch(url, { headers: { 'ngrok-skip-browser-warning': '1' } })
      .then(r => r.blob())
      .then(blob => {
        objectUrl = URL.createObjectURL(blob)
        setBlobUrl(objectUrl)
      })
      .catch(() => setBlobUrl(url)) // fallback to direct url
    return () => { if (objectUrl) URL.revokeObjectURL(objectUrl) }
  }, [url])

  return blobUrl
}
