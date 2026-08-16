/**
 * Client-side layer merge using OffscreenCanvas / Canvas2D.
 * Handles blob:, data:, and http: layer URLs with adjustments applied.
 */
import { LayerData, LayerAdjustments } from '../types'
import { baseImagesUrl } from '../config'

const API_BASE = baseImagesUrl || 'http://localhost:8000'

function resolveUrl(png_path: string): string {
  if (!png_path) return ''
  if (png_path.startsWith('blob:') || png_path.startsWith('data:')) return png_path
  return `${API_BASE}${png_path}`
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise(async (resolve, reject) => {
    try {
      // Fetch through axios so ngrok-skip-browser-warning header is sent,
      // then draw from a local blob URL to avoid CORS issues.
      if (src.startsWith('http://') || src.startsWith('https://')) {
        const res = await fetch(src, { headers: { 'ngrok-skip-browser-warning': '1' } })
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const blob = await res.blob()
        src = URL.createObjectURL(blob)
      }
    } catch { /* fall through to direct load */ }

    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => resolve(img)
    img.onerror = () => reject(new Error(`Failed to load image`))
    img.src = src
  })
}

function buildFilter(adj: LayerAdjustments): string {
  const brightness = adj.brightness / 100
  const contrast = adj.contrast / 100
  const saturation = adj.saturation / 100
  const exposure = Math.pow(2, adj.exposure / 100)
  const sharpness = adj.sharpness > 100 ? (adj.sharpness - 100) / 100 : 0
  return [
    `brightness(${(brightness * exposure).toFixed(3)})`,
    `contrast(${contrast.toFixed(3)})`,
    `saturate(${saturation.toFixed(3)})`,
    `hue-rotate(${adj.hue}deg)`,
    sharpness > 0 ? `contrast(${(1 + sharpness * 0.3).toFixed(3)})` : '',
  ].filter(Boolean).join(' ')
}

export async function mergeLayersClient(
  layers: LayerData[],
  canvasWidth: number,
  canvasHeight: number,
  format: 'png' | 'jpeg' | 'webp' = 'png',
  quality = 0.92,
): Promise<Blob> {
  const canvas = document.createElement('canvas')
  canvas.width = canvasWidth
  canvas.height = canvasHeight
  const ctx = canvas.getContext('2d')!

  // White background for jpeg, transparent for png/webp
  if (format === 'jpeg') {
    ctx.fillStyle = '#ffffff'
    ctx.fillRect(0, 0, canvasWidth, canvasHeight)
  }

  const visible = [...layers]
    .filter(l => l.visible && l.png_path)
    .sort((a, b) => a.z_index - b.z_index)

  for (const layer of visible) {
    const url = resolveUrl(layer.png_path)
    if (!url) continue

    let img: HTMLImageElement
    try { img = await loadImage(url) } catch { continue }

    // Draw layer onto a temp canvas to apply transforms + filter
    const tmp = document.createElement('canvas')
    tmp.width = layer.bbox.width
    tmp.height = layer.bbox.height
    const tctx = tmp.getContext('2d')!

    // Apply CSS filter via offscreen canvas trick
    const adj = layer.adjustments
    if (adj) tctx.filter = buildFilter(adj)
    tctx.drawImage(img, 0, 0, layer.bbox.width, layer.bbox.height)
    tctx.filter = 'none'

    // Apply vignette
    if (adj?.vignette > 0) {
      const grad = tctx.createRadialGradient(
        layer.bbox.width / 2, layer.bbox.height / 2, 0,
        layer.bbox.width / 2, layer.bbox.height / 2,
        Math.max(layer.bbox.width, layer.bbox.height) / 2,
      )
      grad.addColorStop(1 - adj.vignette / 100, 'transparent')
      grad.addColorStop(1, `rgba(0,0,0,${adj.vignette / 100})`)
      tctx.fillStyle = grad
      tctx.fillRect(0, 0, layer.bbox.width, layer.bbox.height)
    }

    // Apply fade
    if (adj?.fade > 0) {
      tctx.fillStyle = `rgba(255,255,255,${adj.fade / 100})`
      tctx.fillRect(0, 0, layer.bbox.width, layer.bbox.height)
    }

    const px = layer.bbox.x + layer.position.x
    const py = layer.bbox.y + layer.position.y
    const cx = px + layer.bbox.width / 2
    const cy = py + layer.bbox.height / 2

    ctx.save()
    ctx.globalAlpha = layer.opacity
    ctx.globalCompositeOperation = (layer.blend_mode || 'normal') as GlobalCompositeOperation
    ctx.translate(cx, cy)
    ctx.rotate((layer.rotation * Math.PI) / 180)
    ctx.scale(layer.scale.x, layer.scale.y)
    ctx.drawImage(tmp, -layer.bbox.width / 2, -layer.bbox.height / 2)
    ctx.restore()
  }

  return new Promise((resolve, reject) => {
    canvas.toBlob(
      blob => blob ? resolve(blob) : reject(new Error('Canvas toBlob failed')),
      `image/${format}`,
      quality,
    )
  })
}
