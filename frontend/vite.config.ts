import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { '@': path.resolve(__dirname, './src') },
  },
  server: {
    port: 5173,
    proxy: {
      '/detect': 'http://localhost:8000',
      '/segment': 'http://localhost:8000',
      '/layers': 'http://localhost:8000',
      '/edit': 'http://localhost:8000',
      '/merge': 'http://localhost:8000',
      '/project': 'http://localhost:8000',
      '/export': 'http://localhost:8000',
      '/progress': 'http://localhost:8000',
      '/outputs': 'http://localhost:8000',
      '/temp': 'http://localhost:8000',
    },
  },
})
