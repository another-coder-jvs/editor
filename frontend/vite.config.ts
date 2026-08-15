import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path' 
let baseUrl = null

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { '@': path.resolve(__dirname, './src') },
  },
  server: {
    port: 5173,
    host: "127.0.0.1",
    strictPort:true,
    proxy: {
      '/detect': baseUrl || 'http://localhost:8000',
      '/segment': baseUrl || 'http://localhost:8000',
      '/layers': baseUrl || 'http://localhost:8000',
      '/edit': baseUrl || 'http://localhost:8000',
      '/merge': baseUrl || 'http://localhost:8000',
      '/project': baseUrl || 'http://localhost:8000',
      '/export': baseUrl || 'http://localhost:8000',
      '/progress': baseUrl || 'http://localhost:8000',
      '/outputs': baseUrl || 'http://localhost:8000',
      '/text': baseUrl || 'http://localhost:8000',
      '/temp': baseUrl || 'http://localhost:8000',
    },
  },
})
