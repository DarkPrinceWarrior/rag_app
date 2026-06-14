import path from 'node:path'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { tanstackRouter } from '@tanstack/router-plugin/vite'

// SPA веб-приложения (roadmap § 7). Прод-сборка отдаётся FastAPI как статика
// (см. deploy/build_web.sh). В dev /api проксируется на API (через SSH-туннель
// на localhost:8100). Оффлайн: все зависимости бандлятся, внешних загрузок нет.
export default defineConfig({
  plugins: [
    tanstackRouter({ target: 'react', autoCodeSplitting: true }),
    react(),
    tailwindcss(),
  ],
  resolve: {
    alias: { '@': path.resolve(import.meta.dirname, './src') },
  },
  server: {
    proxy: {
      '/api': 'http://localhost:8100',
      '/healthz': 'http://localhost:8100',
    },
  },
})
