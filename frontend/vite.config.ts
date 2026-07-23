import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import { fileURLToPath } from 'node:url'

const BACKEND = process.env.NEXTEXT_BACKEND_ORIGIN ?? 'http://localhost:8000'

export default defineConfig({
  plugins: [react()],
  base: '/nextext/',
  resolve: {
    alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) },
  },
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      // SPA emits /nextext/api/... ; backend serves /api/... — strip here in dev
      // (in prod the app's own nginx strips it).
      '/nextext/api': {
        target: BACKEND,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/nextext/, ''),
      },
    },
  },
  test: {
    environment: 'happy-dom',
    // vmForks isolates each worker in its own process+VM, which allows happy-dom
    // to properly provide localStorage/sessionStorage on Node ≥26 (where Node itself
    // adds a stub `localStorage` to globalThis that vitest's getWindowKeys otherwise
    // blocks happy-dom from overriding).
    pool: 'vmForks',
    setupFiles: ['./src/test/setup.ts'],
  },
})
