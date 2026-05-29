import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { npzOpenPlugin } from './vite.npzOpenPlugin'

// In dev, the browser may load the UI via the machine's hostname (e.g. 129.x:5173). Requests to
// localhost:11435 would then hit the user's laptop, not this host. Proxies keep fetches same-origin.
const superdecTarget = process.env.VITE_DEV_PROXY_SUPERDEC ?? 'http://127.0.0.1:11435'
const superflexTarget = process.env.VITE_DEV_PROXY_SUPERFLEX ?? 'http://127.0.0.1:11436'
const trellisTarget = process.env.VITE_DEV_PROXY_TRELLIS ?? 'http://127.0.0.1:11437'
const spaceflowTarget = process.env.VITE_DEV_PROXY_SPACEFLOW ?? 'http://127.0.0.1:11438'
const ollamaTarget = process.env.VITE_DEV_PROXY_OLLAMA ?? 'http://127.0.0.1:11434'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), npzOpenPlugin()],
  server: {
    watch: {
      usePolling: true,
      interval: 1000,
    },
    proxy: {
      '/superdec': { target: superdecTarget, changeOrigin: true },
      '/superflex': { target: superflexTarget, changeOrigin: true },
      '/trellis': { target: trellisTarget, changeOrigin: true },
      '/spaceflow': { target: spaceflowTarget, changeOrigin: true },
      '/api': { target: ollamaTarget, changeOrigin: true },
    },
  },
})
