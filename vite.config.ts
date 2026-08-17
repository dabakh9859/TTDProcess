import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Port 1420 is what tauri.conf.json expects for `devUrl` once you wire the
// dev command back up. Keep them in sync if you change it.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    watch: {
      // src-tauri holds 4.6 GB of build artifacts and the embedded Python
      // runtime — never let the dev server walk it.
      ignored: ['**/src-tauri/**', '**/recovered/**'],
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    target: 'chrome105',
    sourcemap: true,
  },
})
