import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Public assets (camera frames, heatmaps, OSD videos) are placed/symlinked
// under public/ by setup-assets.sh and served at the web root.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5180,
    host: true,
    // Don't file-watch the static asset dirs — heatmaps/feeds and especially the
    // churning live HLS segments otherwise exhaust the inotify limit (ENOSPC) and
    // crash the dev server. They're served statically; they need no HMR.
    watch: {
      ignored: [
        '**/public/live/**',
        '**/public/feeds/**',
        '**/public/heatmaps/**',
        '**/public/frames/**',
      ],
    },
  },
})
