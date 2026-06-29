import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { writeFileSync, mkdirSync } from 'fs'
import { resolve } from 'path'

// Dev-only autosave: the ROI editor POSTs {name, text} here and the config is
// written straight to ../configs/analytics/ — no manual Export, no extra server.
function analyticsSaver() {
  return {
    name: 'analytics-saver',
    configureServer(server: any) {
      server.middlewares.use('/__save-analytics', (req: any, res: any) => {
        if (req.method !== 'POST') { res.statusCode = 405; return res.end() }
        let body = ''
        req.on('data', (c: any) => (body += c))
        req.on('end', () => {
          try {
            const { name, text } = JSON.parse(body)
            const safe = String(name).replace(/[^A-Za-z0-9_.-]/g, '_')
            const dir = resolve(process.cwd(), '../configs/analytics')
            mkdirSync(dir, { recursive: true })
            writeFileSync(resolve(dir, safe), String(text))
            res.setHeader('content-type', 'application/json')
            res.end(JSON.stringify({ ok: true, path: `configs/analytics/${safe}` }))
          } catch (e) {
            res.statusCode = 400; res.end(JSON.stringify({ ok: false, error: String(e) }))
          }
        })
      })
    },
  }
}

// Public assets (camera frames, heatmaps, OSD videos) are placed/symlinked
// under public/ by setup-assets.sh and served at the web root.
export default defineConfig({
  plugins: [react(), analyticsSaver()],
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
