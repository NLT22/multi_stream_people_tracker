// Capture the Ask view's image → identity function for the report.
// Uploads a real person crop into Person Search, waits for candidates,
// clicks the top one, and screenshots the resulting dwell/BEV/timeline.
// Needs: webui dev (:5180) + RAG API (:8077) running, and a crop path in CROP.
import puppeteer from 'puppeteer-core'

const URL = 'http://localhost:5180/#ask'
const CROP = process.env.CROP
const OUT = '../report/latex/Images'
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

const b = await puppeteer.launch({
  executablePath: '/usr/bin/google-chrome',
  headless: 'new', args: ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage'],
})
const p = await b.newPage()
await p.setViewport({ width: 1600, height: 1000, deviceScaleFactor: 2 })
await p.goto(URL, { waitUntil: 'networkidle2' })
await sleep(600)

// upload the crop into Person Search (hidden file input)
const input = await p.waitForSelector('.ask__drop input[type=file]')
await input.uploadFile(CROP)

// wait for candidate list, then click the top candidate
await p.waitForSelector('.ask__cand', { timeout: 8000 })
await sleep(500)
await p.screenshot({ path: `${OUT}/webui_ask_image_candidates.png` })
await p.click('.ask__cand')                       // top match -> loads dwell/BEV/timeline
await p.waitForSelector('.ask__person', { timeout: 8000 })
await sleep(900)
await p.screenshot({ path: `${OUT}/webui_ask_image_search.png` })
console.log('shot: candidates + person detail')
await b.close()
