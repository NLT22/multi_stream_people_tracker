// Capture the Ask view's image → identity function for the report.
// Uploads a real person crop into Person Search, waits for candidates, clicks the
// top one, and screenshots the resulting dwell/BEV/timeline.
// Needs: webui dev (:5180) + RAG API (:8077) running, and a crop path in CROP.
// Env: CROP=<jpg>  OUTNAME=<png stem>  MTMC=1 (toggle to the MTMC dataset first)
import puppeteer from 'puppeteer-core'

const BASE = 'http://localhost:5180/'
const CROP = process.env.CROP
const OUTNAME = process.env.OUTNAME || 'webui_ask_image_search'
const OUT = '../report/latex/Images'
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
const clickByText = (p, re) => p.evaluate((src) => {
  const rx = new RegExp(src, 'i')
  const b = [...document.querySelectorAll('button')].find((x) => rx.test(x.textContent || ''))
  if (b) b.click(); return !!b
}, re.source)

const b = await puppeteer.launch({
  executablePath: '/usr/bin/google-chrome',
  headless: 'new', args: ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage'],
})
const p = await b.newPage()
await p.setViewport({ width: 1600, height: 1000, deviceScaleFactor: 2 })
await p.goto(BASE, { waitUntil: 'networkidle2' })
await sleep(500)
if (process.env.MTMC) { await clickByText(p, /^MTMC$/); await sleep(500) }
await clickByText(p, /Ask$/)
await sleep(700)

// upload the crop into Person Search (hidden file input)
const input = await p.waitForSelector('.ask__drop input[type=file]')
await input.uploadFile(CROP)

// wait for candidate list, then click the top candidate
await p.waitForSelector('.ask__cand', { timeout: 8000 })
await sleep(500)
await p.click('.ask__cand')                       // top match -> loads dwell/BEV/timeline
await p.waitForSelector('.ask__person', { timeout: 8000 })
await sleep(900)
await p.screenshot({ path: `${OUT}/${OUTNAME}.png` })
console.log('shot:', OUTNAME)
await b.close()
