// Capture the Ask chat answering a natural-language question via gpt-4o.
// Needs: webui dev (:5180) + RAG API (:8077, with OPENAI_API_KEY in .env) running.
import puppeteer from 'puppeteer-core'

const URL = 'http://localhost:5180/#ask'
const Q = process.env.Q || 'which area got the most attention?'
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

const input = await p.waitForSelector('.ask__input')
await input.type(Q)
await p.click('.ask__send')

// wait until "analysing…" is gone AND both bubbles (question + answer) are rendered
await p.waitForFunction(() => {
  return document.querySelectorAll('.ask__dots').length === 0 &&
         document.querySelectorAll('.ask__msg').length >= 2
}, { timeout: 30000 })
await sleep(600)
await p.screenshot({ path: `${OUT}/webui_ask_chat.png` })
console.log('shot: chat answer')
await b.close()
