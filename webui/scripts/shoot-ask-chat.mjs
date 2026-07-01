// Capture the Ask chat answering a natural-language question via gpt-4o.
// Needs: webui dev (:5180) + RAG API (:8077, with OPENAI_API_KEY in .env) running.
// Env: Q=<question>  OUTNAME=<png stem>  MTMC=1 (toggle to the MTMC dataset first)
import puppeteer from 'puppeteer-core'

const BASE = 'http://localhost:5180/'
const Q = process.env.Q || 'which area got the most attention?'
const OUTNAME = process.env.OUTNAME || 'webui_ask_chat'
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

const input = await p.waitForSelector('.ask__input')
await input.type(Q)
await p.click('.ask__send')
await p.waitForFunction(() => {
  return document.querySelectorAll('.ask__dots').length === 0 &&
         document.querySelectorAll('.ask__msg').length >= 2
}, { timeout: 30000 })
await sleep(600)
await p.screenshot({ path: `${OUT}/${OUTNAME}.png` })
console.log('shot:', OUTNAME)
await b.close()
