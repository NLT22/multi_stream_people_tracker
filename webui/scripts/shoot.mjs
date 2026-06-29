// Capture webui screenshots for the report. Drives the installed Chrome via
// puppeteer-core, seeds the W022 custom-analytics zones (reconstructed from the
// saved nvdsanalytics configs) into localStorage, then shoots the ROI editor.
import puppeteer from 'puppeteer-core'

const N = (px, py) => ({ x: px / 1920, y: py / 1080 })
const pts = (arr) => { const o = []; for (let i = 0; i < arr.length; i += 2) o.push(N(arr[i], arr[i + 1])); return o }

// reconstructed from configs/analytics/nvdsanalytics_0..3.txt
const ROIS = [
  { id: 'roi-w0', name: 'OC-1', kind: 'overcrowd', cameraId: 'cam-0',
    points: pts([859,102, 594,1037, 1422,1063, 1038,98]), analytics: ['occupancy'], threshold: 2, enabled: true },
  { id: 'roi-w1', name: 'Line-1', kind: 'counting', cameraId: 'cam-1',
    points: pts([746,422, 1006,424]), direction: pts([859,357, 868,499]), analytics: ['lineCrossing'], enabled: true },
  { id: 'roi-w2', name: 'Zone-1', kind: 'detection', cameraId: 'cam-2',
    points: pts([765,37, 659,529, 1221,545, 1113,41]), analytics: ['counting'], enabled: true },
  { id: 'roi-w3', name: 'Restricted-1', kind: 'restricted', cameraId: 'cam-3',
    points: pts([739,224, 647,530, 1194,551, 1123,206]), analytics: ['intrusion'], enabled: true },
]

const URL = 'http://localhost:5180/'
const OUT = '../report/latex/Images'
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
const clickByText = async (p, re) => p.evaluate((src) => {
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
await p.goto(URL, { waitUntil: 'networkidle2' })
await p.evaluate((rois) => localStorage.setItem('sentinel.rois.mtmc', JSON.stringify(rois)), ROIS)
await p.reload({ waitUntil: 'networkidle2' })
await clickByText(p, /^MTMC$/)            // dataset toggle
await sleep(500)
await clickByText(p, /ROI Editor/)
await sleep(900)

for (const [cam, tag] of [['cam-0', 'overcrowd'], ['cam-1', 'linecross'], ['cam-2', 'zone'], ['cam-3', 'restricted']]) {
  try { await p.select('.roi__camsel', cam) } catch (e) { console.log('select fail', cam, String(e)) }
  await sleep(700)
  // select the region so the inspector shows it
  await p.evaluate(() => { const it = document.querySelector('.roi__item'); if (it) it.dispatchEvent(new MouseEvent('click', { bubbles: true })) })
  await sleep(400)
  await p.screenshot({ path: `${OUT}/webui_roi_w022_${tag}.png` })
  console.log('shot', tag)
}
// overview shot of the editor (full)
await p.select('.roi__camsel', 'cam-0').catch(() => {})
await sleep(500)
await p.screenshot({ path: `${OUT}/webui_custom_analytics_w022.png` })
console.log('shot overview')
await b.close()
