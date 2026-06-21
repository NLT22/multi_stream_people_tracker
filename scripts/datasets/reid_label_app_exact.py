"""Manual ReID grouping UI for official MMPTracking ReID crop manifests.

This is the original-dataset equivalent of the retired 10-minute label app.
Build crops from the official MMPTracking zip tree first:

    ./venv/bin/python scripts/datasets/mmp_exact_to_reid.py \
      --mmp-root dataset/MMPTracking \
      --output-dir dataset/mmp_exact_reid_original \
      --splits train \
      --sample-rate 20

Then run:

    ./venv/bin/python scripts/datasets/reid_label_app_exact.py \
      --crop-root dataset/mmp_exact_reid_original

Open http://localhost:8000 and save labels to reid_labels_exact/.
Saved labels are {pid_key: group_id}, where pid_key is time/scene/raw_pid.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse


def env_of(scene: str) -> str:
    parts = scene.split("_")
    return "_".join(parts[:-1]) if parts and parts[-1].isdigit() else scene


def top_relpaths(rows: list[dict[str, str]], n: int) -> list[str]:
    rows = sorted(rows, key=lambda r: int(r.get("frame", 0)))
    if len(rows) <= n:
        return [r["rel_path"] for r in rows]
    out: list[str] = []
    for i in range(n):
        idx = round(i * (len(rows) - 1) / max(1, n - 1))
        out.append(rows[idx]["rel_path"])
    return out


def load_tracks(crop_root: Path, split: str) -> tuple[dict[str, list[dict[str, str]]], dict[str, str]]:
    manifest = crop_root / split / "manifest.csv"
    if not manifest.exists():
        raise SystemExit(f"manifest not found: {manifest}")

    tracks: dict[str, list[dict[str, str]]] = defaultdict(list)
    scenes: dict[str, str] = {}
    with manifest.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            key = row.get("pid_key")
            if not key:
                key = f"{row.get('time', split)}/{row['scene']}/{row.get('raw_pid', row['pid'])}"
            tracks[key].append(row)
            scenes[key] = row["scene"]
    print(f"[exact-label] loaded {len(tracks)} tracks from {manifest}")
    return tracks, scenes


PAGE = """<!doctype html><html><head><meta charset=utf-8><title>Exact MMP ReID Labeler</title>
<style>
:root{--imgh:92px}
body{font-family:sans-serif;margin:0;background:#1e1e1e;color:#ddd}
#bar{position:sticky;top:0;background:#111;padding:8px 12px;z-index:10;border-bottom:1px solid #333}
#bar a{color:#6cf;margin-right:10px;text-decoration:none}
#bar a.cur{color:#fc6;font-weight:bold}
button{background:#2a2a2a;color:#ddd;border:1px solid #555;padding:5px 10px;border-radius:4px;cursor:pointer;margin:0 3px}
button:hover{background:#3a3a3a}
#cols{display:flex;flex-wrap:wrap;gap:10px;padding:12px;align-items:flex-start}
.col{background:#262626;border:1px solid #444;border-radius:6px;min-width:170px;max-width:460px}
.colh{padding:6px 8px;background:#333;border-bottom:1px solid #444;cursor:pointer;font-weight:bold;border-radius:6px 6px 0 0}
.colh:hover{background:#3d5}
.col.junk .colh{background:#622}
.cards{display:flex;flex-wrap:wrap;gap:6px;padding:6px;min-height:30px}
.card{border:3px solid transparent;border-radius:4px;background:#111;padding:2px;cursor:pointer}
.card.sel{border-color:#6cf;box-shadow:0 0 6px #6cf}
.imgs{display:flex;flex-wrap:wrap;gap:2px;max-width:calc(var(--imgh) * 4.5)}
.card img{height:var(--imgh);width:auto;display:block}
.cap{font-size:10px;color:#9c9;text-align:center;max-width:380px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
#status{color:#6f6;margin-left:10px}
</style></head><body>
<div id=bar>
  <span id=envs></span> | <span id=stat></span>
  | imgs/track <select id=nsel onchange="setN(this.value)">
    <option>3</option><option selected>5</option><option>8</option>
    <option>12</option><option>20</option><option>30</option></select>
  | size <select onchange="setSize(this.value)">
    <option>72</option><option selected>92</option><option>120</option><option>160</option><option>220</option></select>
  <button onclick=newPerson()>+ New person</button>
  <button onclick=moveSel('JUNK')>Discard selected</button>
  <button onclick=clearSel()>Clear</button>
  <button onclick=save()>SAVE</button><span id=status></span>
</div>
<div id=cols></div>
<script>
let ENV=null, tracks=[], assign={}, sel=new Set(), N=5;
async function fetchTracks(){const r=await fetch('/api/tracks?env='+ENV+'&n='+N);return await r.json()}
async function load(env){
  ENV=env; const d=await fetchTracks(); tracks=d.tracks; assign={};
  const sv=d.saved||{};
  tracks.forEach((t,i)=>{assign[t.key]=(t.key in sv)?sv[t.key]:('P'+(i+1))});
  sel.clear(); render();
  document.querySelectorAll('#envs a').forEach(a=>a.className=(a.dataset.env==env?'cur':''));
}
async function setN(n){N=+n; const d=await fetchTracks(); tracks=d.tracks; render()}
function setSize(px){document.documentElement.style.setProperty('--imgh', px+'px')}
function groups(){const g={}; tracks.forEach(t=>{const k=assign[t.key]||'P0'; (g[k]=g[k]||[]).push(t)}); return g}
function groupSort(a,b){if(a=='JUNK')return 1;if(b=='JUNK')return -1;return (+a.slice(1)||0)-(+b.slice(1)||0)}
function render(){
  const g=groups(), keys=Object.keys(g).sort(groupSort), cont=document.getElementById('cols'); cont.innerHTML='';
  keys.forEach(k=>{
    const col=document.createElement('div'); col.className='col'+(k=='JUNK'?' junk':'');
    const h=document.createElement('div'); h.className='colh'; h.textContent=(k=='JUNK'?'DISCARD':k)+' ('+g[k].length+')';
    h.onclick=()=>moveSel(k); col.appendChild(h);
    const cd=document.createElement('div'); cd.className='cards';
    g[k].forEach(t=>{
      const c=document.createElement('div'); c.className='card'+(sel.has(t.key)?' sel':'');
      c.onclick=()=>{sel.has(t.key)?sel.delete(t.key):sel.add(t.key); render()};
      const im=document.createElement('div'); im.className='imgs';
      t.crops.forEach(cp=>{const i=document.createElement('img'); i.src=cp; im.appendChild(i)});
      c.appendChild(im);
      const cap=document.createElement('div'); cap.className='cap'; cap.textContent=t.key;
      c.appendChild(cap); cd.appendChild(c);
    });
    col.appendChild(cd); cont.appendChild(col);
  });
  document.getElementById('stat').textContent=tracks.length+' tracks, '+keys.filter(k=>k!='JUNK').length+' people';
}
function moveSel(k){if(sel.size==0)return; sel.forEach(key=>assign[key]=k); sel.clear(); render()}
function newPerson(){let m=0; Object.values(assign).forEach(v=>{if(v[0]=='P')m=Math.max(m,+v.slice(1)||0)}); moveSel('P'+(m+1))}
function clearSel(){sel.clear(); render()}
async function save(){
  const r=await fetch('/api/save?env='+ENV,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(assign)});
  const d=await r.json(); document.getElementById('status').textContent=' saved '+d.n+' tracks @ '+new Date().toLocaleTimeString();
}
window.onload=()=>{
  const e=document.getElementById('envs');
  ENVLIST.forEach(en=>{const a=document.createElement('a');a.textContent=en;a.dataset.env=en;a.href='#';a.onclick=()=>load(en);e.appendChild(a)});
  load(ENVLIST[0]);
}
</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-root", type=Path, default=Path("dataset/mmp_exact_reid_original"))
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--out-dir", type=Path, default=Path("reid_labels_exact"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    crop_root = args.crop_root.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tracks, scenes = load_tracks(crop_root, args.split)
    envs = sorted({env_of(scene) for scene in scenes.values()})

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *vals):
            return

        def _send_json(self, obj) -> None:
            data = json.dumps(obj).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query)
            if parsed.path == "/":
                html = PAGE.replace("ENVLIST", json.dumps(envs)).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)
                return
            if parsed.path == "/api/tracks":
                env = qs.get("env", [envs[0]])[0]
                n = int(qs.get("n", ["5"])[0])
                out = []
                for key in sorted(tracks):
                    scene = scenes[key]
                    if env_of(scene) != env:
                        continue
                    out.append(
                        {
                            "key": key,
                            "scene": scene,
                            "crops": ["/crop?p=" + quote(p) for p in top_relpaths(tracks[key], n)],
                        }
                    )
                label_path = args.out_dir / f"labels_{env}.json"
                saved = json.load(label_path.open()) if label_path.exists() else {}
                self._send_json({"tracks": out, "saved": saved})
                return
            if parsed.path == "/crop":
                rel = qs.get("p", [""])[0]
                path = (crop_root / rel).resolve()
                if not str(path).startswith(str(crop_root)) or not path.exists():
                    self.send_error(404)
                    return
                data = path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            self.send_error(404)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/api/save":
                self.send_error(404)
                return
            env = parse_qs(parsed.query).get("env", ["unknown"])[0]
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length) or b"{}")
            path = args.out_dir / f"labels_{env}.json"
            with path.open("w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, sort_keys=True)
            self._send_json({"ok": True, "n": len(data), "path": str(path)})

    print(f"[exact-label] open http://{args.host}:{args.port}")
    print(f"[exact-label] labels -> {args.out_dir}")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
