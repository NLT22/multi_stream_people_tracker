"""Tiny local web app for assisted manual ReID identity grouping (zero deps).

MMPTracking person_id is scene-local, so each real person is split into many
`(scene, person_id)` scene-tracks. This app shows each scene-track as 3 crops and
lets you group them into real people by clicking, then saves the labels.

Usage:
    python scripts/datasets/reid_label_app.py            # serves http://localhost:8000
    # open the URL, pick an environment, group the cards, Save.
Outputs: reid_labels/labels_<env>.json   {trackKey: group_id}   (git-tracked, portable)
Then:    python scripts/datasets/apply_reid_labels.py  (builds the consolidated cache)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

CACHE = os.path.abspath("dataset/MMPTracking_10minute_reid_cache")
# Optional auto-clustering proposal (gitignored output/). If absent, fall back to
# the crop-cache manifest (built from the dataset) so the app is portable across
# machines with no output/ files — each scene-track simply starts as its own group.
PROPOSAL = "output/reid_consolidation/train_consolidated_manifest.csv"
CACHE_MANIFEST = os.path.join(CACHE, "train", "manifest.csv")
# Tracked (git-committable) so the manual labels are portable across machines.
OUTDIR = "reid_labels"
os.makedirs(OUTDIR, exist_ok=True)


def env_of(scene: str) -> str:
    return "_".join(scene.split("_")[1:-1])


def _load_tracks():
    """Group crops by scene-track (scene, orig_id) and pick a starting group id.
    Prefers the auto-clustering proposal; else uses the crop-cache manifest
    (no proposal -> each scene-track is its own group)."""
    if os.path.exists(PROPOSAL):
        src, gidcol = PROPOSAL, "gid"
    elif os.path.exists(CACHE_MANIFEST):
        src, gidcol = CACHE_MANIFEST, "pid"  # no proposal -> pid (each track separate)
    else:
        raise SystemExit(
            f"[reid] Need either {PROPOSAL} or {CACHE_MANIFEST}.\n"
            f"  Build the crop cache first:\n"
            f"    python -m scripts.datasets.build_reid_crop_cache "
            f"--src-root dataset/MMPTracking_10minute "
            f"--output-dir dataset/MMPTracking_10minute_reid_cache --exclude-retail")
    paths: dict[tuple, list[str]] = defaultdict(list)
    gid: dict[tuple, int] = {}
    for r in csv.DictReader(open(src)):
        oid = r.get("orig_id") or r.get("orig_pid") or r.get("pid")
        key = (r["scene"], oid)
        paths[key].append(r["rel_path"])
        gid[key] = int(r[gidcol])
    print(f"[reid] tracks loaded from {src}")
    return paths, gid


_track_paths, _track_gid = _load_tracks()

_env_cache: dict[str, list] = {}


def top_relpaths(paths: list[str], n: int = 3) -> list[str]:
    """n crops spread across time, largest JPEG (proxy for biggest/clearest) per chunk."""
    import numpy as np
    paths = sorted(paths)
    n = max(1, n)
    chunks = np.array_split(paths, n) if len(paths) >= n else [paths]
    out = []
    for ch in chunks:
        sub = list(ch)
        best, bs = None, -1
        for p in sub[:: max(1, len(sub) // 12)] or sub:
            try:
                s = os.path.getsize(os.path.join(CACHE, p))
            except OSError:
                s = -1
            if s > bs:
                bs, best = s, p
        if best:
            out.append(best)
    return out


def env_tracks(env: str, n: int = 3) -> list:
    ck = (env, n)
    if ck not in _env_cache:
        ks = sorted([k for k in _track_paths if env_of(k[0]) == env],
                    key=lambda k: (_track_gid[k], k[0], int(k[1])))
        out = []
        for idx, k in enumerate(ks, 1):
            scene, op = k
            out.append({"key": f"{scene}|{op}", "idx": idx, "scene": scene,
                        "orig_id": op, "proposed": _track_gid[k],
                        "crops": top_relpaths(_track_paths[k], n)})
        _env_cache[ck] = out
    return _env_cache[ck]


ENVS = sorted({env_of(s) for s, _ in _track_paths})

PAGE = """<!doctype html><html><head><meta charset=utf-8><title>ReID labeler</title>
<style>
:root{--imgh:84px}
body{font-family:sans-serif;margin:0;background:#1e1e1e;color:#ddd}
#bar{position:sticky;top:0;background:#111;padding:8px 12px;z-index:10;border-bottom:1px solid #333}
#bar a{color:#6cf;margin-right:10px;text-decoration:none}
#bar a.cur{color:#fc6;font-weight:bold}
button{background:#2a2a2a;color:#ddd;border:1px solid #555;padding:5px 10px;border-radius:4px;cursor:pointer;margin:0 3px}
button:hover{background:#3a3a3a}
#cols{display:flex;flex-wrap:wrap;gap:10px;padding:12px;align-items:flex-start}
.col{background:#262626;border:1px solid #444;border-radius:6px;min-width:150px;max-width:420px}
.colh{padding:6px 8px;background:#333;border-bottom:1px solid #444;cursor:pointer;font-weight:bold;border-radius:6px 6px 0 0}
.colh:hover{background:#3d5}
.col.junk .colh{background:#622}
.cards{display:flex;flex-wrap:wrap;gap:6px;padding:6px;min-height:30px}
.card{border:3px solid transparent;border-radius:4px;background:#111;padding:2px;cursor:pointer}
.card.sel{border-color:#6cf;box-shadow:0 0 6px #6cf}
.card .imgs{display:flex;flex-wrap:wrap;gap:2px;max-width:calc(var(--imgh) * 3.4)}
.card img{height:var(--imgh);width:auto;display:block}
.cap{font-size:10px;color:#9c9;text-align:center}
#status{color:#6f6;margin-left:10px}
</style></head><body>
<div id=bar>
  <span id=envs></span> | <span id=stat></span>
  | imgs/track <select id=nsel onchange="setN(this.value)">
    <option>3</option><option>4</option><option selected>5</option><option>6</option>
    <option>8</option><option>10</option><option>12</option><option>16</option></select>
  | size <select id=szsel onchange="setSize(this.value)">
    <option>60</option><option>72</option><option selected>84</option><option>110</option>
    <option>140</option><option>180</option><option>240</option></select>
  <button onclick=newPerson()>+ New person</button>
  <button onclick=moveSel('JUNK')>Discard selected</button>
  <button onclick=clearSel()>Clear selection</button>
  <button onclick=save()>SAVE</button><span id=status></span>
  <div style="font-size:12px;color:#888;margin-top:4px">Click cards to select (blue). Click a person's header to move selected cards there. Same real person = one column.</div>
</div>
<div id=cols></div>
<script>
let ENV=null, tracks=[], assign={}, sel=new Set(), N=5;
function qs(){return new URLSearchParams(location.search)}
async function fetchTracks(){const r=await fetch('/api/tracks?env='+ENV+'&n='+N);return await r.json()}
async function load(env){
  ENV=env; const d=await fetchTracks();
  tracks=d.tracks; assign={};
  // restore saved or use proposal
  const sv=d.saved||{};
  tracks.forEach(t=>{assign[t.key]= (t.key in sv)? sv[t.key] : ('P'+t.proposed)});
  sel.clear(); render();
  document.querySelectorAll('#envs a').forEach(a=>a.className=(a.dataset.env==env?'cur':''));
}
async function setN(n){ N=+n; const d=await fetchTracks(); tracks=d.tracks;
  tracks.forEach(t=>{if(!(t.key in assign))assign[t.key]='P'+t.proposed}); render(); }
function setSize(px){ document.documentElement.style.setProperty('--imgh', px+'px'); }
function groups(){
  const g={}; tracks.forEach(t=>{const k=assign[t.key]; (g[k]=g[k]||[]).push(t)}); return g;
}
function render(){
  const g=groups(); const keys=Object.keys(g).sort((a,b)=>{
    if(a=='JUNK')return 1; if(b=='JUNK')return -1;
    return (+a.slice(1)||0)-(+b.slice(1)||0)});
  const cont=document.getElementById('cols'); cont.innerHTML='';
  keys.forEach(k=>{
    const col=document.createElement('div'); col.className='col'+(k=='JUNK'?' junk':'');
    const h=document.createElement('div'); h.className='colh';
    h.textContent=(k=='JUNK'?'DISCARD':k)+' ('+g[k].length+')';
    h.onclick=()=>moveSel(k); col.appendChild(h);
    const cd=document.createElement('div'); cd.className='cards';
    g[k].forEach(t=>{
      const c=document.createElement('div'); c.className='card'+(sel.has(t.key)?' sel':'');
      c.onclick=()=>{sel.has(t.key)?sel.delete(t.key):sel.add(t.key); render()};
      const im=document.createElement('div'); im.className='imgs';
      t.crops.forEach(cp=>{const i=document.createElement('img'); i.src='/crop?p='+encodeURIComponent(cp); im.appendChild(i)});
      c.appendChild(im);
      const cap=document.createElement('div'); cap.className='cap'; cap.textContent='#'+t.idx+' '+t.scene.split('_')[0]+' id'+t.orig_id;
      c.appendChild(cap); cd.appendChild(c);
    });
    col.appendChild(cd); cont.appendChild(col);
  });
  document.getElementById('stat').textContent=tracks.length+' tracks, '+
     (keys.filter(k=>k!='JUNK').length)+' people'+(g['JUNK']?', '+g['JUNK'].length+' discarded':'');
}
function moveSel(k){ if(sel.size==0)return; sel.forEach(key=>assign[key]=k); sel.clear(); render(); }
function newPerson(){ let m=0; Object.values(assign).forEach(v=>{if(v[0]=='P')m=Math.max(m,+v.slice(1)||0)});
  const k='P'+(m+1); if(sel.size){moveSel(k)}else{alert('Select cards first, then they go to '+k)} }
function clearSel(){sel.clear();render()}
async function save(){
  const r=await fetch('/api/save?env='+ENV,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(assign)});
  const d=await r.json(); document.getElementById('status').textContent=' saved '+d.n+' tracks @ '+new Date().toLocaleTimeString();
}
window.onload=()=>{
  const e=document.getElementById('envs'); ENVLIST.forEach(en=>{const a=document.createElement('a');a.textContent=en;a.dataset.env=en;a.href='#';a.onclick=()=>load(en);e.appendChild(a)});
  load(qs().get('env')||ENVLIST[0]);
};
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/":
            html = PAGE.replace("ENVLIST", json.dumps(ENVS))
            self._send(200, "text/html; charset=utf-8", html.encode())
        elif u.path == "/api/tracks":
            env = q.get("env", [ENVS[0]])[0]
            n = int(q.get("n", ["3"])[0])
            saved = {}
            sp = f"{OUTDIR}/labels_{env}.json"
            if os.path.exists(sp):
                saved = json.load(open(sp))
            self._send(200, "application/json",
                       json.dumps({"tracks": env_tracks(env, n), "saved": saved}).encode())
        elif u.path == "/crop":
            rel = q.get("p", [""])[0]
            fp = os.path.abspath(os.path.join(CACHE, rel))
            if not fp.startswith(CACHE) or not os.path.exists(fp):
                self._send(404, "text/plain", b"no")
                return
            self._send(200, "image/jpeg", open(fp, "rb").read())
        else:
            self._send(404, "text/plain", b"no")

    def do_POST(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/api/save":
            env = q.get("env", [""])[0]
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n) or b"{}")
            json.dump(data, open(f"{OUTDIR}/labels_{env}.json", "w"), indent=0)
            self._send(200, "application/json", json.dumps({"n": len(data)}).encode())
        else:
            self._send(404, "text/plain", b"no")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    print(f"Environments: {ENVS}")
    print(f"Serving ReID labeler at  http://localhost:{args.port}   (Ctrl+C to stop)")
    print(f"Saves to {OUTDIR}/labels_<env>.json")
    ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()


if __name__ == "__main__":
    main()
