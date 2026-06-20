"""
GT Editor — visual ground-truth cleaner for MMPTracking_short retail scenes.

Shows each person_id's bounding boxes overlaid on the video.
Per person you can: keep all, delete entirely, or mark one or more DELETE
RANGES (frames where the annotation is wrong/phantom — those frames are
removed; everything outside the ranges is kept).

Controls
--------
  n / →       next person
  p / ←       prev person
  d           delete person ENTIRELY (red)
  k           reset to KEEP ALL (green)
  r           mark a DELETE RANGE:
                press r  → sets range start at current frame  (yellow)
                press r  → sets range end   at current frame  → range confirmed
              repeat to add more ranges for the same person
  u           undo last confirmed range for current person
  s           save  →  gt_<cam>_clean.csv  (backs up original first)
  q / Esc     quit  (prompts save)

  Space       play / pause
  . / ,       step +1 / -1 frame
  0–9         jump to 0%–90% of video

Usage
-----
  python scripts/datasets/gt_editor.py \\
      --scene dataset/MMPTracking_short/retail_0 \\
      --cam   cam1

  # Continue editing a previously saved _clean.csv:
  python scripts/datasets/gt_editor.py \\
      --scene dataset/MMPTracking_short/retail_0 \\
      --cam   cam1 --use-clean
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false;qt.qpa.*=false")

import cv2
import numpy as np
import pandas as pd


# ── colours (BGR) ────────────────────────────────────────────────────────────
_PALETTE = [
    (0, 200, 255), (0, 255, 100), (255, 100, 0), (200, 0, 255),
    (0, 255, 255), (255, 200, 0), (100, 255, 0), (255, 0, 150),
    (0, 150, 255), (150, 255, 0),
]
_COL_KEEP   = (0,  220,   0)
_COL_DELETE = (0,   0,  220)
_COL_RANGE  = (0, 200,  255)   # confirmed delete range bar
_COL_PEND   = (0, 220,  220)   # pending range start


def _pal(idx: int) -> tuple:
    return _PALETTE[idx % len(_PALETTE)]


# ── per-person state ──────────────────────────────────────────────────────────

class PersonState:
    def __init__(self, pid: int):
        self.pid            = pid
        self.delete_all     = False
        self.delete_ranges: list[tuple[int, int]] = []  # (start, end) inclusive
        self.pending_start: int | None = None           # range being built

    @property
    def status(self) -> str:
        if self.delete_all:
            return "delete"
        if self.delete_ranges or self.pending_start is not None:
            return "range"
        return "keep"

    def frame_is_deleted(self, f: int) -> bool:
        if self.delete_all:
            return True
        return any(s <= f <= e for s, e in self.delete_ranges)

    def hud(self) -> str:
        if self.delete_all:
            return f"ID {self.pid}  [DELETE ALL]"
        parts = [f"[{s}-{e}]" for s, e in self.delete_ranges]
        pending = f"  pending start={self.pending_start}" if self.pending_start is not None else ""
        if parts:
            return f"ID {self.pid}  DELETE ranges: {' '.join(parts)}{pending}"
        if self.pending_start is not None:
            return f"ID {self.pid}  [range start={self.pending_start} → scrub then press r]"
        return f"ID {self.pid}  [keep all]"


# ── main ──────────────────────────────────────────────────────────────────────

def run_editor(scene_dir: Path, cam: str, use_clean: bool) -> None:
    suffix   = "_clean" if use_clean else ""
    gt_path  = scene_dir / f"gt_{cam}{suffix}.csv"
    vid_path = scene_dir / f"{cam}.mp4"

    if not gt_path.exists():
        raise FileNotFoundError(gt_path)
    if not vid_path.exists():
        raise FileNotFoundError(vid_path)

    gt         = pd.read_csv(gt_path)
    person_ids = sorted(gt["person_id"].unique())
    n_persons  = len(person_ids)
    states     = {pid: PersonState(pid) for pid in person_ids}

    cap      = cv2.VideoCapture(str(vid_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0

    win = f"GT Editor - {scene_dir.name}/{cam}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 580)

    cur_frame  = 0
    cur_person = 0
    playing    = False
    _cache: dict[int, np.ndarray] = {}

    # ── helpers ───────────────────────────────────────────────────────────────

    def _seek(f: int) -> None:
        nonlocal cur_frame
        cur_frame = max(0, min(n_frames - 1, f))

    def _read(f: int) -> np.ndarray:
        if f not in _cache:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            _cache[f] = img if ok else np.zeros((360, 640, 3), dtype=np.uint8)
            # evict oldest only when cache is large AND it is not the current frame
            if len(_cache) > 150:
                evict = min(k for k in _cache if k != f)
                del _cache[evict]
        return _cache[f].copy()

    def _draw(fi: int) -> np.ndarray:
        img = _read(fi)
        pid = person_ids[cur_person]
        st  = states[pid]

        # faint boxes for all persons
        for i, p in enumerate(person_ids):
            rows = gt[(gt["frame"] == fi) & (gt["person_id"] == p)]
            col  = _pal(i)
            for _, r in rows.iterrows():
                x, y, w, h = int(r.left), int(r.top), int(r.width), int(r.height)
                cv2.rectangle(img, (x, y), (x+w, y+h), col, 1)

        # current person — bright, colour by status at this frame
        rows = gt[(gt["frame"] == fi) & (gt["person_id"] == pid)]
        if st.frame_is_deleted(fi):
            box_col = _COL_DELETE
            label   = f"{pid} [DEL]"
        else:
            box_col = _COL_KEEP
            label   = str(pid)
        for _, r in rows.iterrows():
            x, y, w, h = int(r.left), int(r.top), int(r.width), int(r.height)
            cv2.rectangle(img, (x, y), (x+w, y+h), box_col, 2)
            cv2.putText(img, label, (x, max(y - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_col, 2)

        # ── timeline bar ─────────────────────────────────────────────────────
        W   = img.shape[1]
        bh  = 16
        bar = np.full((bh, W, 3), (0, 100, 0), dtype=np.uint8)  # green base

        def _fx(f: int) -> int:
            return max(1, min(W - 2, int(f / max(n_frames - 1, 1) * (W - 1))))

        # confirmed delete ranges — red
        for s, e in st.delete_ranges:
            bar[:, _fx(s):_fx(e)+1] = (0, 0, 200)

        # pending range — yellow
        if st.pending_start is not None:
            x1, x2 = sorted([_fx(st.pending_start), _fx(fi)])
            bar[:, x1:x2+1] = (0, 200, 200)

        # playhead — white, 2px wide, clamped away from edges
        px = max(1, min(W - 3, _fx(fi)))
        bar[:, px:px+2] = (255, 255, 255)

        # ── HUD ──────────────────────────────────────────────────────────────
        hud = [
            f"Frame {fi}/{n_frames-1}    Person {cur_person+1}/{n_persons}",
            st.hud(),
            "n/p=next/prev  d=del all  k=keep all  r=del range  u=undo range  s=save  q=quit",
            "Space=play/pause   ./,=step   0-9=jump%",
        ]
        for i, line in enumerate(hud):
            yp = 18 + i * 20
            cv2.putText(img, line, (6, yp), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3)
            cv2.putText(img, line, (6, yp), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

        # ── person status strip ───────────────────────────────────────────────
        strip_h  = 18
        strip    = np.zeros((strip_h, img.shape[1], 3), dtype=np.uint8)
        sw       = img.shape[1]
        we       = sw // max(n_persons, 1)
        for i, p in enumerate(person_ids):
            s    = states[p]
            col  = _COL_DELETE if s.delete_all else (_COL_RANGE if s.delete_ranges or s.pending_start is not None else _COL_KEEP)
            x1   = i * we
            # last cell extends to full width to avoid right-edge gap
            x2   = sw - 1 if i == n_persons - 1 else (i + 1) * we - 1
            cv2.rectangle(strip, (x1, 0), (x2, strip_h), col, -1 if i == cur_person else 1)
            cv2.putText(strip, str(p), (x1 + 3, 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1)

        _last_img_h[0] = img.shape[0]
        return np.vstack([img, bar, strip])

    _last_img_h  = [360]
    _mouse_down  = [False]
    img_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640

    def _try_seek_from_bar(x: int, y: int) -> None:
        bar_top = _last_img_h[0]
        bar_bot = bar_top + 16      # matches bh=16
        if bar_top <= y <= bar_bot:
            f = int(x / max(img_w - 1, 1) * (n_frames - 1))
            _seek(f)

    def _on_mouse(event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            _mouse_down[0] = True
            _try_seek_from_bar(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            _mouse_down[0] = False
        elif event == cv2.EVENT_MOUSEMOVE and _mouse_down[0]:
            _try_seek_from_bar(x, y)

    cv2.setMouseCallback(win, _on_mouse)

    print(f"\n[gt_editor] {scene_dir.name}/{cam}  {n_persons} persons  {n_frames} frames")
    print(f"[gt_editor] d=delete all  k=keep  r=mark delete range  u=undo  s=save  q=quit")

    while True:
        if playing:
            cur_frame = min(cur_frame + 1, n_frames - 1)
            if cur_frame == n_frames - 1:
                playing = False

        cv2.imshow(win, _draw(cur_frame))

        wait_ms = max(1, int(1000 / fps)) if playing else 30
        key = cv2.waitKey(wait_ms) & 0xFF

        pid = person_ids[cur_person]
        st  = states[pid]

        if key in (ord('q'), 27):
            break

        elif key in (ord('n'), 83):         # next →
            cur_person = (cur_person + 1) % n_persons

        elif key in (ord('p'), 81):         # prev ←
            cur_person = (cur_person - 1) % n_persons

        elif key == ord('d'):
            st.delete_all    = True
            st.delete_ranges = []
            st.pending_start = None
            print(f"  ID {pid} → DELETE ALL")

        elif key == ord('k'):
            st.delete_all    = False
            st.delete_ranges = []
            st.pending_start = None
            print(f"  ID {pid} → KEEP ALL")

        elif key == ord('r'):
            if st.delete_all:
                print(f"  ID {pid}: already marked delete-all, press k first to reset")
            elif st.pending_start is None:
                st.pending_start = cur_frame
                print(f"  ID {pid}: range start={cur_frame}  →  scrub to end frame, press r again")
            else:
                s = min(st.pending_start, cur_frame)
                e = max(st.pending_start, cur_frame)
                st.delete_ranges.append((s, e))
                st.pending_start = None
                print(f"  ID {pid}: delete range [{s}–{e}] added  (total {len(st.delete_ranges)} ranges)")

        elif key == ord('u'):
            if st.pending_start is not None:
                st.pending_start = None
                print(f"  ID {pid}: pending range cancelled")
            elif st.delete_ranges:
                removed = st.delete_ranges.pop()
                print(f"  ID {pid}: removed range {removed}")
            else:
                print(f"  ID {pid}: nothing to undo")

        elif key == ord(' '):
            playing = not playing

        elif key == ord('.'):
            _seek(cur_frame + 1)

        elif key == ord(','):
            _seek(cur_frame - 1)

        elif ord('0') <= key <= ord('9'):
            _seek(int((key - ord('0')) / 10.0 * n_frames))

        elif key == ord('s'):
            _save(gt, states, person_ids, gt_path, scene_dir, cam)
            # flash confirmation on screen
            flash = _draw(cur_frame)
            cv2.putText(flash, "SAVED", (flash.shape[1]//2 - 60, flash.shape[0]//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 6)
            cv2.putText(flash, "SAVED", (flash.shape[1]//2 - 60, flash.shape[0]//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)
            cv2.imshow(win, flash)
            cv2.waitKey(800)

    cv2.destroyAllWindows()
    cap.release()


def _save(gt: pd.DataFrame, states: dict, person_ids: list,
          gt_path: Path, scene_dir: Path, cam: str) -> None:
    out_path = scene_dir / f"gt_{cam}_clean.csv"
    backup   = scene_dir / f"gt_{cam}_original.csv"
    if not backup.exists():
        shutil.copy(gt_path, backup)
        print(f"  [save] backed up original → {backup.name}", flush=True)

    keep_rows = []
    for pid in person_ids:
        st = states[pid]
        if st.delete_all:
            print(f"  [save] ID {pid} — deleted entirely")
            continue
        rows = gt[gt["person_id"] == pid].copy()
        if st.delete_ranges:
            mask = pd.Series(False, index=rows.index)
            for s, e in st.delete_ranges:
                mask |= (rows["frame"] >= s) & (rows["frame"] <= e)
            removed = mask.sum()
            rows = rows[~mask]
            print(f"  [save] ID {pid} — removed {removed} rows across "
                  f"{len(st.delete_ranges)} range(s), {len(rows)} kept")
        keep_rows.append(rows)

    out = pd.concat(keep_rows).sort_values(["frame", "person_id"]) if keep_rows \
          else pd.DataFrame(columns=gt.columns)
    out.to_csv(out_path, index=False)
    print(f"  [save] → {out_path.name}  ({len(out)} rows, "
          f"{out['person_id'].nunique() if len(out) else 0} persons)", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Visual GT editor for MMPTracking retail scenes")
    p.add_argument("--scene", required=True,
                   help="Scene directory, e.g. dataset/MMPTracking_short/retail_0")
    p.add_argument("--cam", default=None,
                   help="Camera name to start from, e.g. cam1 (default: first cam in folder)")
    p.add_argument("--use-clean", action="store_true",
                   help="Load gt_<cam>_clean.csv instead of original")
    args = p.parse_args()

    scene_dir = Path(args.scene)

    # Discover all cameras in the scene (sorted: cam1, cam2, ...)
    all_cams = sorted(
        p.stem for p in scene_dir.glob("cam*.mp4")
        if (scene_dir / f"gt_{p.stem}.csv").exists()
                    or (scene_dir / f"gt_{p.stem}_clean.csv").exists()
    )
    if not all_cams:
        print(f"[gt_editor] No cam*.mp4 + gt_cam*.csv found in {scene_dir}")
        return

    start_cam = args.cam if args.cam else all_cams[0]
    if start_cam not in all_cams:
        print(f"[gt_editor] Camera '{start_cam}' not found. Available: {all_cams}")
        return

    start_idx = all_cams.index(start_cam)
    print(f"[gt_editor] Scene: {scene_dir.name}  —  cameras: {all_cams}")
    print(f"[gt_editor] Starting from {start_cam}  ({start_idx+1}/{len(all_cams)})")

    for cam in all_cams[start_idx:]:
        print(f"\n{'='*60}")
        print(f"  Camera {cam}  ({all_cams.index(cam)+1}/{len(all_cams)})")
        print(f"{'='*60}")
        run_editor(scene_dir, cam, args.use_clean)
        print(f"  [gt_editor] Done with {cam}.")


if __name__ == "__main__":
    main()
