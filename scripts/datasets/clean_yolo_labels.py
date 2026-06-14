#!/usr/bin/env python3
"""Model-assisted cleaning of MMP YOLO labels.

Diagnosis: MMP GT labels every person's *projected* box in every camera,
including people fully occluded by walls/shelves/other people. Those boxes sit
over background → the detector learns "background = person" → false positives.

Fix: an INDEPENDENT COCO-pretrained verifier (YOLO11x) judges whether a person
is actually visible in each labeled box. Keep a GT box iff a verifier person
detection (conf >= conf_thr) overlaps it (IoU >= iou_thr). Also drop boxes
>occ_thr contained by a *closer* (lower) GT box (person-on-person occlusion).

Operates directly on the extracted YOLO dataset (images/ + labels/), writing a
cleaned copy. Emits per-split drop stats + a QA montage of dropped vs kept crops.

Usage:
  python scripts/datasets/clean_yolo_labels.py \
      --src dataset/mmp_yolo_10minute --dst dataset/mmp_yolo_10minute_clean \
      --verifier yolo11x.pt --iou 0.4 --conf 0.25 [--limit 500] [--splits train]
"""
from __future__ import annotations
import argparse, random, shutil
from pathlib import Path

import numpy as np
import cv2
from ultralytics import YOLO

IMG_W, IMG_H = 640, 360


def yolo_to_xyxy(cx, cy, w, h, W=IMG_W, H=IMG_H):
    return ((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H)


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def contain_frac(a, b):  # fraction of a covered by b
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    aa = (a[2]-a[0])*(a[3]-a[1])
    return inter / aa if aa > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--verifier", default="yolo11x.pt")
    ap.add_argument("--iou", type=float, default=0.4)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--occ", type=float, default=0.6, help="drop if contained > this by a closer box")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--limit", type=int, default=0, help="0 = all; else first N images/split (dry run)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--qa-dir", default=None, help="dir for dropped/kept crop montage")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    model = YOLO(args.verifier)
    qa_drop, qa_keep = [], []
    grand = {"boxes": 0, "kept": 0, "drop_occ": 0, "drop_unverified": 0, "imgs": 0}

    for split in args.splits:
        img_dir = src / "images" / split
        lbl_dir = src / "labels" / split
        out_img = dst / "images" / split; out_img.mkdir(parents=True, exist_ok=True)
        out_lbl = dst / "labels" / split; out_lbl.mkdir(parents=True, exist_ok=True)
        imgs = sorted(img_dir.glob("*.jpg"))
        if args.limit:
            imgs = imgs[:args.limit]
        st = {"boxes": 0, "kept": 0, "drop_occ": 0, "drop_unverified": 0}
        for i in range(0, len(imgs), args.batch):
            chunk = imgs[i:i+args.batch]
            res = model.predict([str(p) for p in chunk], conf=args.conf, classes=[0],
                                verbose=False, imgsz=640)
            for p, r in zip(chunk, res):
                vboxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
                lbl = lbl_dir / (p.stem + ".txt")
                gt = []
                if lbl.exists():
                    for ln in lbl.read_text().splitlines():
                        c = ln.split()
                        if len(c) != 5: continue
                        gt.append(yolo_to_xyxy(*map(float, c[1:])))
                # occlusion filter: drop if >occ contained by a closer (lower bottom) box
                keep_occ = [True]*len(gt)
                for a in range(len(gt)):
                    for b in range(len(gt)):
                        if a==b: continue
                        if gt[b][3] > gt[a][3] and contain_frac(gt[a], gt[b]) > args.occ:
                            keep_occ[a] = False; break
                out_lines = []
                for k, box in enumerate(gt):
                    st["boxes"] += 1
                    if not keep_occ[k]:
                        st["drop_occ"] += 1; continue
                    best = max((iou(box, vb) for vb in vboxes), default=0.0)
                    if best >= args.iou:
                        st["kept"] += 1
                        cx=((box[0]+box[2])/2)/IMG_W; cy=((box[1]+box[3])/2)/IMG_H
                        w=(box[2]-box[0])/IMG_W; h=(box[3]-box[1])/IMG_H
                        out_lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                        if args.qa_dir and len(qa_keep) < 60 and random.random() < 0.05:
                            qa_keep.append((str(p), box))
                    else:
                        st["drop_unverified"] += 1
                        if args.qa_dir and len(qa_drop) < 60 and random.random() < 0.3:
                            qa_drop.append((str(p), box))
                (out_lbl / (p.stem + ".txt")).write_text("\n".join(out_lines))
                tgt = out_img / p.name
                if not tgt.exists():
                    tgt.symlink_to(p.resolve())
            grand["imgs"] += len(chunk)
            if (i//args.batch) % 20 == 0:
                print(f"  [{split}] {i+len(chunk)}/{len(imgs)} imgs ...", flush=True)
        for k in st: grand[k] += st[k]
        print(f"[{split}] boxes={st['boxes']} kept={st['kept']} "
              f"drop_occ={st['drop_occ']} drop_unverified={st['drop_unverified']} "
              f"({100*(st['drop_occ']+st['drop_unverified'])/max(1,st['boxes']):.1f}% dropped)")

    # dataset.yaml
    if (src/"dataset.yaml").exists():
        y = (src/"dataset.yaml").read_text().replace(str(src), str(dst)).replace(src.name, dst.name)
        (dst/"dataset.yaml").write_text(y)

    # QA montage
    if args.qa_dir:
        qd = Path(args.qa_dir); qd.mkdir(parents=True, exist_ok=True)
        for name, items in [("dropped", qa_drop), ("kept", qa_keep)]:
            crops = []
            for path, box in items[:48]:
                im = cv2.imread(path)
                if im is None: continue
                x1,y1,x2,y2 = [int(v) for v in box]
                x1,y1=max(0,x1),max(0,y1); x2,y2=min(IMG_W,x2),min(IMG_H,y2)
                if x2<=x1 or y2<=y1: continue
                c = cv2.resize(im[y1:y2, x1:x2], (64,128))
                crops.append(c)
            if crops:
                rows=[]
                for r in range(0, len(crops), 8):
                    row = crops[r:r+8]
                    while len(row)<8: row.append(np.zeros((128,64,3),np.uint8))
                    rows.append(np.hstack(row))
                cv2.imwrite(str(qd/f"montage_{name}.jpg"), np.vstack(rows))
                print(f"QA: {qd}/montage_{name}.jpg ({len(crops)} crops)")

    tot = grand["boxes"]
    print(f"\n[TOTAL] {grand['imgs']} imgs, {tot} boxes -> kept {grand['kept']} "
          f"({100*grand['kept']/max(1,tot):.1f}%), dropped occ={grand['drop_occ']} "
          f"unverified={grand['drop_unverified']} ({100*(tot-grand['kept'])/max(1,tot):.1f}%)")


if __name__ == "__main__":
    main()
