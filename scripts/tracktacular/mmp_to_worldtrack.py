#!/usr/bin/env python3
"""Convert one MMPTracking scene into the Wildtrack on-disk layout that
TrackTacular's PedestrianDataset expects.

Output (per scene):
    <out>/Image_subsets/C{1..N}/<frame:08d>.png      # frames from cam*.mp4
    <out>/annotations_positions/<frame:08d>.json       # [{personID, positionID, views:[{xmin..}]}]
    <out>/calibrations.json                            # copied (env calibration)

MMP sources:
    frames     : dataset/MMPTracking_10minute/<split>/<scene>/cam{1..N}.mp4
    img bbox   : .../labels/<session>/<scene_inst>.zip  -> rgb_<frame>_<cam>.json {pid:[x1,y1,x2,y2]}
    bev gt     : .../topdown_labels/<session>/<scene_inst>.zip -> topdown_<frame>.csv (pid,gx,gy,h)
    calibration: .../calibrations/<env>/calibrations.json

positionID = round(gx) * GRID_NY + round(gy)   (inverted by the adapter).
"""
from __future__ import annotations
import argparse, json, os, zipfile
from pathlib import Path
import cv2

GRID_NY = 256  # worldgrid N_row (gy divisor); must match the adapter


def _scene_parts(scene: str):
    # "63am_industry_safety_0" -> session=63am, env=industry_safety, inst=industry_safety_0
    session, rest = scene.split("_", 1)
    env = rest.rsplit("_", 1)[0]
    return session, env, rest


def _read_zip_jsons(zip_path: Path):
    """frame(int) -> {cam(int): {pid:[x1,y1,x2,y2]}}  from rgb_<f>_<cam>.json."""
    out: dict[int, dict[int, dict]] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            base = os.path.basename(name)
            if not (base.startswith("rgb_") and base.endswith(".json")):
                continue
            _, fr, cam = base[:-5].split("_")
            d = json.loads(zf.read(name))
            out.setdefault(int(fr), {})[int(cam)] = d
    return out


def _read_topdown(zip_path: Path):
    """frame(int) -> list[(pid, gx, gy)] from topdown_<f>.csv."""
    out: dict[int, list] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            base = os.path.basename(name)
            if not (base.startswith("topdown_") and base.endswith(".csv")):
                continue
            fr = int(base[len("topdown_"):-4])
            rows = []
            for line in zf.read(name).decode().splitlines():
                p = line.split(",")
                if len(p) >= 3:
                    rows.append((int(float(p[0])), float(p[1]), float(p[2])))
            out[fr] = rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="e.g. 63am_industry_safety_0")
    ap.add_argument("--split", default="train")
    ap.add_argument("--mmp-root", default="dataset/MMPTracking")
    ap.add_argument("--short-root", default="dataset/MMPTracking_10minute")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frame-step", type=int, default=2)
    args = ap.parse_args()

    session, env, inst = _scene_parts(args.scene)
    root = Path(args.mmp_root)
    base = root / ("MMPTracking_training/train" if args.split == "train"
                   else "MMPTracking_validation/validation")
    calib = json.load(open(base / "calibrations" / env / "calibrations.json"))
    cam_ids = sorted(c["CameraId"] for c in calib["Cameras"])   # ascending -> C1..CN
    n_cam = len(cam_ids)

    print(f"[convert] {args.scene}: env={env} cams={cam_ids} step={args.frame_step}")
    img_bbox = _read_zip_jsons(base / "labels" / session / f"{inst}.zip")
    topdown = _read_topdown(base / "topdown_labels" / session / f"{inst}.zip")

    out = Path(args.out)
    (out / "annotations_positions").mkdir(parents=True, exist_ok=True)
    for ci in range(n_cam):
        (out / "Image_subsets" / f"C{ci + 1}").mkdir(parents=True, exist_ok=True)
    json.dump(calib, open(out / "calibrations.json", "w"))

    # open mp4s
    scene_dir = Path(args.short_root) / args.split / args.scene
    caps = {cam: cv2.VideoCapture(str(scene_dir / f"cam{cam}.mp4")) for cam in cam_ids}
    n_frames = int(min(c.get(cv2.CAP_PROP_FRAME_COUNT) for c in caps.values()))
    keep = set(range(0, n_frames, args.frame_step))

    written = 0
    for fr in range(n_frames):
        frames = {}
        for cam in cam_ids:
            ok, im = caps[cam].read()
            frames[cam] = im if ok else None
        if fr not in keep or fr not in topdown:
            continue
        # images
        for ci, cam in enumerate(cam_ids):
            if frames[cam] is not None:
                cv2.imwrite(str(out / "Image_subsets" / f"C{ci + 1}" / f"{fr:08d}.png"),
                            frames[cam])
        # annotations
        peds = []
        cam_lbl = img_bbox.get(fr, {})
        for pid, gx, gy in topdown[fr]:
            views = []
            for cam in cam_ids:
                d = cam_lbl.get(cam, {})
                if str(pid) in d:
                    x1, y1, x2, y2 = d[str(pid)]
                    views.append({"viewNum": cam, "xmin": int(x1), "ymin": int(y1),
                                  "xmax": int(x2), "ymax": int(y2)})
                else:
                    views.append({"viewNum": cam, "xmin": -1, "ymin": -1,
                                  "xmax": -1, "ymax": -1})
            peds.append({"personID": int(pid),
                         "positionID": int(round(gx)) * GRID_NY + int(round(gy)),
                         "views": views})
        json.dump(peds, open(out / "annotations_positions" / f"{fr:08d}.json", "w"))
        written += 1

    for c in caps.values():
        c.release()
    print(f"[convert] done: {written} frames, {n_cam} cams -> {out}")


if __name__ == "__main__":
    main()
