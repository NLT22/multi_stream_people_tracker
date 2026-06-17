#!/usr/bin/env python3
"""Stage 3 of the AIC23 pipeline: their STCRA (run_stcra.py id_reassignment) run
VERBATIM on MMP, on top of the anchor-clustered output (stage 2). Completes their
full pipeline: SCT -> anchor-guided clustering -> STCRA.

Their image2world(feet, homography) is replaced by the precomputed MMP world coords
from tracklet_bev.csv (same thing: foot point projected to the ground plane via the
camera calibration). cam_weight uniform; distance gates in mm (MMP world units).
Everything else (the merge / outlier-removal / nearest-trajectory reassignment with
the conf = 1 - d_best/d_cur gate, 3 shrinking passes) is their code.
"""
from __future__ import annotations
import argparse, csv, collections
from pathlib import Path
import numpy as np
import pandas as pd


# ---- VERBATIM from STCRA/run_stcra.py ---------------------------------------
def get_iou(box1, box2):
    box1, box2 = box1[:4], box2[:4]
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2]); y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    inter = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    a1 = box1[2] * box1[3]; a2 = box2[2] * box2[3]
    return inter / (a1 + a2 - inter)


def id_reassignment(tracks_list, cam_weight, dis_thr=200, conf_thr=0.7, try_second_nearest=True):
    # loc layout: [x,y,w,h, worldx, worldy, score]  (world precomputed)
    merge_tracks = collections.defaultdict(dict)
    reassign_duplicate_ids = []
    for cam_name, tracks in tracks_list.items():
        for tid, track in tracks.items():
            for fid, location in track.items():
                if len(location) > 1:
                    if get_iou(location[0], location[1]) >= 0.75:
                        keep = location[0] if location[0][6] >= location[1][6] else location[1]
                        tracks_list[cam_name][tid][fid] = [keep]
                    else:
                        reassign_duplicate_ids.append((cam_name, tid, fid, location[0]))
                        reassign_duplicate_ids.append((cam_name, tid, fid, location[1]))
                        tracks_list[cam_name][tid][fid] = []
                elif len(location):
                    xworld, yworld = location[0][4], location[0][5]
                    merge_tracks[tid].setdefault(fid, []).append((cam_name, [xworld, yworld], location[0]))

    avg_tracks = collections.defaultdict(dict)
    reassign_outliers = []
    for tid, track in merge_tracks.items():
        for fid, loc_list in track.items():
            if len(loc_list) > 2:
                coord = np.asarray([l[1] for l in loc_list]); cam = [l[0] for l in loc_list]
                loc = [l[2] for l in loc_list]
                dist = [np.linalg.norm(coord - coord[i], axis=1) for i in range(len(coord))]
                delete_index = []
                for i in range(len(coord)):
                    if np.all(dist[i][0:i] > dis_thr) and np.all(dist[i][i + 1:] > dis_thr):
                        delete_index.append(i); reassign_outliers.append((cam[i], tid, fid, loc[i]))
                if len(delete_index) != len(coord):
                    loc_list = list(np.delete(np.asarray(loc_list, dtype=object), delete_index, axis=0))
            avg_loc = np.sum([np.multiply(l[1], cam_weight[l[0]]) for l in loc_list], axis=0) \
                / np.sum([cam_weight[l[0]] for l in loc_list])
            avg_tracks[tid][fid] = avg_loc

    def _reassign(items, allow_same, delete_from):
        cnt = 0
        for cam_name, tid, fid, location in items:
            xworld, yworld = location[4], location[5]
            dist_list, id_list = [], []
            if fid not in avg_tracks[tid].keys() and not delete_from:
                tracks_list[cam_name][tid][fid] = [location]; continue
            for c_tid, track in avg_tracks.items():
                if fid in track:
                    dist_list.append(np.linalg.norm(np.array([xworld, yworld]) - np.array(track[fid])))
                    id_list.append(c_tid)
            if not dist_list:
                continue
            mi = int(np.argmin(dist_list)); cid = id_list[mi]
            try:
                conf = 1 - dist_list[mi] / dist_list[id_list.index(tid)]
            except (ValueError, ZeroDivisionError):
                conf = 1.0
            if fid not in tracks_list[cam_name][cid] or not tracks_list[cam_name][cid][fid]:
                if conf >= conf_thr or (allow_same and cid == tid):
                    tracks_list[cam_name][cid][fid] = [location]
                    if delete_from and cid != tid:
                        tracks_list[cam_name][tid].pop(fid, None)
                    cnt += 1
        return cnt

    d = _reassign(reassign_duplicate_ids, allow_same=True, delete_from=False)
    o = _reassign(reassign_outliers, allow_same=False, delete_from=True)
    print(f"  [stcra] dis_thr={dis_thr} conf_thr={conf_thr}: dup_reassign={d} outlier_reassign={o}")
    return tracks_list
# -----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor-dir", required=True, help="stage-2 output (cam_*_predictions.csv)")
    ap.add_argument("--bev", required=True, help="tracklet_bev.csv (world coords)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--passes", default="2500:0.65,2000:0.70,1500:0.75")
    args = ap.parse_args()

    bev = pd.read_csv(args.bev)
    world = {(int(r.cam_id), int(r.frame_no_cam), int(r.local_track_id)): (r.world_x, r.world_y)
             for r in bev.itertuples()}
    anchor = Path(args.anchor_dir)
    # build tracks_list {cam: {gid: {frame: [loc]}}}; remember row order to rewrite
    tracks_list = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    rows_by_cam = {}
    cams = []
    for fp in sorted(anchor.glob("cam_*_predictions.csv")):
        src = int(fp.stem.split("_")[1]); cams.append(src)
        rows = list(csv.DictReader(open(fp)))
        rows_by_cam[src] = rows
        for r in rows:
            f, l, g = int(r["frame_no_cam"]), int(r["local_track_id"]), int(float(r["global_id"]))
            if g < 0:
                continue
            wx_wy = world.get((src, f, l))
            if wx_wy is None:
                continue
            loc = [float(r["left"]), float(r["top"]), float(r["width"]), float(r["height"]),
                   wx_wy[0], wx_wy[1], 1.0]
            tracks_list[src][g][f].append(loc)

    cam_weight = collections.defaultdict(lambda: 1.0)  # uniform (no per-cam coverage prior)
    passes = [(float(p.split(":")[0]), float(p.split(":")[1])) for p in args.passes.split(",")]
    for dis_thr, conf_thr in passes:
        tracks_list = id_reassignment(tracks_list, cam_weight, dis_thr, conf_thr)

    # rebuild (cam,frame,ltid) -> new gid from tracks_list, then rewrite cam preds
    newgid = {}
    for src, gids in tracks_list.items():
        for g, frames in gids.items():
            for f, locs in frames.items():
                for _ in locs:
                    pass  # gid set below via reverse lookup
    # reverse: a (cam,frame,ltid) detection's new gid = the gid whose [cam][gid][frame] holds its world
    # simplest: map by (cam,frame,worldx,worldy)
    loc2gid = {}
    for src, gids in tracks_list.items():
        for g, frames in gids.items():
            for f, locs in frames.items():
                for loc in locs:
                    loc2gid[(src, f, round(loc[4], 1), round(loc[5], 1))] = g
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    for src in cams:
        with open(out / f"cam_{src}_predictions.csv", "w", newline="") as fo:
            w = csv.DictWriter(fo, fieldnames=["frame_no_cam", "cam_id", "local_track_id",
                                               "global_id", "left", "top", "width", "height"])
            w.writeheader()
            for r in rows_by_cam[src]:
                f, l = int(r["frame_no_cam"]), int(r["local_track_id"])
                ww = world.get((src, f, l))
                if ww is not None:
                    g = loc2gid.get((src, f, round(ww[0], 1), round(ww[1], 1)), int(float(r["global_id"])))
                    r = {**r, "global_id": g}
                w.writerow(r)
    print(f"[stcra] -> {out}")


if __name__ == "__main__":
    main()
