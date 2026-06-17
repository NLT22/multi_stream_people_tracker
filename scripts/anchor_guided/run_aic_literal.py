#!/usr/bin/env python3
"""Run the AIC23 authors' anchor-guided clustering on MMP, reusing their code
VERBATIM (nms_fast, get_box_dist, and the per-frame Hungarian + sliding-window(15)
majority-vote main loop copied from BoT-SORT/tools/aic_hungarian_cluster.py).

Only the I/O glue is MMP-specific: anchor frames are sampled evenly and k is fixed
(oracle), instead of their per-AIC-scene hardcoded `threshold` table. Inputs are
produced by build_aic_inputs.py. Output: cam_<src>_predictions.csv (global IDs) in
--out-dir, ready for metrics_mmp.
"""
from __future__ import annotations
import argparse, csv, collections, pickle, sys
from pathlib import Path
import numpy as np
from collections import Counter
from scipy.spatial import distance
from sklearn.cluster import AgglomerativeClustering
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aic_types import Tracklet  # noqa: F401 (needed for unpickling)

n = 15            # sliding-window vote length (their default)
nms_thres = 1     # their default (no suppression)


# ---- VERBATIM from aic_hungarian_cluster.py ---------------------------------
def nms_fast(boxes, probs=None, overlapThresh=0.3):
    if len(boxes) == 0:
        return [], []
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    pick = []
    x1 = boxes[:, 2]; y1 = boxes[:, 3]; x2 = boxes[:, 4]; y2 = boxes[:, 5]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2
    if probs is not None:
        idxs = probs
    idxs = np.argsort(idxs)
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]]); yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]]); yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1); h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
    return boxes[pick].astype("float"), pick


def get_box_dist(feat, anchors):
    box_dist = []
    for idx in anchors:
        dists = []
        for anchor in anchors[idx]:
            anchor = anchor / np.linalg.norm(anchor)
            dists += [distance.cosine(feat, anchor)]
        dist = sum(dists) / len(dists)
        box_dist.append(dist)
    return box_dist
# -----------------------------------------------------------------------------


# Vectorized equivalent of get_box_dist (numerically identical): mean over the
# bank of cosine_distance(feat, a) = 1 - feat_hat . mean_a(a_hat). Precompute the
# per-anchor mean of L2-normalized members; then box_dist = 1 - feat_hat @ M.T.
def anchor_means(anchors):
    M = []
    for idx in sorted(anchors):
        A = np.stack([a / (np.linalg.norm(a) or 1.0) for a in anchors[idx]])
        m = A.mean(0)
        M.append(m)
    return np.stack(M).astype(np.float32)   # (k, D)


def box_dist_fast(feat, M):
    fh = feat / (np.linalg.norm(feat) or 1.0)
    return (1.0 - M @ fh).tolist()


def get_anchor(work, scene, k, n_anchor_frames=40):
    """Their get_anchor, adapted: sample anchor frames evenly, force k clusters."""
    detections = np.genfromtxt(work / "test_det" / f"{scene}.txt", delimiter=",", dtype=str)
    embeddings = np.array(np.load(work / "test_emb" / f"{scene}.npy", allow_pickle=True).tolist())
    frames = np.unique(detections[:, 1].astype(int))
    pick = frames[np.linspace(0, len(frames) - 1, n_anchor_frames).astype(int)]
    all_embs = None
    for frame in pick:
        inds = detections[:, 1] == str(frame)
        fdet, femb = detections[inds], embeddings[inds]
        for cam in np.unique(detections[:, 0]):
            ci = fdet[:, 0] == cam
            cam_det = fdet[ci][:, 1:].astype("float")
            cam_emb = femb[ci]
            cam_det, p = nms_fast(cam_det, None, nms_thres)
            cam_emb = cam_emb[p]
            if len(cam_det) == 0:
                continue
            all_embs = cam_emb if all_embs is None else np.vstack((all_embs, cam_emb))
    clustering = AgglomerativeClustering(n_clusters=k).fit(all_embs)
    anchors = collections.defaultdict(list)
    for gid in range(k):
        for j in range(len(all_embs)):
            if gid == clustering.labels_[j]:
                anchors[gid].append(all_embs[j])
    return anchors


def cluster_cam(work, scene, cam, anchors, M, out_dir, src):
    with open(work / "tracklet" / f"{scene}_{cam}.pkl", "rb") as f:
        tracklets = pickle.load(f)
    mapper = collections.defaultdict(list)
    global_id_mapper = collections.defaultdict(list)
    for trk_id in tracklets:
        trk = tracklets[trk_id]
        box_dist = None
        for feat in trk.features:
            box_dist = box_dist_fast(feat, M)   # == get_box_dist(feat, anchors)
            mapper[trk_id].append(box_dist)
        mapper[trk_id].append(box_dist)
    sct = np.genfromtxt(work / "SCT" / f"{scene}_{cam}.txt", delimiter=",", dtype=None)
    counter = collections.defaultdict(int)
    cur_frame = -1
    cost_matrix = None
    for frame_id, trk_id, x, y, w, h, score, _, _, _ in sct:
        if len(tracklets[trk_id].features) == 0:
            continue
        if frame_id != cur_frame and cost_matrix is None:
            cost_matrix = []; frame_trk_ids = []; cur_frame = frame_id
        elif frame_id != cur_frame and cost_matrix is not None:
            cost_matrix = np.array(cost_matrix)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for row, col in zip(row_ind, col_ind):
                global_id_mapper[frame_trk_ids[row]].append(col)
            cost_matrix = []; frame_trk_ids = []; cur_frame = frame_id
        cost_matrix.append(mapper[trk_id][counter[trk_id]])
        frame_trk_ids.append(trk_id)
        counter[trk_id] += 1
    new_global_id_mapper = collections.defaultdict(list)
    for trk_id in global_id_mapper:
        ids = global_id_mapper[trk_id]; new_ids = []; cur_ids = []
        for id in ids:
            cur_ids.append(id)
            if len(cur_ids) == n:
                new_ids += [Counter(cur_ids).most_common(1)[0][0]] * n
                cur_ids = []
        if len(cur_ids) > 0:
            if len(new_ids) > 0:
                new_ids += [new_ids[-1]] * (len(cur_ids) + 1)
            else:
                new_ids += [Counter(cur_ids).most_common(1)[0][0]] * (len(cur_ids) + 1)
        new_global_id_mapper[trk_id] = new_ids + [new_ids[-1]] * n if new_ids else []
    # write cam predictions (frame, gid, box) for metrics_mmp
    counter2 = collections.defaultdict(int)
    out = open(out_dir / f"cam_{src}_predictions.csv", "w", newline="")
    wr = csv.writer(out)
    wr.writerow(["frame_no_cam", "cam_id", "local_track_id", "global_id",
                 "left", "top", "width", "height"])
    for frame_id, trk_id, x, y, w, h, score, _, _, _ in sct:
        gm = new_global_id_mapper.get(trk_id, [])
        if len(gm) == 0:
            continue
        gid = (gm[-1] + 1) if counter2[trk_id] >= len(gm) else (gm[counter2[trk_id]] + 1)
        wr.writerow([int(frame_id), src, int(trk_id), int(gid), x, y, w, h])
        counter2[trk_id] += 1
    out.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--k", type=int, required=True)
    args = ap.parse_args()
    work = Path(args.work); out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    anchors = get_anchor(work, args.scene, args.k)
    M = anchor_means(anchors)
    print(f"[aic-literal] built {len(anchors)} anchors (k={args.k})")
    cams = sorted(int(p.stem.split("_")[-1]) for p in (work / "SCT").glob(f"{args.scene}_*.txt"))
    for src in cams:
        cluster_cam(work, args.scene, src, anchors, M, out, src)
        print(f"  wrote cam_{src}_predictions.csv")
    print(f"[aic-literal] -> {out}")


if __name__ == "__main__":
    main()
