"""Verify the long-range Global-ID consolidation (src.eval.reid_reentry_merge).

Two invariants:
  * a temporally-disjoint fragment of the SAME person (re-entry, or the same
    person split across cameras) is merged when appearance matches;
  * two ids that share a (camera, frame) are NEVER merged (one detection per
    camera => different people), even with identical appearance.
"""

import csv
import tempfile

import numpy as np

from src.eval.reid_reentry_merge import merge


def _write(d, tracklets, cam_rows):
    with open(f"{d}/tracklets.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tracklet_id", "cam_id", "local_track_id", "global_id",
                    "start_frame", "end_frame", "num_detections", "num_embeddings",
                    "mean_width", "mean_height", "mean_area"])
        for t in tracklets:
            w.writerow(t)
    with open(f"{d}/cam_0_predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_no_cam", "cam_id", "local_track_id", "global_id",
                    "left", "top", "width", "height"])
        w.writerows(cam_rows)
    open(f"{d}/tracklet_bev.csv", "w").write(
        "tracklet_id,frame_no_cam,cam_id,local_track_id,global_id,world_x,world_y\n")


def _emb(*vecs):
    return np.stack([np.array(list(v) + [0] * (256 - len(v)), dtype=np.float32) for v in vecs])


def test_reentry_merges_and_cooccurrence_blocked(tmp_path=None):
    d = tmp_path or tempfile.mkdtemp()
    d = str(d)
    # T0(gid1) & T1(gid2): same person (identical emb), disjoint frames, same cam.
    # T2(gid3): different person, co-occurs with gid1 in cam0 at the same frames.
    _write(
        d,
        tracklets=[
            (0, 0, 0, 1, 0, 100, 100, 100, 50, 100, 5000),
            (1, 0, 1, 2, 200, 300, 100, 100, 50, 100, 5000),
            (2, 0, 2, 3, 0, 100, 100, 100, 50, 100, 5000),
        ],
        cam_rows=(
            [[fr, 0, 0, 1, 10, 10, 50, 100] for fr in range(0, 101)]
            + [[fr, 0, 1, 2, 10, 10, 50, 100] for fr in range(200, 301)]
            + [[fr, 0, 2, 3, 300, 10, 50, 100] for fr in range(0, 101)]
        ),
    )
    np.savez(f"{d}/tracklet_embeddings.npz",
             tracklet_ids=np.array([0, 1, 2]),
             embeddings=_emb([1, 0, 0], [1, 0, 0], [0, 1, 0]))

    remap, nb, na, _ = merge(d, threshold=0.7)
    assert remap[2] == remap[1], "re-entry fragment G2 should merge into G1"
    assert remap[3] != remap[1], "G3 shares a (cam,frame) with G1 -> must stay separate"
    assert (nb, na) == (3, 2)


if __name__ == "__main__":
    test_reentry_merges_and_cooccurrence_blocked()
    print("ok")
