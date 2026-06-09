"""Pure-Python unit tests for the micro-batch cross-camera fusion logic.

No DeepStream / GPU needed — exercises the clustering primitives directly.

Run:   python -m pytest tests/test_fusion.py -v
  or:  python tests/test_fusion.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.eval import offline_merge as om
from src.reid.micro_batch_fusion import MicroBatchFusion, _compress_remap
from src.reid import fusion_bridge


def _tracklet(tid, cam, gid, start, end, ndet, nemb):
    return {
        "tracklet_id": tid, "cam_id": cam, "local_track_id": tid,
        "global_id": gid, "start_frame": start, "end_frame": end,
        "num_detections": ndet, "num_embeddings": nemb,
    }


def _vec(direction, dim=8):
    v = np.zeros(dim, dtype=np.float32)
    v[direction] = 1.0
    return v


# --------------------------------------------------------------- remap compress
def test_compress_remap_chains():
    remap = {3: 2, 2: 1}
    _compress_remap(remap)
    assert remap[3] == 1 and remap[2] == 1


def test_compress_remap_noop_on_roots():
    remap = {5: 5}
    _compress_remap(remap)
    assert remap[5] == 5


# ------------------------------------------------------------- temporal conflict
def test_intervals_conflict_same_cam_overlap():
    # same camera, overlapping time -> conflict (cannot be one person)
    assert om._intervals_conflict([(0, 0, 100)], [(0, 50, 150)], 0) is True


def test_intervals_conflict_different_cam():
    # different cameras at the same time -> NOT a conflict (one person, two views)
    assert om._intervals_conflict([(0, 0, 100)], [(1, 0, 100)], 0) is False


def test_intervals_conflict_same_cam_disjoint():
    assert om._intervals_conflict([(0, 0, 40)], [(0, 60, 100)], 0) is False


# ----------------------------------------------------- short-tracklet filtering
def test_build_gid_summaries_filters_short_tracklets():
    # tracklet with too few detections is ignored for the embedding mean
    tracklets = [
        _tracklet(1, 0, 1, 0, 100, ndet=20, nemb=20),
        _tracklet(2, 0, 2, 0, 100, ndet=3, nemb=20),   # < min_tracklet_detections
    ]
    emb = {1: _vec(0), 2: _vec(1)}
    gids, vectors, _ = om._build_gid_summaries(
        tracklets, emb, min_gid_embeddings=4, min_tracklet_detections=6)
    assert gids == [1]                 # gid 2 dropped (short tracklet)
    assert vectors.shape == (1, 8)


def test_build_gid_summaries_filters_low_embedding_gid():
    tracklets = [_tracklet(1, 0, 1, 0, 100, ndet=20, nemb=2)]   # 2 < min_gid_embeddings
    emb = {1: _vec(0)}
    gids, _, _ = om._build_gid_summaries(
        tracklets, emb, min_gid_embeddings=4, min_tracklet_detections=6)
    assert gids == []


# --------------------------------------------------------- engine merge / block
def _engine():
    return MicroBatchFusion(interval_frames=125, threshold=0.55, margin=0.02,
                            min_gid_embeddings=4, min_tracklet_detections=6,
                            geo_weight=0.0)


def test_engine_merges_cross_camera_duplicate():
    f = _engine()
    # same person, two cameras, identical appearance -> should merge
    f.ingest_tracklet(1, 0, 1, 1, 0, 100, 20, 20, _vec(0))
    f.ingest_tracklet(2, 1, 2, 2, 0, 100, 20, 20, _vec(0))
    f.step(100)
    f.flush(100)
    remap = f.final_remap()
    assert f.resolve(2) == 1            # gid 2 merged into min root 1
    assert f.resolve(1) == 1
    assert remap.get(2) == 1


def test_engine_blocks_same_camera_lookalikes():
    f = _engine()
    # two look-alikes in the SAME camera at the same time -> must NOT merge
    f.ingest_tracklet(1, 0, 1, 1, 0, 100, 20, 20, _vec(0))
    f.ingest_tracklet(2, 0, 2, 2, 0, 100, 20, 20, _vec(0))   # same cam, same time
    f.step(100)
    f.flush(100)
    assert f.resolve(1) == 1
    assert f.resolve(2) == 2            # stayed separate despite identical appearance


def test_engine_no_merge_below_threshold():
    f = _engine()
    f.ingest_tracklet(1, 0, 1, 1, 0, 100, 20, 20, _vec(0))
    f.ingest_tracklet(2, 1, 2, 2, 0, 100, 20, 20, _vec(3))   # orthogonal -> sim 0
    f.step(100)
    f.flush(100)
    assert f.resolve(1) == 1 and f.resolve(2) == 2


def test_recluster_matches_streaming():
    # recluster() (one-shot) and step()+flush() should agree on a clean merge
    f1 = _engine()
    f2 = _engine()
    for f in (f1, f2):
        f.ingest_tracklet(1, 0, 1, 1, 0, 100, 20, 20, _vec(0))
        f.ingest_tracklet(2, 1, 2, 2, 0, 100, 20, 20, _vec(0))
    f1.step(100); f1.flush(100)
    f2.recluster()
    assert f1.resolve(2) == f2.resolve(2) == 1


# ------------------------------------------------------------- fusion_bridge
def test_build_records_from_exporter():
    exp = {(0, 5, 1): {"sum_embedding": [2.0, 0.0], "num_embeddings": 2,
                       "start_frame": 0, "end_frame": 100, "num_detections": 20}}
    tid_by_key = {}
    recs = fusion_bridge.build_records(exp, {}, {}, tid_by_key, 100)
    assert len(recs) == 1
    tid, cam, local, gid, start, end, ndet, nemb, emb = recs[0]
    assert (cam, local, gid, ndet, nemb) == (0, 5, 1, 20, 2)
    assert abs(float(emb[0]) - 1.0) < 1e-6        # mean [1,0] then L2-normalized


def test_build_records_skips_negative_gid():
    exp = {(0, 5, -1): {"sum_embedding": [1.0], "num_embeddings": 1,
                        "start_frame": 0, "end_frame": 10, "num_detections": 5}}
    assert fusion_bridge.build_records(exp, {}, {}, {}, 10) == []


def test_build_records_gallery_fallback():
    gal = {(0, 5): {"tracklet_id": 7, "gid": 3, "fusion_emb_sum": None,
                    "fusion_emb_count": 0, "start_frame": 0, "end_frame": 50,
                    "num_detections": 12}}
    recs = fusion_bridge.build_records(None, gal, {(0, 5): 3}, {}, 50)
    assert len(recs) == 1 and recs[0][3] == 3 and recs[0][8] is None


def test_accumulate_geo_samples():
    pts = {}
    rows = [{"gid": 1, "src": 0, "foot_world": (100.0, 200.0)}]
    fusion_bridge.accumulate_geo(rows, None, 10, 10, pts, 5)   # 10 % 5 == 0 -> kept
    assert pts[1][10] == [(0, 100.0, 200.0)]
    fusion_bridge.accumulate_geo(rows, None, 7, 7, pts, 5)     # 7 % 5 != 0 -> skipped
    assert 7 not in pts.get(1, {})


# ----------------------------------------------------------------- run standalone
if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {fn.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
