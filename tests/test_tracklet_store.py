"""Unit tests for src/reid/tracklet_store.py (TrackletStore).

Run:  python tests/test_tracklet_store.py   (or python -m pytest)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.reid.config import ReIDConfig
from src.reid.tracklet_store import TrackletStore


def _store(**over):
    cfg = ReIDConfig(**over)
    return TrackletStore(cfg)


def test_update_creates_and_counts():
    s = _store()
    t1 = s.update((0, 5), 0, 5, [1.0, 0.0], frame_count=10, initial_gid=3)
    assert t1["tracklet_id"] == 1 and t1["gid"] == 3 and t1["num_detections"] == 1
    t2 = s.update((0, 5), 0, 5, [1.0, 0.0], frame_count=11)
    assert t2 is t1 and t2["num_detections"] == 2          # same tracklet refreshed
    assert s.update((0, 6), 0, 6, [1.0], frame_count=11)["tracklet_id"] == 2


def test_embedding_capped():
    s = _store(tracklet_max_embeddings=3, tracklet_min_embeddings_for_match=1,
               tracklet_embedding_interval=1)
    for f in range(10):
        s.update((0, 5), 0, 5, [1.0, 0.0], frame_count=f)
    assert len(s.tracklets[(0, 5)]["embeddings"]) == 3      # rolling window capped


def test_quality_gate_blocks_sampling():
    s = _store()
    t = s.update((0, 5), 0, 5, [1.0, 0.0], frame_count=0, quality_ok=False)
    assert t["sampled_this_frame"] is False and t["embeddings"] == []


def test_fusion_sum_accumulates_uncapped():
    s = _store(tracklet_max_embeddings=2, tracklet_min_embeddings_for_match=1,
               tracklet_embedding_interval=1)
    for f in range(5):
        s.update((0, 5), 0, 5, [1.0, 0.0], frame_count=f, use_fusion=True)
    t = s.tracklets[(0, 5)]
    assert t["fusion_emb_count"] == 5                       # uncapped sum
    assert len(t["embeddings"]) == 2                        # but rolling list capped


def test_tracklet_embedding_fallback():
    s = _store(tracklet_min_embeddings_for_match=3)
    t = s.update((0, 5), 0, 5, [1.0, 0.0], frame_count=0)   # only 1 embedding
    assert s.tracklet_embedding(t, fallback=[9.0]) == [9.0]


def test_tracklet_embedding_mean_when_enough():
    s = _store(tracklet_min_embeddings_for_match=2, tracklet_embedding_interval=1)
    s.update((0, 5), 0, 5, [2.0, 0.0], frame_count=0)
    t = s.update((0, 5), 0, 5, [2.0, 0.0], frame_count=1)
    m = s.tracklet_embedding(t)
    assert abs(m[0] - 1.0) < 1e-6                           # normalized mean of [2,0]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"  PASS {fn.__name__}")
        except Exception as e:
            failed += 1; print(f"  FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
