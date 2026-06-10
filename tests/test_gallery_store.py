"""Unit tests for src/reid/gallery_store.py (GalleryStore).

Run:  python tests/test_gallery_store.py   (or python -m pytest)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.reid.config import ReIDConfig
from src.reid.gallery_store import GalleryStore


def _gs(**over):
    return GalleryStore(ReIDConfig(**over))


def test_allocate_skips_existing():
    s = _gs()
    s.gallery[1] = {"embedding": [], "age": 0}
    s.gallery[2] = {"embedding": [], "age": 0}
    assert s.allocate_gid() == 3            # 1,2 taken -> 3


def test_single_vector_update_and_score():
    s = _gs(gallery_max_prototypes=0)            # single-vector mode
    s.update(1, [1.0, 0.0], src=0)
    assert s.score(1, [1.0, 0.0]) > 0.99    # identical -> ~1
    assert s.score(1, [0.0, 1.0]) < 0.01    # orthogonal -> ~0


def test_rank_orders_by_similarity():
    s = _gs(gallery_max_prototypes=0)
    s.update(1, [1.0, 0.0], src=0)
    s.update(2, [0.0, 1.0], src=0)
    ranked = s.rank([0.9, 0.1])
    assert ranked[0][0] == 1                     # closer to gid 1


def test_prototype_mode_adds_per_source():
    s = _gs(gallery_max_prototypes=24, prototype_add_threshold=0.72)
    s.update(1, [1.0, 0.0], src=0)
    s.update(1, [0.0, 1.0], src=1)      # different view/src -> new prototype
    assert len(s.gallery[1]["prototypes"]) == 2


def test_merge_entries_single_vector():
    s = _gs(gallery_max_prototypes=0)
    s.update(1, [1.0, 0.0], src=0)
    s.update(2, [0.0, 1.0], src=0)
    s.merge(2, 1)               # fold 2 into 1 (no crash; keeps len-ok vec)
    assert 1 in s.gallery


def test_score_unknown_gid_zero():
    s = _gs()
    assert s.score(99, [1.0]) == 0.0


def test_new_entry_shape_matches_mode():
    assert "prototypes" in _gs(gallery_max_prototypes=24).new_entry()
    assert "embedding" in _gs(gallery_max_prototypes=0).new_entry()


def test_expire_removes_old_and_returns_gids():
    s = _gs()
    s.gallery[1] = {"embedding": [], "age": 5}
    s.gallery[2] = {"embedding": [], "age": 1}
    expired = s.expire(max_age=3)
    assert expired == [1]
    assert 1 not in s.gallery and 2 in s.gallery


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
