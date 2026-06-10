"""Unit tests for the gallery probe mixins (assignment / conflict / merge).

These exercise the identity logic WITHOUT DeepStream/pyservicemaker by composing
the mixins onto a plain harness that holds the same shared state the probe sets
up (stores + track_to_gid + cfg + geometry). This is the verification net that
makes refactoring this logic (row dict -> DetectionRow attrs, service objects)
safe — the live pipeline IDF1 is too noisy to catch subtle regressions.

Run:  python tests/test_gallery_mixins.py   (or python -m pytest)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.reid.config import ReIDConfig
from src.reid.detection_row import DetectionRow
from src.reid.gallery_store import GalleryStore
from src.reid.tracklet_store import TrackletStore
from src.reid.gallery_assignment import GalleryAssignmentMixin
from src.reid.gallery_conflict import GalleryConflictMixin
from src.reid.gallery_merge import GalleryMergeMixin
from src.reid.gallery_rows import GalleryRowsMixin


class Harness(GalleryRowsMixin, GalleryConflictMixin,
              GalleryAssignmentMixin, GalleryMergeMixin):
    """Same shared state as CrossCameraGalleryProbe, minus the DeepStream base."""

    def __init__(self, **cfg_over):
        # single-vector gallery (gallery_max_prototypes=0) -> simple cosine scoring
        cfg_over.setdefault("gallery_max_prototypes", 0)
        self._cfg = ReIDConfig(**cfg_over)
        self._gs = GalleryStore(self._cfg)
        self._gallery = self._gs.gallery
        self._ts = TrackletStore(self._cfg)
        self._tracklets = self._ts.tracklets
        self._track_to_gid = {}
        self._geometry = None
        self._debug_similarity = False
        self._frame_count = 0


def _row(track_id, emb, **over):
    base = dict(src=0, track_id=track_id, track_key=(0, track_id), rect={},
                raw_embedding=emb, embedding=emb)
    base.update(over)
    return DetectionRow(**base)


# ---- assignment ----------------------------------------------------------

def test_find_or_create_empty_gallery_allocates_new():
    h = Harness()
    gid = h._find_or_create([1.0, 0.0], src=0, track_id=1, log=False)
    assert gid in h._gallery


def test_find_or_create_matches_existing():
    h = Harness(similarity_threshold=0.5)
    h._gs.update(1, [1.0, 0.0], src=0)
    assert h._find_or_create([1.0, 0.0], src=0, track_id=2, log=False) == 1


def test_match_blocked_below_threshold():
    h = Harness(similarity_threshold=0.9)
    h._gs.update(1, [1.0, 0.0], src=0)
    ranked = h._gs.rank([0.5, 0.5])           # cosine ~0.707 < 0.9
    allowed, reason = h._is_gid_match_allowed([0.5, 0.5], 1, None, ranked)
    assert not allowed and reason == "below_threshold"


def test_greedy_assigns_new_gid_and_updates_gallery():
    h = Harness(similarity_threshold=0.5)
    row = _row(1, [1.0, 0.0], allow_new_gid=True, update_gallery=True)
    h._assign_new_tracks_greedy([row], log=False)
    assert row.gid is not None and row.gid in h._gallery and row.gallery_updated


def test_hungarian_one_to_one_no_duplicate_gid_in_stream():
    h = Harness(similarity_threshold=0.5)
    h._gs.update(1, [1.0, 0.0], src=0)        # one existing gid
    r1 = _row(10, [1.0, 0.0], allow_new_gid=True)
    r2 = _row(11, [1.0, 0.0], allow_new_gid=True)   # same stream, same look
    h._assign_new_tracks_with_hungarian([r1, r2], log=False)
    assert r1.gid is not None and r2.gid is not None
    assert r1.gid != r2.gid                   # cannot share a gid in one camera


# ---- conflict ------------------------------------------------------------

def test_conflict_releases_weaker_same_stream_duplicate():
    h = Harness()
    strong = _row(1, [1.0, 0.0], gid=1, tracklet_len=5)
    weak = _row(2, [0.9, 0.1], gid=1, tracklet_len=2)
    h._mark_duplicate_known_conflicts([strong, weak])
    assert (strong.gid == 1) != (weak.gid == 1)   # exactly one keeps gid 1
    assert None in (strong.gid, weak.gid)
    released = strong if strong.gid is None else weak
    assert released.release_previous_gid and released.suppress_gallery_update


# ---- merge ---------------------------------------------------------------

def test_candidate_gids_excludes_and_caps():
    h = Harness()
    for g in (1, 2, 3):
        h._gallery[g] = {"embedding": [], "age": g}
    cands = h._candidate_gids(exclude={2}, max_count=10)
    assert 2 not in cands and set(cands) == {1, 3}


def test_merge_gid_folds_mapping_and_tracklets():
    h = Harness()
    h._gs.update(1, [1.0, 0.0], src=0)
    h._gs.update(2, [0.0, 1.0], src=0)
    h._track_to_gid[(0, 5)] = 2
    h._tracklets[(0, 5)] = {"gid": 2, "age": 0, "embeddings": []}
    h._merge_gid(2, 1)
    assert 2 not in h._gallery
    assert h._track_to_gid[(0, 5)] == 1 and h._tracklets[(0, 5)]["gid"] == 1


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
