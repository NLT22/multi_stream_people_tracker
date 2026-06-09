"""Pin the ReIDConfig production defaults so they cannot silently drift.

These values are the deployed tuning that produces the regression-anchored
results; a change here must be deliberate. Run:
  python tests/test_config.py   (or: python -m pytest tests/test_config.py)
"""

import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.reid.config import ReIDConfig

EXPECTED = {
    "similarity_threshold": 0.68,
    "gallery_max_age": 1800,
    "gallery_max_prototypes": 24,
    "id_switch_margin": 0.12,
    "match_ambiguity_margin": 0.06,
    "global_id_merge_threshold": 0.76,
    "tracklet_max_embeddings": 8,
    "tracklet_min_embeddings_for_match": 3,
    "geo_weight": 0.35,
    "geo_assignment_mode": "weight_only",
    "use_micro_batch_fusion": False,
    "micro_batch_fusion_interval": 125,
    "micro_batch_fusion_threshold": 0.55,
    "micro_batch_fusion_min_gid_embeddings": 4,
    "micro_batch_fusion_min_tracklet_detections": 6,
    "micro_batch_fusion_geo_weight": 0.25,
    "enable_embedding_quality_gate": True,
    "use_hungarian_assignment": True,
    "enforce_unique_global_per_stream": True,
}


def test_production_defaults_pinned():
    c = ReIDConfig()
    for field, val in EXPECTED.items():
        assert getattr(c, field) == val, f"{field}={getattr(c, field)} != {val}"


def test_from_args_overrides_and_clamps():
    # minimal args object with the attributes from_args reads
    args = types.SimpleNamespace(
        similarity_threshold=-5.0,           # clamped to >= 0
        gallery_max_age=0,                   # clamped to >= 1
        assignment_max_candidates=10,
        disable_id_stickiness=True,          # -> enable_id_stickiness False
        id_switch_margin=0.2,
        allow_ambiguous_match=False,
        match_ambiguity_margin=0.05,
        disable_global_merge=True,           # -> enable_global_id_merge False
        global_merge_threshold=0.8,
        global_merge_min_embeddings=5,
        global_merge_margin=0.03,
        global_merge_interval=7,
        global_merge_max_candidates=40,
        disable_tracklet=False,
        tracklet_embedding_interval=3,
        disable_embedding_quality_gate=False,
        tracklet_window=10,
        tracklet_min_embeddings=2,
        tracklet_max_age=900,
        geo_weight=2.0,                      # clamped to <= 1
        geometry_assignment_mode="close_reid_only",
        geometry_reid_margin=0.5,
        micro_batch_fusion=True,
        fusion_interval=50,
        fusion_threshold=0.6,
    )
    c = ReIDConfig.from_args(args)
    assert c.similarity_threshold == 0.0          # clamped
    assert c.gallery_max_age == 1                 # clamped
    assert c.enable_id_stickiness is False
    assert c.enable_global_id_merge is False
    assert c.geo_weight == 1.0                     # clamped
    assert c.geo_assignment_mode == "close_reid_only"
    assert c.use_micro_batch_fusion is True
    assert c.micro_batch_fusion_interval == 50


def test_use_prototypes():
    assert ReIDConfig(gallery_max_prototypes=24).use_prototypes() is True
    assert ReIDConfig(gallery_max_prototypes=0).use_prototypes() is False


def test_summary_lines():
    s = ReIDConfig().summary()
    assert "similarity threshold=0.68" in s
    assert "micro_batch_fusion=False" in s


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
