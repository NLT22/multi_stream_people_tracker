"""Unit tests for src/reid/quality.py (embedding-quality gate, pure logic).

Run:  python tests/test_quality.py   (or: python -m pytest tests/test_quality.py)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.reid import quality

GATE = dict(enabled=True, edge_margin_ratio=0.005, min_height_ratio=0.04,
            min_area_ratio=0.0008, min_aspect=0.12, max_aspect=0.95,
            max_overlap_iou=0.25)


def _row(left, top, w, h, src=0, emb=(1.0,), frame=(640.0, 360.0)):
    return {"src": src, "raw_embedding": list(emb),
            "rect": {"left": left, "top": top, "width": w, "height": h,
                     "frame_w": frame[0], "frame_h": frame[1]}}


def test_rect_iou_identical():
    a = {"left": 0, "top": 0, "width": 10, "height": 10}
    assert abs(quality.rect_iou(a, a) - 1.0) < 1e-9


def test_rect_iou_disjoint():
    a = {"left": 0, "top": 0, "width": 10, "height": 10}
    b = {"left": 100, "top": 100, "width": 10, "height": 10}
    assert quality.rect_iou(a, b) == 0.0


def test_no_embedding_rejected():
    r = _row(100, 100, 30, 80, emb=())
    ok, reason = quality.embedding_quality(r, [r], **GATE)
    assert ok is False and reason == "no_embedding"


def test_disabled_passes():
    r = _row(0, 0, 1, 1)   # would fail every gate
    ok, reason = quality.embedding_quality(r, [r], **{**GATE, "enabled": False})
    assert ok is True and reason == "disabled"


def test_edge_crop_rejected():
    r = _row(0.0, 100, 30, 80)   # left on the frame edge
    ok, reason = quality.embedding_quality(r, [r], **GATE)
    assert ok is False and reason == "edge_crop"


def test_small_height_rejected():
    r = _row(100, 100, 30, 5)    # 5/360 < 0.04
    ok, reason = quality.embedding_quality(r, [r], **GATE)
    assert ok is False and reason in ("small_height", "small_area")


def test_good_crop_passes():
    r = _row(200, 100, 40, 110)  # centered, full-body aspect
    ok, reason = quality.embedding_quality(r, [r], **GATE)
    assert ok is True and reason == "ok"


def test_overlap_rejected():
    a = _row(200, 100, 40, 110, src=0)
    b = _row(205, 105, 40, 110, src=0)   # same cam, big overlap
    ok, reason = quality.embedding_quality(a, [a, b], **GATE)
    assert ok is False and reason.startswith("overlap_iou")


def test_overlap_other_camera_ignored():
    a = _row(200, 100, 40, 110, src=0)
    b = _row(205, 105, 40, 110, src=1)   # different cam -> not counted
    ok, reason = quality.embedding_quality(a, [a, b], **GATE)
    assert ok is True and reason == "ok"


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
