"""Unit tests for src/reid/detection_row.py (DetectionRow).

Locks the dict-compat shim so the dataclass stays a drop-in replacement for the
old row dict while call sites migrate to attribute access.

Run:  python tests/test_detection_row.py   (or python -m pytest)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.reid.detection_row import DetectionRow


def _row(**over):
    base = dict(src=0, track_id=5, track_key=(0, 5), rect={"left": 1.0},
                raw_embedding=[1.0, 0.0])
    base.update(over)
    return DetectionRow(**base)


def test_attr_and_item_access_agree():
    r = _row()
    assert r.src == 0 and r["src"] == 0
    assert r["rect"]["left"] == 1.0
    assert r.raw_embedding == [1.0, 0.0]


def test_item_set_updates_attr():
    r = _row()
    r["gid"] = 7
    assert r.gid == 7 and r["gid"] == 7
    r.previous_gid = 3
    assert r["previous_gid"] == 3


def test_get_with_default_matches_dict():
    r = _row()
    assert r.get("gid") is None            # field present, default None
    assert r.get("gid", 9) is None         # present -> field value, not default
    assert r.get("does_not_exist", "d") == "d"


def test_defaults_present():
    r = _row()
    for name, expect in [("embedding", []), ("tracklet_len", 0),
                         ("gid", None), ("embedding_quality_ok", False),
                         ("embedding_quality_reason", ""),
                         ("gallery_updated", False), ("defer_assignment", False)]:
        assert r[name] == expect, name


def test_independent_embedding_lists():
    a, b = _row(), _row()
    a.embedding.append(1.0)
    assert b.embedding == []                # default_factory, not shared


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
