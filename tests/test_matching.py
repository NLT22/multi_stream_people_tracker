"""Unit tests for src/reid/matching.py (pure helpers extracted from gallery.py).

Run:  python tests/test_matching.py   (or: python -m pytest tests/test_matching.py)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.reid.matching import _cosine_similarity, _mean_embedding, max_weight_assignment


def test_cosine_identical():
    assert abs(_cosine_similarity([1, 0, 0], [1, 0, 0]) - 1.0) < 1e-6


def test_cosine_orthogonal():
    assert abs(_cosine_similarity([1, 0], [0, 1])) < 1e-6


def test_cosine_empty_is_zero():
    assert _cosine_similarity([], [1, 0]) == 0.0
    assert _cosine_similarity(None, [1, 0]) == 0.0


def test_mean_embedding_normalized():
    m = _mean_embedding([[2.0, 0.0], [2.0, 0.0]])
    assert abs(m[0] - 1.0) < 1e-6 and abs(m[1]) < 1e-6   # averaged + L2-normalized


def test_mean_embedding_skips_mismatched_dims():
    m = _mean_embedding([[1.0, 0.0], [1.0, 0.0, 0.0]])   # second wrong dim dropped
    assert len(m) == 2


def test_hungarian_diagonal():
    # max-weight on the diagonal -> identity assignment
    assert max_weight_assignment([[5, 1], [1, 5]]) == [0, 1]


def test_hungarian_swap():
    assert max_weight_assignment([[1, 5], [5, 1]]) == [1, 0]


def test_hungarian_rectangular():
    # 2 rows, 3 cols (extra "new id" slots) -> each row picks its best column
    a = max_weight_assignment([[9, 0, 0], [0, 0, 9]])
    assert a[0] == 0 and a[1] == 2


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
