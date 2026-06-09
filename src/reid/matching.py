"""Pure, stateless matching helpers used by the cross-camera gallery.
Extracted from src/reid/gallery.py (no shared state — unit-testable).
"""

import math

import numpy as np


def _cosine_similarity(a, b) -> float:
    """Cosine similarity between two embedding vectors. Returns 0.0–1.0.

    Accepts list or np.ndarray for either argument. Use len() rather than
    truthiness so multi-element np arrays (gallery storage) don't raise
    "truth value of an array is ambiguous".
    """
    if a is None or b is None or len(a) == 0 or len(b) == 0:
        return 0.0
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    score = float(np.dot(va, vb) / (norm_a * norm_b))
    if not math.isfinite(score):
        return 0.0
    return score


def _mean_embedding(embeddings: list[list[float]]) -> list[float]:
    """Average same-sized embeddings and L2-normalize the result."""
    valid = [e for e in embeddings if e]
    if not valid:
        return []

    dim = len(valid[0])
    same_dim = [e for e in valid if len(e) == dim]
    if not same_dim:
        return []

    arr = np.array(same_dim, dtype=np.float32)   # shape (n, dim)
    mean = arr.mean(axis=0)                        # shape (dim,)
    norm = np.linalg.norm(mean)
    if norm == 0.0:
        return mean.tolist()
    return (mean / norm).tolist()


def max_weight_assignment(weights: list[list[float]]) -> list[int]:
    """
    Hungarian assignment for max-weight rectangular matrices.

    Returns a list where result[row] = assigned column. The implementation uses
    the classic O(n^2*m) shortest augmenting path form for min-cost assignment,
    converting max weights to costs internally. It assumes columns >= rows; the
    caller always provides enough "new identity" dummy columns.
    """
    if not weights:
        return []

    n = len(weights)
    m = len(weights[0])
    if m < n:
        raise ValueError("Hungarian assignment requires columns >= rows")

    max_w = max(max(row) for row in weights)
    cost = [[max_w - w for w in row] for row in weights]

    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (m + 1)
        used = [False] * (m + 1)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j

            for j in range(0, m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = [-1] * n
    for j in range(1, m + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1
    return assignment


