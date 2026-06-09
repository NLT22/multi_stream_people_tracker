"""Global-identity store: per-Global-ID appearance memory (prototypes or single
vector), scoring, allocation, and merge.

Owns the `global_id -> entry` dict and `_next_gid`. Separated from the DeepStream
probe so the matching/scoring/prototype logic is plain Python and unit-testable.
The probe holds a GalleryStore and aliases `self._gallery = store.gallery`.
"""

from __future__ import annotations

import math

import numpy as np

from src.reid.matching import _cosine_similarity


class GalleryStore:
    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self.gallery: dict[int, dict] = {}   # global_id -> entry
        self._next_gid = 1
        self.frame_count = 0                  # set by the probe each batch

    # -------------------------------------------------------------- entry mode
    def _use_prototypes(self) -> bool:
        return self._cfg.gallery_max_prototypes > 0

    def _new_gallery_entry(self) -> dict:
        if self._use_prototypes():
            return {"prototypes": [], "age": 0}
        return {"embedding": [], "age": 0}

    @staticmethod
    def _best_prototype_score(embedding, entry: dict,
                              src: int | None = None) -> float:
        prototypes = entry.get("prototypes", [])
        if src is not None:
            prototypes = [p for p in prototypes if p.get("src") == src]
        if not prototypes or embedding is None or len(embedding) == 0:
            return 0.0
        # Vectorized cosine against all prototypes at once.
        q = np.asarray(embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0.0:
            return 0.0
        mat = np.asarray([p["embedding"] for p in prototypes], dtype=np.float32)
        denom = np.linalg.norm(mat, axis=1) * q_norm
        sims = np.where(denom > 0.0, (mat @ q) / np.where(denom > 0.0, denom, 1.0), 0.0)
        best = float(sims.max())
        return best if math.isfinite(best) else 0.0

    # ------------------------------------------------------------- scoring
    def _rank_gallery(self, embedding: list[float]) -> list[tuple[int, float]]:
        """Global IDs ranked by single-embedding or prototype similarity."""
        if not embedding:
            return []
        scores = []
        for gid, entry in self.gallery.items():
            if self._use_prototypes():
                score = self._best_prototype_score(embedding, entry)
            else:
                score = _cosine_similarity(embedding, entry.get("embedding", []))
            scores.append((gid, score))
        return sorted(scores, key=lambda item: item[1], reverse=True)

    def _score_gid(self, gid: int, embedding: list[float]) -> float:
        if not embedding or gid not in self.gallery:
            return 0.0
        entry = self.gallery[gid]
        if self._use_prototypes():
            return self._best_prototype_score(embedding, entry)
        return _cosine_similarity(embedding, entry.get("embedding", []))

    # ------------------------------------------------------------- mutation
    def _allocate_new_gid(self) -> int:
        while self._next_gid in self.gallery:
            self._next_gid += 1
        gid = self._next_gid
        self._next_gid += 1
        return gid

    def _merge_gallery_entries(self, source_gid: int, target_gid: int) -> None:
        source = self.gallery.get(source_gid, self._new_gallery_entry())
        target = self.gallery.setdefault(target_gid, self._new_gallery_entry())
        target["age"] = min(target.get("age", 0), source.get("age", 0))

        if not self._use_prototypes():
            source_embedding = source.get("embedding", [])
            if len(source_embedding) > 0:        # len(), not truthiness (np array)
                target["embedding"] = source_embedding
            return

        target_prototypes = target.setdefault("prototypes", [])
        target_prototypes.extend(source.get("prototypes", []))
        target_prototypes.sort(key=lambda p: p.get("last_seen", 0))
        if len(target_prototypes) > self._cfg.gallery_max_prototypes:
            del target_prototypes[:-self._cfg.gallery_max_prototypes]

    def _update_gallery(self, gid: int, embedding: list[float], src: int) -> None:
        """Refresh a Global ID using single-vector or prototype mode."""
        entry = self.gallery.setdefault(gid, self._new_gallery_entry())
        entry["age"] = 0
        if not embedding:
            return

        if not self._use_prototypes():
            entry["embedding"] = np.asarray(embedding, dtype=np.float32)
            return

        prototypes = entry["prototypes"]
        same_src_score = self._best_prototype_score(embedding, entry, src=src)
        all_score = self._best_prototype_score(embedding, entry)
        has_src = any(p.get("src") == src for p in prototypes)
        should_add = (
            not prototypes
            or not has_src
            or same_src_score < self._cfg.prototype_add_threshold
            or all_score < self._cfg.prototype_add_threshold
        )
        if not should_add:
            return

        prototypes.append({
            "embedding": np.asarray(embedding, dtype=np.float32),
            "src": src,
            "last_seen": self.frame_count,
        })
        if len(prototypes) > self._cfg.gallery_max_prototypes:
            del prototypes[:-self._cfg.gallery_max_prototypes]
