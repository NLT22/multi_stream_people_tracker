"""Same-stream duplicate-GID conflict resolution before assignment.

Extracted from gallery.py as a mixin of CrossCameraGalleryProbe. These methods
operate on the probe's shared state (self._gs, self._track_to_gid,
self._tracklets, self._cfg, ...); the split is by concern, not ownership.
"""

from __future__ import annotations


class GalleryConflictMixin:
    def _mark_duplicate_known_conflicts(self, rows: list[dict]) -> None:
        """Release weaker same-stream duplicate GIDs before assignment.

        A single Global ID cannot represent two simultaneous tracks in the same
        camera. Keep the stronger holder stable and send the weaker row back
        through Hungarian assignment so it can take another existing ID or open
        a new one. This prevents same-frame duplicate GIDs from poisoning IDF1.
        """
        active: dict[tuple[int, int], dict] = {}
        for row in rows:
            gid = row.gid
            if gid is None:
                continue

            key = (row.src, gid)
            existing = active.get(key)
            if existing is None:
                active[key] = row
                continue

            existing_score = self._gs.score(gid, existing.embedding)
            row_score = self._gs.score(gid, row.embedding)

            if self._prefer_conflict_gallery_update(
                row, row_score, existing, existing_score
            ):
                suppressed = existing
                active[key] = row
            else:
                suppressed = row

            active[key].identity_conflict = True
            suppressed.identity_conflict = True
            suppressed.suppress_gallery_update = True
            suppressed.release_previous_gid = True
            suppressed.previous_gid = gid
            suppressed.gid = None
            if self._debug_similarity:
                print(
                    f"  [Re-ID conflict] Cam{suppressed['src']}#{suppressed['track_id']} "
                    f"duplicate_known_gid=G{gid} "
                    f"held_by=Cam{active[key]['src']}#{active[key]['track_id']} "
                    f"suppressed_gallery_update_score={self._gs.score(gid, suppressed['embedding']):.3f} "
                    f"held_score={self._gs.score(gid, active[key]['embedding']):.3f}"
                )

    @staticmethod
    def _prefer_conflict_gallery_update(candidate: dict, candidate_score: float,
                                        incumbent: dict,
                                        incumbent_score: float) -> bool:
        """Return True when candidate should be the gallery updater."""
        candidate_len = candidate.get("tracklet_len", 0)
        incumbent_len = incumbent.get("tracklet_len", 0)
        if candidate_len != incumbent_len:
            return candidate_len > incumbent_len
        if candidate_score != incumbent_score:
            return candidate_score > incumbent_score
        return candidate.get("track_id", 0) < incumbent.get("track_id", 0)
