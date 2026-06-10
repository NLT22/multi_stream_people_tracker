"""Stable duplicate Global-ID merging across difficult cross views.

Extracted from gallery.py as a mixin of CrossCameraGalleryProbe. These methods
operate on the probe's shared state (self._gs, self._track_to_gid,
self._tracklets, self._cfg, ...); the split is by concern, not ownership.
"""

from __future__ import annotations


class GalleryMergeMixin:
    def _merge_duplicate_global_ids(self, rows: list[dict], log: bool) -> None:
        """Merge stable duplicate Global IDs created by difficult cross views."""
        active_by_src: dict[int, set[int]] = {}
        for row in rows:
            active_by_src.setdefault(row.src, set()).add(row.gid)

        for row in rows:
            source_gid = row.gid
            if source_gid is None or source_gid not in self._gallery:
                continue
            if self._gallery[source_gid].get("age", 0) > 1:
                continue
            if row.tracklet_len < self._cfg.global_id_merge_min_tracklet_embeddings:
                continue
            if not row.embedding:
                continue

            candidate = self._best_merge_candidate(
                source_gid, row, active_by_src)
            if candidate is None:
                continue

            target_gid, score, runner_up = candidate
            self._merge_gid(source_gid, target_gid)
            for update_row in rows:
                if update_row.gid == source_gid:
                    update_row.gid = target_gid
                    update_row.previous_gid = target_gid
                    self._track_to_gid[update_row.track_key] = target_gid
                    self._tracklets[update_row.track_key]["gid"] = target_gid

            if self._debug_similarity or log:
                print(
                    f"  [Re-ID merge] G{source_gid} -> G{target_gid} "
                    f"score={score:.3f} runner_up={runner_up:.3f} "
                    f"tracklet_len={row['tracklet_len']} "
                    f"Cam{row['src']}#{row['track_id']}"
                )

    def _best_merge_candidate(self, source_gid: int, row: dict,
                              active_by_src: dict[int, set[int]]
                              ) -> tuple[int, float, float] | None:
        candidates = self._candidate_gids(
            exclude=active_by_src.get(row.src, set()),
            max_count=self._cfg.global_id_merge_max_candidates,
            only_older_than=source_gid,
        )

        scores = []
        for target_gid in candidates:
            reid_score = self._gs.score(target_gid, row.embedding)
            scores.append((
                target_gid,
                self._blend_geo_score(reid_score, row, target_gid),
            ))

        if not scores:
            return None

        scores.sort(key=lambda item: item[1], reverse=True)
        target_gid, best_score = scores[0]
        runner_up = scores[1][1] if len(scores) > 1 else 0.0
        if best_score < self._cfg.global_id_merge_threshold:
            return None
        if runner_up > 0.0 and best_score < runner_up + self._cfg.global_id_merge_margin:
            return None
        return target_gid, best_score, runner_up

    def _candidate_gids(self, exclude: set[int] | None = None,
                        max_count: int = 80,
                        only_older_than: int | None = None) -> list[int]:
        """Return a bounded list of recent gallery IDs for expensive matching."""
        exclude = exclude or set()
        candidates = []
        for gid, entry in self._gallery.items():
            if gid in exclude:
                continue
            if only_older_than is not None and gid >= only_older_than:
                continue
            candidates.append((entry.get("age", 0), gid))

        candidates.sort(key=lambda item: (item[0], -item[1]))
        return [gid for _, gid in candidates[:max_count]]

    def _merge_gid(self, source_gid: int, target_gid: int) -> None:
        if source_gid == target_gid or source_gid not in self._gallery:
            return
        if target_gid not in self._gallery:
            self._gallery[target_gid] = self._gs.new_entry()

        self._gs.merge(source_gid, target_gid)
        for track_key, gid in list(self._track_to_gid.items()):
            if gid == source_gid:
                self._track_to_gid[track_key] = target_gid
        for tracklet in self._tracklets.values():
            if tracklet.get("gid") == source_gid:
                tracklet["gid"] = target_gid
        del self._gallery[source_gid]
