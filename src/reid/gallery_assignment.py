"""Global-ID assignment: greedy / Hungarian, match gating, geo blending.

Extracted from gallery.py as a mixin of CrossCameraGalleryProbe. These methods
operate on the probe's shared state (self._gs, self._track_to_gid,
self._tracklets, self._cfg, ...); the split is by concern, not ownership.
"""

from __future__ import annotations

from src.reid.matching import max_weight_assignment
from src.reid.geometry import GroundPlaneGeometry

class GalleryAssignmentMixin:
    def _find_or_create(self, embedding: list[float], src: int,
                        track_id: int, log: bool,
                        tracklet_len: int = 0,
                        previous_gid: int | None = None) -> int:
        """Match embedding against gallery; return existing or new global_id."""
        ranked = self._gs._rank_gallery(embedding)
        best_gid = ranked[0][0] if ranked else -1
        best_score = ranked[0][1] if ranked else 0.0
        allowed, block_reason = self._is_gid_match_allowed(
            embedding, best_gid, previous_gid, ranked)

        matched = best_gid != -1 and allowed
        reason = "no_embedding" if not embedding else (
            "empty_gallery" if best_gid == -1 else "below_threshold"
        )
        if best_gid != -1 and not allowed:
            reason = block_reason
        should_log_similarity = self._debug_similarity or (log and matched)
        if should_log_similarity:
            status = "MATCH" if matched else "NEW"
            display_reason = "above_threshold" if matched else reason
            top = ", ".join(
                f"G{gid}={score:.3f}" for gid, score in ranked[:self._cfg.debug_top_k]
            ) or "none"
            print(
                f"  [Re-ID similarity] Cam{src}#{track_id} "
                f"best_gid={best_gid if best_gid != -1 else 'None'} "
                f"max_similarity={best_score:.3f} "
                f"threshold={self._cfg.similarity_threshold:.3f} "
                f"tracklet_len={tracklet_len} "
                f"previous_gid={previous_gid if previous_gid is not None else 'None'} "
                f"status={status} reason={display_reason} top{self._cfg.debug_top_k}=[{top}]"
            )

        if matched:
            if log:
                print(f"  [Re-ID] Cam{src}#{track_id} → G#{best_gid} "
                      f"(similarity={best_score:.3f})")
            return best_gid

        # If the track already has a known global ID still alive in the gallery,
        # keep it rather than minting a new one. This prevents ID explosion when
        # a track's embedding temporarily dips below threshold (occlusion, blur).
        if previous_gid is not None and previous_gid in self._gallery:
            if should_log_similarity:
                print(f"  [Re-ID] Cam{src}#{track_id} → G#{previous_gid} "
                      f"(sticky: below threshold but known track)")
            return previous_gid

        # New person
        gid = self._gs._allocate_new_gid()
        self._gallery[gid] = self._gs._new_gallery_entry()
        return gid

    def _assign_new_tracks_greedy(self, rows: list[dict], log: bool) -> None:
        for row in rows:
            if row["gid"] is None and not row.get("defer_assignment"):
                if row.get("allow_new_gid"):
                    row["gid"] = self._find_or_create(
                        row["embedding"], row["src"], row["track_id"], log,
                        row["tracklet_len"], row.get("previous_gid"))
                else:
                    ranked = self._gs._rank_gallery(row["embedding"])
                    best_gid = ranked[0][0] if ranked else -1
                    allowed, _ = self._is_gid_match_allowed(
                        row["embedding"], best_gid, row.get("previous_gid"),
                        ranked)
                    if allowed:
                        row["gid"] = best_gid
                    else:
                        continue
                # Greedy fallback: once a new track is assigned, later
                # detections in the same tiled frame can match it.
                self._gs._update_gallery(
                    row["gid"],
                    row["raw_embedding"] if row.get("update_gallery") else [],
                    row["src"],
                )
                row["gallery_updated"] = True

    def _assign_new_tracks_with_hungarian(self, rows: list[dict],
                                          log: bool) -> None:
        """
        Assign new tracks per stream with one-to-one Global ID constraints.

        Known local tracks keep their existing global ID. New tracks in the same
        stream compete for currently available global IDs plus one private
        "new ID" slot per track. This prevents the physically impossible state
        where one global ID appears twice in one camera frame, while still
        allowing different cameras to match the same global ID.
        """
        rows_by_src: dict[int, list[dict]] = {}
        occupied_by_src: dict[int, set[int]] = {}
        for row in rows:
            src = row["src"]
            if row["gid"] is None and not row.get("defer_assignment"):
                rows_by_src.setdefault(src, []).append(row)
            else:
                if row["gid"] is not None:
                    occupied_by_src.setdefault(src, set()).add(row["gid"])

        for src, new_rows in rows_by_src.items():
            occupied = occupied_by_src.setdefault(src, set())
            existing_gids = self._candidate_gids(
                exclude=occupied,
                max_count=self._cfg.global_assignment_max_candidates,
            )
            columns = [("gid", gid) for gid in existing_gids]
            columns += [("new", i) for i in range(len(new_rows))]

            weights = []
            for row in new_rows:
                # Pre-compute scores and ranked list once per row (not per cell).
                # Without this, ranked is recomputed for every (row, gid) pair
                # → O(rows × gids²) calls to _score_gid instead of O(rows × gids).
                scores_for_row = {
                    gid: self._gs._score_gid(gid, row["embedding"])
                    for gid in existing_gids
                }
                ranked = sorted(scores_for_row.items(),
                                key=lambda item: item[1], reverse=True)
                row_weights = []
                best_reid_score = ranked[0][1] if ranked else 0.0
                for kind, value in columns:
                    if kind == "gid":
                        reid_score = scores_for_row[value]
                        assignment_score = self._assignment_score(
                            reid_score, best_reid_score, row, value)
                        allowed, _ = self._is_gid_match_allowed(
                            row["embedding"], value, row.get("previous_gid"),
                            ranked)
                        row_weights.append(assignment_score if allowed else -1.0)
                    else:
                        row_weights.append(0.0)
                weights.append(row_weights)

            assignment = max_weight_assignment(weights)
            for row_idx, col_idx in enumerate(assignment):
                row = new_rows[row_idx]
                kind, value = columns[col_idx]
                if kind == "gid":
                    gid = value
                    score = self._gs._score_gid(gid, row["embedding"])
                    row["gid"] = gid
                    occupied.add(gid)
                    status = "MATCH"
                    reason = "hungarian"
                else:
                    if not row.get("allow_new_gid"):
                        row["gid"] = None
                        score = 0.0
                        status = "DEFER"
                        reason = row.get(
                            "embedding_quality_reason",
                            "low_quality_new_track",
                        )
                        if self._debug_similarity:
                            ranked = [
                                (gid, self._gs._score_gid(gid, row["embedding"]))
                                for gid in existing_gids
                            ]
                            ranked.sort(key=lambda item: item[1], reverse=True)
                            top = ", ".join(
                                f"G{gid}={s:.3f}" for gid, s in ranked[:self._cfg.debug_top_k]
                            ) or "none"
                            print(
                                f"  [Re-ID Hungarian] Cam{src}#{row['track_id']} "
                                f"assigned=None score={score:.3f} "
                                f"threshold={self._cfg.similarity_threshold:.3f} "
                                f"tracklet_len={row['tracklet_len']} "
                                f"quality={row.get('embedding_quality_reason')} "
                                f"status={status} reason={reason} "
                                f"top{self._cfg.debug_top_k}=[{top}]"
                            )
                        continue

                    gid = self._gs._allocate_new_gid()
                    self._gallery[gid] = self._gs._new_gallery_entry()
                    row["gid"] = gid
                    occupied.add(gid)
                    score = self._gs._score_gid(gid, row["embedding"])
                    status = "NEW"
                    reason = "new_slot"

                if self._debug_similarity:
                    ranked = [
                        (gid, self._gs._score_gid(gid, row["embedding"]))
                        for gid in existing_gids
                    ]
                    ranked.sort(key=lambda item: item[1], reverse=True)
                    top = ", ".join(
                        f"G{gid}={s:.3f}" for gid, s in ranked[:self._cfg.debug_top_k]
                    ) or "none"
                    print(
                        f"  [Re-ID Hungarian] Cam{src}#{row['track_id']} "
                        f"assigned=G{row['gid']} score={score:.3f} "
                        f"threshold={self._cfg.similarity_threshold:.3f} "
                        f"tracklet_len={row['tracklet_len']} "
                        f"previous_gid={row.get('previous_gid') if row.get('previous_gid') is not None else 'None'} "
                        f"status={status} reason={reason} top{self._cfg.debug_top_k}=[{top}]"
                    )

            for row in new_rows:
                if row["gid"] is not None:
                    self._gs._update_gallery(
                        row["gid"],
                        row["raw_embedding"] if row.get("update_gallery") else [],
                        row["src"],
                    )
                    row["gallery_updated"] = True

    def _is_gid_match_allowed(self, embedding: list[float],
                              candidate_gid: int | None,
                              previous_gid: int | None,
                              ranked: list[tuple[int, float]]) -> tuple[bool, str]:
        """Apply threshold, stickiness, and ambiguity gates for a candidate ID."""
        if not embedding:
            return False, "no_embedding"
        if candidate_gid is None or candidate_gid == -1:
            return False, "empty_gallery"
        if candidate_gid not in self._gallery:
            return False, "stale_candidate"

        candidate_score = self._gs._score_gid(candidate_gid, embedding)
        if candidate_score < self._cfg.similarity_threshold:
            return False, "below_threshold"

        if (
            self._cfg.enable_id_stickiness
            and previous_gid is not None
            and previous_gid in self._gallery
            and candidate_gid != previous_gid
        ):
            previous_score = self._gs._score_gid(previous_gid, embedding)
            if candidate_score < previous_score + self._cfg.id_switch_margin:
                return (
                    False,
                    f"switch_margin(prev=G{previous_gid},"
                    f"prev_score={previous_score:.3f})",
                )

        if (
            self._cfg.enable_ambiguous_match_rejection
            and candidate_gid != previous_gid
        ):
            runner_up = max(
                (score for gid, score in ranked if gid != candidate_gid),
                default=0.0,
            )
            if runner_up > 0.0 and candidate_score < runner_up + self._cfg.match_ambiguity_margin:
                return False, f"ambiguous(runner_up={runner_up:.3f})"

        return True, "ok"

    def _blend_geo_score(self, reid_score: float, row: dict,
                         candidate_gid: int) -> float:
        """
        Blend geometry score into the ReID score for cross-camera pairs.

        Same-camera assignments are left unchanged (geometry doesn't help
        when the local tracker already handles intra-camera identity).
        Returns reid_score unchanged when geometry is disabled or unavailable.
        """
        if self._geometry is None or self._cfg.geo_weight <= 0.0:
            return reid_score

        foot_q = row.get("foot_world")
        if foot_q is None:
            return reid_score

        # Look for the best geo score among all tracklets mapped to this gid
        # that come from a *different* source.
        best_geo = 0.0
        for (t_src, _t_id), t_gid in self._track_to_gid.items():
            if t_gid != candidate_gid:
                continue
            if t_src == row["src"]:
                continue   # same camera — skip
            foot_t = self._tracklets.get((t_src, _t_id), {}).get("foot_world")
            g = GroundPlaneGeometry.geo_score(foot_q, foot_t)
            if g > best_geo:
                best_geo = g

        if best_geo == 0.0:
            return reid_score

        return (1.0 - self._cfg.geo_weight) * reid_score + self._cfg.geo_weight * best_geo

    def _assignment_score(self, reid_score: float, best_reid_score: float,
                          row: dict, candidate_gid: int) -> float:
        if self._cfg.geo_assignment_mode == "close_reid_only":
            if best_reid_score - reid_score > self._cfg.geo_reid_margin:
                return reid_score
        return self._blend_geo_score(reid_score, row, candidate_gid)
