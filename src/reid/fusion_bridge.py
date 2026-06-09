"""Bridge between the gallery's tracklet evidence and MicroBatchFusion.

Turns the gallery's (or exporter's) tracklet state into the per-segment record
list the streaming fusion engine ingests, and runs one fresh-engine pass to get
the raw->stable Global-ID remap. Kept out of the DeepStream probe so the
record-building is pure and unit-testable (see tests/test_fusion.py).
"""

from __future__ import annotations

import numpy as np


# (tracklet_id, cam_id, local_track_id, gid, start, end, num_det, num_emb, mean_emb)
def build_records(exporter_tracklets, gallery_tracklets, track_to_gid,
                  tid_by_key: dict, frame_count: int) -> list[tuple]:
    """Build fusion ingest records, one per clean single-gid tracklet segment.

    Prefers the exporter's per-(cam, local, gid) summaries (the exact vectors the
    offline pass clusters); falls back to the gallery's end-state tracklets when
    not exporting (OSD-only live mode). `tid_by_key` is mutated to assign stable
    tracklet ids to exporter keys.
    """
    records: list[tuple] = []
    if exporter_tracklets:
        for key, entry in exporter_tracklets.items():
            cam_id, local_track_id, gid = key
            if gid < 0:
                continue
            tid = tid_by_key.setdefault(key, len(tid_by_key))
            mean = _norm_mean(entry.get("sum_embedding"),
                              entry.get("num_embeddings", 0), already_summed=True)
            records.append((
                tid, cam_id, local_track_id, gid,
                entry.get("start_frame", 0),
                entry.get("end_frame", frame_count),
                entry.get("num_detections", 0),
                entry.get("num_embeddings", 0), mean,
            ))
    else:
        for (src, tid_key), tracklet in gallery_tracklets.items():
            raw_gid = track_to_gid.get((src, tid_key), tracklet.get("gid"))
            if raw_gid is None:
                continue
            emb_count = tracklet.get("fusion_emb_count", 0)
            mean = _norm_mean(tracklet.get("fusion_emb_sum"), emb_count,
                              already_summed=False)
            records.append((
                tracklet["tracklet_id"], src, tid_key, raw_gid,
                tracklet.get("start_frame", 0),
                tracklet.get("end_frame", frame_count),
                tracklet.get("num_detections", 0), emb_count, mean,
            ))
    return records


def _norm_mean(emb_sum, count, *, already_summed):
    """L2-normalized mean embedding from a running sum, or None."""
    if emb_sum is None or count <= 0:
        return None
    v = np.asarray(emb_sum, dtype=np.float32)
    if already_summed:
        v = v / count
    norm = float(np.linalg.norm(v))
    return (v / norm).astype(np.float32) if norm > 0.0 else None


def run_fusion_pass(records: list[tuple], cfg, geometry_points,
                    geo_weight: float, geo_min_overlaps: int) -> dict[int, int]:
    """Replay records through a FRESH engine in end_frame order; return the remap.

    A fresh engine per tick re-derives the authoritative remap over all
    evidence-so-far, firing sticky merges at the moment evidence completes
    (identical to the validated src.eval.online_fusion path).
    """
    from src.reid.micro_batch_fusion import MicroBatchFusion
    engine = MicroBatchFusion(
        interval_frames=cfg.micro_batch_fusion_interval,
        threshold=cfg.micro_batch_fusion_threshold,
        margin=cfg.micro_batch_fusion_margin,
        min_gid_embeddings=cfg.micro_batch_fusion_min_gid_embeddings,
        min_tracklet_detections=cfg.micro_batch_fusion_min_tracklet_detections,
        geo_weight=geo_weight,
        geo_min_overlaps=geo_min_overlaps,
        geometry_points=geometry_points,
    )
    for rec in sorted(records, key=lambda r: r[5]):  # by end_frame
        tid, cam, local, gid, start, end, ndet, nemb, emb = rec
        engine.ingest_tracklet(tid, cam, local, gid, start, end, ndet, nemb, emb)
        engine.step(end)
    if records:
        engine.flush(max(r[5] for r in records))
    return engine.final_remap()


def accumulate_geo(rows: list[dict], frame_numbers, frame_meta_frame_number,
                   frame_count: int, geometry_points: dict, sample_step: int) -> None:
    """Accumulate per-(gid, frame) world foot positions into geometry_points.

    Mirrors offline_merge._load_geometry_points but built live, keyed by the raw
    Global ID and sampled every `sample_step` frames. Mutates geometry_points.
    """
    for row in rows:
        gid = row["gid"]
        foot = row.get("foot_world")
        if gid is None or gid < 0 or foot is None:
            continue
        src = row["src"]
        if frame_numbers is not None:
            frame_no = frame_numbers.get(src, frame_count)
        else:
            frame_no = frame_meta_frame_number
        if frame_no % sample_step != 0:
            continue
        geometry_points.setdefault(gid, {}).setdefault(
            frame_no, []).append((src, float(foot[0]), float(foot[1])))
