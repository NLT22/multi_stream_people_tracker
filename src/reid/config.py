"""Typed configuration for the cross-camera ReID gallery + micro-batch fusion.

Replaces the module-level tuning globals that used to live in gallery.py. Build
one with `ReIDConfig.from_args(args)` and pass it into CrossCameraGalleryProbe.
Field defaults are the production tuning values (see tests/test_config.py, which
pins them so they cannot silently drift).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ReIDConfig:
    # --- 1. Embedding quality gate ------------------------------------------
    enable_embedding_quality_gate: bool = True
    reid_edge_margin_ratio: float = 0.005
    reid_min_bbox_height_ratio: float = 0.04
    reid_min_bbox_area_ratio: float = 0.0008
    reid_min_bbox_aspect_ratio: float = 0.12
    reid_max_bbox_aspect_ratio: float = 0.95
    reid_max_overlap_iou_for_update: float = 0.25

    # --- 2. Tracklet (local per-camera memory) ------------------------------
    use_tracklet_embedding: bool = True
    tracklet_embedding_interval: int = 5
    tracklet_max_embeddings: int = 8
    tracklet_min_embeddings_for_match: int = 3
    tracklet_max_age: int = 1800

    # --- 3. Gallery (global identity store) ---------------------------------
    gallery_max_age: int = 1800
    gallery_max_prototypes: int = 24
    prototype_add_threshold: float = 0.72

    # --- 4. Match threshold -------------------------------------------------
    similarity_threshold: float = 0.68

    # --- 5. Assignment ------------------------------------------------------
    use_hungarian_assignment: bool = True
    global_assignment_max_candidates: int = 80
    enforce_unique_global_per_stream: bool = True

    # --- 6. ID stickiness & ambiguity guard ---------------------------------
    enable_id_stickiness: bool = True
    id_switch_margin: float = 0.12
    enable_ambiguous_match_rejection: bool = True
    match_ambiguity_margin: float = 0.06

    # --- 7. Global-ID merge (online duplicate resolution) -------------------
    enable_global_id_merge: bool = True
    global_id_merge_threshold: float = 0.76
    global_id_merge_min_tracklet_embeddings: int = 6
    global_id_merge_margin: float = 0.04
    global_id_merge_interval: int = 5
    global_id_merge_max_candidates: int = 80

    # --- 8. Debug -----------------------------------------------------------
    debug_top_k: int = 3

    # --- 9. Ground-plane geometry -------------------------------------------
    geo_weight: float = 0.35
    geo_assignment_mode: str = "weight_only"   # weight_only | close_reid_only
    geo_reid_margin: float = 1.0

    # --- 10. Micro-batch cross-camera fusion --------------------------------
    use_micro_batch_fusion: bool = False
    micro_batch_fusion_interval: int = 125
    micro_batch_fusion_threshold: float = 0.55
    micro_batch_fusion_margin: float = 0.02
    micro_batch_fusion_min_gid_embeddings: int = 4
    micro_batch_fusion_min_tracklet_detections: int = 6
    micro_batch_fusion_geo_weight: float = 0.25
    micro_batch_fusion_geo_min_overlaps: int = 8
    micro_batch_fusion_geo_sample_step: int = 5

    # ----------------------------------------------------------------- builder
    @classmethod
    def from_args(cls, args) -> "ReIDConfig":
        """Build a config from parsed CLI args (mirrors the old configure_from_args)."""
        c = cls()
        c.similarity_threshold = max(0.0, args.similarity_threshold)
        c.gallery_max_age = max(1, args.gallery_max_age)
        c.global_assignment_max_candidates = max(1, args.assignment_max_candidates)
        c.enable_id_stickiness = not args.disable_id_stickiness
        c.id_switch_margin = max(0.0, args.id_switch_margin)
        c.enable_ambiguous_match_rejection = not args.allow_ambiguous_match
        c.match_ambiguity_margin = max(0.0, args.match_ambiguity_margin)
        c.enable_global_id_merge = not args.disable_global_merge
        c.global_id_merge_threshold = max(0.0, args.global_merge_threshold)
        c.global_id_merge_min_tracklet_embeddings = max(1, args.global_merge_min_embeddings)
        c.global_id_merge_margin = max(0.0, args.global_merge_margin)
        c.global_id_merge_interval = max(1, args.global_merge_interval)
        c.global_id_merge_max_candidates = max(1, args.global_merge_max_candidates)
        c.use_tracklet_embedding = not args.disable_tracklet
        c.tracklet_embedding_interval = max(1, args.tracklet_embedding_interval)
        c.enable_embedding_quality_gate = not args.disable_embedding_quality_gate
        c.tracklet_max_embeddings = max(1, args.tracklet_window)
        c.tracklet_min_embeddings_for_match = max(1, args.tracklet_min_embeddings)
        c.tracklet_max_age = max(1, args.tracklet_max_age)
        _gw = getattr(args, "geo_weight", None)
        if _gw is not None:
            c.geo_weight = max(0.0, min(1.0, float(_gw)))
        _mode = getattr(args, "geometry_assignment_mode", c.geo_assignment_mode)
        if _mode in {"weight_only", "close_reid_only"}:
            c.geo_assignment_mode = _mode
        _margin = getattr(args, "geometry_reid_margin", None)
        if _margin is not None:
            c.geo_reid_margin = max(0.0, float(_margin))
        c.use_micro_batch_fusion = bool(getattr(args, "micro_batch_fusion", False))
        _fi = getattr(args, "fusion_interval", None)
        if _fi is not None:
            c.micro_batch_fusion_interval = max(1, int(_fi))
        _ft = getattr(args, "fusion_threshold", None)
        if _ft is not None:
            c.micro_batch_fusion_threshold = max(0.0, float(_ft))
        return c

    def use_prototypes(self) -> bool:
        return self.gallery_max_prototypes > 0

    def summary(self) -> str:
        """Multi-line summary for startup logs (same lines as the old config_summary)."""
        return "\n".join([
            f"[reid] Re-ID similarity threshold={self.similarity_threshold}",
            f"[reid] gallery_max_age={self.gallery_max_age}",
            f"[reid] assignment_max_candidates={self.global_assignment_max_candidates}",
            (f"[reid] id_stickiness={self.enable_id_stickiness} "
             f"switch_margin={self.id_switch_margin} "
             f"ambiguous_match_rejection={self.enable_ambiguous_match_rejection} "
             f"ambiguity_margin={self.match_ambiguity_margin}"),
            (f"[reid] global_id_merge={self.enable_global_id_merge} "
             f"threshold={self.global_id_merge_threshold} "
             f"min_tracklet_embeddings={self.global_id_merge_min_tracklet_embeddings} "
             f"margin={self.global_id_merge_margin} "
             f"interval={self.global_id_merge_interval} "
             f"max_candidates={self.global_id_merge_max_candidates}"),
            (f"[reid] tracklet_embedding={self.use_tracklet_embedding} "
             f"window={self.tracklet_max_embeddings} "
             f"min_embeddings={self.tracklet_min_embeddings_for_match} "
             f"sample_interval={self.tracklet_embedding_interval} "
             f"max_age={self.tracklet_max_age}"),
            (f"[reid] embedding_quality_gate={self.enable_embedding_quality_gate} "
             f"edge_margin={self.reid_edge_margin_ratio} "
             f"max_overlap_iou={self.reid_max_overlap_iou_for_update}"),
            (f"[reid] geo_weight={self.geo_weight} "
             f"assignment_mode={self.geo_assignment_mode} "
             f"reid_margin={self.geo_reid_margin}"),
            (f"[reid] micro_batch_fusion={self.use_micro_batch_fusion} "
             f"interval={self.micro_batch_fusion_interval} "
             f"threshold={self.micro_batch_fusion_threshold}"),
        ])
