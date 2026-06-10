"""Dataset/source selection: turn parsed args into a SourcePlan.

Extracted from src/main.py so the entry point is just: parse args -> build
source plan -> build run config -> run. Behavior is unchanged.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from src.dataset.mta import MtaDataset
from src.dataset.mmp_tracking import MMPTrackingDataset, MMPTrackingShortDataset
from src.dataset.wildtrack import WildtrackDataset


@dataclass
class SourcePlan:
    sources: list[str]
    gt_by_cam: dict | None
    gt_snap_frames: int | None
    gt_scale: tuple[float, float]
    geometry: object | None


def build_source_plan(args, reid_config) -> SourcePlan:
    """Resolve sources + optional GT/geometry from the selected dataset flag."""
    sources = args.sources
    gt_by_cam = None
    gt_snap_frames = None   # None = exact frame lookup (MTA); int = snap window (Wildtrack)
    gt_scale = (1.0, 1.0)
    geometry = None

    exclusive = [args.mta_dataset, args.wildtrack_dataset, args.mmp_dataset,
                 args.mmp_short_dataset]
    if sum(bool(x) for x in exclusive) > 1:
        print("[ERROR] --mta-dataset, --wildtrack-dataset, and --mmp-dataset are mutually exclusive.")
        sys.exit(1)

    if args.mta_dataset:
        try:
            _mta_path = Path(args.mta_dataset)
            mta = MtaDataset(str(_mta_path.parent), split=_mta_path.name)
            sources = mta.get_video_uris()
            print(f"[reid] MTA dataset: {args.mta_dataset} → {len(sources)} camera(s)")
            if args.show_gt:
                gt_by_cam = mta.load_all_gt()
                print(f"[reid] Loading GT annotations for {len(gt_by_cam)} camera(s)")
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
            sys.exit(1)
    elif args.wildtrack_dataset:
        try:
            wt = WildtrackDataset(args.wildtrack_dataset)
            sources = wt.get_video_uris()
            print(f"[reid] Wildtrack dataset: {args.wildtrack_dataset} "
                  f"→ {len(sources)} camera(s)")
            if args.show_gt:
                max_sec = (args.wildtrack_minutes * 60.0
                           if args.wildtrack_minutes is not None else None)
                gt_by_cam = wt.load_all_gt(max_seconds=max_sec)
                # Wildtrack: annotations every ~30 video frames; snap to nearest slot
                from src.dataset.wildtrack import FRAMES_PER_ANN
                gt_snap_frames = round(FRAMES_PER_ANN)
                print(f"[reid] Loading Wildtrack GT for {len(gt_by_cam)} camera(s) "
                      f"(annotated: {wt.annotated_duration_seconds:.0f}s)")
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
            sys.exit(1)
    elif args.mmp_dataset:
        try:
            if ":" not in args.mmp_dataset:
                print("[ERROR] --mmp-dataset must be 'ROOT:SCENE', e.g. "
                      "'dataset/MMPTracking:lobby_0'")
                sys.exit(1)
            mmp_root, mmp_scene = args.mmp_dataset.split(":", 1)
            mmp = MMPTrackingDataset(mmp_root, mmp_scene, split=args.mmp_split)
            sources = mmp.get_video_uris()
            print(f"[reid] MMPTracking scene '{mmp_scene}' → {len(sources)} camera(s)")
            if args.show_gt:
                gt_by_cam = {
                    source_id: mmp.load_gt(cam_id)
                    for source_id, cam_id in enumerate(mmp.get_cam_ids())
                }
                print(f"[reid] Loading GT annotations for {len(gt_by_cam)} camera(s)")
        except (FileNotFoundError, ValueError) as e:
            print(f"[ERROR] {e}")
            sys.exit(1)
    elif args.mmp_short_dataset:
        try:
            if ":" not in args.mmp_short_dataset:
                print("[ERROR] --mmp-short-dataset must be 'ROOT:SCENE', e.g. "
                      "'dataset/MMPTracking_short:lobby_0'")
                sys.exit(1)
            short_root, short_scene = args.mmp_short_dataset.split(":", 1)
            mmp_s = MMPTrackingShortDataset(short_root, short_scene)
            sources = mmp_s.get_video_uris()
            print(f"[reid] MMPTracking_short scene '{short_scene}' → {len(sources)} camera(s)")
            if args.show_gt:
                gt_by_cam = {
                    source_id: mmp_s.load_gt(cam_id)
                    for source_id, cam_id in enumerate(mmp_s.get_cam_ids())
                }
                gt_scale = (
                    1920.0 / MMPTrackingShortDataset.IMG_W,
                    1080.0 / MMPTrackingShortDataset.IMG_H,
                )
                print(f"[reid] Loading GT annotations for {len(gt_by_cam)} camera(s)")
            # Load calibration and build GroundPlaneGeometry when available
            if not args.no_calibration:
                try:
                    from src.reid.geometry import GroundPlaneGeometry
                    calib = mmp_s.load_calibration()
                    geometry = GroundPlaneGeometry(calib)
                    n_cams = len(calib.get("Cameras", []))
                    print(f"[reid] Ground-plane geometry loaded: {n_cams} camera(s), "
                          f"geo_weight={reid_config.geo_weight}")
                except FileNotFoundError as cal_err:
                    print(f"[reid] Calibration not found ({cal_err}); "
                          f"running without geometry assistance.")
        except (FileNotFoundError, ValueError) as e:
            print(f"[ERROR] {e}")
            sys.exit(1)
    elif args.show_gt:
        print("[WARNING] --show-gt requires --mta-dataset, --wildtrack-dataset, "
              "--mmp-dataset, or --mmp-short-dataset; ignoring.")

    return SourcePlan(sources=sources, gt_by_cam=gt_by_cam,
                      gt_snap_frames=gt_snap_frames, gt_scale=gt_scale,
                      geometry=geometry)
