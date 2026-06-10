"""Dataset/source selection: turn parsed args into a SourcePlan.

Extracted from src/main.py so the entry point is just: parse args -> build
source plan -> build run config -> run. Behavior is unchanged.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from src.dataset.mmp_tracking import MMPTrackingDataset, MMPTrackingShortDataset


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
    gt_snap_frames = None
    gt_scale = (1.0, 1.0)
    geometry = None

    if args.mmp_dataset and args.mmp_short_dataset:
        print("[ERROR] --mmp-dataset and --mmp-short-dataset are mutually exclusive.")
        sys.exit(1)

    if args.mmp_dataset:
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
        print("[WARNING] --show-gt requires --mmp-dataset or "
              "--mmp-short-dataset; ignoring.")

    return SourcePlan(sources=sources, gt_by_cam=gt_by_cam,
                      gt_snap_frames=gt_snap_frames, gt_scale=gt_scale,
                      geometry=geometry)
