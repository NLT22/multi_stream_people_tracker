"""Entry point for the cross-camera ReID pipeline.

Thin orchestration only:
  - src/config/args.py     : CLI parsing
  - src/config/runtime.py  : defaults from YAML + gallery tuning
  - src/pipeline/runner.py : pipeline assembly + run()

  python -m src.main --config configs/pipeline_mmp_10cam_quality.yaml \
      --mmp-short-dataset dataset/MMPTracking_short:lobby_0 --no-display"""

import sys
from pathlib import Path

from src.dataset.mta import MtaDataset
from src.dataset.mmp_tracking import MMPTrackingDataset, MMPTrackingShortDataset
from src.dataset.wildtrack import WildtrackDataset
from src.reid import gallery
from src.config.args import parse_args
from src.pipeline.runner import run


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    # Build the typed ReID/Global-ID config from CLI args.
    reid_config = gallery.configure_from_args(args)
    enforce_unique = (
        reid_config.enforce_unique_global_per_stream
        and not args.allow_duplicate_gid_per_stream
    )
    use_hungarian = (
        reid_config.use_hungarian_assignment and not args.disable_hungarian
    )

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

    run(sources, args.nvinfer_config, args.tracker_config,
        args.tile_w, args.tile_h, args.debug_similarity, use_hungarian,
        enforce_unique, args.save_video, args.record_bitrate, args.no_display,
        batch_size=args.batch_size, gpu_id=args.gpu_id,
        tracker_width=args.tracker_width,
        tracker_height=args.tracker_height,
        tracker_sub_batches=args.tracker_sub_batches,
        max_sources=args.max_sources,
        force_rebuild_engine=args.force_rebuild_engine,
        trim_seconds=args.trim_seconds, trim_start=args.trim_start,
        pretiler=args.pretiler, no_tiler=args.no_tiler,
        show_trajectories=args.show_trajectories,
        trajectory_history=args.trajectory_history,
        trajectory_sample_interval=args.trajectory_sample_interval,
        trajectory_max_segments=args.trajectory_max_segments,
        export_predictions=args.export_predictions,
        disable_gallery=args.disable_gallery,
        osd_enabled=args.osd_enabled,
        gt_by_cam=gt_by_cam,
        gt_snap_frames=gt_snap_frames,
        gt_scale=gt_scale,
        no_sync=args.no_sync,
        loop_video=args.loop_video,
        reid_sgie_config=args.reid_sgie_config,
        geometry=geometry,
        reid_config=reid_config)



if __name__ == "__main__":
    main()
