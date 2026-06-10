"""Entry point for the cross-camera ReID pipeline.

Thin orchestration only:
  - src/config/args.py     : CLI parsing
  - src/config/runtime.py  : defaults from YAML + gallery tuning
  - src/pipeline/runner.py : pipeline assembly + run()

  python -m src.main --config configs/pipelines/pipeline_mmp_10cam_quality.yaml \
      --mmp-short-dataset dataset/MMPTracking_short:lobby_0 --no-display"""

from src.reid import gallery
from src.config.args import parse_args
from src.pipeline.runner import run
from src.pipeline.run_config import PipelineRunConfig
from src.pipeline.source_plan import build_source_plan


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

    plan = build_source_plan(args, reid_config)
    sources = plan.sources
    gt_by_cam = plan.gt_by_cam
    gt_snap_frames = plan.gt_snap_frames
    gt_scale = plan.gt_scale
    geometry = plan.geometry

    run(PipelineRunConfig(
        sources=sources,
        nvinfer_config=args.nvinfer_config,
        tracker_config=args.tracker_config,
        tile_w=args.tile_w,
        tile_h=args.tile_h,
        debug_similarity=args.debug_similarity,
        use_hungarian_assignment=use_hungarian,
        enforce_unique_per_stream=enforce_unique,
        save_video=args.save_video,
        record_bitrate=args.record_bitrate,
        no_display=args.no_display,
        batch_size=args.batch_size,
        gpu_id=args.gpu_id,
        tracker_width=args.tracker_width,
        tracker_height=args.tracker_height,
        tracker_sub_batches=args.tracker_sub_batches,
        max_sources=args.max_sources,
        force_rebuild_engine=args.force_rebuild_engine,
        trim_seconds=args.trim_seconds,
        trim_start=args.trim_start,
        pretiler=args.pretiler,
        no_tiler=args.no_tiler,
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
        reid_config=reid_config,
    ))



if __name__ == "__main__":
    main()
