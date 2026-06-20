"""Pipeline assembly + run orchestration (the production builder).

`run(PipelineRunConfig)` builds the GStreamer/pyservicemaker graph
(sources → mux → nvinfer → nvtracker → probes → tiler → osd → sink) and runs it."""

import sys
from pathlib import Path

import pyservicemaker as psm

from src.pipeline.model_utils import (
    deepstream_tracker_lib_path,
    infer_person_class_id,
)
from src.pipeline.engine_prep import prepare_nvinfer_config
from src.pipeline.recording import add_recording_branch, compute_grid
from src.pipeline.run_config import PipelineRunConfig
from src.eval.export import PredictionExporter
from src.eval.gt_overlay import GtOverlayProbe
from src.pipeline.sources import resolve_sources, trim_sources
from src.reid import gallery
from src.reid.visualization import TrajectoryVisualizer
from src.utils.platform_utils import get_sink_element


DEFAULT_CONFIG_PATH = "configs/pipelines/pipeline_mmp_nvdcf_online_sgie.yaml"


def run(config: PipelineRunConfig):
    # Unpack into locals so the assembly body below is unchanged. Field names
    # match the old positional parameters one-for-one.
    sources = config.sources
    nvinfer_config = config.nvinfer_config
    tracker_config = config.tracker_config
    tile_w = config.tile_w
    tile_h = config.tile_h
    debug_similarity = config.debug_similarity
    use_hungarian_assignment = config.use_hungarian_assignment
    enforce_unique_per_stream = config.enforce_unique_per_stream
    save_video = config.save_video
    record_bitrate = config.record_bitrate
    no_display = config.no_display
    batch_size = config.batch_size
    gpu_id = config.gpu_id
    tracker_width = config.tracker_width
    tracker_height = config.tracker_height
    tracker_sub_batches = config.tracker_sub_batches
    max_sources = config.max_sources
    force_rebuild_engine = config.force_rebuild_engine
    trim_seconds = config.trim_seconds
    trim_start = config.trim_start
    pretiler = config.pretiler
    no_tiler = config.no_tiler
    show_trajectories = config.show_trajectories
    trajectory_history = config.trajectory_history
    trajectory_sample_interval = config.trajectory_sample_interval
    trajectory_max_segments = config.trajectory_max_segments
    export_predictions = config.export_predictions
    disable_gallery = config.disable_gallery
    osd_enabled = config.osd_enabled
    gt_by_cam = config.gt_by_cam
    gt_snap_frames = config.gt_snap_frames
    gt_scale = config.gt_scale
    no_sync = config.no_sync
    loop_video = config.loop_video
    reid_sgie_config = config.reid_sgie_config
    nvdsanalytics_config = config.nvdsanalytics_config
    geometry = config.geometry
    reid_config = config.reid_config

    if reid_config is None:
        from src.reid.config import ReIDConfig
        reid_config = ReIDConfig()
    # pretiler=True guarantees exact source_id and frame_number — no geometric
    # guessing from tile coordinates.  Force it whenever:
    #   - --no-tiler: tiler absent, only pre-tiler position makes sense
    # NOTE: --export-predictions and --show-gt do NOT force pretiler anymore.
    #   SourceIdCollectorProbe (pre-tiler) already fills frame_numbers exactly,
    #   and CrossCameraGalleryProbe (post-tiler) uses that dict for the exporter.
    #   Forcing pretiler breaks ReID because NvDeepSORT's ReID tensor is only
    #   accessible via obj_reid_items post-tracker, which the two-probe path
    #   handles correctly through SourceIdCollectorProbe.
    pretiler = pretiler or no_tiler
    try:
        uris, is_live = resolve_sources(sources)
    except (FileNotFoundError, ValueError, NotADirectoryError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    if max_sources is not None:
        if max_sources < 1:
            print("[ERROR] --max-sources must be >= 1")
            sys.exit(1)
        original_count = len(uris)
        uris = uris[:max_sources]
        is_live = any(u.startswith("rtsp://") for u in uris)
        if len(uris) < original_count:
            print(
                f"[reid] max_sources={max_sources}: using first "
                f"{len(uris)}/{original_count} resolved source(s)"
            )

    # Optionally pre-cut each file source to a fixed length before the pipeline.
    if trim_seconds:
        uris = trim_sources(uris, trim_seconds, trim_start)

    n = len(uris)
    # nvstreammux feeds one frame per source per batch, so the inference batch
    # must be at least the stream count. Default to exactly n.
    batch = max(n, batch_size) if batch_size else n
    rows, cols = compute_grid(n)
    person_class_id = infer_person_class_id(nvinfer_config)

    # Prepare a runtime nvinfer config + engine matching this batch.
    runtime_nvinfer_config = prepare_nvinfer_config(
        nvinfer_config, batch, gpu_id, force_rebuild_engine)
    total_w, total_h = tile_w * cols, tile_h * rows
    print(f"[reid] {n} stream(s) → {rows}×{cols} grid  canvas={total_w}×{total_h}")
    print(f"[reid] batch_size={batch} gpu_id={gpu_id} live_source={is_live}")
    print(f"[reid] nvinfer runtime config={runtime_nvinfer_config}")
    tracker_extra = (
        f" sub_batches={tracker_sub_batches}" if tracker_sub_batches else "")
    print(
        f"[reid] tracker={tracker_config} "
        f"({tracker_width}x{tracker_height}{tracker_extra})"
    )
    print(f"[reid] person_class_id={person_class_id} inferred from {nvinfer_config}")
    print(f"[reid] debug_similarity={debug_similarity}")
    print(f"[reid] use_hungarian_assignment={use_hungarian_assignment}")
    print(f"[reid] enforce_unique_per_stream={enforce_unique_per_stream}")
    print(f"[reid] show_trajectories={show_trajectories}")
    print(f"[reid] gallery_enabled={not disable_gallery} osd_enabled={osd_enabled}")
    print(gallery.config_summary(reid_config))
    if save_video:
        print(f"[reid] save_video={save_video}")
    if export_predictions:
        print(f"[reid] export_predictions={export_predictions}")
    if disable_gallery and export_predictions:
        print("[reid] WARNING: --disable-gallery ignores --export-predictions.")

    id_map: dict[int, int] = {}
    embeddings: dict[tuple, list] = {}  # (source_id, object_id) → embedding vector

    # With micro-batch fusion on, the export represents the CONVERGED
    # authoritative Global IDs: rows are buffered and the final cross-camera
    # remap is applied at close (analytics-export semantics — the same result a
    # post-pass would produce). The live OSD still shows near-realtime
    # provisional IDs. delay_frames is set effectively unbounded so a merge
    # decided late still corrects all of a track's earlier frames.
    #   NOTE: buffers all rows in memory until close — fine for evaluation
    #   clips; for very long / 20-cam production streams prefer the bounded
    #   `src.eval.online_fusion` post-pass instead.
    export_delay_frames = (
        10 ** 9 if reid_config.use_micro_batch_fusion else 0
    )
    exporter = (
        PredictionExporter(export_predictions, delay_frames=export_delay_frames,
                           emb_flush_frames=config.live_buffered_window)
        if export_predictions and not disable_gallery else None
    )

    pipeline = psm.Pipeline("reid-pipeline")

    mux_props = {
        "batch-size": batch, "batched-push-timeout": 40000,
        "width": 1920, "height": 1080, "gpu-id": gpu_id,
    }
    if is_live:
        mux_props["live-source"] = 1
    pipeline.add("nvstreammux", "mux", mux_props)
    for i, uri in enumerate(uris):
        name = f"source_{i}"
        src_props = {"uri": uri, "gpu-id": gpu_id}
        if is_live:
            src_props["live-source"] = 1
        if loop_video and not is_live:
            src_props["file-loop"] = 1
        pipeline.add("nvurisrcbin", name, src_props)
        pipeline.link((name, "mux"), ("", "sink_%u"))

    pipeline.add("nvinfer", "pgie", {
        "config-file-path": runtime_nvinfer_config,
        "batch-size": batch, "gpu-id": gpu_id,
    })
    pipeline.attach("pgie", "measure_fps_probe", "fps_probe")

    tracker_props = {
        "ll-lib-file": deepstream_tracker_lib_path(),
        "ll-config-file": tracker_config,
        "tracker-width": tracker_width, "tracker-height": tracker_height,
        "gpu-id": gpu_id,
    }
    if tracker_sub_batches:
        tracker_props["sub-batches"] = tracker_sub_batches
    pipeline.add("nvtracker", "tracker", tracker_props)

    # Optional decoupled ReID: a secondary nvinfer (SGIE) extracts a per-person
    # embedding and attaches it as output-tensor-meta, keeping the tracker on the
    # reidType:0 path. NOTE: this was the workaround for the *old* throughput
    # bottleneck — the VPI visual tracker (visualTrackerType:2 ~ 5 FPS/cam). With
    # Legacy DCF (visualTrackerType:1) the in-tracker ReID (reidType:2) config
    # runs ~350 frames/s aggregate (≈17 FPS/cam at 20 cams), so the decoupled
    # SGIE is no longer required for 20cam@10FPS. Kept for flexibility.
    # `reid_src_element` is the element whose objects carry ReID embeddings.
    reid_src_element = "tracker"
    if reid_sgie_config:
        import yaml as _yaml
        _sgie_raw = _yaml.safe_load(Path(reid_sgie_config).read_text()) or {}
        _sgie_batch = int(_sgie_raw.get("property", {}).get("batch-size", 64))
        runtime_sgie_config = prepare_nvinfer_config(
            reid_sgie_config, _sgie_batch, gpu_id, force_rebuild_engine)
        pipeline.add("nvinfer", "sgie_reid", {
            "config-file-path": runtime_sgie_config,
            "batch-size": _sgie_batch,
            "gpu-id": gpu_id,
            "process-mode": 2,
        })
        reid_src_element = "sgie_reid"
        print(f"[reid] decoupled ReID SGIE enabled "
              f"(batch={_sgie_batch}): {runtime_sgie_config}")

    # Optional gst-nvdsanalytics: ROI occupancy / line-crossing / overcrowding
    # on the tracked objects. Counts attach as frame user meta (read by
    # AnalyticsProbe) and, with osd-mode=2, are drawn on the video.
    analytics_probe = None
    if nvdsanalytics_config:
        from src.pipeline.analytics import AnalyticsProbe
        pipeline.add("nvdsanalytics", "analytics",
                     {"config-file": nvdsanalytics_config, "enable": 1})
        analytics_probe = AnalyticsProbe(
            print_interval=60,
            export_path=(f"{export_predictions}/analytics.csv"
                         if export_predictions else None),
        )
        pipeline.attach("analytics", psm.Probe("analytics_probe", analytics_probe))
        print(f"[reid] nvdsanalytics enabled: {nvdsanalytics_config}")

    trajectory_visualizer = None
    if show_trajectories:
        trajectory_visualizer = TrajectoryVisualizer(
            tile_w, tile_h, cols, n,
            max_points=trajectory_history,
            sample_interval=trajectory_sample_interval,
            max_segments_per_track=trajectory_max_segments,
            pretiler=pretiler,
        )

    # frame_numbers/frame_sizes: shared dicts filled by
    # SourceIdCollectorProbe (pre-tiler) and read by CrossCameraGalleryProbe
    # (post-tiler) so the exporter records the correct per-source frame index
    # and source-space bbox coordinates.
    frame_numbers: dict = {}
    frame_sizes: dict = {}

    gallery_probe = None
    if disable_gallery:
        print("[reid] gallery disabled: tracker-only realtime path")
    else:
        gallery_probe = gallery.CrossCameraGalleryProbe(
            id_map, embeddings, person_class_id, tile_w, tile_h, cols, n,
            debug_similarity=debug_similarity,
            use_hungarian_assignment=use_hungarian_assignment,
            enforce_unique_per_stream=enforce_unique_per_stream,
            pretiler=pretiler,
            extract_embeddings=pretiler,
            trajectory_visualizer=trajectory_visualizer,
            exporter=exporter,
            frame_numbers=frame_numbers if not pretiler else None,
            frame_sizes=frame_sizes if not pretiler else None,
            geometry=geometry,
            config=reid_config)

        if pretiler:
            # One pre-tiler probe on the tracker: exact source_id (no geometric
            # guessing), extracts embeddings + matches + sets labels in one pass.
            print(f"[reid] pretiler mode: gallery runs on {reid_src_element} "
                  f"(no src guessing)")
            pipeline.attach(reid_src_element,
                            psm.Probe("reid_probe", gallery_probe))
        else:
            # Two-probe path: SourceIdCollectorProbe fills id_map pre-tiler
            # (source_id exact), CrossCameraGalleryProbe reads id_map post-tiler.
            # source_id is resolved from id_map — no geometric tile guessing.
            print("[reid] two-probe mode: source_id via id_map (pre-tiler exact)")
            pipeline.attach(reid_src_element, psm.Probe(
                "src_collector",
                gallery.SourceIdCollectorProbe(
                    id_map, embeddings, person_class_id, debug=debug_similarity,
                    frame_numbers=frame_numbers, frame_sizes=frame_sizes),
            ))

    if gt_by_cam:
        pipeline.attach("tracker", psm.Probe(
            "gt_overlay", GtOverlayProbe(
                gt_by_cam,
                snap_frames=gt_snap_frames,
                scale_x=gt_scale[0],
                scale_y=gt_scale[1],
            )))
        print(f"[reid] GT overlay enabled for {len(gt_by_cam)} camera(s) "
              f"(green boxes = ground truth)")

    if not no_tiler:
        pipeline.add("nvmultistreamtiler", "tiler", {
            "rows": rows, "columns": cols,
            "width": total_w, "height": total_h, "gpu-id": gpu_id,
        })
        if not pretiler and gallery_probe is not None:
            pipeline.attach("tiler", psm.Probe("reid_probe", gallery_probe))

    pipeline.link("mux", "pgie")
    pipeline.link("pgie", "tracker")
    # Insert the ReID SGIE into the data path (tracker -> sgie_reid -> ...) so it
    # runs on tracked objects and downstream elements see the embedding tensors.
    tracker_tail = "tracker"
    if reid_src_element == "sgie_reid":
        pipeline.link("tracker", "sgie_reid")
        tracker_tail = "sgie_reid"
    # nvdsanalytics runs on tracked objects, before the tiler.
    if analytics_probe is not None:
        pipeline.link(tracker_tail, "analytics")
        tracker_tail = "analytics"
    visual_tail = tracker_tail
    if no_tiler:
        # Headless throughput: skip the tiler entirely.
        visual_tail = tracker_tail
    else:
        pipeline.link(tracker_tail, "tiler")
        visual_tail = "tiler"

    if osd_enabled:
        pipeline.add("nvosdbin", "osd", {
            "gpu-id": gpu_id,
            "process-mode": 1,
            "display-text": 1,
            "display-bbox": 1,
            "text-size": 18,
        })
        pipeline.link(visual_tail, "osd")
        visual_tail = "osd"

    # sync=0: render as-fast-as-possible (no timestamp throttling).
    # Use for RTSP, high-fps sources (MTA=41fps), or slow GPUs.
    sink_sync = 0 if (is_live or no_sync) else 1

    if save_video and not no_display:
        pipeline.add("tee", "output_tee")
        # leaky display queue: if the encoder branch stalls, the live view keeps
        # moving instead of the whole tee dead-locking.
        pipeline.add("queue", "display_queue",
                     {"leaky": 2, "max-size-buffers": 5})
        pipeline.add(get_sink_element(), "sink",
                     {"sync": sink_sync, "qos": 0, "async": 0})
        pipeline.link(visual_tail, "output_tee", "display_queue", "sink")
        written_path = add_recording_branch(
            pipeline, "output_tee", save_video, record_bitrate,
            canvas_w=total_w, canvas_h=total_h, gpu_id=gpu_id)
    elif save_video:
        written_path = add_recording_branch(
            pipeline, visual_tail, save_video, record_bitrate,
            canvas_w=total_w, canvas_h=total_h, gpu_id=gpu_id)
    elif no_display:
        # Headless: drop frames as fast as possible, no window opened.
        pipeline.add("fakesink", "sink", {"sync": 0, "async": 0})
        pipeline.link(visual_tail, "sink")
    else:
        pipeline.add(get_sink_element(), "sink", {"sync": sink_sync, "qos": 0})
        pipeline.link(visual_tail, "sink")

    try:
        pipeline.start()
        print("[reid] Running. Gallery stats print every 60 frames.")
        print("[reid] Labels show GID:<global_id>; bbox color follows GID.")
        if save_video:
            print(f"[reid] Recording annotated video to: {written_path}")
        print("[reid] Press Ctrl+C to stop.")
        pipeline.wait()
    except KeyboardInterrupt:
        print("\n[reid] Stopped.")
        if gallery_probe is not None:
            total_gids = gallery_probe._next_gid - 1
            print(f"[reid] Total unique global IDs assigned: {total_gids}")
    finally:
        pipeline.stop()
        if exporter is not None:
            exporter.close()
            print(f"[reid] Predictions exported to: {export_predictions}")
        if analytics_probe is not None:
            analytics_probe.close()

