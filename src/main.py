"""
=============================================================================
Cross-Camera Person Re-Identification Pipeline (flexible stream count)
=============================================================================

Links the same physical person across cameras with one stable Global ID, even
though the per-camera tracker assigns a different local ID in each stream:

                 cam0 ──→ track_id=42  ─┐
                                         ├─ embedding match → GID:1
                 cam1 ──→ track_id=7   ─┘

TWO IDENTITY LAYERS:
  1. NvDeepSORT tracker (default: configs/tracker/nvdeepsort_reid_swin.yaml)
     Uses a Swin-Tiny ReID model for local association and exports ReID tensors
     for the cross-camera gallery. NvDCF accuracy remains useful for A/B tests
     when local bbox overlap is the main issue.
  2. CrossCameraGalleryProbe (src/reid/gallery.py)
     Reads tracker ReID embeddings and matches each (cam, local_id) against a
     cross-camera gallery to assign a Global ID. Stabilized with:
       - tracklet embedding averaging (noisy frame-level vectors)
       - per-identity prototypes (different camera views)
       - Hungarian one-to-one assignment per stream
       - ID stickiness + ambiguity rejection (stop label bouncing)
       - online Global ID merge (repair cross-view splits)
       - bounded candidate search (keep long videos responsive)

PIPELINE TOPOLOGY:
  [src_0..N] → [mux] → [nvinfer] → [nvtracker/DeepSORT+Swin] → [tiler]
                                          │                    │
                                 [SourceIdCollectorProbe] [CrossCameraGalleryProbe]
                                  (pre-tiler: source_id)  (post-tiler: draw labels)
                                                                ↓
                                                          [nvosdbin] → [sink]

FLEXIBLE INPUT + DYNAMIC ENGINE:
  Runs with ANY number of streams. The detector batch size defaults to the
  stream count; a runtime nvinfer config is generated next to the original and
  pointed at the matching per-batch TensorRT engine. Stale engines (older than
  their ONNX) are cleaned; engines for other batch sizes are kept as a cache.
  The ReID tracker engine is independent of stream count and left untouched.

  Inputs accepted by --sources: a .txt list file, a folder of videos, or one /
  more video paths or URIs (rtsp:// auto-enables live mode).

RUN:
  python -m src.main
  python -m src.main --sources configs/sources/video_files.txt
  python -m src.main --sources dataset/mtmc_12cam/videos
  python -m src.main --sources cam0.mp4 cam1.mp4 rtsp://host/stream
  python -m src.main --sources dataset/mtmc_4cam/videos --debug-similarity
  python -m src.main --sources <dir> --force-rebuild-engine
=============================================================================
"""

import argparse
import sys
from pathlib import Path

import pyservicemaker as psm
import yaml

from src.pipeline.model_utils import (
    deepstream_tracker_lib_path,
    infer_person_class_id,
)
from src.pipeline.engine_prep import prepare_nvinfer_config
from src.pipeline.recording import add_recording_branch, compute_grid
from src.dataset.mta import MtaDataset
from src.dataset.mmp_tracking import MMPTrackingDataset, MMPTrackingShortDataset
from src.dataset.wildtrack import WildtrackDataset
from src.eval.export import PredictionExporter
from src.eval.gt_overlay import GtOverlayProbe
from src.pipeline.sources import resolve_sources, trim_sources
from src.reid import gallery
from src.reid.visualization import TrajectoryVisualizer
from src.utils.platform_utils import get_sink_element


DEFAULT_CONFIG_PATH = "configs/pipeline.yaml"


def _load_defaults(config_path: str) -> dict:
    """Read pipeline.yaml and turn it into CLI defaults for this app."""
    defaults = {
        "sources": ["configs/sources/video_files.txt"],
        "nvinfer_config": "configs/models/nvinfer_yolov11_people.yml",
        "tracker_config": "configs/tracker/nvdeepsort_reid_swin.yaml",
        "tile_w": 1280,
        "tile_h": 720,
        "batch_size": None,
        "max_sources": None,
        "gpu_id": 0,
        "pretiler": False,
        "no_tiler": False,
        "save_video": None,
        "record_bitrate": 8000000,
        "no_display": False,
        "show_trajectories": True,
        "trajectory_history": 96,
        "trajectory_sample_interval": 20,
        "trajectory_max_segments": 24,
        # ReID/gallery tuning — mirrors gallery.py module defaults
        "similarity_threshold":              gallery.SIMILARITY_THRESHOLD,
        "gallery_max_age":                   gallery.GALLERY_MAX_AGE,
        "assignment_max_candidates":         gallery.GLOBAL_ASSIGNMENT_MAX_CANDIDATES,
        "disable_id_stickiness":             not gallery.ENABLE_ID_STICKINESS,
        "id_switch_margin":                  gallery.ID_SWITCH_MARGIN,
        "allow_ambiguous_match":             not gallery.ENABLE_AMBIGUOUS_MATCH_REJECTION,
        "match_ambiguity_margin":            gallery.MATCH_AMBIGUITY_MARGIN,
        "disable_global_merge":              not gallery.ENABLE_GLOBAL_ID_MERGE,
        "global_merge_threshold":            gallery.GLOBAL_ID_MERGE_THRESHOLD,
        "global_merge_min_embeddings":       gallery.GLOBAL_ID_MERGE_MIN_TRACKLET_EMBEDDINGS,
        "global_merge_margin":              gallery.GLOBAL_ID_MERGE_MARGIN,
        "global_merge_interval":             gallery.GLOBAL_ID_MERGE_INTERVAL,
        "global_merge_max_candidates":       gallery.GLOBAL_ID_MERGE_MAX_CANDIDATES,
        "disable_tracklet":                  not gallery.USE_TRACKLET_EMBEDDING,
        "tracklet_embedding_interval":       gallery.TRACKLET_EMBEDDING_INTERVAL,
        "disable_embedding_quality_gate":    not gallery.ENABLE_EMBEDDING_QUALITY_GATE,
        "tracklet_window":                   gallery.TRACKLET_MAX_EMBEDDINGS,
        "tracklet_min_embeddings":           gallery.TRACKLET_MIN_EMBEDDINGS_FOR_MATCH,
        "tracklet_max_age":                  gallery.TRACKLET_MAX_AGE,
    }

    path = Path(config_path)
    if not path.exists():
        return defaults

    raw = yaml.safe_load(path.read_text()) or {}
    source_configs = {
        "video_files": "configs/sources/video_files.txt",
        "folder_input": "configs/sources/folder_input.yaml",
        "rtsp_cameras": "configs/sources/rtsp_cameras.txt",
    }
    source_configs.update(raw.get("source_configs", {}) or {})
    source_mode = raw.get("source_mode", "video_files")
    defaults["sources"] = [source_configs.get(source_mode, defaults["sources"][0])]

    detection = raw.get("detection", {}) or {}
    tracker = raw.get("tracker", {}) or {}
    display = raw.get("display", {}) or {}
    runtime = raw.get("runtime", {}) or {}

    defaults["nvinfer_config"] = detection.get(
        "config_file", defaults["nvinfer_config"])
    defaults["tracker_config"] = tracker.get(
        "config_file", defaults["tracker_config"])
    defaults["tile_w"] = display.get("tile_width", defaults["tile_w"])
    defaults["tile_h"] = display.get("tile_height", defaults["tile_h"])
    defaults["batch_size"] = raw.get("batch_size", defaults["batch_size"])
    defaults["max_sources"] = runtime.get(
        "max_sources", defaults["max_sources"])
    defaults["gpu_id"] = raw.get("gpu_id", defaults["gpu_id"])
    defaults["pretiler"] = runtime.get("pretiler", defaults["pretiler"])
    defaults["no_tiler"] = runtime.get("no_tiler", defaults["no_tiler"])
    defaults["save_video"] = runtime.get("save_video", defaults["save_video"])
    defaults["record_bitrate"] = runtime.get(
        "record_bitrate", defaults["record_bitrate"])
    defaults["no_display"] = runtime.get("no_display", defaults["no_display"])
    defaults["show_trajectories"] = runtime.get(
        "show_trajectories", defaults["show_trajectories"])
    defaults["trajectory_history"] = runtime.get(
        "trajectory_history", defaults["trajectory_history"])
    defaults["trajectory_sample_interval"] = runtime.get(
        "trajectory_sample_interval", defaults["trajectory_sample_interval"])
    defaults["trajectory_max_segments"] = runtime.get(
        "trajectory_max_segments", defaults["trajectory_max_segments"])

    # reid: section — all keys map directly to their defaults dict counterpart
    reid = raw.get("reid", {}) or {}
    _bool = lambda v: bool(v)
    for key, yaml_key in [
        ("similarity_threshold",           "similarity_threshold"),
        ("gallery_max_age",                "gallery_max_age"),
        ("assignment_max_candidates",      "assignment_max_candidates"),
        ("id_switch_margin",               "id_switch_margin"),
        ("match_ambiguity_margin",         "match_ambiguity_margin"),
        ("global_merge_threshold",         "global_merge_threshold"),
        ("global_merge_min_embeddings",    "global_merge_min_embeddings"),
        ("global_merge_margin",            "global_merge_margin"),
        ("global_merge_interval",          "global_merge_interval"),
        ("global_merge_max_candidates",    "global_merge_max_candidates"),
        ("tracklet_embedding_interval",    "tracklet_embedding_interval"),
        ("tracklet_window",                "tracklet_window"),
        ("tracklet_min_embeddings",        "tracklet_min_embeddings"),
        ("tracklet_max_age",               "tracklet_max_age"),
    ]:
        if yaml_key in reid:
            defaults[key] = reid[yaml_key]
    # Boolean toggles stored as positive flags in YAML for readability
    if "id_stickiness" in reid:
        defaults["disable_id_stickiness"] = not reid["id_stickiness"]
    if "ambiguous_match_rejection" in reid:
        defaults["allow_ambiguous_match"] = not reid["ambiguous_match_rejection"]
    if "global_merge" in reid:
        defaults["disable_global_merge"] = not reid["global_merge"]
    if "tracklet_embedding" in reid:
        defaults["disable_tracklet"] = not reid["tracklet_embedding"]
    if "embedding_quality_gate" in reid:
        defaults["disable_embedding_quality_gate"] = not reid["embedding_quality_gate"]

    return defaults


def run(sources: list[str], nvinfer_config: str, tracker_config: str,
        tile_w: int, tile_h: int, debug_similarity: bool,
        use_hungarian_assignment: bool, enforce_unique_per_stream: bool,
        save_video: str | None, record_bitrate: int, no_display: bool,
        batch_size: int | None = None, gpu_id: int = 0,
        max_sources: int | None = None,
        force_rebuild_engine: bool = False,
        trim_seconds: float | None = None, trim_start: float = 0.0,
        pretiler: bool = False, no_tiler: bool = False,
        show_trajectories: bool = True,
        trajectory_history: int = 96,
        trajectory_sample_interval: int = 20,
        trajectory_max_segments: int = 24,
        export_predictions: str | None = None,
        gt_by_cam: dict | None = None,
        gt_snap_frames: int | None = None,
        no_sync: bool = False,
        geometry=None):
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
    print(f"[reid] tracker={tracker_config} (ReID engine independent of stream count)")
    print(f"[reid] person_class_id={person_class_id} inferred from {nvinfer_config}")
    print(f"[reid] debug_similarity={debug_similarity}")
    print(f"[reid] use_hungarian_assignment={use_hungarian_assignment}")
    print(f"[reid] enforce_unique_per_stream={enforce_unique_per_stream}")
    print(f"[reid] show_trajectories={show_trajectories}")
    print(gallery.config_summary())
    if save_video:
        print(f"[reid] save_video={save_video}")
    if export_predictions:
        print(f"[reid] export_predictions={export_predictions}")

    id_map: dict[int, int] = {}
    embeddings: dict[tuple, list] = {}  # (source_id, object_id) → embedding vector

    exporter = PredictionExporter(export_predictions) if export_predictions else None

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
        pipeline.add("nvurisrcbin", name, src_props)
        pipeline.link((name, "mux"), ("", "sink_%u"))

    pipeline.add("nvinfer", "pgie", {
        "config-file-path": runtime_nvinfer_config,
        "batch-size": batch, "gpu-id": gpu_id,
    })
    pipeline.attach("pgie", "measure_fps_probe", "fps_probe")

    pipeline.add("nvtracker", "tracker", {
        "ll-lib-file": deepstream_tracker_lib_path(),
        "ll-config-file": tracker_config,
        "tracker-width": 640, "tracker-height": 384,
        "gpu-id": gpu_id,
    })

    trajectory_visualizer = None
    if show_trajectories:
        trajectory_visualizer = TrajectoryVisualizer(
            tile_w, tile_h, cols, n,
            max_points=trajectory_history,
            sample_interval=trajectory_sample_interval,
            max_segments_per_track=trajectory_max_segments,
            pretiler=pretiler,
        )

    # frame_numbers: shared dict source_id → frame_number, filled by
    # SourceIdCollectorProbe (pre-tiler) and read by CrossCameraGalleryProbe
    # (post-tiler) so the exporter records the correct per-source frame index.
    frame_numbers: dict = {}

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
        geometry=geometry)

    if pretiler:
        # One pre-tiler probe on the tracker: exact source_id (no geometric
        # guessing), extracts embeddings + matches + sets labels in one pass.
        print("[reid] pretiler mode: gallery runs on tracker (no src guessing)")
        pipeline.attach("tracker", psm.Probe("reid_probe", gallery_probe))
    else:
        # Two-probe path: SourceIdCollectorProbe fills id_map pre-tiler
        # (source_id exact), CrossCameraGalleryProbe reads id_map post-tiler.
        # source_id is resolved from id_map — no geometric tile guessing.
        print("[reid] two-probe mode: source_id via id_map (pre-tiler exact)")
        pipeline.attach("tracker", psm.Probe(
            "src_collector",
            gallery.SourceIdCollectorProbe(
                id_map, embeddings, person_class_id, debug=debug_similarity,
                frame_numbers=frame_numbers),
        ))

    if gt_by_cam:
        pipeline.attach("tracker", psm.Probe(
            "gt_overlay", GtOverlayProbe(gt_by_cam, snap_frames=gt_snap_frames)))
        print(f"[reid] GT overlay enabled for {len(gt_by_cam)} camera(s) "
              f"(green boxes = ground truth)")

    if not no_tiler:
        pipeline.add("nvmultistreamtiler", "tiler", {
            "rows": rows, "columns": cols,
            "width": total_w, "height": total_h, "gpu-id": gpu_id,
        })
        if not pretiler:
            pipeline.attach("tiler", psm.Probe("reid_probe", gallery_probe))

    pipeline.add("nvosdbin", "osd", {
        "gpu-id": gpu_id,
        "process-mode": 1,
        "display-text": 1,
        "display-bbox": 1,
        "text-size": 18,
    })
    pipeline.link("mux", "pgie")
    pipeline.link("pgie", "tracker")
    if no_tiler:
        # Headless throughput: skip the tiler entirely.
        pipeline.link("tracker", "osd")
    else:
        pipeline.link("tracker", "tiler")
        pipeline.link("tiler", "osd")

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
        pipeline.link("osd", "output_tee", "display_queue", "sink")
        written_path = add_recording_branch(
            pipeline, "output_tee", save_video, record_bitrate,
            canvas_w=total_w, canvas_h=total_h, gpu_id=gpu_id)
    elif save_video:
        written_path = add_recording_branch(
            pipeline, "osd", save_video, record_bitrate,
            canvas_w=total_w, canvas_h=total_h, gpu_id=gpu_id)
    elif no_display:
        # Headless: drop frames as fast as possible, no window opened.
        pipeline.add("fakesink", "sink", {"sync": 0, "async": 0})
        pipeline.link("osd", "sink")
    else:
        pipeline.add(get_sink_element(), "sink", {"sync": sink_sync, "qos": 0})
        pipeline.link("osd", "sink")

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
        total_gids = gallery_probe._next_gid - 1
        print(f"[reid] Total unique global IDs assigned: {total_gids}")
    finally:
        pipeline.stop()
        if exporter is not None:
            exporter.close()
            print(f"[reid] Predictions exported to: {export_predictions}")


def build_arg_parser(defaults: dict) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Flexible cross-camera Re-ID pipeline (any stream count, "
                    "auto engine/config per batch)")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH,
                        help="Pipeline YAML used for default values. CLI flags "
                             "override it. Default: configs/pipeline.yaml")
    parser.add_argument(
        "--sources", nargs="+", default=defaults["sources"],
        help="One .txt list file, OR one folder of videos, OR one/more "
             "video paths / URIs (rtsp:// auto-enables live mode).")
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"],
                        help="Inference batch size. Default = number of streams. "
                             "Clamped to be >= stream count.")
    parser.add_argument("--max-sources", type=int,
                        default=defaults["max_sources"],
                        help="Load only the first N resolved sources/videos. "
                             "Default = use all sources.")
    parser.add_argument("--gpu-id", type=int, default=defaults["gpu_id"],
                        help="GPU device id used across the whole pipeline.")
    parser.add_argument("--pretiler", action="store_true",
                        default=defaults["pretiler"],
                        help="Run the gallery on the tracker (before the tiler). "
                             "Uses exact source_id instead of guessing it from "
                             "tile coordinates. Recommended at scale.")
    parser.add_argument("--no-tiler", action="store_true",
                        default=defaults["no_tiler"],
                        help="Headless throughput: drop the tiler entirely "
                             "(implies --pretiler). For many-camera benchmarking.")
    parser.add_argument("--trim-seconds", type=float, default=None,
                        help="Pre-cut each file source to this many seconds of "
                             "footage (ffmpeg stream-copy, cached) before running. "
                             "Gives a fixed-length clip regardless of GPU speed.")
    parser.add_argument("--trim-start", type=float, default=0.0,
                        help="Start offset (seconds) for --trim-seconds. "
                             "Default 0 = from the beginning.")
    parser.add_argument("--force-rebuild-engine", action="store_true",
                        help="Delete the current-batch detector engine and "
                             "rebuild it on this run.")
    parser.add_argument("--nvinfer-config", default=defaults["nvinfer_config"],
                        help="nvinfer config. Default comes from pipeline.yaml. "
                             "Alternatives: configs/models/nvinfer_yolov8_people.yml, "
                             "configs/models/nvinfer_yolov11_people.yml, "
                             "configs/models/nvinfer_peoplenet.yml, "
                             "configs/models/nvinfer_trafficcamnet.yml")
    parser.add_argument("--tracker-config", default=defaults["tracker_config"],
                        help="Tracker config. Default comes from pipeline.yaml. "
                             "Recommended demo: nvdeepsort_reid_swin.yaml. "
                             "Alternatives: nvdcf_accuracy.yaml, "
                             "nvdeepsort_reid.yaml, nvdcf_perf.yaml")
    parser.add_argument("--tile-w", type=int, default=defaults["tile_w"])
    parser.add_argument("--tile-h", type=int, default=defaults["tile_h"])
    parser.add_argument("--debug-similarity", action="store_true",
                        help="Print max cosine similarity for every new track")
    parser.add_argument("--show-trajectories", action="store_true",
                        default=defaults["show_trajectories"],
                        help="Draw recent tracker paths as colored OSD lines")
    parser.add_argument("--no-trajectories", action="store_false",
                        dest="show_trajectories",
                        help="Disable trajectory overlays")
    parser.add_argument("--trajectory-history", type=int,
                        default=defaults["trajectory_history"],
                        help="Max sampled points kept per local track for OSD paths")
    parser.add_argument("--trajectory-sample-interval", type=int,
                        default=defaults["trajectory_sample_interval"],
                        help="Append one trajectory point per local track every N batches")
    parser.add_argument("--trajectory-max-segments", type=int,
                        default=defaults["trajectory_max_segments"],
                        help="Max recent line segments drawn per visible local track")
    parser.add_argument("--similarity-threshold", type=float,
                        default=defaults["similarity_threshold"],
                        help="Min cosine similarity to accept a gallery match "
                             f"(default: {gallery.SIMILARITY_THRESHOLD})")
    parser.add_argument("--gallery-max-age", type=int,
                        default=defaults["gallery_max_age"],
                        help="Drop inactive Global IDs after this many batches")
    parser.add_argument("--disable-hungarian", action="store_true",
                        help="Use greedy gallery matching instead of per-stream "
                             "Hungarian assignment")
    parser.add_argument("--assignment-max-candidates", type=int,
                        default=defaults["assignment_max_candidates"],
                        help="Limit gallery IDs scanned by Hungarian assignment")
    parser.add_argument("--allow-duplicate-gid-per-stream", action="store_true",
                        help="Disable the guard that releases duplicate known "
                             "Global IDs within one stream before Hungarian")
    parser.add_argument("--disable-id-stickiness", action="store_true",
                        default=defaults["disable_id_stickiness"],
                        help="Allow a known local track to switch Global IDs "
                             "without requiring an extra similarity margin")
    parser.add_argument("--id-switch-margin", type=float,
                        default=defaults["id_switch_margin"],
                        help="Extra similarity required before switching from "
                             "a previous Global ID to another one")
    parser.add_argument("--allow-ambiguous-match", action="store_true",
                        default=defaults["allow_ambiguous_match"],
                        help="Allow matching an existing Global ID even when "
                             "the runner-up ID is very close")
    parser.add_argument("--match-ambiguity-margin", type=float,
                        default=defaults["match_ambiguity_margin"],
                        help="Best existing Global ID must beat runner-up by "
                             "this margin before it is accepted")
    parser.add_argument("--disable-global-merge", action="store_true",
                        default=defaults["disable_global_merge"],
                        help="Disable online merging of duplicate Global IDs")
    parser.add_argument("--global-merge-threshold", type=float,
                        default=defaults["global_merge_threshold"],
                        help="Similarity threshold to merge one Global ID into another")
    parser.add_argument("--global-merge-min-embeddings", type=int,
                        default=defaults["global_merge_min_embeddings"],
                        help="Minimum local tracklet embeddings before merge is considered")
    parser.add_argument("--global-merge-margin", type=float,
                        default=defaults["global_merge_margin"],
                        help="Merge candidate must beat runner-up by this margin")
    parser.add_argument("--global-merge-interval", type=int,
                        default=defaults["global_merge_interval"],
                        help="Run duplicate Global ID merge every N batches")
    parser.add_argument("--global-merge-max-candidates", type=int,
                        default=defaults["global_merge_max_candidates"],
                        help="Limit older Global IDs scanned per merge candidate")
    parser.add_argument("--disable-tracklet", action="store_true",
                        default=defaults["disable_tracklet"],
                        help="Use only current-frame embeddings, without local "
                             "tracklet averaging")
    parser.add_argument("--tracklet-embedding-interval", type=int,
                        default=defaults["tracklet_embedding_interval"],
                        help="Store one ReID embedding per local track every N "
                             "batches after warmup. Larger values reduce noisy "
                             "appearance drift.")
    parser.add_argument("--disable-embedding-quality-gate",
                        action="store_true",
                        default=defaults["disable_embedding_quality_gate"],
                        help="Allow border/overlap/small/aspect-ratio-poor crops "
                             "to update tracklets and the Global ID gallery.")
    parser.add_argument("--tracklet-window", type=int,
                        default=defaults["tracklet_window"],
                        help="Number of recent embeddings kept per (camera, track)")
    parser.add_argument("--tracklet-min-embeddings", type=int,
                        default=defaults["tracklet_min_embeddings"],
                        help="Minimum embeddings before using the averaged tracklet "
                             "vector for matching")
    parser.add_argument("--tracklet-max-age", type=int,
                        default=defaults["tracklet_max_age"],
                        help="Drop inactive local tracklets after this many batches")
    parser.add_argument("--save-video", nargs="?", const="output/videos/reid.mp4",
                        default=defaults["save_video"],
                        help="Save annotated output MP4. Default path when no value is "
                             "given: output/videos/reid.mp4")
    parser.add_argument("--record-bitrate", type=int,
                        default=defaults["record_bitrate"],
                        help="H.264 recording bitrate in bits/sec")
    parser.add_argument("--no-display", action="store_true",
                        default=defaults["no_display"],
                        help="Only valid with --save-video: record without opening a window")
    parser.add_argument("--no-sync", action="store_true",
                        help="Disable sink clock sync (sync=0). Prevents buffer drops on "
                             "high-fps sources like MTA (41fps) or slow GPUs. "
                             "Video plays as fast as the pipeline processes.")
    parser.add_argument("--mta-dataset", default=None, metavar="PATH",
                        help="Path to an MTA split folder (e.g. dataset/mta/MTA_ext_short/test). "
                             "Auto-loads all cam_*.mp4 files as sources, overriding --sources.")
    parser.add_argument("--export-predictions", default=None, metavar="DIR",
                        help="Write per-camera prediction CSVs to this directory "
                             "for offline evaluation with src.eval.metrics.")
    parser.add_argument("--show-gt", action="store_true",
                        help="Overlay ground-truth boxes (green) on the display. "
                             "Requires --mta-dataset or --wildtrack-dataset.")
    parser.add_argument("--wildtrack-dataset", default=None, metavar="PATH",
                        help="Path to the Wildtrack root folder "
                             "(e.g. dataset/Wildtrack). "
                             "Auto-loads cam1.mp4…cam7.mp4 as sources, "
                             "overriding --sources. Incompatible with --mta-dataset.")
    parser.add_argument("--wildtrack-minutes", type=float, default=None,
                        help="Limit Wildtrack playback/GT to this many minutes "
                             "(max: ~3.3 min = full annotated range).")
    parser.add_argument("--mmp-dataset", default=None, metavar="ROOT:SCENE",
                        help="MMPTracking scene to run: 'ROOT:SCENE', e.g. "
                             "'dataset/MMPTracking:lobby_0'. "
                             "Auto-loads cam MP4s as sources, overriding --sources.")
    parser.add_argument("--mmp-split", default="64pm",
                        help="MMPTracking split subfolder (default: 64pm).")
    parser.add_argument("--mmp-short-dataset", default=None, metavar="ROOT:SCENE",
                        help="MMPTracking_short scene: 'ROOT:SCENE', e.g. "
                             "'dataset/MMPTracking_short:lobby_0'. "
                             "Pre-built 1-min clips with GT CSVs. "
                             "Auto-loads camN.mp4 as sources.")
    parser.add_argument("--no-calibration", action="store_true",
                        help="Disable ground-plane geometry assistance even when "
                             "calibration data is available (--mmp-short-dataset). "
                             "Useful for A/B comparison against pure-ReID baseline.")
    parser.add_argument("--geo-weight", type=float, default=None,
                        help="Geometry blend weight [0.0–1.0]. "
                             "0 = pure ReID (same as --no-calibration), "
                             "1 = pure geometry. Default: 0.35.")
    return parser


def parse_args(argv: list[str] | None = None):
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    config_args, _ = config_parser.parse_known_args(argv)
    defaults = _load_defaults(config_args.config)
    parser = build_arg_parser(defaults)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    # Apply all ReID/Global-ID tuning overrides onto the gallery module.
    gallery.configure_from_args(args)
    enforce_unique = (
        gallery.ENFORCE_UNIQUE_GLOBAL_PER_STREAM
        and not args.allow_duplicate_gid_per_stream
    )
    use_hungarian = (
        gallery.USE_HUNGARIAN_ASSIGNMENT and not args.disable_hungarian
    )

    sources = args.sources
    gt_by_cam = None
    gt_snap_frames = None   # None = exact frame lookup (MTA); int = snap window (Wildtrack)
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
                gt_by_cam = mmp.load_all_gt()
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
                gt_by_cam = mmp_s.load_all_gt()
                print(f"[reid] Loading GT annotations for {len(gt_by_cam)} camera(s)")
            # Load calibration and build GroundPlaneGeometry when available
            if not args.no_calibration:
                try:
                    from src.reid.geometry import GroundPlaneGeometry
                    calib = mmp_s.load_calibration()
                    geometry = GroundPlaneGeometry(calib)
                    n_cams = len(calib.get("Cameras", []))
                    print(f"[reid] Ground-plane geometry loaded: {n_cams} camera(s), "
                          f"geo_weight={gallery.GEO_WEIGHT}")
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
        max_sources=args.max_sources,
        force_rebuild_engine=args.force_rebuild_engine,
        trim_seconds=args.trim_seconds, trim_start=args.trim_start,
        pretiler=args.pretiler, no_tiler=args.no_tiler,
        show_trajectories=args.show_trajectories,
        trajectory_history=args.trajectory_history,
        trajectory_sample_interval=args.trajectory_sample_interval,
        trajectory_max_segments=args.trajectory_max_segments,
        export_predictions=args.export_predictions,
        gt_by_cam=gt_by_cam,
        gt_snap_frames=gt_snap_frames,
        no_sync=args.no_sync,
        geometry=geometry)


if __name__ == "__main__":
    main()
