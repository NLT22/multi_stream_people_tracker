"""CLI argument parsing for the pipeline.
Extracted from src/main.py."""

import argparse

from src.reid.config import ReIDConfig
from src.config.runtime import DEFAULT_CONFIG_PATH, _load_defaults


def build_arg_parser(defaults: dict) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Flexible cross-camera Re-ID pipeline (any stream count, "
                    "auto engine/config per batch)")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH,
                        help="Pipeline YAML used for default values. CLI flags "
                             "override it. Default: configs/pipelines/pipeline.yaml")
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
                        help="nvinfer config. Default comes from the pipeline YAML. "
                             "MMP detector: configs/models/nvinfer_yolov11_mmp.yml; "
                             "generic people: configs/models/nvinfer_yolov11_people.yml")
    parser.add_argument("--reid-sgie-config", default=defaults["reid_sgie_config"],
                        help="Optional secondary nvinfer (SGIE) config that "
                             "extracts per-person ReID embeddings as "
                             "output-tensor-meta. Use with a reidType:0 perf "
                             "tracker for realtime cross-camera ReID. "
                             "E.g. configs/models/nvinfer_reid_swin_sgie.yml")
    parser.add_argument("--nvdsanalytics-config",
                        default=defaults["nvdsanalytics_config"],
                        help="Optional gst-nvdsanalytics config (ROI occupancy, "
                             "line-crossing, overcrowding). Counts are printed, "
                             "exported (with --export-predictions), and drawn on the "
                             "video. E.g. configs/analytics/nvdsanalytics_mmp.txt")
    parser.add_argument("--tracker-config", default=defaults["tracker_config"],
                        help="Tracker config. Default comes from pipeline.yaml. "
                             "Recommended demo: nvdeepsort_reid_swin.yaml. "
                             "Alternatives: nvdcf_accuracy.yaml, "
                             "nvdeepsort_reid.yaml, nvdcf_perf.yaml")
    parser.add_argument("--tracker-width", type=int,
                        default=defaults["tracker_width"],
                        help="nvtracker input width. Match detector input for "
                             "best performance.")
    parser.add_argument("--tracker-height", type=int,
                        default=defaults["tracker_height"],
                        help="nvtracker input height. Match detector input for "
                             "best performance.")
    parser.add_argument("--tracker-sub-batches",
                        default=defaults["tracker_sub_batches"],
                        help="Optional nvtracker sub-batches string, e.g. "
                             "'5:5:5:5' for four tracker instances.")
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
                             f"(default: {ReIDConfig().similarity_threshold})")
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
    parser.add_argument("--micro-batch-fusion", action="store_true",
                        default=defaults["micro_batch_fusion"],
                        help="Enable live micro-batch cross-camera fusion: a "
                             "MicroBatchFusion engine clusters tracklet embeddings "
                             "across cameras every --fusion-interval frames and "
                             "remaps displayed/exported Global IDs. This is the "
                             "production MTMC architecture (decoupled perception + "
                             "delayed fusion), replacing per-frame online merge.")
    parser.add_argument("--fusion-interval", type=int,
                        default=defaults["fusion_interval"],
                        help="Frames between micro-batch fusion passes "
                             "(125 = 5s at 25 FPS)")
    parser.add_argument("--fusion-threshold", type=float,
                        default=defaults["fusion_threshold"],
                        help="Cosine similarity gate for cross-camera fusion merges")
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
    parser.add_argument("--geometry-assignment-mode",
                        choices=["weight_only", "close_reid_only"],
                        default=defaults["geometry_assignment_mode"],
                        help="How calibration geometry affects Hungarian assignment")
    parser.add_argument("--geometry-reid-margin", type=float,
                        default=defaults["geometry_reid_margin"],
                        help="For close_reid_only mode, apply geometry only to "
                             "candidates within this ReID score margin of best")
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
    parser.add_argument("--disable-gallery", action="store_true",
                        default=defaults["disable_gallery"],
                        help="Skip cross-camera gallery probes. Use for pure "
                             "tracker-only realtime throughput.")
    parser.add_argument("--enable-gallery", action="store_false",
                        dest="disable_gallery",
                        help="Enable cross-camera gallery probes even if the "
                             "selected YAML disables them.")
    parser.add_argument("--disable-osd", action="store_false",
                        dest="osd_enabled",
                        default=defaults["osd_enabled"],
                        help="Skip nvosdbin. Use with --no-display for pure "
                             "throughput or backend ingest.")
    parser.add_argument("--enable-osd", action="store_true",
                        dest="osd_enabled",
                        help="Enable nvosdbin even if the selected YAML "
                             "disables it.")
    parser.add_argument("--no-sync", action="store_true",
                        default=defaults["no_sync"],
                        help="Disable sink clock sync (sync=0). Prevents buffer drops on "
                             "high-fps sources like MTA (41fps) or slow GPUs. "
                             "Video plays as fast as the pipeline processes.")
    parser.add_argument("--sync", action="store_false", dest="no_sync",
                        help="Use sink clock sync even if the selected YAML "
                             "sets runtime.no_sync.")
    parser.add_argument("--loop-video", action="store_true", default=False,
                        help="Loop file sources indefinitely (file-loop=1 on nvurisrcbin). "
                             "Useful for benchmarking so short videos do not end "
                             "before the FPS probe fires.")
    parser.add_argument("--export-predictions", default=None, metavar="DIR",
                        help="Write per-camera prediction CSVs to this directory "
                             "for offline evaluation with src.eval.metrics_mmp.")
    parser.add_argument("--show-gt", action="store_true",
                        help="Overlay ground-truth boxes (green) on the display. "
                             "Requires --mmp-dataset or --mmp-short-dataset.")
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

