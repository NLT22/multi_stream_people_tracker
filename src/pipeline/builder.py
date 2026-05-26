"""
PipelineBuilder — assembles the full DeepStream pipeline from config.

WHY A BUILDER CLASS:
  As the pipeline grows (sources → mux → infer → tracker → osd → tiler → sink),
  the wiring logic becomes long and error-prone.
  The builder encapsulates each stage as a method, making it easy to:
    - Enable/disable stages via config flags
    - Understand which elements exist and how they are connected
    - Reuse the same logic from both milestones and main.py

PIPELINE TOPOLOGY (fully assembled, Milestone 7+):

    [nvurisrcbin_0] ─┐
    [nvurisrcbin_1] ─┼─→ [nvstreammux] → [nvinfer/pgie] → [nvtracker]
    [nvurisrcbin_N] ─┘                                         │
                                                                ↓
                                           [nvosdbin] ← ─ ─ ─ ─
                                               │
                                               ↓
                                    [nvmultistreamtiler]
                                               │
                                               ↓
                                       [nveglglessink]

HOW TO USE (Milestone 7+):
    config = PipelineConfig.from_yaml("configs/pipeline.yaml")
    uris   = load_uris(config.source_mode, config.active_source_config)
    config.num_sources = len(uris)

    builder  = PipelineBuilder(config, uris)
    pipeline = builder.build()
    pipeline.start()
    pipeline.wait()
"""

import math
import pyservicemaker as psm

from src.config.loader import PipelineConfig
from src.utils.platform_utils import get_sink_element, get_sink_properties
from src.pipeline.probes import PersonOSDProbe, PersonCountProbe


class PipelineBuilder:

    def __init__(self, config: PipelineConfig, uris: list[str]):
        self.config = config
        self.uris = uris
        self.num_sources = len(uris)
        self._pipeline: psm.Pipeline | None = None

    # ── Public API ──────────────────────────────────────────────────────────

    def build(self) -> psm.Pipeline:
        """Build and return a configured (but not started) pipeline."""
        cfg = self.config

        self._pipeline = psm.Pipeline("people-tracker")

        self._add_sources()
        self._add_muxer()

        last_element = "mux"

        if cfg.detection.enabled:
            self._add_inference()
            self._pipeline.link(last_element, "pgie")
            last_element = "pgie"

        if cfg.tracker.enabled and cfg.detection.enabled:
            self._add_tracker()
            self._pipeline.link(last_element, "tracker")
            last_element = "tracker"

        if cfg.display.osd_enabled:
            self._add_osd(upstream=last_element)
            last_element = "osd"

        if cfg.display.tiled_display and self.num_sources > 1:
            self._add_tiler()
            self._pipeline.link(last_element, "tiler")
            last_element = "tiler"

        self._add_sink(upstream=last_element)

        return self._pipeline

    # ── Stage builders ──────────────────────────────────────────────────────

    def _add_sources(self):
        """
        Add one nvurisrcbin per source URI and link each to nvstreammux.

        WHY nvurisrcbin:
          It handles file://, rtsp://, and http:// transparently.
          It auto-detects the codec and sets up demux+parse+decode internally.
          You never need to know whether the video is H.264 or H.265.

        WHY "sink_%u" (NOT "sink_0", "sink_1", etc.):
          nvstreammux uses GStreamer "request pads" — pads that are created
          on demand. The template "sink_%u" tells GStreamer to allocate the
          next available numbered sink pad.
          Using a literal "sink_0" bypasses this mechanism and WILL fail.
        """
        cfg = self.config
        is_live = cfg.source_mode == "rtsp_cameras"

        for i, uri in enumerate(self.uris):
            src_name = f"source_{i}"
            src_props = {"uri": uri, "gpu-id": cfg.gpu_id}
            if is_live:
                src_props["live-source"] = 1

            self._pipeline.add("nvurisrcbin", src_name, src_props)

            # CRITICAL: use "sink_%u" — GStreamer request pad template
            self._pipeline.link((src_name, "mux"), ("", "sink_%u"))

    def _add_muxer(self):
        """
        Add nvstreammux — the batching hub.

        WHY nvstreammux IS REQUIRED even for 1 stream:
          All DeepStream metadata (NvDsBatchMeta) is attached at the muxer.
          nvinfer, nvtracker, and OSD all expect batched buffers with
          NvDsBatchMeta attached. Without the muxer, they have nothing to
          read from or write to.

        Key properties:
          batch-size            : must equal number of sources
          batched-push-timeout  : how long to wait for a full batch (µs)
                                  For files: 40000 µs is fine.
                                  For RTSP:  set live-source=1 instead.
          width/height          : output resolution sent to nvinfer
        """
        cfg = self.config
        mux_props = {
            "batch-size": self.num_sources,
            "batched-push-timeout": 40000,
            "width": 1920,
            "height": 1080,
            "gpu-id": cfg.gpu_id,
        }
        if cfg.source_mode == "rtsp_cameras":
            mux_props["live-source"] = 1

        self._pipeline.add("nvstreammux", "mux", mux_props)

    def _add_inference(self):
        """
        Add nvinfer — TensorRT inference engine.

        The config file (YAML) specifies:
          - Which ONNX model to use
          - Precision (FP16 for RTX 3050Ti)
          - Batch size
          - Class labels

        On FIRST RUN: nvinfer builds a TensorRT engine from the ONNX file.
        This takes 1-3 minutes. Subsequent runs load the cached .engine file.

        WHY FPS probe attaches to nvinfer (NOT the sink):
          Attaching measure_fps_probe to a sink element raises RuntimeError.
          nvinfer is the correct attachment point.
        """
        cfg = self.config
        self._pipeline.add("nvinfer", "pgie", {
            "config-file-path": cfg.detection.config_file,
            "batch-size": self.num_sources,
            "gpu-id": cfg.gpu_id,
        })
        # Built-in FPS probe — prints throughput every 5 seconds
        # Attach to nvinfer, NEVER to a sink element
        self._pipeline.attach("pgie", "measure_fps_probe", "fps_probe")

    def _add_tracker(self):
        """
        Add nvtracker — multi-object tracking.

        WHY tracking is separate from detection:
          nvinfer detects objects independently in each frame — it has no
          memory of previous frames. nvtracker takes detection bounding boxes
          and assigns persistent object_id integers across frames.

        tracker-width/height should match nvinfer input dimensions for
        best accuracy when using input-tensor-meta (not required here).
        """
        cfg = self.config
        self._pipeline.add("nvtracker", "tracker", {
            "ll-lib-file": (
                "/opt/nvidia/deepstream/deepstream/lib/"
                "libnvds_nvmultiobjecttracker.so"
            ),
            "ll-config-file": cfg.tracker.config_file,
            "tracker-width":  cfg.tracker.tracker_width,
            "tracker-height": cfg.tracker.tracker_height,
            "gpu-id": cfg.gpu_id,
        })

    def _add_osd(self, upstream: str):
        """
        Add nvosdbin + custom OSD probe.

        nvosdbin reads DisplayMeta added by probes and draws:
          - Bounding rectangles around detected objects
          - Text labels (class name + tracking ID)
          - Any custom overlays added by PersonOSDProbe

        process-mode: 1 = GPU rendering (faster, requires NVMM buffers)
        process-mode: 0 = CPU rendering (slower, for debugging)
        """
        # Attach the custom label probe BEFORE the OSD element
        # Custom probes must be wrapped: psm.Probe("name", instance)
        self._pipeline.attach(upstream, psm.Probe("osd_probe", PersonOSDProbe()))

        self._pipeline.add("nvosdbin", "osd", {
            "gpu-id": self.config.gpu_id,
            "process-mode": 1,
        })
        self._pipeline.link(upstream, "osd")

    def _add_tiler(self):
        """
        Add nvmultistreamtiler — NxN grid display.

        WHY tiling:
          nvstreammux batches multiple streams into one buffer.
          Without the tiler, only one stream (the last) appears in the sink.
          The tiler arranges all streams side-by-side in a grid.

        Grid size is computed from stream count: e.g., 4 streams → 2×2 grid.
        """
        cfg = self.config
        rows, cols = _compute_tile_grid(self.num_sources)
        total_w = cfg.display.tile_width * cols
        total_h = cfg.display.tile_height * rows

        self._pipeline.add("nvmultistreamtiler", "tiler", {
            "rows":    rows,
            "columns": cols,
            "width":   total_w,
            "height":  total_h,
            "gpu-id":  cfg.gpu_id,
        })

    def _add_sink(self, upstream: str):
        """
        Add the display sink.

        nveglglessink (x86_64): EGL/OpenGL window
        nv3dsink (Jetson):      3D hardware compositor

        sync=1: render at source frame rate (smooth playback for files)
        sync=0: render as fast as possible (for live/RTSP sources)
        """
        cfg = self.config
        is_live = cfg.source_mode == "rtsp_cameras"
        sink_type = get_sink_element()
        sink_props = get_sink_properties(is_live=is_live)

        self._pipeline.add(sink_type, "sink", sink_props)
        self._pipeline.link(upstream, "sink")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _compute_tile_grid(n: int) -> tuple[int, int]:
    """
    Compute (rows, cols) for an NxN-ish grid that fits n streams.

    Examples:
        1 stream  → (1, 1)
        2 streams → (1, 2)
        3 streams → (2, 2)  [one empty cell]
        4 streams → (2, 2)
        6 streams → (2, 3)
        9 streams → (3, 3)
    """
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols
