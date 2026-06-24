"""Sidecar ReID: decouples Swin/ReID inference from the GStreamer data path.

Architecture
------------
  MetaCaptureOp  (BatchMetadataOperator — attaches to tracker pad)
    → records (frame_num, src, oid, bbox, pipeline_size, source_size) per person
    → puts entries in a bounded metadata queue (non-blocking)

  VideoReaderPool  (one thread per source)
    → each thread reads its video file sequentially via cv2.VideoCapture
    → when metadata arrives for a frame, extracts + normalises the crop
    → drops the crop into the ONNX inference queue

  ReIDWorker  (daemon thread)
    → drains the inference queue in mini-batches
    → runs ONNX inference (CUDAExecutionProvider, falls back to CPU)
    → writes (src, oid) → embedding into a thread-safe cache

  SourceIdCollectorProbe reads from the cache via SidecarReID.get_embedding()
  instead of SGIE tensor-output-meta.

Why not use buffer.extract()?
-----------------------------
  pyservicemaker's Buffer.extract() only supports plain-RGB surfaces. The
  nvstreammux output format is RGBA (or NV12), so extract() raises
  "Only RGB format is supported for being extracted as a tensor".
  Using cv2.VideoCapture avoids this constraint entirely.

Limitations
-----------
  Works for offline video files only (not live RTSP). Each source file is
  opened by its own reader thread and consumed at the same rate as the
  pipeline. The bbox coordinates are scaled from pipeline (mux) space to
  source (video) space using the sizes reported by DeepStream metadata.

Preprocessing
-------------
  Matches the SGIE config (nvinfer_reid_swin_sgie_all.yml):
    model-color-format: 0 → RGB input
    offsets: 123.675;116.28;103.53 (ImageNet mean × 255, RGB)
    net-scale-factor: 0.01735207   (≈ 1/255/std, single-scalar approx.)
  We use per-channel std for slightly better accuracy.
"""
from __future__ import annotations

import queue
import threading
import traceback
from pathlib import Path

import cv2
import numpy as np
import pyservicemaker as psm

# ImageNet RGB normalisation
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD  = np.array([0.229, 0.224, 0.225], np.float32)

REID_H = 256
REID_W = 128
MIN_CROP_W = 8
MIN_CROP_H = 16

# Module-level ort import so GStreamer callback threads can access it
# via module globals (required when running via DeepStream's pyservicemaker
# which may not propagate venv sys.path into C-created threads).
try:
    import onnxruntime as _ort
    _ORT_OK = True
except ImportError:
    _ORT_OK = False


# ---------------------------------------------------------------------------
# Metadata capture (BatchMetadataOperator — no surface access)
# ---------------------------------------------------------------------------

class MetaCaptureOp(psm.BatchMetadataOperator):
    """Records per-person bbox + frame info into a metadata queue.

    Runs in the GStreamer data path (pre-tiler), cheap — no GPU work.
    Each item in the queue is:
      (src, oid, frame_num, bbox_left, bbox_top, bbox_w, bbox_h,
       pipeline_w, pipeline_h, source_w, source_h)
    """

    def __init__(self, meta_queue: queue.Queue, person_class_id: int):
        super().__init__()
        self._queue = meta_queue
        self._person_class_id = person_class_id
        self._frames_seen = 0
        self._dropped = 0

    def handle_metadata(self, batch_meta):
        try:
            self._handle(batch_meta)
        except Exception:
            traceback.print_exc()

    def _handle(self, batch_meta):
        self._frames_seen += 1
        for frame_meta in batch_meta.frame_items:
            src = frame_meta.source_id
            fnum = frame_meta.frame_number
            pw = frame_meta.pipeline_width or 0
            ph = frame_meta.pipeline_height or 0
            sw = frame_meta.source_width or 0
            sh = frame_meta.source_height or 0
            for obj_meta in frame_meta.object_items:
                if obj_meta.class_id != self._person_class_id:
                    continue
                r = obj_meta.rect_params
                item = (src, obj_meta.object_id, fnum,
                        float(r.left), float(r.top),
                        float(r.width), float(r.height),
                        pw, ph, sw, sh)
                try:
                    self._queue.put_nowait(item)
                except queue.Full:
                    self._dropped += 1

    @property
    def dropped(self) -> int:
        return self._dropped


# ---------------------------------------------------------------------------
# Per-source video reader thread
# ---------------------------------------------------------------------------

class _SourceReader(threading.Thread):
    """Reads one video source sequentially; extracts crops on demand."""

    def __init__(self, src_id: int, uri: str,
                 meta_queue: queue.Queue, crop_queue: queue.Queue,
                 stop_evt: threading.Event):
        super().__init__(daemon=True, name=f"reid-reader-{src_id}")
        self._src = src_id
        self._uri = uri
        self._meta = meta_queue    # incoming: items for THIS source
        self._crops = crop_queue   # outgoing: (src, oid, crop_chw) to ONNX
        self._stop_evt = stop_evt  # NOT _stop: that name shadows Thread._stop()
        self._curr_fnum = -1
        self._curr_frame: np.ndarray | None = None

    def run(self):
        cap = cv2.VideoCapture(self._uri)
        if not cap.isOpened():
            print(f"[sidecar reader {self._src}] cannot open {self._uri}")
            return

        while not self._stop_evt.is_set():
            try:
                item = self._meta.get(timeout=0.01)
            except queue.Empty:
                continue

            (src, oid, fnum, bx, by, bw, bh, pw, ph, sw, sh) = item

            # Advance to the requested frame
            while self._curr_fnum < fnum and not self._stop_evt.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                self._curr_frame = frame
                self._curr_fnum += 1

            if self._curr_frame is None or self._curr_fnum != fnum:
                continue

            # Scale bbox from pipeline (mux) space → source (video) frame space
            fh, fw = self._curr_frame.shape[:2]
            if pw > 0 and ph > 0:
                xs = fw / pw
                ys = fh / ph
            else:
                xs = ys = 1.0

            x1 = max(0, int(bx * xs))
            y1 = max(0, int(by * ys))
            x2 = min(fw, int((bx + bw) * xs))
            y2 = min(fh, int((by + bh) * ys))
            if (x2 - x1) < MIN_CROP_W or (y2 - y1) < MIN_CROP_H:
                continue

            crop_bgr = self._curr_frame[y1:y2, x1:x2]
            crop_rgb = cv2.cvtColor(
                cv2.resize(crop_bgr, (REID_W, REID_H)), cv2.COLOR_BGR2RGB)
            crop_f = crop_rgb.astype(np.float32) / 255.0
            crop_n = ((crop_f - _MEAN) / _STD).transpose(2, 0, 1)  # (3,H,W)

            try:
                self._crops.put_nowait((src, oid, crop_n))
            except queue.Full:
                pass

        cap.release()


class VideoReaderPool:
    """One reader thread per source URI. Routes meta items to the right reader."""

    def __init__(self, sources: list[str], meta_queue: queue.Queue,
                 crop_queue: queue.Queue, stop_evt: threading.Event):
        # per-source private queues
        self._src_queues: dict[int, queue.Queue] = {
            i: queue.Queue(maxsize=500) for i in range(len(sources))
        }
        self._readers = [
            _SourceReader(i, uri, self._src_queues[i], crop_queue, stop_evt)
            for i, uri in enumerate(sources)
        ]
        self._meta_queue = meta_queue
        self._stop = stop_evt
        self._router = threading.Thread(
            target=self._route, daemon=True, name="reid-meta-router")

    def _route(self):
        """Fan-out: dispatch metadata items to the correct per-source queue."""
        while not self._stop.is_set():  # VideoReaderPool._stop is not a Thread attr
            try:
                item = self._meta_queue.get(timeout=0.01)
                src = item[0]
                if src in self._src_queues:
                    try:
                        self._src_queues[src].put_nowait(item)
                    except queue.Full:
                        pass
            except queue.Empty:
                continue

    def start(self):
        self._router.start()
        for r in self._readers:
            r.start()

    def stop(self):
        for r in self._readers:
            r.join(timeout=1.0)
        self._router.join(timeout=1.0)


# ---------------------------------------------------------------------------
# ONNX inference worker
# ---------------------------------------------------------------------------

class ReIDWorker(threading.Thread):
    """Daemon thread: drains crop queue → ONNX inference → fills embedding cache."""

    def __init__(self, onnx_path: str, crop_queue: queue.Queue,
                 embedding_cache: dict, cache_lock: threading.Lock,
                 batch_size: int = 32, poll_s: float = 0.002):
        super().__init__(daemon=True, name="reid-sidecar-infer")
        self._onnx_path = onnx_path
        self._queue = crop_queue
        self._cache = embedding_cache
        self._lock = cache_lock
        self._batch_size = batch_size
        self._poll_s = poll_s
        self._stop_evt = threading.Event()
        self._inferred = 0

    def run(self):
        if not _ORT_OK:
            print("[sidecar] onnxruntime not available — worker idle (run via venv python)")
            return
        sess = _ort.InferenceSession(
            self._onnx_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        iname = sess.get_inputs()[0].name
        print(f"[sidecar] infer worker started  model={Path(self._onnx_path).name}"
              f"  batch={self._batch_size}"
              f"  provider={sess.get_providers()[0]}")

        while not self._stop_evt.is_set():
            keys, crops = self._drain()
            if not keys:
                self._stop_evt.wait(timeout=self._poll_s)
                continue
            batch = np.stack(crops, 0).astype(np.float32)
            try:
                out = sess.run(None, {iname: batch})[0]
                norms = np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
                out = out / norms
                with self._lock:
                    for (src, oid), emb in zip(keys, out):
                        self._cache[(src, oid)] = emb.tolist()
                self._inferred += len(keys)
            except Exception:
                traceback.print_exc()

    def _drain(self):
        keys, crops = [], []
        try:
            src, oid, crop = self._queue.get(timeout=self._poll_s)
            keys.append((src, oid))
            crops.append(crop)
        except queue.Empty:
            return keys, crops
        while len(keys) < self._batch_size:
            try:
                src, oid, crop = self._queue.get_nowait()
                keys.append((src, oid))
                crops.append(crop)
            except queue.Empty:
                break
        return keys, crops

    def stop(self):
        self._stop_evt.set()

    @property
    def inferred(self) -> int:
        return self._inferred


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------

class SidecarReID:
    """Coordinator: owns queues, cache, threads, and probe operators.

    Usage::

        sidecar = SidecarReID(onnx_path, sources, person_class_id)
        sidecar.start()

        # Attach metadata capture to tracker (BatchMetadataOperator):
        pipeline.attach("tracker", psm.Probe("meta_capture", sidecar.meta_op))

        # Pass sidecar to SourceIdCollectorProbe:
        probe = SourceIdCollectorProbe(..., sidecar=sidecar)

        pipeline.start(); pipeline.wait()
        sidecar.stop()
    """

    def __init__(self, onnx_path: str, sources: list[str], person_class_id: int,
                 meta_queue_size: int = 1000, crop_queue_size: int = 200,
                 batch_size: int = 32):
        self._onnx_path = onnx_path
        self._sources = sources
        self.embedding_cache: dict[tuple[int, int], list[float]] = {}
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()

        self._meta_queue: queue.Queue = queue.Queue(maxsize=meta_queue_size)
        self._crop_queue: queue.Queue = queue.Queue(maxsize=crop_queue_size)

        # Probe operator (BatchMetadataOperator) attached to tracker pad
        self.meta_op = MetaCaptureOp(self._meta_queue, person_class_id)

        self._reader_pool = VideoReaderPool(
            sources, self._meta_queue, self._crop_queue, self._stop_evt)
        self._worker = ReIDWorker(
            onnx_path, self._crop_queue, self.embedding_cache, self._lock,
            batch_size=batch_size)

    def start(self):
        self._worker.start()
        self._reader_pool.start()
        print(f"[sidecar] started  sources={len(self._sources)}  onnx={self._onnx_path}")

    def stop(self):
        self._stop_evt.set()
        self._worker.stop()
        self._reader_pool.stop()
        self._worker.join(timeout=3.0)
        print(f"[sidecar] stopped  inferred={self._worker.inferred}"
              f"  meta_dropped={self.meta_op.dropped}")

    def get_embedding(self, src: int, oid: int) -> list[float]:
        """Thread-safe read of the latest embedding for (src, oid)."""
        with self._lock:
            return self.embedding_cache.get((src, oid), [])
