"""Pre-tiler metadata extraction probe.

SourceIdCollectorProbe runs where frame_meta.source_id is exact (pre-tiler)
and reads ReID embedding tensors off the tracker metadata. Self-contained:
only DeepStream metadata + optional torch, no gallery tuning state.
"""

import traceback

import pyservicemaker as psm

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class SourceIdCollectorProbe(psm.BatchMetadataOperator):
    """
    Pre-tiler: source_id is valid here — collect it for each tracked person.
    Also extracts Re-ID embeddings from tracker ReID metadata OR from a sidecar.

    Fills shared dicts:
      embeddings: (source_id, object_id) → embedding vector (list[float]) or []

    When `sidecar` is provided, embeddings are read from SidecarReID.get_embedding()
    instead of SGIE tensor-output-meta (CropCaptureOperator populates the cache
    on the same buffer, one probe ahead in the attachment chain).
    """

    def __init__(self, id_map: dict, embeddings: dict, person_class_id: int,
                 debug: bool = False, frame_numbers: dict | None = None,
                 frame_sizes: dict | None = None, sidecar=None):
        super().__init__()
        self._id_map = id_map
        self._embeddings = embeddings
        self._frame_numbers = frame_numbers  # source_id → frame_number (for exporter)
        self._frame_sizes = frame_sizes      # source_id → (width, height)
        self._person_class_id = person_class_id
        self._debug = debug
        self._sidecar = sidecar  # SidecarReID | None
        self._frame_count = 0
        self._persons_seen = 0
        self._embeddings_seen = 0
        self._object_reid_metas = 0
        self._object_tensor_metas = 0
        self._frame_tensor_metas = 0
        self._debug_failures_printed = 0

    def handle_metadata(self, batch_meta):
        try:
            self._handle_metadata(batch_meta)
        except Exception:
            print("[reid ERROR] SourceIdCollectorProbe failed:")
            traceback.print_exc()

    def _handle_metadata(self, batch_meta):
        self._frame_count += 1
        batch_persons = 0
        batch_embeddings = 0
        batch_obj_reids = 0
        batch_obj_tensors = 0
        batch_frame_tensors = 0

        # The shared dicts are a per-batch handoff to the post-tiler gallery
        # probe, which runs synchronously on the same buffer right after this
        # one. Clearing here bounds memory: without it, every (src, object_id)
        # ever seen would accumulate forever on long/multi-camera videos.
        self._embeddings.clear()
        self._id_map.clear()
        if self._frame_numbers is not None:
            self._frame_numbers.clear()
        if self._frame_sizes is not None:
            self._frame_sizes.clear()

        for frame_meta in batch_meta.frame_items:
            src = frame_meta.source_id
            if self._frame_numbers is not None:
                self._frame_numbers[src] = frame_meta.frame_number
            if self._frame_sizes is not None:
                size = self._source_frame_size(frame_meta)
                if size is not None:
                    self._frame_sizes[src] = size
            frame_tensor_count = self._count_iter(frame_meta.tensor_items)
            batch_frame_tensors += frame_tensor_count
            for obj_meta in frame_meta.object_items:
                if obj_meta.class_id != self._person_class_id:
                    continue
                oid = obj_meta.object_id
                self._id_map[oid] = src
                batch_persons += 1

                # Extract Re-ID embedding: either from the sidecar cache
                # (populated by CropCaptureOperator one probe ahead) or from
                # the SGIE tensor-output-meta attached by nvinfer SGIE.
                if self._sidecar is not None:
                    embedding = self._sidecar.get_embedding(src, oid)
                    obj_reid_count = obj_tensor_count = 0
                    reason = "sidecar"
                    if embedding:
                        batch_embeddings += 1
                else:
                    embedding, obj_reid_count, obj_tensor_count, reason = (
                        self._extract_embedding(obj_meta))
                    batch_obj_reids += obj_reid_count
                    batch_obj_tensors += obj_tensor_count
                    if embedding:
                        batch_embeddings += 1
                    elif self._debug and self._debug_failures_printed < 12:
                        print(
                            f"  [Re-ID tensor debug] Cam{src}#{oid} "
                            f"embedding=empty reason={reason} "
                            f"obj_reid_items={obj_reid_count} "
                            f"obj_tensor_items={obj_tensor_count} "
                            f"frame_tensor_items={frame_tensor_count} "
                            f"torch_available={_TORCH_AVAILABLE}"
                        )
                        self._debug_failures_printed += 1
                self._embeddings[(src, oid)] = embedding

        self._persons_seen += batch_persons
        self._embeddings_seen += batch_embeddings
        self._object_reid_metas += batch_obj_reids
        self._object_tensor_metas += batch_obj_tensors
        self._frame_tensor_metas += batch_frame_tensors

        if self._debug and self._frame_count % 60 == 0:
            print(
                f"[reid tensor debug] frame={self._frame_count:06d} "
                f"batch_persons={batch_persons} "
                f"batch_embeddings={batch_embeddings} "
                f"batch_obj_reid_items={batch_obj_reids} "
                f"batch_obj_tensor_items={batch_obj_tensors} "
                f"batch_frame_tensor_items={batch_frame_tensors} "
                f"total_embeddings={self._embeddings_seen}/{self._persons_seen} "
                f"torch_available={_TORCH_AVAILABLE}"
            )

    @staticmethod
    def _count_iter(items) -> int:
        return sum(1 for _ in items)

    # Maximum plausible source resolution — anything larger is the MUX
    # output size being mis-reported as the per-source size.
    _MAX_SOURCE_W = 1280.0
    _MAX_SOURCE_H = 720.0

    @staticmethod
    def _source_frame_size(frame_meta) -> tuple[float, float] | None:
        width_names = ("source_frame_width", "frame_width", "source_width", "width")
        height_names = ("source_frame_height", "frame_height", "source_height", "height")
        width = next(
            (float(getattr(frame_meta, name)) for name in width_names
             if hasattr(frame_meta, name) and getattr(frame_meta, name)),
            None,
        )
        height = next(
            (float(getattr(frame_meta, name)) for name in height_names
             if hasattr(frame_meta, name) and getattr(frame_meta, name)),
            None,
        )
        if width and height:
            # Reject MUX-level dimensions (e.g. 1920×1080) that DeepStream
            # sometimes reports as the per-source size for early-initialised
            # sources.  Values larger than _MAX_SOURCE_W/H are the mux output
            # size leaking into source metadata; treat them as unknown.
            if (width > SourceIdCollectorProbe._MAX_SOURCE_W
                    or height > SourceIdCollectorProbe._MAX_SOURCE_H):
                return None
            return width, height
        return None

    @staticmethod
    def _extract_embedding(obj_meta) -> tuple[list[float], int, int, str]:
        reid_count = 0
        try:
            for reid_meta in obj_meta.obj_reid_items:
                reid_count += 1
                reid = reid_meta.as_obj_reid()
                feature = reid.feature_vector
                if callable(feature):
                    feature = feature()
                embedding = list(feature) if feature is not None else []
                if embedding:
                    return embedding, reid_count, 0, f"ok_obj_reid_dim_{len(embedding)}"
            if reid_count > 0:
                return [], reid_count, 0, "empty_obj_reid_feature_vector"
        except Exception as e:
            return [], reid_count, 0, f"obj_reid_{type(e).__name__}: {e}"

        tensor_count = 0
        if not _TORCH_AVAILABLE:
            return [], reid_count, tensor_count, "torch_unavailable"

        try:
            for tensor_meta in obj_meta.tensor_items:
                tensor_count += 1
                layers = tensor_meta.as_tensor_output().get_layers()
                if not layers:
                    continue
                raw = next(iter(layers.values()))
                feat = torch.utils.dlpack.from_dlpack(raw)
                embedding = feat.cpu().numpy().flatten().tolist()
                if embedding:
                    return embedding, reid_count, tensor_count, f"ok_tensor_dim_{len(embedding)}"
            return [], reid_count, tensor_count, "no_reid_or_tensor_layers"
        except Exception as e:
            return [], reid_count, tensor_count, f"tensor_{type(e).__name__}: {e}"
