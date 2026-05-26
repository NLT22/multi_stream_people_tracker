"""
=============================================================================
MILESTONE 8 — Person Re-Identification (Concept Guide / Stub)
=============================================================================

THIS FILE IS A LEARNING GUIDE — not runnable working code.
It explains HOW to extend the pipeline for cross-camera Re-ID,
with stub classes showing exactly where each piece of code goes.

WHY RE-ID MATTERS:
  Milestones 1–7 track each person within ONE camera with a persistent ID.
  BUT: if Person #42 walks out of camera 0 and into camera 1,
  camera 1 assigns them a completely different ID (e.g. Person #7).

  Person Re-Identification solves the cross-camera identity problem:
    "Person #42 in cam0 = Person #7 in cam1 = the SAME physical person."

  Use cases:
    • Count unique shoppers across a store's cameras (not double-count)
    • Track a suspect from building entrance to exit
    • Measure cross-zone dwell time in retail analytics

=============================================================================
APPROACH: NvDeepSORT Tracker (built into DeepStream)
=============================================================================

NvDeepSORT runs a visual Re-ID model on each tracked person crop.
The feature vector (embedding) is used to match people across:
  - missed frames (brief occlusion)
  - camera handoff (cross-camera Re-ID with shared gallery)

Step 1 — Download the Re-ID model:
  ngc registry model download-version "nvidia/tao/reidentificationnet:deployable_v1.0"

Step 2 — Build TensorRT engine from the downloaded model.

Step 3 — Create a NvDeepSORT tracker config (stub below).

Step 4 — In pipeline.yaml, change:
  tracker:
    config_file: configs/tracker/nvdeepsort_reid.yaml
"""

# ── STUB: configs/tracker/nvdeepsort_reid.yaml ───────────────────────────────
NVDEEPSORT_CONFIG = """
# configs/tracker/nvdeepsort_reid.yaml  (STUB — fill in model paths)

BaseConfig:
  minDetectorConfidence: 0.2

ReIDModel:
  reidEngineFilePath: /path/to/reid_model_b16_gpu0_fp16.engine  # TODO
  reidFeatureSize: 128        # embedding dimension
  batchSize: 16

TargetManagement:
  maxTargetsPerStream: 50
  enableBboxUnClipping: true
  maxShadowTrackingAge: 60    # keep lost track 2 sec before dropping ID

  # Enable cross-camera matching using shared embedding gallery
  enableCrossStreamReid: false  # TODO: set true for multi-camera Re-ID
  reidMatchingThreshold: 0.5    # cosine similarity threshold

DataAssociator:
  attributeName: deepsort
"""

# ── STUB: Secondary GIE for custom Re-ID feature extraction ──────────────────
SGIE_REID_CONFIG = """
# configs/models/nvinfer_reid_sgie.yml  (STUB)

property:
  gpu-id: 0
  onnx-file: /path/to/reid_backbone.onnx           # TODO
  model-engine-file: ../../engine_cache/reid_b16_fp16.engine

  process-mode: 2       # secondary: operates on detected object crops
  operate-on-gie-id: 1  # process objects from pgie (TrafficCamNet)
  operate-on-class-ids: 2  # class_id=2 = Person only

  output-tensor-meta: 1  # expose raw embedding tensor to probe
  network-mode: 2        # FP16
  batch-size: 16
  gie-unique-id: 2
  network-type: 0

  labelfile-path: ../labels/people_only_labels.txt
"""

import pyservicemaker as psm


# ── STUB: Custom cross-camera Re-ID probe ────────────────────────────────────
class ReIDMatchingProbe(psm.BatchMetadataOperator):
    """
    STUB: Match persons across cameras using embedding similarity.

    Pipeline placement:
      [...] → nvtracker → sgie(reid) → [this probe] → nvosdbin → ...

    The sgie writes embedding tensors into tensor_items.
    This probe reads them and matches against a global gallery.
    """

    def __init__(self, similarity_threshold: float = 0.5):
        super().__init__()
        self._threshold = similarity_threshold
        # global_id → embedding vector
        self._gallery: dict[int, list[float]] = {}
        self._next_global_id = 1

    def handle_metadata(self, batch_meta):
        """
        TODO: Implement the following steps for each detected person:
          1. Extract Re-ID embedding from tensor_items
          2. Compare against gallery using cosine similarity
          3. If match found (similarity > threshold): assign existing global_id
          4. Else: assign new global_id, add to gallery
          5. Write global_id as custom text overlay (DisplayMeta)
        """
        for frame_meta in batch_meta.frame_items:
            for obj_meta in frame_meta.object_items:
                if obj_meta.class_id != 2:  # 2 = Person
                    continue
                # TODO: extract embedding
                # embedding = self._get_embedding(frame_meta, obj_meta)
                # global_id = self._match(embedding)
                # draw on screen: f"G#{global_id}"
                pass

    def _get_embedding(self, frame_meta, obj_meta) -> list[float]:
        """
        TODO: Extract Re-ID feature vector from TensorOutputUserMetadata.

        import torch
        for tensor_meta in frame_meta.tensor_items:
            t = tensor_meta.get_tensor(0).clone()
            feat = torch.utils.dlpack.from_dlpack(t)
            return feat.cpu().numpy().flatten().tolist()
        """
        raise NotImplementedError

    def _cosine_similarity(self, a: list, b: list) -> float:
        """TODO: compute dot(a,b) / (|a| * |b|)"""
        raise NotImplementedError

    def _match(self, embedding: list) -> int:
        """
        TODO: Find closest gallery entry.
        If similarity > threshold → return that global_id.
        Else → add new entry, return new global_id.
        """
        raise NotImplementedError


# ── Learning resources ────────────────────────────────────────────────────────
RESOURCES = """
Resources for implementing Re-ID:

  NGC models:
    ngc registry model list --query reidentification

  DeepStream sample configs:
    /opt/nvidia/deepstream/deepstream-9.0/samples/configs/deepstream-app/

  Implementation order:
    1. Complete Milestones 1-7 first
    2. Download ReID model from NGC
    3. Fill in nvdeepsort_reid.yaml paths
    4. Set tracker.config_file in pipeline.yaml
    5. Observe more stable IDs across occlusions
    6. For cross-camera: implement ReIDMatchingProbe gallery matching
"""

if __name__ == "__main__":
    print(__doc__)
    print("\n--- NvDeepSORT config stub ---")
    print(NVDEEPSORT_CONFIG)
    print("--- Resources ---")
    print(RESOURCES)
