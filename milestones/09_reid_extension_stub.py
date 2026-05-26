"""
=============================================================================
MILESTONE 9 — Person Re-Identification Extension (STUB / Concept Guide)
=============================================================================

THIS FILE CONTAINS NO WORKING CODE.
It is a learning guide explaining HOW to extend the pipeline for Re-ID,
with stub functions showing WHERE each piece of code would go.

WHY RE-ID MATTERS:
  Milestones 1-8 give each person a unique ID *within one camera*.
  If person #42 walks out of camera 0 and into camera 1, they get a NEW ID.

  Person Re-Identification solves the cross-camera identity problem:
  "Person #42 in camera 0 is the SAME as Person #7 in camera 1."

  Use cases:
    - Retail: count unique shoppers across multiple store zones
    - Security: track a suspect from entrance to exit
    - Crowd analysis: measure dwell time across areas

APPROACHES:
  1. NvDeepSORT (built-in): Uses a visual ReID model inside nvtracker.
     The tracker runs a ReID feature extractor on each tracked person
     and uses those features for cross-frame and cross-camera matching.

  2. Custom tensor probe (advanced): Extract appearance features from
     a secondary GIE and implement your own matching logic.

  3. NVIDIA MTMC (Multi-Target Multi-Camera): Enterprise feature in
     DeepStream SDK, requires ReID model from NGC.

=============================================================================
APPROACH 1: NvDeepSORT Tracker (Recommended Starting Point)
=============================================================================

To enable DeepSORT-style ReID:

Step 1 — Download the ReID model from NGC:
  ngc registry model download-version \
      "nvidia/tao/reidentificationnet:deployable_v1.0"

Step 2 — Replace tracker config in pipeline.yaml:
  tracker:
    config_file: configs/tracker/nvdeepsort_reid.yaml

Step 3 — Create configs/tracker/nvdeepsort_reid.yaml:
  (stub below)
"""

# ── STUB: NvDeepSORT tracker config (configs/tracker/nvdeepsort_reid.yaml) ───
NVDEEPSORT_CONFIG_STUB = """
# configs/tracker/nvdeepsort_reid.yaml
# This is a STUB — fill in actual model paths after downloading from NGC

BaseConfig:
  minDetectorConfidence: 0.2

ReIDModel:
  # Path to the ReID feature extractor model
  # TODO: Set after downloading from NGC
  reidEngineFilePath: /path/to/reid_model.engine

  # ReID feature vector dimension (typically 128 or 512)
  reidFeatureSize: 128

  # Batch size for ReID inference
  batchSize: 16

TargetManagement:
  maxTargetsPerStream: 50
  enableBboxUnClipping: true

  # Cross-camera ID assignment settings (enable for multi-camera)
  enableCrossStreamReid: false   # TODO: set true for cross-camera ReID
  reidMatchingThreshold: 0.5    # lower = stricter match

DataAssociator:
  attributeName: deepsort
"""

"""
=============================================================================
APPROACH 2: Custom Re-ID with Secondary GIE (Advanced)
=============================================================================

The hook points in the pipeline for custom ReID:

Pipeline:
  [mux] → [pgie/detector] → [nvtracker] → [sgie/reid_feature_extractor]
                                                         ↓
                                              [CustomReIDMatchingProbe]
                                                         ↓
                                                    [nvosdbin] ...

The secondary GIE (sgie) runs a ReID model on each tracked person crop.
The custom probe reads the feature vectors and performs cross-camera matching.
"""


# ── STUB: Secondary GIE config for ReID feature extraction ───────────────────
SGIE_REID_CONFIG_STUB = """
# configs/models/nvinfer_reid_sgie.yml
# STUB — fill in after downloading ReID model

property:
  gpu-id: 0
  onnx-file: /path/to/reid_backbone.onnx
  model-engine-file: /path/to/reid_backbone.engine

  # process-mode: 2 = secondary (operates on objects detected by pgie)
  process-mode: 2

  # Operate on objects detected by pgie (gie-unique-id: 1)
  operate-on-gie-id: 1

  # Only process person class (class_id=2 for TrafficCamNet)
  operate-on-class-ids: 2

  # Output raw tensor metadata instead of detection bboxes
  # This gives us the ReID feature vector
  output-tensor-meta: 1

  network-mode: 2    # FP16
  batch-size: 16     # ReID runs on object crops, can batch more
  gie-unique-id: 2   # secondary GIE uses a different unique-id
  network-type: 0
"""


# ── STUB: Custom ReID matching probe ─────────────────────────────────────────
import pyservicemaker as psm


class ReIDMatchingProbe(psm.BatchMetadataOperator):
    """
    STUB: Cross-camera person re-identification probe.

    This would:
    1. Extract ReID feature vectors from tensor metadata (written by sgie)
    2. Match features against a gallery of known person embeddings
    3. Assign a consistent global_id across cameras
    4. Update the OSD label to show the global ID

    TODO: Implement after understanding Milestones 1-8 thoroughly.
    """

    def __init__(self):
        super().__init__()
        # Gallery: maps local object_id to ReID feature vector
        # In production this would be shared across camera streams
        self._gallery: dict[int, list] = {}  # {global_id: feature_vector}

    def execute(self, batch_meta):
        """
        TODO: Extract ReID features and match against gallery.

        Pseudo-code:
            for frame_meta in batch_meta.frame_items:
                for tensor_meta in frame_meta.tensor_items:
                    feature_vector = self._extract_reid_features(tensor_meta)
                    global_id = self._match_to_gallery(feature_vector)
                    # Write global_id somewhere visible (custom user meta or label)
        """
        # STUB — not implemented
        pass

    def _extract_reid_features(self, tensor_meta) -> list:
        """
        TODO: Convert TensorOutputUserMetadata to a feature vector.

        In pyservicemaker:
            import torch
            tensor = tensor_meta.get_tensor(0).clone()
            feature = torch.utils.dlpack.from_dlpack(tensor)
            return feature.cpu().numpy().flatten().tolist()
        """
        raise NotImplementedError("TODO: implement ReID feature extraction")

    def _match_to_gallery(self, feature: list) -> int:
        """
        TODO: Cosine similarity match against the gallery.

        Return the closest gallery ID if similarity > threshold,
        otherwise add as new entry and return a new global ID.
        """
        raise NotImplementedError("TODO: implement gallery matching")


"""
=============================================================================
RESOURCES FOR IMPLEMENTING RE-ID
=============================================================================

1. NGC Models:
   - ReIdentificationNet:  ngc registry model list --query reidentification
   - NvDeepSORT:           Part of DeepStream SDK tracker

2. DeepStream SDK samples:
   /opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/
   Look for configs using deepsort or reid

3. Key reference documents (in this project):
   - LEARNING_NOTES.md § Tracker algorithms
   - configs/tracker/iou.yaml, nvdcf_perf.yaml (patterns to follow)

4. Next steps in order:
   a. Complete Milestones 1-8 first
   b. Download ReID model from NGC
   c. Create nvdeepsort_reid.yaml tracker config
   d. Switch pipeline.yaml tracker.config_file to it
   e. Observe if IDs are more stable across brief occlusions
   f. For cross-camera: implement the gallery matching in ReIDMatchingProbe
=============================================================================
"""

if __name__ == "__main__":
    print(__doc__)
    print("\nThis milestone is a conceptual guide only.")
    print("Complete Milestones 1-8 first, then return here.")
    print("\nNvDeepSORT config stub:")
    print(NVDEEPSORT_CONFIG_STUB)
    print("\nSGIE ReID config stub:")
    print(SGIE_REID_CONFIG_STUB)
