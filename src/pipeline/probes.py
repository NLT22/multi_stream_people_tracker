"""
Probe callbacks for metadata inspection and OSD annotation.

WHY PROBES EXIST:
  In GStreamer/DeepStream, you cannot modify the pipeline mid-stream by
  inserting processing steps. Instead, you attach "probes" — callbacks
  that fire every time a buffer passes through a pad. Probes let you:
    - Read detection/tracking metadata (object boxes, IDs, confidence)
    - Add custom OSD overlays (text, rectangles) via DisplayMeta
    - Filter or modify metadata before the next element sees it
    - Log statistics without changing the pipeline structure

  pyservicemaker wraps this in BatchMetadataOperator:
    class MyProbe(psm.BatchMetadataOperator):
        def handle_metadata(self, batch_meta):
            ...  # runs once per buffer (= once per batch of frames)

HOW ITERATORS WORK (CRITICAL):
  batch_meta.frame_items   → iterator, NOT a list  → no len(), no second pass
  frame_meta.object_items  → iterator, NOT a list  → same rules
  If you need multiple passes, convert once: objects = list(frame_meta.object_items)
  But avoid this — it allocates memory for every frame.

PERSON CLASS IDs:
  TrafficCamNet: class_id 2 = Person
  YOLOv8 COCO / PeopleNet: class_id 0 = Person
  Pass the selected model's person class id into each reusable probe.
"""

import pyservicemaker as psm

from src.pipeline.model_utils import set_object_label


# ── Constants ──────────────────────────────────────────────────────────────────
PERSON_CLASS_ID_TRAFFICCAMNET = 2   # TrafficCamNet label index for "Person"
PERSON_CLASS_ID_COCO          = 0   # COCO label index for "person"
PERSON_CLASS_ID_DEFAULT       = PERSON_CLASS_ID_COCO


class PersonCountProbe(psm.BatchMetadataOperator):
    """
    Milestone 8: Count detected/tracked persons per frame and print to console.

    Attach this to nvtracker (or nvinfer if no tracker) with:
        pipeline.attach("tracker", psm.Probe("count_probe", PersonCountProbe()))
    """

    def __init__(self, person_class_id: int = PERSON_CLASS_ID_DEFAULT):
        super().__init__()
        self._person_class_id = person_class_id

    def handle_metadata(self, batch_meta):
        """
        Called once per buffer (one batch of frames from nvstreammux).
        Each batch contains one frame per source stream.
        """
        for frame_meta in batch_meta.frame_items:  # ITERATOR — not a list!
            person_count = 0

            for obj_meta in frame_meta.object_items:  # ITERATOR — not a list!
                if obj_meta.class_id == self._person_class_id:
                    person_count += 1

                    # TODO (Milestone 8): Log tracking ID, bounding box, confidence
                    # print(
                    #     f"  object_id={obj_meta.object_id}"
                    #     f"  conf={obj_meta.confidence:.2f}"
                    #     f"  rect=({obj_meta.rect_params.left:.0f},"
                    #     f"{obj_meta.rect_params.top:.0f},"
                    #     f"{obj_meta.rect_params.width:.0f},"
                    #     f"{obj_meta.rect_params.height:.0f})"
                    # )

            print(
                f"[src={frame_meta.source_id:02d} "
                f"frame={frame_meta.frame_number:06d}] "
                f"persons={person_count}"
            )


class PersonOSDProbe(psm.BatchMetadataOperator):
    """
    Milestone 6: Override built-in object labels for tracked persons.

    This probe runs BEFORE nvosdbin. It updates ObjectMetadata.label so the
    built-in per-object OSD text is stable even with many people on screen.

    Attach this between tracker and osd:
        pipeline.attach("tracker", psm.Probe("osd_probe", PersonOSDProbe()))

    WHY NOT USE extra DisplayMeta text?
      DisplayMeta text has a small per-meta capacity. Updating the object label
      avoids dropped labels when a tiled frame contains many people.
    """

    def __init__(self, person_class_id: int = PERSON_CLASS_ID_DEFAULT):
        super().__init__()
        self._person_class_id = person_class_id

    def handle_metadata(self, batch_meta):
        for frame_meta in batch_meta.frame_items:
            for obj_meta in frame_meta.object_items:
                if obj_meta.class_id != self._person_class_id:
                    # TODO (Milestone 6): Decide whether to show non-person classes
                    # Options: skip them, use a different color, etc.
                    continue

                # Build label text — object_id is the tracker-assigned ID
                # NOTE: object_id is 0 until Milestone 5 adds nvtracker
                label = f"Person #{obj_meta.object_id}"

                # TODO (Milestone 6): Add confidence to label
                # label = f"Person #{obj_meta.object_id} ({obj_meta.confidence:.0%})"
                set_object_label(obj_meta, label)


class MetadataExtractorProbe(psm.BatchMetadataOperator):
    """
    Milestone 8: Full metadata traversal for learning/debugging.

    Attach to nvinfer or nvtracker to see all metadata fields.
    Use this to understand the data model before building real logic.
    """

    def __init__(self, person_class_id: int = PERSON_CLASS_ID_DEFAULT):
        super().__init__()
        self._person_class_id = person_class_id
        self._frame_count = 0


    def handle_metadata(self, batch_meta):
        for frame_meta in batch_meta.frame_items:
            self._frame_count += 1

            persons = []
            for obj_meta in frame_meta.object_items:
                if obj_meta.class_id == self._person_class_id:
                    persons.append({
                        "object_id": obj_meta.object_id,         # tracker-assigned ID
                        "confidence": round(obj_meta.confidence, 3),
                        "class_id": obj_meta.class_id,
                        "label": obj_meta.label,                 # string label from labelfile
                        "left": round(obj_meta.rect_params.left, 1),
                        "top": round(obj_meta.rect_params.top, 1),
                        "width": round(obj_meta.rect_params.width, 1),
                        "height": round(obj_meta.rect_params.height, 1),
                    })

            if persons:
                print(
                    f"[src={frame_meta.source_id} frame={frame_meta.frame_number}] "
                    f"{len(persons)} person(s):"
                )
                for p in persons:
                    print(
                        f"  ID={p['object_id']:4d}  conf={p['confidence']:.3f}"
                        f"  box=({p['left']:.0f},{p['top']:.0f},"
                        f"{p['width']:.0f}x{p['height']:.0f})"
                    )

            # TODO (Milestone 8): Save to JSON/CSV
            # import json
            # with open(f"output/frame_{self._frame_count:06d}.json", "w") as f:
            #     json.dump({"source": frame_meta.source_id,
            #                "frame": frame_meta.frame_number,
            #                "persons": persons}, f)
