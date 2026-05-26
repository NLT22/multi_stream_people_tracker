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
  YOLOv8 (people-only model): class_id 0 = Person
  The PERSON_CLASS_ID constant below must match your model.
"""

import pyservicemaker as psm


# ── Constants ──────────────────────────────────────────────────────────────────
PERSON_CLASS_ID_TRAFFICCAMNET = 2   # TrafficCamNet label index for "Person"
PERSON_CLASS_ID_PEOPLE_ONLY   = 0   # YOLOv8 people-only model


class PersonCountProbe(psm.BatchMetadataOperator):
    """
    Milestone 8: Count detected/tracked persons per frame and print to console.

    Attach this to nvtracker (or nvinfer if no tracker) with:
        pipeline.attach("tracker", psm.Probe("count_probe", PersonCountProbe()))
    """

    PERSON_CLASS_ID = PERSON_CLASS_ID_TRAFFICCAMNET

    def handle_metadata(self, batch_meta):
        """
        Called once per buffer (one batch of frames from nvstreammux).
        Each batch contains one frame per source stream.
        """
        for frame_meta in batch_meta.frame_items:  # ITERATOR — not a list!
            person_count = 0

            for obj_meta in frame_meta.object_items:  # ITERATOR — not a list!
                if obj_meta.class_id == self.PERSON_CLASS_ID:
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
    Milestone 6: Draw custom bounding box labels for tracked persons.

    This probe runs BEFORE nvosdbin. It adds text overlays to DisplayMeta
    which nvosdbin then renders onto the video.

    Attach this between tracker and osd:
        pipeline.attach("tracker", psm.Probe("osd_probe", PersonOSDProbe()))

    WHY NOT USE nvosdbin's built-in display?
      nvosdbin auto-draws class labels from the label file. The custom probe
      lets you control the exact text, color, and position — for example,
      showing "Person #42 (conf=0.87)" instead of just "Person".
    """

    PERSON_CLASS_ID = PERSON_CLASS_ID_TRAFFICCAMNET

    def handle_metadata(self, batch_meta):
        for frame_meta in batch_meta.frame_items:
            display_meta = psm.DisplayMeta(frame_meta)

            for obj_meta in frame_meta.object_items:
                if obj_meta.class_id != self.PERSON_CLASS_ID:
                    # TODO (Milestone 6): Decide whether to show non-person classes
                    # Options: skip them, use a different color, etc.
                    continue

                # Build label text — object_id is the tracker-assigned ID
                # NOTE: object_id is 0 until Milestone 5 adds nvtracker
                label = f"Person #{obj_meta.object_id}"

                # TODO (Milestone 6): Add confidence to label
                # label = f"Person #{obj_meta.object_id} ({obj_meta.confidence:.0%})"

                # Text position: top-left corner of the bounding box
                x = int(obj_meta.rect_params.left)
                y = max(0, int(obj_meta.rect_params.top) - 20)  # above the box

                display_meta.add_text(
                    psm.Text(
                        label,
                        x=x,
                        y=y,
                        font=psm.Font(psm.FontFamily.Sans, 14),
                        color=psm.Color(0.0, 1.0, 0.0, 1.0),  # green RGBA
                    )
                )

                # TODO (Milestone 6): Add a custom colored rectangle instead of
                # relying on nvosdbin's default rectangle rendering
                # display_meta.add_rect(psm.Rect(...))


class MetadataExtractorProbe(psm.BatchMetadataOperator):
    """
    Milestone 8: Full metadata traversal for learning/debugging.

    Attach to nvinfer or nvtracker to see all metadata fields.
    Use this to understand the data model before building real logic.
    """

    PERSON_CLASS_ID = PERSON_CLASS_ID_TRAFFICCAMNET

    def __init__(self):
        super().__init__()
        self._frame_count = 0

    def handle_metadata(self, batch_meta):
        for frame_meta in batch_meta.frame_items:
            self._frame_count += 1

            persons = []
            for obj_meta in frame_meta.object_items:
                if obj_meta.class_id == self.PERSON_CLASS_ID:
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
