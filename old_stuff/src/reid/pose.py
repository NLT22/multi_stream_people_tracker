"""YOLO11n-pose ankle-keypoint extraction for pose-based foot points (#2).

NOT WIRED into the live pipeline — this is the ready-but-inactive implementation
of improvement #2 (pose-based feet). The ground-plane geometry currently uses the
bbox bottom-centre as the foot point, which is biased under occlusion, bbox
jitter, or frame-edge truncation. Ankle keypoints give a far better foot pixel;
`GroundPlaneGeometry.bbox_foot_pose()` consumes them.

Two intended consumers (neither active yet):
  1. Offline experiments — run YOLO11n-pose on frames, attach ankle keypoints to
     detections, feed `bbox_foot_pose`, and A/B the geometry vs the bbox-bottom
     baseline. This module + ultralytics is all that needs.
  2. DeepStream SGIE (production realtime) — run the pose model as a secondary
     GIE on tracked person crops (configs/models/nvinfer_yolo11n_pose_sgie.yml),
     read the keypoint tensor in a probe, and call `bbox_foot_pose`. That path
     additionally needs a YOLO-pose keypoint output parser (see the config).

ultralytics is a training-time dependency (not in the DeepStream runtime venv),
so the import is guarded: importing this module never breaks the live pipeline;
`PoseAnkleEstimator(...)` only fails if you actually try to use it without
ultralytics installed.
"""

from __future__ import annotations

# COCO-17 keypoint indices used for the foot point.
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

DEFAULT_POSE_WEIGHTS = "yolo11n-pose.pt"


class PoseAnkleEstimator:
    """Thin wrapper over ultralytics YOLO11n-pose returning ankle keypoints.

    Usage (offline experiment):

        est = PoseAnkleEstimator()
        # frame_bgr: np.ndarray HxWx3 (OpenCV BGR)
        people = est.estimate(frame_bgr)          # list of dicts per person
        for p in people:
            foot = geometry.bbox_foot_pose(
                cam_id, *p["bbox_ltwh"], keypoints=p["keypoints"])
    """

    def __init__(self, weights: str = DEFAULT_POSE_WEIGHTS,
                 conf: float = 0.25, device: str | None = None) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as e:  # pragma: no cover - depends on env
            raise ImportError(
                "PoseAnkleEstimator needs ultralytics "
                "(`pip install ultralytics`). It is a training-time dependency, "
                "not part of the DeepStream runtime venv."
            ) from e
        self._model = YOLO(weights)
        self._conf = conf
        self._device = device

    def estimate(self, frame_bgr) -> list[dict]:
        """Return per-person dicts: {bbox_ltwh, keypoints (COCO-17 x,y,conf)}.

        Only the person class is returned (YOLO-pose is person-only).
        """
        kwargs = {"conf": self._conf, "verbose": False}
        if self._device is not None:
            kwargs["device"] = self._device
        results = self._model(frame_bgr, **kwargs)
        out: list[dict] = []
        for res in results:
            if res.keypoints is None or res.boxes is None:
                continue
            kpts = res.keypoints.data.cpu().numpy()      # (N, 17, 3)
            boxes = res.boxes.xyxy.cpu().numpy()         # (N, 4)
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                out.append({
                    "bbox_ltwh": (float(x1), float(y1),
                                  float(x2 - x1), float(y2 - y1)),
                    "keypoints": [
                        (float(x), float(y), float(c)) for x, y, c in kpts[i]
                    ],
                })
        return out


def ankle_foot_pixel(bbox_ltwh: tuple[float, float, float, float],
                     keypoints: list[tuple[float, float, float]] | None,
                     conf_thresh: float = 0.3) -> tuple[float, float]:
    """Standalone foot-pixel helper mirroring GroundPlaneGeometry.foot_pixel.

    Kept here too so offline scripts can compute the pose foot point without a
    calibration object.
    """
    left, top, width, height = bbox_ltwh
    if keypoints is not None:
        ankles = [
            (keypoints[i][0], keypoints[i][1])
            for i in (LEFT_ANKLE, RIGHT_ANKLE)
            if i < len(keypoints) and keypoints[i][2] >= conf_thresh
        ]
        if ankles:
            return (
                sum(a[0] for a in ankles) / len(ankles),
                sum(a[1] for a in ankles) / len(ankles),
            )
    return left + width / 2.0, top + height
