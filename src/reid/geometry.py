"""
Ground-plane geometry helpers for calibration-assisted cross-camera matching.

Given MMPTracking calibration (pinhole, no distortion), projects person foot
points from 2D image space → world XY (mm, Z=0 floor plane).  The resulting
world positions can be compared across cameras so that two tracks whose feet
are close in 3D receive a geometry bonus in the ReID similarity score.

Coordinate conventions (matching calibrations.json):
  - Camera model: pinhole,  p = K @ (R @ P_world + t)
  - World axes: X=U (mm), Y=V (mm), Z=W (mm, up = negative)
  - Floor = Z=0 plane (persons stand on it)
  - Translation stored in calibrations.json is t in camera space (not camera centre)

Usage:
    calib = ds.load_calibration()
    geo = GroundPlaneGeometry(calib)
    world_xy = geo.foot_to_world(cam_id, pixel_u, pixel_v)   # (X, Y) mm or None
    score    = geo.geo_score(world_xy_a, world_xy_b)          # 0-1 float
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


# Distance (mm) at which two tracks are considered "same location" → score = 1.0
_CLOSE_MM = 300.0    # 30 cm
# Distance at which the score decays to ~0.  Persons > 2 m apart get near-zero bonus.
_FAR_MM   = 2000.0   # 2 m


class GroundPlaneGeometry:
    """
    Per-environment calibration geometry.

    Args:
        calibration: dict returned by MMPTrackingShortDataset.load_calibration()
                     Must contain "Cameras" list with IntrinsicParameters and
                     ExtrinsicParameters per camera.
    """

    def __init__(self, calibration: dict) -> None:
        self._cams: dict[int, dict] = {}   # cam_id → {"K", "R", "t"}
        for cam in calibration.get("Cameras", []):
            cid = cam["CameraId"]
            intr = cam["IntrinsicParameters"]
            extr = cam["ExtrinsicParameters"]

            K = np.array([
                [intr["Fx"], 0.0,        intr["Cx"]],
                [0.0,        intr["Fy"], intr["Cy"]],
                [0.0,        0.0,        1.0       ],
            ], dtype=np.float64)

            R = np.array(extr["Rotation"], dtype=np.float64).reshape(3, 3)
            t = np.array(extr["Translation"], dtype=np.float64)   # shape (3,)

            self._cams[cid] = {"K": K, "R": R, "t": t}

    def has_camera(self, cam_id: int) -> bool:
        return cam_id in self._cams

    def foot_to_world(
        self, cam_id: int, u: float, v: float
    ) -> Optional[tuple[float, float]]:
        """
        Unproject pixel (u, v) onto the Z=0 ground plane.

        Returns (X_mm, Y_mm) in world coordinates, or None if the ray is
        parallel to the floor (degenerate camera pointing straight up/down).
        """
        cam = self._cams.get(cam_id)
        if cam is None:
            return None

        K, R, t = cam["K"], cam["R"], cam["t"]

        # Ray direction in camera space (normalised)
        K_inv = np.linalg.inv(K)
        ray_cam = K_inv @ np.array([u, v, 1.0], dtype=np.float64)

        # Transform ray into world space.
        # p_cam = R @ p_world + t  →  p_world = R^T @ (p_cam - t)
        # Camera centre in world:  C = -R^T @ t
        # Ray in world direction:  d = R^T @ ray_cam  (unit direction)
        R_t = R.T
        C_world = -R_t @ t                    # camera centre (mm)
        d_world = R_t @ ray_cam               # ray direction in world

        # Intersect with Z=0: C_world[2] + λ * d_world[2] = 0
        dz = d_world[2]
        if abs(dz) < 1e-6:
            return None   # ray nearly parallel to floor

        lam = -C_world[2] / dz
        if lam < 0:
            return None   # intersection is behind the camera

        X = C_world[0] + lam * d_world[0]
        Y = C_world[1] + lam * d_world[1]
        return float(X), float(Y)

    @staticmethod
    def geo_score(
        xy_a: Optional[tuple[float, float]],
        xy_b: Optional[tuple[float, float]],
    ) -> float:
        """
        Geometry similarity score in [0, 1].

        Returns 0.0 if either position is unknown.
        Decays from 1.0 (same spot) to 0.0 (> _FAR_MM apart).
        Uses a smooth Gaussian-like decay so small errors don't hard-clip.
        """
        if xy_a is None or xy_b is None:
            return 0.0
        dx = xy_a[0] - xy_b[0]
        dy = xy_a[1] - xy_b[1]
        dist = math.sqrt(dx * dx + dy * dy)

        # Normalize: 0 at _CLOSE_MM, 1 at _FAR_MM
        t = max(0.0, (dist - _CLOSE_MM) / max(1.0, _FAR_MM - _CLOSE_MM))
        # Smooth decay: score = exp(-3 * t^2)  →  1.0 at t=0, ~0.05 at t=1
        return math.exp(-3.0 * t * t)

    def bbox_foot(
        self,
        cam_id: int,
        left: float, top: float,
        width: float, height: float,
    ) -> Optional[tuple[float, float]]:
        """
        Convenience: foot point = bottom-centre of bbox, unprojected to world.
        """
        u = left + width / 2.0
        v = top + height
        return self.foot_to_world(cam_id, u, v)

    # --- Pose-based foot point (#2, NOT wired into the live pipeline yet) -------
    # The bbox bottom-centre is a biased foot estimate under occlusion, bbox
    # jitter, or frame-edge truncation. When ankle keypoints are available
    # (e.g. from a YOLO11n-pose SGIE — see src/reid/pose.py), the midpoint of the
    # confident ankles is a far better foot pixel. Falls back to bbox-bottom when
    # no ankle clears the confidence gate, so it is always at least as good.
    @staticmethod
    def foot_pixel(
        left: float, top: float, width: float, height: float,
        keypoints: "list[tuple[float, float, float]] | None" = None,
        conf_thresh: float = 0.3,
    ) -> tuple[float, float]:
        """Foot pixel (u, v): mean of confident ankles, else bbox bottom-centre.

        keypoints: COCO-17 [(x, y, conf), ...] in image pixels;
        ankle indices are 15 (left) and 16 (right).
        """
        if keypoints is not None:
            ankles = [
                (keypoints[i][0], keypoints[i][1])
                for i in (15, 16)
                if i < len(keypoints) and keypoints[i][2] >= conf_thresh
            ]
            if ankles:
                return (
                    sum(a[0] for a in ankles) / len(ankles),
                    sum(a[1] for a in ankles) / len(ankles),
                )
        return left + width / 2.0, top + height

    def bbox_foot_pose(
        self,
        cam_id: int,
        left: float, top: float,
        width: float, height: float,
        keypoints: "list[tuple[float, float, float]] | None" = None,
        conf_thresh: float = 0.3,
    ) -> Optional[tuple[float, float]]:
        """Pose-aware variant of bbox_foot: ankle-based foot pixel → world."""
        u, v = self.foot_pixel(left, top, width, height, keypoints, conf_thresh)
        return self.foot_to_world(cam_id, u, v)
