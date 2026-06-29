"""MTMC_Tracking_2026 camera calibration adapter (ground-plane back-projection).

Each warehouse ships a `calibration.json` (`calibrationType: cartesian`) with, per
camera, a 3x3 `intrinsicMatrix` K and a 3x4 `extrinsicMatrix` [R|t] mapping WORLD →
CAMERA (X_cam = R·X_world + t). Foot points (bottom-centre of a person bbox) lie on
the ground plane z=0, so a pixel back-projects to a unique world (x, y):

    d_cam   = K⁻¹ · [u, v, 1]
    C       = -Rᵀ · t                 (camera centre in world)
    d_world = Rᵀ · d_cam
    s       = -C_z / d_world_z         (intersect ground z=0)
    world   = (C + s · d_world)[:2]

Validated on W022: GT `3d location` vs back-projected foot point agree to ~0.14
world units. This is the geometry the MMP pipeline never had (overlapping FOV, no
metric calibration) and the reason position-first linking is viable for the disjoint
AICity warehouses but was rejected for MMP.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class WarehouseCalibration:
    def __init__(self, calib_json: str | Path):
        calib = json.load(open(calib_json))
        self._cam: dict[int, dict] = {}
        for s in calib.get("sensors", []):
            if s.get("type") != "camera":
                continue
            cam_id = int(s["id"].split("_")[-1])
            K = np.asarray(s["intrinsicMatrix"], float)
            E = np.asarray(s["extrinsicMatrix"], float)
            R = E[:, :3]
            t = E[:, 3]
            self._cam[cam_id] = {
                "K": K,
                "Kinv": np.linalg.inv(K),
                "R": R,
                "t": t,
                "Rt": R.T,
                "C": -R.T @ t,
            }

    def has(self, cam_id: int) -> bool:
        return int(cam_id) in self._cam

    def foot_to_world(self, cam_id: int, u: float, v: float) -> tuple[float, float] | None:
        c = self._cam.get(int(cam_id))
        if c is None:
            return None
        d_world = c["Rt"] @ (c["Kinv"] @ np.array([u, v, 1.0]))
        if abs(d_world[2]) < 1e-9:
            return None
        s = -c["C"][2] / d_world[2]
        if s <= 0:                       # behind the camera
            return None
        w = c["C"] + s * d_world
        return float(w[0]), float(w[1])

    def world_to_pixel(self, cam_id: int, x: float, y: float, z: float = 0.0):
        """Project a world point (x, y, z) to pixel (u, v). Returns (u, v, depth);
        depth>0 means in front of the camera. Inverse of foot_to_world (foot at z=0).
        Used to synthesize a box in a camera that missed a person from their world
        position localised by another camera (occlusion reprojection)."""
        c = self._cam.get(int(cam_id))
        if c is None:
            return None
        Xc = c["R"] @ np.array([x, y, z]) + c["t"]
        if Xc[2] <= 1e-6:
            return None
        px = c["K"] @ Xc
        return float(px[0] / px[2]), float(px[1] / px[2]), float(Xc[2])
