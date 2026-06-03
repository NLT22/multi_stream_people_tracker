"""
Unit tests for src/reid/geometry.py

Tests:
  1. geo_score decay: same spot → 1.0, far apart → near 0.0
  2. foot_to_world: known pinhole camera, foot point projects to expected world XY
  3. GroundPlaneGeometry with real lobby calibration data
  4. bbox_foot convenience wrapper
  5. Edge cases: unknown cam_id, ray parallel to floor
"""

import math
import json
import sys
import os

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.reid.geometry import GroundPlaneGeometry, _CLOSE_MM, _FAR_MM


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_calib(cam_id: int, K, R, t) -> dict:
    """Build a minimal calibrations.json-shaped dict for one camera."""
    return {
        "Cameras": [{
            "CameraId": cam_id,
            "IntrinsicParameters": {
                "Fx": float(K[0, 0]), "Fy": float(K[1, 1]),
                "Cx": float(K[0, 2]), "Cy": float(K[1, 2]),
            },
            "ExtrinsicParameters": {
                "Rotation": R.flatten().tolist(),
                "Translation": t.tolist(),
            },
        }],
        "Space": {"MinU": -3000, "MaxU": 3000,
                  "MinV": -3000, "MaxV": 3000,
                  "MinW": 0, "MaxW": 2500,
                  "VoxelSizeInMM": 20},
    }


# ── Test 1: geo_score ──────────────────────────────────────────────────────────

def test_geo_score_same_point():
    score = GroundPlaneGeometry.geo_score((0.0, 0.0), (0.0, 0.0))
    assert score == 1.0, f"same point should give 1.0, got {score}"

def test_geo_score_close():
    score = GroundPlaneGeometry.geo_score((0.0, 0.0), (100.0, 0.0))
    assert score > 0.95, f"100 mm apart should give >0.95, got {score}"

def test_geo_score_far():
    score = GroundPlaneGeometry.geo_score((0.0, 0.0), (3000.0, 0.0))
    assert score < 0.05, f"3 m apart should give <0.05, got {score}"

def test_geo_score_none():
    assert GroundPlaneGeometry.geo_score(None, (0.0, 0.0)) == 0.0
    assert GroundPlaneGeometry.geo_score((0.0, 0.0), None) == 0.0
    assert GroundPlaneGeometry.geo_score(None, None) == 0.0

def test_geo_score_at_close_mm():
    # At exactly _CLOSE_MM the exponent is 0 → score should be 1.0
    score = GroundPlaneGeometry.geo_score((0.0, 0.0), (_CLOSE_MM, 0.0))
    assert score == 1.0, f"at _CLOSE_MM score should be 1.0, got {score}"

def test_geo_score_monotone():
    pts = [(0.0, 0.0), (200.0, 0.0), (800.0, 0.0), (2000.0, 0.0), (4000.0, 0.0)]
    scores = [GroundPlaneGeometry.geo_score((0.0, 0.0), p) for p in pts]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i+1], \
            f"geo_score not monotone at index {i}: {scores[i]} < {scores[i+1]}"


# ── Test 2: synthetic camera — camera looking straight down at Z=0 ─────────────
#
# Camera at world (0,0,2000) looking DOWN (cam Z-axis = world -Z):
#   R = [[1,0,0],[0,-1,0],[0,0,-1]]  (180° rotation around X)
#   t = -R @ C_world = (0,0,2000)
# Then p_cam = R @ p_world + t, camera centre = R^T @ (-t) = (0,0,2000) ✓
# Principal ray in camera = (0,0,1) → in world = R^T @ (0,0,1) = (0,0,-1) ↓
# Hits Z=0 at λ=2000 → origin (0,0) ✓

def _down_camera_calib(C_world, f=300.0, cx=320.0, cy=180.0, cam_id=1):
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
    # 180° around X: cam looks along world -Z (down)
    R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
    C = np.asarray(C_world, dtype=np.float64)
    t = -R @ C
    return _make_calib(cam_id, K, R, t)


def test_foot_to_world_camera_looking_down():
    """Principal ray of a camera directly above origin should hit world (0,0)."""
    calib = _down_camera_calib([0.0, 0.0, 2000.0])
    geo = GroundPlaneGeometry(calib)

    xy = geo.foot_to_world(1, 320.0, 180.0)
    assert xy is not None, "expected a valid world point"
    assert abs(xy[0]) < 1.0 and abs(xy[1]) < 1.0, \
        f"principal ray should hit origin, got {xy}"


def test_foot_to_world_offset_pixel():
    """Pixel (cx + f, cy) → world X ≈ camera_height = 2000 mm."""
    f = 300.0
    cx, cy = 320.0, 180.0
    calib = _down_camera_calib([0.0, 0.0, 2000.0], f=f, cx=cx, cy=cy)
    geo = GroundPlaneGeometry(calib)

    # Normalised ray: x = 1, z = 1 → d_cam = (1/√2, 0, 1/√2) unnorm
    # In world (R^T flips z): d_world = (1, 0, -1) unnorm
    # C = (0,0,2000), λ = 2000/1 = 2000, X = 0 + 2000*1 = 2000
    xy = geo.foot_to_world(1, cx + f, cy)
    assert xy is not None, "expected valid world point"
    assert abs(xy[0] - 2000.0) < 2.0, f"expected X≈2000, got {xy[0]:.2f}"
    assert abs(xy[1]) < 2.0, f"expected Y≈0, got {xy[1]:.2f}"


# ── Test 3: unknown cam_id ─────────────────────────────────────────────────────

def test_unknown_cam_id():
    f = 300.0
    K = np.array([[f, 0, 320.], [0, f, 180.], [0, 0, 1.]], dtype=np.float64)
    R = np.eye(3, dtype=np.float64)
    C = np.array([0., 0., 2000.])
    calib = _make_calib(1, K, R, -R @ C)
    geo = GroundPlaneGeometry(calib)

    result = geo.foot_to_world(99, 320.0, 180.0)
    assert result is None, "unknown cam_id should return None"


# ── Test 4: ray parallel to floor (degenerate) ────────────────────────────────

def test_ray_parallel_to_floor():
    """Camera at Z=0 looking horizontally: ray parallel to floor, no intersection."""
    f = 300.0
    K = np.array([[f, 0, 320.], [0, f, 180.], [0, 0, 1.]], dtype=np.float64)
    # Rotate camera 90° around X: now camera looks along world Y (horizontal)
    R = np.array([[1, 0, 0],
                  [0, 0, -1],
                  [0, 1, 0]], dtype=np.float64)
    C = np.array([0., 0., 1000.])
    t = -R @ C
    calib = _make_calib(1, K, R, t)
    geo = GroundPlaneGeometry(calib)

    # Principal ray points along world Y — parallel to Z=0 plane
    result = geo.foot_to_world(1, 320.0, 180.0)
    assert result is None, "horizontal ray should return None"


# ── Test 5: real lobby calibration ────────────────────────────────────────────

def test_real_lobby_calibration():
    calib_path = os.path.join(
        os.path.dirname(__file__), "..",
        "dataset", "MMPTracking_short", "calibrations", "lobby", "calibrations.json"
    )
    if not os.path.exists(calib_path):
        print(f"  [SKIP] lobby calibration not found at {calib_path}")
        return

    with open(calib_path) as f:
        calib = json.load(f)

    geo = GroundPlaneGeometry(calib)
    valid_count = 0

    for cam in calib["Cameras"]:
        cid = cam["CameraId"]
        intr = cam["IntrinsicParameters"]
        # Try a few pixels along image columns (foot=bottom of bbox)
        for dy in [50, 100, 150]:
            xy = geo.foot_to_world(cid, intr["Cx"], intr["Cy"] + dy)
            if xy is None:
                continue
            x, y = xy
            assert math.isfinite(x) and math.isfinite(y), \
                f"cam {cid}: non-finite world coords {xy}"
            valid_count += 1
            print(f"  cam {cid} dy={dy}: world ({x:.0f}, {y:.0f}) mm")
            break   # one valid result per camera is enough

    # At least 1 camera must give a valid result
    assert valid_count >= 1, "No camera produced a valid ground projection"


# ── Test 6: bbox_foot wrapper ─────────────────────────────────────────────────

def test_bbox_foot():
    f = 300.0
    cx, cy = 320.0, 180.0
    calib = _down_camera_calib([0., 0., 2000.], f=f, cx=cx, cy=cy)
    geo = GroundPlaneGeometry(calib)

    # bbox centred at principal ray, bottom at cy → foot pixel = (cx, cy)
    xy_bbox = geo.bbox_foot(1, left=cx - 30, top=cy - 60, width=60, height=60)
    xy_direct = geo.foot_to_world(1, cx, cy)
    assert xy_bbox is not None and xy_direct is not None, \
        f"bbox_foot={xy_bbox}, direct={xy_direct}"
    assert abs(xy_bbox[0] - xy_direct[0]) < 1.0
    assert abs(xy_bbox[1] - xy_direct[1]) < 1.0


# ── Runner ────────────────────────────────────────────────────────────────────

def run_all():
    tests = [
        test_geo_score_same_point,
        test_geo_score_close,
        test_geo_score_far,
        test_geo_score_none,
        test_geo_score_at_close_mm,
        test_geo_score_monotone,
        test_foot_to_world_camera_looking_down,
        test_foot_to_world_offset_pixel,
        test_unknown_cam_id,
        test_ray_parallel_to_floor,
        test_real_lobby_calibration,
        test_bbox_foot,
    ]
    passed = failed = 0
    for t in tests:
        try:
            print(f"  running {t.__name__} ...", end=" ")
            t()
            print("OK")
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed  (total {passed+failed})")
    return failed == 0


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
