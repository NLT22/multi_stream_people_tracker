"""MMPTracking base adapter for TrackTacular (Wildtrack-compatible interface).

Reads a directory produced by scripts/tracktacular/mmp_to_worldtrack.py:
    <root>/calibrations.json
    <root>/Image_subsets/C{1..N}/<frame:08d>.png
    <root>/annotations_positions/<frame:08d>.json

Provides the same attributes/methods PedestrianDataset consumes from Wildtrack.
"""
import os
import glob
import json
import re
import numpy as np
import cv2

GRID_NY = 256  # must match the converter's positionID encoding

# Per-environment grid->world(mm) affine, fitted from foot-point projections
# (maps [gx, gy, 1] -> [world_x_mm, world_y_mm, 1]). See scripts/tracktacular.
_AFFINE_JSON = os.path.join(os.path.dirname(__file__), "affines.json")
if os.path.exists(_AFFINE_JSON):
    _data = json.load(open(_AFFINE_JSON))
    WORLDCOORD_FROM_WORLDGRID = {k: np.array(v["affine"]) for k, v in _data.items()}
else:
    WORLDCOORD_FROM_WORLDGRID = {
        "industry_safety": np.array([[1.706, 33.103, -6301.5],
                                     [36.942, 3.970, -10626.8],
                                     [0.0, 0.0, 1.0]]),
    }
DEFAULT_WORLDCOORD = WORLDCOORD_FROM_WORLDGRID["industry_safety"]


class Mmptracking:
    def __init__(self, root):
        self.root = root
        self.__name__ = 'MMPTracking'
        calib = json.load(open(os.path.join(root, 'calibrations.json')))
        self.cameras = sorted(calib['Cameras'], key=lambda c: c['CameraId'])
        self.num_cam = len(self.cameras)

        # frames present (from annotation files); infer step + count
        ann = sorted(int(os.path.basename(f)[:-5])
                     for f in glob.glob(os.path.join(root, 'annotations_positions', '*.json')))
        self.frame_step = (ann[1] - ann[0]) if len(ann) > 1 else 1
        self.num_frame = ann[-1] + self.frame_step if ann else 0

        # image shape from a sample frame
        sample = sorted(glob.glob(os.path.join(root, 'Image_subsets', 'C1', '*.png')))[0]
        h, w = cv2.imread(sample).shape[:2]
        self.img_shape = [h, w]                 # H,W
        self.worldgrid_shape = [GRID_NY, GRID_NY]  # N_row(gy), N_col(gx)

        env = self._env_from_root(root)
        self.worldcoord_from_worldgrid_mat = WORLDCOORD_FROM_WORLDGRID.get(
            env, DEFAULT_WORLDCOORD)

        self.intrinsic_matrices, self.extrinsic_matrices = zip(
            *[self._intr_extr(c) for c in self.cameras])

    @staticmethod
    def _env_from_root(root):
        name = os.path.basename(os.path.normpath(root))
        for env in WORLDCOORD_FROM_WORLDGRID:
            if env in name:
                return env
        return None

    @staticmethod
    def _intr_extr(cam):
        ip = cam['IntrinsicParameters']
        K = np.array([[ip['Fx'], 0, ip['Cx']],
                      [0, ip['Fy'], ip['Cy']],
                      [0, 0, 1]], dtype=np.float32)
        ep = cam['ExtrinsicParameters']
        R = np.array(ep['Rotation'], dtype=np.float32).reshape(3, 3)
        t = np.array(ep['Translation'], dtype=np.float32).reshape(3, 1)
        Rt = np.hstack((R, t))                   # world->cam, mm
        return K, Rt

    def get_image_fpaths(self, frame_range):
        fpaths = {cam: {} for cam in range(self.num_cam)}
        for cam in range(self.num_cam):
            for fp in glob.glob(os.path.join(self.root, 'Image_subsets',
                                             f'C{cam + 1}', '*.png')):
                frame = int(os.path.basename(fp)[:-4])
                if frame in frame_range:
                    fpaths[cam][frame] = fp
        return fpaths

    def get_worldgrid_from_pos(self, pos):
        pos = int(pos)
        grid_y = pos % GRID_NY
        grid_x = pos // GRID_NY
        return np.array([grid_x, grid_y], dtype=int)
