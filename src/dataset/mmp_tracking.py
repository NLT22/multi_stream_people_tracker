"""
MMPTracking dataset loader.

Folder layout (validation split):
    <root>/validation/images/64pm/<scene>.zip   → rgb_{frame:05d}_{cam_id}.jpg
    <root>/validation/labels/64pm/<scene>.zip   → rgb_{frame:05d}_{cam_id}.json
    <root>/validation/calibrations/<env>/calibrations.json

Each label JSON:  { "<person_id>": [x1, y1, x2, y2], ... }
Filename pattern: rgb_{frame_no:05d}_{cam_id}.{jpg|json}

Extracted layout (produced by extract_scene):
    <extract_root>/<scene>/      → all .jpg files flat
    <extract_root>/<scene>_labels/  → all .json files flat

Videos created by create_scene_videos:
    <videos_root>/<scene>_cam{cam_id}.mp4
"""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path
from typing import Iterator

import pandas as pd


class MMPTrackingDataset:
    """
    Loader for one scene of the MMPTracking validation split.

    Args:
        root:   Path to dataset root containing validation/images/64pm/ etc.
        scene:  Scene name, e.g. "lobby_0", "cafe_shop_2".
        split:  Subfolder inside validation/images/ (default "64pm").
        extract_root: Where to look for / write extracted frames + labels.
                      Defaults to <root>/extracted/.
        videos_root:  Where to look for pre-built scene MP4s.
                      Defaults to <root>/videos/.
    """

    IMG_W = 640
    IMG_H = 360

    def __init__(
        self,
        root: str,
        scene: str,
        split: str = "64pm",
        extract_root: str | None = None,
        videos_root: str | None = None,
    ) -> None:
        self.root = Path(root)
        self.scene = scene
        self.split = split

        # Support both flat root (with validation/ subdir) and
        # root that already contains MMPTracking_validation/validation/
        val_base = self.root / "MMPTracking_validation" / "validation"
        if not val_base.exists():
            val_base = self.root / "validation"
        self._images_zip = val_base / "images" / split / f"{scene}.zip"
        self._labels_zip = val_base / "labels" / split / f"{scene}.zip"
        self._calibrations_dir = val_base / "calibrations"

        self._extract_root = Path(extract_root) if extract_root else self.root / "extracted"
        self._videos_root = Path(videos_root) if videos_root else self.root / "videos"

        self._img_dir = self._extract_root / scene / scene
        self._lbl_dir = self._extract_root / f"{scene}_labels" / scene

        self._cam_ids: list[int] | None = None

    # ------------------------------------------------------------------
    # Camera IDs
    # ------------------------------------------------------------------

    def get_cam_ids(self) -> list[int]:
        """Return sorted list of camera IDs present in the scene."""
        if self._cam_ids is not None:
            return self._cam_ids

        ids: set[int] = set()
        if self._img_dir.exists():
            for p in self._img_dir.iterdir():
                if p.suffix == ".jpg":
                    cam = int(p.stem.rsplit("_", 1)[-1])
                    ids.add(cam)
        elif self._images_zip.exists():
            with zipfile.ZipFile(self._images_zip) as zf:
                for name in zf.namelist():
                    if name.endswith(".jpg"):
                        stem = Path(name).stem
                        cam = int(stem.rsplit("_", 1)[-1])
                        ids.add(cam)
        else:
            raise FileNotFoundError(
                f"Neither extracted dir nor zip found for scene '{self.scene}'. "
                f"Run extract_scene() first or provide images zip at {self._images_zip}"
            )
        self._cam_ids = sorted(ids)
        return self._cam_ids

    # ------------------------------------------------------------------
    # Video sources
    # ------------------------------------------------------------------

    def get_video_paths(self) -> list[Path]:
        """Return sorted list of MP4 paths for each camera in this scene."""
        paths = []
        for cam_id in self.get_cam_ids():
            p = self._videos_root / f"{self.scene}_cam{cam_id}.mp4"
            if not p.exists():
                raise FileNotFoundError(
                    f"Video not found: {p}. "
                    f"Run create_scene_videos() first."
                )
            paths.append(p)
        return paths

    def get_video_uris(self) -> list[str]:
        return [f"file://{p.resolve()}" for p in self.get_video_paths()]

    # ------------------------------------------------------------------
    # Ground-truth annotations
    # ------------------------------------------------------------------

    def load_gt(self, cam_id: int) -> pd.DataFrame:
        """
        Load GT annotations for one camera.

        Returns DataFrame with columns:
            frame (int), person_id (int),
            left (float), top (float), width (float), height (float)
        """
        if not self._lbl_dir.exists():
            raise FileNotFoundError(
                f"Label directory not found: {self._lbl_dir}. "
                f"Run extract_scene(labels=True) first."
            )

        rows = []
        for json_path in sorted(self._lbl_dir.glob(f"rgb_*_{cam_id}.json")):
            stem = json_path.stem                  # rgb_NNNNN_C
            parts = stem.split("_")                # ['rgb', 'NNNNN', 'C']
            frame_no = int(parts[1])

            with open(json_path) as f:
                ann = json.load(f)

            for pid_str, box in ann.items():
                x1, y1, x2, y2 = box
                rows.append({
                    "frame": frame_no,
                    "person_id": int(pid_str),
                    "left": float(x1),
                    "top": float(y1),
                    "width": float(x2 - x1),
                    "height": float(y2 - y1),
                })

        if not rows:
            raise ValueError(
                f"No label files found for cam {cam_id} in {self._lbl_dir}"
            )

        df = pd.DataFrame(rows)
        return df.sort_values("frame").reset_index(drop=True)

    def load_all_gt(self) -> dict[int, pd.DataFrame]:
        """Load GT for all cameras. Returns {cam_id: DataFrame}."""
        return {cam_id: self.load_gt(cam_id) for cam_id in self.get_cam_ids()}

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def load_calibration(self) -> dict:
        """
        Load calibration for this scene's environment.

        Returns the parsed calibrations.json dict:
            { "Cameras": [ { "CameraId": N, "ExtrinsicParameters": {...},
                              "IntrinsicParameters": {...} }, ... ] }
        """
        env = self.scene.rsplit("_", 1)[0]   # e.g. "lobby_0" → "lobby"
        cal_path = self._calibrations_dir / env / "calibrations.json"
        if not cal_path.exists():
            raise FileNotFoundError(f"Calibration not found: {cal_path}")
        with open(cal_path) as f:
            return json.load(f)

    def get_camera_calibration(self, cam_id: int) -> dict | None:
        """Return calibration dict for a specific camera ID, or None."""
        cal = self.load_calibration()
        for cam in cal.get("Cameras", []):
            if cam.get("CameraId") == cam_id:
                return cam
        return None

    # ------------------------------------------------------------------
    # Extraction helpers
    # ------------------------------------------------------------------

    def extract_scene(self, images: bool = True, labels: bool = True) -> None:
        """
        Extract this scene's zip(s) to extract_root.

        images → extract_root/<scene>/<scene>/*.jpg
        labels → extract_root/<scene>_labels/<scene>/*.json
        """
        if images:
            if not self._images_zip.exists():
                raise FileNotFoundError(f"Images zip not found: {self._images_zip}")
            out_dir = self._extract_root / self.scene
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"[MMPTracking] Extracting images for '{self.scene}' → {out_dir}")
            with zipfile.ZipFile(self._images_zip) as zf:
                zf.extractall(out_dir)
            print(f"[MMPTracking] Images extracted.")

        if labels:
            if not self._labels_zip.exists():
                raise FileNotFoundError(f"Labels zip not found: {self._labels_zip}")
            out_dir = self._extract_root / f"{self.scene}_labels"
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"[MMPTracking] Extracting labels for '{self.scene}' → {out_dir}")
            with zipfile.ZipFile(self._labels_zip) as zf:
                zf.extractall(out_dir)
            print(f"[MMPTracking] Labels extracted.")

    def create_scene_videos(
        self,
        fps: int = 25,
        max_seconds: int | None = None,
        overwrite: bool = False,
    ) -> list[Path]:
        """
        Create one MP4 per camera from extracted frames using ffmpeg.

        Frames must be extracted first (extract_scene(images=True)).
        Returns list of created video paths.
        """
        import subprocess

        if not self._img_dir.exists():
            raise FileNotFoundError(
                f"Frames not extracted yet. Run extract_scene(images=True) first. "
                f"Expected at: {self._img_dir}"
            )

        self._videos_root.mkdir(parents=True, exist_ok=True)
        created = []

        for cam_id in self.get_cam_ids():
            out_path = self._videos_root / f"{self.scene}_cam{cam_id}.mp4"
            if out_path.exists() and not overwrite:
                print(f"[MMPTracking] {out_path.name} already exists, skipping.")
                created.append(out_path)
                continue

            pattern = str(self._img_dir / f"rgb_%05d_{cam_id}.jpg")
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", pattern,
                "-c:v", "libx264", "-crf", "20", "-preset", "fast",
                "-pix_fmt", "yuv420p",
            ]
            if max_seconds:
                cmd += ["-t", str(max_seconds)]
            cmd.append(str(out_path))

            print(f"[MMPTracking] Creating {out_path.name} ...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg failed for cam {cam_id}:\n{result.stderr}"
                )
            print(f"[MMPTracking] Done: {out_path}")
            created.append(out_path)

        return created

    # ------------------------------------------------------------------
    # Iterator over all available scenes
    # ------------------------------------------------------------------

    @staticmethod
    def list_scenes(root: str, split: str = "64pm") -> list[str]:
        """Return sorted list of scene names in the given split."""
        r = Path(root)
        # Try MMPTracking_validation/validation first, then plain validation/
        for candidate in [
            r / "MMPTracking_validation" / "validation" / "images" / split,
            r / "validation" / "images" / split,
        ]:
            if candidate.exists():
                return sorted(p.stem for p in candidate.glob("*.zip"))
        return []

    @staticmethod
    def iter_scenes(
        root: str,
        split: str = "64pm",
        **kwargs,
    ) -> Iterator["MMPTrackingDataset"]:
        """Yield one MMPTrackingDataset per available scene."""
        for scene in MMPTrackingDataset.list_scenes(root, split):
            yield MMPTrackingDataset(root, scene, split=split, **kwargs)


class MMPTrackingShortDataset:
    """
    Loader for MMPTracking_short — the pre-built 1-minute clips dataset.

    Layout (created by scripts/create_mmp_short.py):
        <root>/
            <scene>/
                cam<N>.mp4
                gt_cam<N>.csv          ← frame,person_id,left,top,width,height
            calibrations/<env>/calibrations.json
            manifest.json

    Usage:
        ds = MMPTrackingShortDataset("dataset/MMPTracking_short", "lobby_0")
        uris   = ds.get_video_uris()
        gt     = ds.load_all_gt()    # {cam_id: DataFrame}
    """

    IMG_W = 640
    IMG_H = 360

    def __init__(self, root: str, scene: str) -> None:
        self.root = Path(root)
        self.scene = scene
        self._scene_dir = self.root / scene
        if not self._scene_dir.exists():
            raise FileNotFoundError(
                f"Scene directory not found: {self._scene_dir}"
            )
        self._cam_ids: list[int] | None = None

    # ------------------------------------------------------------------
    # Camera IDs
    # ------------------------------------------------------------------

    def get_cam_ids(self) -> list[int]:
        if self._cam_ids is not None:
            return self._cam_ids
        ids = set()
        for p in self._scene_dir.glob("cam*.mp4"):
            ids.add(int(p.stem[3:]))   # cam4 → 4
        self._cam_ids = sorted(ids)
        return self._cam_ids

    # ------------------------------------------------------------------
    # Video sources
    # ------------------------------------------------------------------

    def get_video_paths(self) -> list[Path]:
        paths = []
        for cam_id in self.get_cam_ids():
            p = self._scene_dir / f"cam{cam_id}.mp4"
            if not p.exists():
                raise FileNotFoundError(f"Video not found: {p}")
            paths.append(p)
        return paths

    def get_video_uris(self) -> list[str]:
        return [f"file://{p.resolve()}" for p in self.get_video_paths()]

    # ------------------------------------------------------------------
    # Ground-truth annotations
    # ------------------------------------------------------------------

    def load_gt(self, cam_id: int) -> pd.DataFrame:
        """Load pre-built GT CSV for one camera."""
        csv_path = self._scene_dir / f"gt_cam{cam_id}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"GT CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
        return df.sort_values("frame").reset_index(drop=True)

    def load_all_gt(self) -> dict[int, pd.DataFrame]:
        return {cam_id: self.load_gt(cam_id) for cam_id in self.get_cam_ids()}

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def load_calibration(self) -> dict:
        env = self.scene.rsplit("_", 1)[0]
        cal_path = self.root / "calibrations" / env / "calibrations.json"
        if not cal_path.exists():
            raise FileNotFoundError(f"Calibration not found: {cal_path}")
        with open(cal_path) as f:
            return json.load(f)

    def get_camera_calibration(self, cam_id: int) -> dict | None:
        cal = self.load_calibration()
        for cam in cal.get("Cameras", []):
            if cam.get("CameraId") == cam_id:
                return cam
        return None

    # ------------------------------------------------------------------
    # Scene listing
    # ------------------------------------------------------------------

    @staticmethod
    def list_scenes(root: str) -> list[str]:
        """Return sorted scene names found in this short dataset root."""
        r = Path(root)
        return sorted(
            p.name for p in r.iterdir()
            if p.is_dir() and not p.name.startswith(".")
            and p.name != "calibrations"
            and any(p.glob("cam*.mp4"))
        )

    @staticmethod
    def iter_scenes(root: str) -> Iterator["MMPTrackingShortDataset"]:
        for scene in MMPTrackingShortDataset.list_scenes(root):
            yield MMPTrackingShortDataset(root, scene)
