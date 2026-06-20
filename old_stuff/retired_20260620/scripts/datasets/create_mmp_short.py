"""
Create MMPTracking_short: 1-minute clips with GT for all 24 validation scenes.

Tự extract từ raw zip files — không cần extract trước.

Input layout (raw dataset):
    dataset/MMPTracking/MMPTracking_validation/validation/
        images/64pm/<scene>.zip
        labels/64pm/<scene>.zip
        calibrations/<env>/calibrations.json

Output layout:
    dataset/MMPTracking_short/
        <scene>/
            cam<N>.mp4          ← 1-min video (frames 0–max_frames, 25fps)
            gt_cam<N>.csv       ← frame,person_id,left,top,width,height
        calibrations/<env>/calibrations.json

Run:
    python scripts/datasets/create_mmp_short.py
    python scripts/datasets/create_mmp_short.py --mmp-root dataset/MMPTracking --jobs 3
    python scripts/datasets/create_mmp_short.py --scenes lobby_0 lobby_1 cafe_shop_0
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ALL_SCENES = [
    "cafe_shop_0", "cafe_shop_1", "cafe_shop_2", "cafe_shop_3",
    "industry_safety_0", "industry_safety_1", "industry_safety_2",
    "industry_safety_3", "industry_safety_4",
    "lobby_0", "lobby_1", "lobby_2", "lobby_3",
    "office_0", "office_1", "office_2",
    "retail_0", "retail_1", "retail_2", "retail_3",
    "retail_4", "retail_5", "retail_6", "retail_7",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _val_base(mmp_root: Path) -> Path:
    """Return path to validation/ inside the MMPTracking root."""
    candidate = mmp_root / "MMPTracking_validation" / "validation"
    if candidate.exists():
        return candidate
    candidate2 = mmp_root / "validation"
    if candidate2.exists():
        return candidate2
    raise FileNotFoundError(
        f"Cannot find validation directory under {mmp_root}. "
        f"Expected: MMPTracking_validation/validation/ or validation/"
    )


def _get_cam_ids_from_zip(zip_path: Path) -> list[int]:
    """Scan zip namelist to find unique camera IDs."""
    ids: set[int] = set()
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith(".jpg"):
                stem = Path(name).stem          # rgb_NNNNN_C
                ids.add(int(stem.rsplit("_", 1)[-1]))
    return sorted(ids)


def _extract_frames_to_tmpdir(img_zip: Path, tmp_dir: Path) -> None:
    """Extract all jpg files from zip into tmp_dir/<scene>/."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(img_zip) as zf:
        members = [m for m in zf.namelist() if m.endswith(".jpg")]
        zf.extractall(tmp_dir, members=members)


def _create_video(img_dir: Path, scene: str, cam_id: int,
                  out_path: Path, fps: int, max_frames: int) -> None:
    pattern = str(img_dir / f"rgb_%05d_{cam_id}.jpg")
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-start_number", "0",
        "-i", pattern,
        "-vframes", str(max_frames),
        "-c:v", "libx264", "-crf", "20", "-preset", "fast",
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {scene} cam{cam_id}:\n{result.stderr[-600:]}"
        )


def _create_gt_csv(lbl_zip: Path, scene: str, cam_id: int,
                   out_path: Path, max_frames: int) -> None:
    rows = []
    with zipfile.ZipFile(lbl_zip) as zf:
        for frame_no in range(max_frames):
            name = f"{scene}/rgb_{frame_no:05d}_{cam_id}.json"
            if name not in zf.namelist():
                continue
            ann = json.loads(zf.read(name))
            for pid_str, box in ann.items():
                x1, y1, x2, y2 = box
                rows.append(
                    f"{frame_no},{int(pid_str)},{float(x1)},{float(y1)},"
                    f"{float(x2 - x1)},{float(y2 - y1)}\n"
                )
    out_path.write_text("frame,person_id,left,top,width,height\n" + "".join(rows))


def _copy_calibration(scene: str, val_base: Path, out_root: Path) -> None:
    env = scene.rsplit("_", 1)[0]   # lobby_0 → lobby
    src = val_base / "calibrations" / env / "calibrations.json"
    if not src.exists():
        print(f"  [WARN] calibration not found: {src}")
        return
    dst_dir = out_root / "calibrations" / env
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "calibrations.json"
    if not dst.exists():
        shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# Per-scene worker
# ---------------------------------------------------------------------------

def process_scene(
    scene: str,
    val_base: Path,
    out_root: Path,
    tmp_root: Path,
    fps: int,
    max_frames: int,
    keep_extracted: bool,
) -> dict:
    result = {"scene": scene, "videos": 0, "csvs": 0, "errors": []}

    img_zip = val_base / "images" / "64pm" / f"{scene}.zip"
    lbl_zip = val_base / "labels" / "64pm" / f"{scene}.zip"
    scene_out = out_root / scene
    scene_out.mkdir(parents=True, exist_ok=True)

    if not img_zip.exists():
        result["errors"].append(f"images zip not found: {img_zip}")
        return result
    if not lbl_zip.exists():
        result["errors"].append(f"labels zip not found: {lbl_zip}")
        return result

    # Detect cameras
    try:
        cam_ids = _get_cam_ids_from_zip(img_zip)
    except Exception as e:
        result["errors"].append(f"cam detection: {e}")
        return result

    print(f"[{scene}] cams={cam_ids} — extracting frames ...")

    # Extract images to tmp dir (needed by ffmpeg for frame sequence input)
    tmp_scene = tmp_root / scene
    try:
        _extract_frames_to_tmpdir(img_zip, tmp_scene)
    except Exception as e:
        result["errors"].append(f"extract images: {e}")
        return result

    img_dir = tmp_scene / scene

    # GT CSV (read directly from zip — no need to extract to disk)
    for cam_id in cam_ids:
        csv_path = scene_out / f"gt_cam{cam_id}.csv"
        if csv_path.exists():
            print(f"  [SKIP] {scene}/gt_cam{cam_id}.csv")
        else:
            try:
                _create_gt_csv(lbl_zip, scene, cam_id, csv_path, max_frames)
                result["csvs"] += 1
            except Exception as e:
                result["errors"].append(f"gt cam{cam_id}: {e}")

    # Videos via ffmpeg
    for cam_id in cam_ids:
        vid_path = scene_out / f"cam{cam_id}.mp4"
        if vid_path.exists():
            print(f"  [SKIP] {scene}/cam{cam_id}.mp4")
        else:
            try:
                _create_video(img_dir, scene, cam_id, vid_path, fps, max_frames)
                result["videos"] += 1
                print(f"  [OK]   {scene}/cam{cam_id}.mp4")
            except Exception as e:
                result["errors"].append(f"video cam{cam_id}: {e}")

    # Calibration
    try:
        _copy_calibration(scene, val_base, out_root)
    except Exception as e:
        result["errors"].append(f"calibration: {e}")

    # Clean up extracted frames unless user wants to keep them
    if not keep_extracted:
        shutil.rmtree(tmp_scene, ignore_errors=True)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build MMPTracking_short (1-min clips + GT) from raw validation zips")
    ap.add_argument("--mmp-root", default="dataset/MMPTracking",
                    help="Root of MMPTracking dataset "
                         "(must contain MMPTracking_validation/ or validation/). "
                         "Default: dataset/MMPTracking")
    ap.add_argument("--output", default="dataset/MMPTracking_short",
                    help="Output directory. Default: dataset/MMPTracking_short")
    ap.add_argument("--tmp-dir", default=None,
                    help="Temp dir for extracted frames (deleted after each scene). "
                         "Default: <output>/.tmp_frames")
    ap.add_argument("--scenes", nargs="+", default=ALL_SCENES,
                    help="Scenes to process. Default: all 24 scenes.")
    ap.add_argument("--fps", type=int, default=25,
                    help="Output video framerate. Default: 25")
    ap.add_argument("--max-frames", type=int, default=1500,
                    help="Frames per camera (1500 = 60s at 25fps). Default: 1500")
    ap.add_argument("--jobs", type=int, default=2,
                    help="Parallel scene workers. Default: 2 "
                         "(each scene extracts ~2GB; keep low to avoid I/O bottleneck)")
    ap.add_argument("--keep-extracted", action="store_true",
                    help="Keep extracted frames after building video "
                         "(useful if you want to reuse them later)")
    args = ap.parse_args()

    mmp_root = Path(args.mmp_root)
    out_root = Path(args.output)
    tmp_root = Path(args.tmp_dir) if args.tmp_dir else out_root / ".tmp_frames"

    # Validate input
    try:
        val_base = _val_base(mmp_root)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    scenes = [s for s in args.scenes if s in ALL_SCENES]
    unknown = [s for s in args.scenes if s not in ALL_SCENES]
    if unknown:
        print(f"[WARN] Unknown scenes ignored: {unknown}")
    if not scenes:
        print("[ERROR] No valid scenes to process.")
        sys.exit(1)

    print(f"MMPTracking root : {mmp_root}")
    print(f"Validation base  : {val_base}")
    print(f"Output           : {out_root}")
    print(f"Temp frames      : {tmp_root}")
    print(f"Scenes ({len(scenes)}): {scenes}")
    print(f"Frames/cam       : {args.max_frames} ({args.max_frames/args.fps:.0f}s @ {args.fps}fps)")
    print(f"Parallel jobs    : {args.jobs}")
    print()

    out_root.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)

    errors_total = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(
                process_scene,
                s, val_base, out_root, tmp_root,
                args.fps, args.max_frames, args.keep_extracted,
            ): s
            for s in scenes
        }
        for fut in as_completed(futures):
            res = fut.result()
            scene = res["scene"]
            if res["errors"]:
                print(f"[ERROR] {scene}: {res['errors']}")
                errors_total.extend(res["errors"])
            else:
                print(f"[DONE]  {scene}: {res['videos']} videos, {res['csvs']} GT CSVs")

    # Clean up tmp root if empty
    try:
        tmp_root.rmdir()
    except OSError:
        if not args.keep_extracted:
            shutil.rmtree(tmp_root, ignore_errors=True)

    print()
    if errors_total:
        print(f"Finished with {len(errors_total)} error(s).")
        sys.exit(1)

    manifest = {
        "scenes": scenes,
        "fps": args.fps,
        "max_frames": args.max_frames,
        "duration_seconds": args.max_frames / args.fps,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Done. Dataset at: {out_root.resolve()}")


if __name__ == "__main__":
    main()
