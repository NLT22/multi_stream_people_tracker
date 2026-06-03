"""
Create MMPTracking_short: 1-minute clips (frames 0–1499) with GT for all 24 scenes.

Output layout:
    dataset/MMPTracking_short/
        <scene>/
            cam<N>.mp4                  ← 1-min video (frames 0-1499, 25fps)
            gt_cam<N>.csv               ← GT filtered to frames 0-1499
        calibrations/<env>/calibrations.json  ← copy from original

Usage:
    python scripts/create_mmp_short.py [--scenes scene1 scene2 ...]
                                        [--fps 25]
                                        [--max-frames 1500]
                                        [--jobs 4]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


EXTRACT_ROOT = Path("dataset/MMPTracking/extracted")
CALIB_ROOT   = Path("dataset/MMPTracking/MMPTracking_validation/validation/calibrations")
OUT_ROOT     = Path("dataset/MMPTracking_short")

ALL_SCENES = [
    "cafe_shop_0", "cafe_shop_1", "cafe_shop_2", "cafe_shop_3",
    "industry_safety_0", "industry_safety_1", "industry_safety_2",
    "industry_safety_3", "industry_safety_4",
    "lobby_0", "lobby_1", "lobby_2", "lobby_3",
    "office_0", "office_1", "office_2",
    "retail_0", "retail_1", "retail_2", "retail_3",
    "retail_4", "retail_5", "retail_6", "retail_7",
]


def get_cam_ids(scene: str) -> list[int]:
    img_dir = EXTRACT_ROOT / scene / scene
    ids = set()
    for p in img_dir.iterdir():
        if p.suffix == ".jpg":
            ids.add(int(p.stem.rsplit("_", 1)[-1]))
    return sorted(ids)


def create_video(scene: str, cam_id: int, fps: int, max_frames: int) -> Path:
    """ffmpeg: encode first max_frames frames of cam_id → cam<N>.mp4."""
    img_dir  = EXTRACT_ROOT / scene / scene
    out_dir  = OUT_ROOT / scene
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"cam{cam_id}.mp4"

    if out_path.exists():
        print(f"  [SKIP] {scene}/cam{cam_id}.mp4 already exists")
        return out_path

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
            f"ffmpeg failed for {scene} cam{cam_id}:\n{result.stderr[-500:]}"
        )
    return out_path


def create_gt_csv(scene: str, cam_id: int, max_frames: int) -> Path:
    """Parse JSON labels for frames 0..max_frames-1 and write gt_cam<N>.csv."""
    lbl_dir  = EXTRACT_ROOT / f"{scene}_labels" / scene
    out_dir  = OUT_ROOT / scene
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"gt_cam{cam_id}.csv"

    if out_path.exists():
        print(f"  [SKIP] {scene}/gt_cam{cam_id}.csv already exists")
        return out_path

    rows = []
    for frame_no in range(max_frames):
        json_path = lbl_dir / f"rgb_{frame_no:05d}_{cam_id}.json"
        if not json_path.exists():
            continue
        with open(json_path) as f:
            ann = json.load(f)
        for pid_str, box in ann.items():
            x1, y1, x2, y2 = box
            rows.append(
                f"{frame_no},{int(pid_str)},{float(x1)},{float(y1)},"
                f"{float(x2 - x1)},{float(y2 - y1)}\n"
            )

    with open(out_path, "w") as f:
        f.write("frame,person_id,left,top,width,height\n")
        f.writelines(rows)

    return out_path


def copy_calibration(scene: str) -> None:
    """Copy calibration JSON for the scene's environment."""
    env = scene.rsplit("_", 1)[0]          # lobby_0 → lobby
    src = CALIB_ROOT / env / "calibrations.json"
    if not src.exists():
        print(f"  [WARN] calibration not found: {src}")
        return
    dst_dir = OUT_ROOT / "calibrations" / env
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "calibrations.json"
    if not dst.exists():
        shutil.copy2(src, dst)


def process_scene(scene: str, fps: int, max_frames: int) -> dict:
    result = {"scene": scene, "videos": [], "csvs": [], "errors": []}
    try:
        cam_ids = get_cam_ids(scene)
        print(f"[{scene}] {len(cam_ids)} cams: {cam_ids}")

        for cam_id in cam_ids:
            try:
                vp = create_video(scene, cam_id, fps, max_frames)
                result["videos"].append(str(vp))
            except Exception as e:
                result["errors"].append(f"video cam{cam_id}: {e}")

            try:
                cp = create_gt_csv(scene, cam_id, max_frames)
                result["csvs"].append(str(cp))
            except Exception as e:
                result["errors"].append(f"gt cam{cam_id}: {e}")

        copy_calibration(scene)
    except Exception as e:
        result["errors"].append(str(e))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=ALL_SCENES)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--max-frames", type=int, default=1500,
                    help="1 minute at 25fps = 1500 frames")
    ap.add_argument("--jobs", type=int, default=2,
                    help="Parallel scene workers (each runs ffmpeg sequentially per cam)")
    args = ap.parse_args()

    scenes = [s for s in args.scenes if s in ALL_SCENES]
    missing = [s for s in args.scenes if s not in ALL_SCENES]
    if missing:
        print(f"[WARN] Unknown scenes ignored: {missing}")

    print(f"Creating MMPTracking_short: {len(scenes)} scenes, "
          f"{args.max_frames} frames @ {args.fps}fps "
          f"({args.max_frames/args.fps:.0f}s) → {OUT_ROOT}")

    errors_total = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(process_scene, s, args.fps, args.max_frames): s
            for s in scenes
        }
        for fut in as_completed(futures):
            res = fut.result()
            scene = res["scene"]
            if res["errors"]:
                print(f"[ERROR] {scene}: {res['errors']}")
                errors_total.extend(res["errors"])
            else:
                print(f"[OK]    {scene}: "
                      f"{len(res['videos'])} videos, {len(res['csvs'])} GT CSVs")

    print()
    if errors_total:
        print(f"Finished with {len(errors_total)} error(s).")
        sys.exit(1)
    else:
        # Write a simple manifest
        manifest = {
            "scenes": scenes,
            "fps": args.fps,
            "max_frames": args.max_frames,
            "duration_seconds": args.max_frames / args.fps,
        }
        (OUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"Done. Dataset at: {OUT_ROOT.resolve()}")
        print(f"Manifest: {OUT_ROOT / 'manifest.json'}")


if __name__ == "__main__":
    main()
