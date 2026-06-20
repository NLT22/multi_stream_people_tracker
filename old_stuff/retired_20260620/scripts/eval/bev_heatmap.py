"""Bird's-eye-view occupancy heatmap from exported tracklet_bev.csv.

The pipeline (PredictionExporter) writes per-detection world-plane foot positions
to <pred_dir>/tracklet_bev.csv when geometry/calibration is available. This
aggregates those world (x, y) points into a 2D occupancy / dwell heatmap on the
ground plane. No model retraining needed — it reuses the homography foot-point
projection already in the pipeline.

Run:
    # 1. export predictions for a scene (writes tracklet_bev.csv)
    python -m src.main --config configs/pipelines/pipeline_mmp_nvdcf_realtime_baseline.yaml \
        --mmp-short-dataset dataset/MMPTracking_short:lobby_0 \
        --no-display --no-sync --export-predictions output/eval/lobby0
    # 2. render the heatmap
    python scripts/eval/bev_heatmap.py --pred-dir output/eval/lobby0
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _load_points(bev_csv: Path, gid: int | None):
    xs, ys = [], []
    with open(bev_csv, newline="") as f:
        for row in csv.DictReader(f):
            if gid is not None and int(float(row["global_id"])) != gid:
                continue
            xs.append(float(row["world_x"]))
            ys.append(float(row["world_y"]))
    return np.asarray(xs), np.asarray(ys)


def _smooth(h, sigma):
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(h, sigma=sigma)
    except ImportError:
        return h


def render_camera_overlay(pred_dir: Path, cam: int, video: Path, out: str,
                          gid: int | None, heat_sigma: float, frame_idx: int):
    """Overlay an occupancy heatmap on a real camera frame (image space).

    Uses the per-camera bbox foot points (left+w/2, top+h) from
    cam_<cam>_predictions.csv — already in the video's pixel space — so it lands
    on the actual scene, no world projection needed.
    """
    import cv2
    import matplotlib.cm as cm

    csv_path = pred_dir / f"cam_{cam}_predictions.csv"
    if not csv_path.exists():
        raise SystemExit(f"[bev] no {csv_path}")
    fx, fy = [], []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if gid is not None and int(float(row["global_id"])) != gid:
                continue
            l, t, w, h = (float(row["left"]), float(row["top"]),
                          float(row["width"]), float(row["height"]))
            fx.append(l + w / 2)
            fy.append(t + h)
    if not fx:
        raise SystemExit("[bev] no detections for that camera/gid")

    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx if frame_idx >= 0 else n // 2)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"[bev] could not read frame from {video}")
    H, W = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)

    acc = np.zeros((H, W), np.float32)
    for x, y in zip(fx, fy):
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < W and 0 <= yi < H:
            acc[yi, xi] += 1.0
    acc = _smooth(acc, heat_sigma)
    if acc.max() > 0:
        norm = (acc / acc.max()) ** 0.5            # power-norm for visibility
    else:
        norm = acc
    heat = (cm.turbo(norm)[..., :3] * 255.0)        # RGB heat
    alpha = np.clip(norm * 1.4, 0, 1)[..., None] * 0.7
    blend = (frame_rgb * (1 - alpha) + heat * alpha).clip(0, 255).astype(np.uint8)

    cv2.imwrite(out, cv2.cvtColor(blend, cv2.COLOR_RGB2BGR))
    print(f"[bev] wrote {out}  (camera overlay: {len(fx)} foot points on "
          f"{video.name} {W}x{H})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True, type=Path,
                    help="Export dir containing tracklet_bev.csv")
    ap.add_argument("--bins", type=int, default=110)
    ap.add_argument("--sigma", type=float, default=2.0, help="Gaussian smoothing")
    ap.add_argument("--gid", type=int, default=None, help="Only this global id")
    ap.add_argument("--iqr-k", type=float, default=1.5,
                    help="Tukey-fence crop: keep points within "
                         "[Q1-k*IQR, Q3+k*IQR] per axis. Homography foot-point "
                         "projection is heavy-tailed (edge/horizon boxes shoot to "
                         "~infinity); this auto-zooms to the room. 0 = no crop.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--overlay-video", default=None, type=Path,
                    help="Overlay the heatmap on a real camera frame from this "
                         "video (image-space) instead of the world BEV plot.")
    ap.add_argument("--overlay-cam", type=int, default=0,
                    help="source_id of cam_<N>_predictions.csv matching the video.")
    ap.add_argument("--heat-sigma", type=float, default=9.0,
                    help="Heatmap blur in pixels for the camera overlay.")
    ap.add_argument("--frame", type=int, default=-1,
                    help="Frame index for the overlay background (-1 = middle).")
    args = ap.parse_args()

    if args.overlay_video is not None:
        out = args.out or str(args.pred_dir / f"overlay_cam{args.overlay_cam}.png")
        render_camera_overlay(args.pred_dir, args.overlay_cam, args.overlay_video,
                              out, args.gid, args.heat_sigma, args.frame)
        return

    bev_csv = args.pred_dir / "tracklet_bev.csv"
    if not bev_csv.exists():
        raise SystemExit(
            f"[bev] no tracklet_bev.csv in {args.pred_dir} — re-export with a scene "
            f"that has calibration (geometry must be active).")

    xs, ys = _load_points(bev_csv, args.gid)
    if xs.size == 0:
        raise SystemExit("[bev] no BEV points (check --gid)")

    raw = xs.size
    if args.iqr_k > 0:
        def _fence(v):
            q1, q3 = np.percentile(v, [25, 75])
            iqr = q3 - q1
            return q1 - args.iqr_k * iqr, q3 + args.iqr_k * iqr
        xlo, xhi = _fence(xs)
        ylo, yhi = _fence(ys)
        keep = (xs >= xlo) & (xs <= xhi) & (ys >= ylo) & (ys <= yhi)
        xs, ys = xs[keep], ys[keep]
        print(f"[bev] dropped {raw - xs.size}/{raw} heavy-tail outliers "
              f"(Tukey k={args.iqr_k}); room ≈ "
              f"{(xhi - xlo) / 1000:.1f}×{(yhi - ylo) / 1000:.1f} m")

    h, xe, ye = np.histogram2d(xs, ys, bins=args.bins)
    h = _smooth(h, args.sigma)

    # Human-readable crowd-heatmap look:
    #  - white background (hide near-empty cells) instead of a black canvas
    #  - PowerNorm so a single high-dwell cell doesn't crush everything to ~0
    #  - colorbar clipped at the 99th percentile of occupied cells
    from matplotlib.colors import PowerNorm
    pos = h[h > 0]
    vmax = float(np.percentile(pos, 99)) if pos.size else 1.0
    masked = np.ma.masked_where(h.T < vmax * 0.04, h.T)

    out = args.out or str(args.pred_dir / "bev_heatmap.png")
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_facecolor("white")
    im = ax.imshow(masked, origin="lower", extent=[xe[0], xe[-1], ye[0], ye[-1]],
                   cmap="turbo", norm=PowerNorm(gamma=0.5, vmax=vmax),
                   aspect="equal", interpolation="bilinear")
    fig.colorbar(im, ax=ax, label="occupancy (person-frames, ~time spent)")
    ax.set_xlabel("world X (mm)")
    ax.set_ylabel("world Y (mm)")
    title = f"BEV occupancy — {args.pred_dir.name} ({xs.size} points)"
    if args.gid is not None:
        title += f" — GID {args.gid}"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"[bev] wrote {out}  ({xs.size} points, {args.bins}x{args.bins} bins, "
          f"X[{xs.min():.0f},{xs.max():.0f}] Y[{ys.min():.0f},{ys.max():.0f}] mm)")


if __name__ == "__main__":
    main()
