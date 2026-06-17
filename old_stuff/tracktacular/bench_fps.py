#!/usr/bin/env python3
"""Clean inference-only throughput benchmark for TrackTacular (SegNet/bilinear).

Times ONLY model.forward (encoder x S cams + BEV lift + decoder) on the GPU:
warmup excluded, no disk I/O, no tracking, no metrics. Reports multi-camera
timesteps/s (= the realtime rate the whole rig runs at) for several camera counts.

Run from repo root:  python scripts/tracktacular/bench_fps.py
"""
import sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path("reference/TrackTacular/WorldTrack").resolve()))
from world_track import WorldTrackModel
from datasets.mmptracking_dataset import Mmptracking
from datasets.pedestrian_dataset import PedestrianDataset

RES = (64, 4, 64)
BOUNDS = (0, 256, 0, 256, 0, 60)
H, W = 720, 1280
WARMUP, ITERS = 8, 40


def get_real_sample():
    base = Mmptracking("dataset/worldtrack/mmp_industry_safety_0")
    ds = PedestrianDataset(base, is_train=False, resolution=RES, bounds=BOUNDS)
    item, _ = ds[0]
    return item  # img (4,3,H,W), intrinsic (4,4,4), extrinsic (4,4,4), ref_T_global (4,4)


def tile_to(t, s):
    """Repeat camera dim (dim0) to length s."""
    reps = (s + t.shape[0] - 1) // t.shape[0]
    return t.repeat(reps, *([1] * (t.dim() - 1)))[:s]


@torch.no_grad()
def bench(s, sample):
    m = WorldTrackModel(model_name='segnet', resolution=RES, bounds=BOUNDS,
                        num_cameras=s, use_temporal_cache=False,
                        depth=(32, 250, 4000)).cuda().eval()
    item = {
        'img': tile_to(sample['img'], s).unsqueeze(0).cuda(),
        'intrinsic': tile_to(sample['intrinsic'], s).unsqueeze(0).cuda(),
        'extrinsic': tile_to(sample['extrinsic'], s).unsqueeze(0).cuda(),
        'ref_T_global': sample['ref_T_global'].unsqueeze(0).cuda(),
        'frame': torch.tensor([-1]),
    }
    for _ in range(WARMUP):
        m(item)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(ITERS):
        m(item)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / ITERS
    peak = torch.cuda.max_memory_allocated() / 1e9
    del m, item
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    return dt, peak


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    print(f"SegNet/bilinear, input {H}x{W}/cam, BEV {RES}, batch 1, "
          f"warmup {WARMUP} excluded, mean of {ITERS}\n")
    sample = get_real_sample()
    print(f"{'cams':>5} {'ms/timestep':>12} {'timesteps/s':>12} "
          f"{'cam-frames/s':>13} {'peakVRAM(GB)':>13}")
    for s in (4, 5, 6, 7, 20):
        try:
            dt, peak = bench(s, sample)
            print(f"{s:5d} {dt*1000:12.1f} {1/dt:12.2f} {s/dt:13.1f} {peak:13.2f}")
        except RuntimeError as e:
            print(f"{s:5d}   FAILED: {str(e)[:60]}")


if __name__ == "__main__":
    main()
