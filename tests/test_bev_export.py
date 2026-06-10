import csv
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.eval.offline_merge import _load_geometry_points


def test_load_geometry_points_prefers_exported_bev():
    with tempfile.TemporaryDirectory() as d:
        pred_dir = Path(d)
        with open(pred_dir / "tracklet_bev.csv", "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "tracklet_id", "frame_no_cam", "cam_id", "local_track_id",
                    "global_id", "world_x", "world_y",
                ],
            )
            writer.writeheader()
            writer.writerow({
                "tracklet_id": 0,
                "frame_no_cam": 10,
                "cam_id": 2,
                "local_track_id": 7,
                "global_id": 4,
                "world_x": 123.5,
                "world_y": 456.75,
            })
            writer.writerow({
                "tracklet_id": 1,
                "frame_no_cam": 11,
                "cam_id": 3,
                "local_track_id": 8,
                "global_id": 5,
                "world_x": 999.0,
                "world_y": 999.0,
            })

        points = _load_geometry_points(pred_dir, None, None, sample_step=5)

        assert points == {4: {10: [(2, 123.5, 456.75)]}}


if __name__ == "__main__":
    test_load_geometry_points_prefers_exported_bev()
    print("PASS")
