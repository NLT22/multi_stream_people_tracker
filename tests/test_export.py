"""Pure-Python unit tests for the delayed-flush prediction exporter.

Covers the near-realtime remap semantics added for micro-batch fusion:
rows are buffered for `delay_frames` and the latest Global-ID remap is applied
at flush time.

Run:   python -m pytest tests/test_export.py -v
  or:  python tests/test_export.py
"""

import csv
import os
import sys
import tempfile
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.eval.export import PredictionExporter


def _read_gids(out_dir, cam_id):
    path = os.path.join(out_dir, f"cam_{cam_id}_predictions.csv")
    with open(path, newline="") as f:
        return [r["global_id"] for r in csv.DictReader(f)]


def test_delay_zero_writes_all_rows():
    with tempfile.TemporaryDirectory() as d:
        e = PredictionExporter(d, delay_frames=0)
        e.record(1, 0, 5, 3, 10, 10, 20, 40)
        e.record(2, 0, 5, 3, 10, 10, 20, 40)
        e.tick(2)
        e.close()
        assert _read_gids(d, 0) == ["3", "3"]


def test_delayed_remap_applied_retroactively():
    # row recorded under raw gid 8; a later merge {8:2} arrives before flush
    with tempfile.TemporaryDirectory() as d:
        e = PredictionExporter(d, delay_frames=5)
        e.record(1, 0, 5, 8, 10, 10, 20, 40)
        e.tick(3, {})                 # safe_frame = -2 -> nothing flushed yet
        e.record(6, 0, 5, 8, 10, 10, 20, 40)
        e.tick(8, {8: 2})             # safe_frame = 3 -> flush frame 1 with remap
        e.close()                     # flush remainder with final remap
        assert _read_gids(d, 0) == ["2", "2"]


def test_remap_chain_resolves():
    with tempfile.TemporaryDirectory() as d:
        e = PredictionExporter(d, delay_frames=0)
        e.record(1, 0, 5, 9, 10, 10, 20, 40)
        e.tick(1, {9: 7, 7: 4})       # chain 9->7->4 must resolve to 4
        e.close()
        assert _read_gids(d, 0) == ["4"]


def test_negative_gid_preserved():
    with tempfile.TemporaryDirectory() as d:
        e = PredictionExporter(d, delay_frames=0)
        e.record(1, 0, 5, None, 10, 10, 20, 40)   # -> -1
        e.tick(1, {})
        e.close()
        assert _read_gids(d, 0) == ["-1"]


def test_tracklet_summary_written():
    with tempfile.TemporaryDirectory() as d:
        e = PredictionExporter(d, delay_frames=0)
        for fno in range(0, 30):
            e.record(fno, 0, 5, 3, 10, 10, 20, 40, embedding=[1.0, 0.0])
        e.tick(40)
        e.close()
        tpath = os.path.join(d, "tracklets.csv")
        assert os.path.exists(tpath)
        rows = list(csv.DictReader(open(tpath)))
        assert len(rows) == 1
        assert int(rows[0]["num_detections"]) == 30


def test_tracklet_bev_written_when_foot_world_available():
    with tempfile.TemporaryDirectory() as d:
        e = PredictionExporter(d, delay_frames=0)
        e.record(
            10, 2, 7, 4, 10, 10, 20, 40,
            embedding=[1.0, 0.0],
            foot_world=(1234.5, 6789.0),
        )
        e.tick(10)
        e.close()
        path = os.path.join(d, "tracklet_bev.csv")
        assert os.path.exists(path)
        rows = list(csv.DictReader(open(path)))
        assert len(rows) == 1
        assert rows[0]["frame_no_cam"] == "10"
        assert rows[0]["cam_id"] == "2"
        assert rows[0]["global_id"] == "4"
        assert rows[0]["world_x"] == "1234.5"
        assert rows[0]["world_y"] == "6789.0"


def test_live_embedding_chunks_are_uncompressed_and_close_flushes_tail():
    with tempfile.TemporaryDirectory() as d:
        e = PredictionExporter(d, delay_frames=0, emb_flush_frames=100)
        e.record(
            1, 0, 5, 3, 10, 10, 20, 40,
            det_embedding=[1.0, 0.0],
        )
        e.tick(100)
        e.record(
            101, 0, 5, 3, 10, 10, 20, 40,
            det_embedding=[0.0, 1.0],
        )
        e.close()

        chunk0 = os.path.join(d, "det_emb_chunk_0000.npz")
        chunk1 = os.path.join(d, "det_emb_chunk_0001.npz")
        assert os.path.exists(chunk0)
        assert os.path.exists(chunk1)
        assert not os.path.exists(os.path.join(d, "detection_embeddings.npz"))
        for path in (chunk0, chunk1):
            with zipfile.ZipFile(path) as zf:
                assert all(
                    info.compress_type == zipfile.ZIP_STORED
                    for info in zf.infolist()
                )


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {fn.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
