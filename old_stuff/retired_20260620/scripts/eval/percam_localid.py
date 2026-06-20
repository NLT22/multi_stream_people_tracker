"""Per-camera IDF1 using local_track_id (raw tracker, no gallery) vs global_id.
Makes a temp pred dir with global_id:=local_track_id and runs metrics per-camera."""
import sys, shutil, subprocess, tempfile, pandas as pd
from pathlib import Path

pred_dir = Path(sys.argv[1]); short_root = sys.argv[2]; scene = sys.argv[3]
tmp = Path(tempfile.mkdtemp())
for f in pred_dir.glob("cam_*_predictions.csv"):
    df = pd.read_csv(f)
    df["global_id"] = df["local_track_id"]   # eval raw tracker id
    df.to_csv(tmp / f.name, index=False)
print("=== PER-CAMERA using LOCAL track id (raw tracker) ===")
subprocess.run(["python","-m","src.eval.metrics_mmp","--short-root",short_root,
                "--scene",scene,"--pred-dir",str(tmp)])
shutil.rmtree(tmp, ignore_errors=True)
