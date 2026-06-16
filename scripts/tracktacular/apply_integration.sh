#!/usr/bin/env bash
# Apply the MMPTracking integration into the (gitignored) TrackTacular clone.
# Idempotent. Run from repo root: bash scripts/tracktacular/apply_integration.sh
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
TT="$REPO/reference/TrackTacular/WorldTrack"
SRC="$REPO/scripts/tracktacular"

# 1. adapter + config + affines into the clone
cp "$SRC/mmptracking_dataset.py" "$TT/datasets/mmptracking_dataset.py"
cp "$SRC/d_mmp_industry.yml" "$TT/configs/d_mmp_industry.yml"
[ -f "$SRC/affines.json" ] && cp "$SRC/affines.json" "$TT/datasets/affines.json"

# 2. guard the mmcv-only bevformer import (segnet/mvdet/liftnet need no mmcv)
python3 - "$TT/models/__init__.py" <<'PY'
import sys, io
p = sys.argv[1]
s = open(p).read()
if "except ModuleNotFoundError" not in s:
    s = s.replace(
        "from models.bevformernet import Bevformernet",
        "try:\n    from models.bevformernet import Bevformernet\n"
        "except ModuleNotFoundError:\n    Bevformernet = None")
    open(p, "w").write(s)
print("models/__init__.py guarded")
PY

# 3. add the mmp branch to the datamodule's setup()
python3 - "$TT/datasets/pedestrian_datamodule.py" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
if "mmptracking_dataset" not in s:
    s = s.replace(
        "from datasets.wildtrack_dataset import Wildtrack",
        "from datasets.wildtrack_dataset import Wildtrack\n"
        "from datasets.mmptracking_dataset import Mmptracking")
    s = s.replace(
        "        elif 'multiviewx' in self.dataset.lower():\n"
        "            base = MultiviewX(self.data_dir)",
        "        elif 'multiviewx' in self.dataset.lower():\n"
        "            base = MultiviewX(self.data_dir)\n"
        "        elif 'mmp' in self.dataset.lower():\n"
        "            base = Mmptracking(self.data_dir)")
    open(p, "w").write(s)
print("datamodule patched")
PY
# 4. torch>=2.x: Sampler.__init__ no longer takes data_source
sed -i 's/        super().__init__(data_source)/        super().__init__()/' \
    "$TT/datasets/sampler.py"

# 5. guard datasets/__init__ imports missing in this clone version
cat > "$TT/datasets/__init__.py" <<'PY'
from .pedestrian_datamodule import PedestrianDataModule
try:
    from .synthehicle_datamodule import SynthehicleDataModule
except ModuleNotFoundError:
    SynthehicleDataModule = None
PY

echo "integration applied to $TT"
