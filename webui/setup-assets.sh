#!/usr/bin/env bash
# Populate webui/public/ with real pipeline assets:
#   frames/   <- still camera frames (small, copied)
#   heatmaps/ <- occupancy/footfall/dwell PNGs (small, copied)
#   feeds/    <- per-zone OSD videos (large, symlinked)
# Re-run any time the demo outputs are regenerated. Safe to run repeatedly.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PUB="$HERE/public"
DEMO="$ROOT/output/demo"
IMAGES="$ROOT/report/latex/Images"

SCENES=(cafe_shop lobby office industry_safety retail)

mkdir -p "$PUB/frames" "$PUB/heatmaps" "$PUB/feeds"

echo "→ camera still frames"
cp -f "$IMAGES"/orig_*.jpg "$PUB/frames/" 2>/dev/null \
  && echo "  copied $(ls "$PUB/frames" | wc -l) frames" \
  || echo "  WARN: no orig_*.jpg under $IMAGES (run the frame extraction first)"

# Quadrant offsets: the per-zone OSD video is a 2×2 tile (1280×720); each camera
# is a 640×360 crop (source 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right).
declare -A OFF=( [1]="0:0" [2]="640:0" [3]="0:360" [4]="640:360" )

echo "→ heatmaps + per-camera OSD feeds per scene"
for s in "${SCENES[@]}"; do
  if [[ -d "$DEMO/$s/heatmap" ]]; then
    mkdir -p "$PUB/heatmaps/$s"
    cp -f "$DEMO/$s/heatmap"/*.png "$PUB/heatmaps/$s/" 2>/dev/null || true
    echo "  heatmaps/$s ✓"
  else
    echo "  WARN: missing $DEMO/$s/heatmap"
  fi

  feed="$DEMO/$s/${s}_live_buffered_osd.mp4"
  if [[ -f "$feed" ]]; then
    for n in 1 2 3 4; do
      out="$PUB/feeds/${s}_cam${n}.mp4"
      [[ -f "$out" ]] && continue                       # skip if already cropped
      ffmpeg -y -i "$feed" -filter:v "crop=640:360:${OFF[$n]}" \
        -c:v libx264 -preset veryfast -crf 23 -an "$out" -loglevel error \
        && echo "  feeds/${s}_cam${n}.mp4 ✓" || echo "  WARN: crop ${s}_cam${n} failed"
    done
  else
    echo "  WARN: missing $feed"
  fi
done

echo "✓ assets ready under public/"
