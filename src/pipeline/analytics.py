"""Read gst-nvdsanalytics frame metadata (ROI occupancy, line-crossing,
overcrowding) and print/export per-camera counts.

This is the pipeline-native NVIDIA analytic (nvdsanalytics) — counts/events in
configured regions, attached as frame user meta. It complements the offline
occupancy heatmap (scripts/eval/camera_heatmap.py), which is a density map.

The plugin uses the bounding-box bottom-centre (foot point) for all rules, so
ROI/line geometry is in foot-point space — the same point the heatmap uses.
"""

import csv
import traceback

import pyservicemaker as psm


class AnalyticsProbe(psm.BatchMetadataOperator):
    """Reads NvDsAnalyticsFrameMeta per camera: obj_in_roi_cnt, obj_lc_cum_cnt,
    oc_status, obj_cnt. Prints a summary every `print_interval` frames and,
    if `export_path` is set, writes a per-frame CSV at close()."""

    def __init__(self, print_interval: int = 60, export_path: str | None = None):
        super().__init__()
        self._interval = max(1, print_interval)
        self._export_path = export_path
        self._frames = 0
        self._rows: list[dict] = []
        self._last: dict[int, dict] = {}   # source_id -> latest analytics snapshot

    def handle_metadata(self, batch_meta):
        try:
            self._handle(batch_meta)
        except Exception:
            print("[analytics ERROR] AnalyticsProbe failed:")
            traceback.print_exc()

    def _handle(self, batch_meta):
        self._frames += 1
        for frame_meta in batch_meta.frame_items:
            src = frame_meta.source_id
            af = None
            for item in frame_meta.nvdsanalytics_frame_items:
                # pyservicemaker exposes a dedicated analytics iterator; items are
                # AnalyticsFrameMeta (or a UserMetadata to cast).
                af = item if hasattr(item, "obj_in_roi_cnt") else item.as_nvdsanalytics_frame()
                break
            if af is None:
                continue

            roi = dict(af.obj_in_roi_cnt)
            lc = dict(af.obj_lc_cum_cnt)
            oc = [k for k, v in dict(af.oc_status).items() if v]
            total = sum(dict(af.obj_cnt).values())
            self._last[src] = {"roi": roi, "lc": lc, "oc": oc, "total": total}

            if self._export_path is not None:
                self._rows.append({
                    "frame": frame_meta.frame_number,
                    "cam": src,
                    "total": total,
                    "roi": ";".join(f"{k}={v}" for k, v in roi.items()),
                    "lc_cumulative": ";".join(f"{k}={v}" for k, v in lc.items()),
                    "overcrowded": ";".join(oc),
                })

        if self._frames % self._interval == 0 and self._last:
            parts = []
            for src in sorted(self._last):
                d = self._last[src]
                roi = " ".join(f"{k}:{v}" for k, v in d["roi"].items()) or "-"
                lc = " ".join(f"{k}:{v}" for k, v in d["lc"].items()) or "-"
                oc = "  OVERCROWDED:" + ",".join(d["oc"]) if d["oc"] else ""
                parts.append(f"cam{src}[n={d['total']} roi({roi}) lc({lc}){oc}]")
            print(f"[analytics f{self._frames}] " + "   ".join(parts))

    def close(self):
        if self._last:
            print("[analytics] final cumulative line-crossing counts:")
            for src in sorted(self._last):
                lc = self._last[src]["lc"]
                print(f"  cam{src}: " + (", ".join(f"{k}={v}" for k, v in lc.items()) or "(none)"))
        if self._export_path and self._rows:
            with open(self._export_path, "w", newline="") as f:
                w = csv.DictWriter(
                    f, fieldnames=["frame", "cam", "total", "roi", "lc_cumulative", "overcrowded"])
                w.writeheader()
                w.writerows(self._rows)
            print(f"[analytics] exported {len(self._rows)} rows -> {self._export_path}")
