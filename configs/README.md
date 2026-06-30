# Config Reference

The active configs are the production YOLO11 + NvDCF + SGIE ReID path.

## Pipeline Presets (MMP production)

| Preset | Use |
|--------|-----|
| `pipeline_mmp_nvdcf_online_sgie_reid0.yaml` | **Recommended default.** NvDCF internal ReID off; SGIE drives global IDs. Lower VRAM / more FPS. |
| `pipeline_mmp_nvdcf_online_sgie.yaml` | Quality preset. Keeps NvDCF internal ReID. **Ties reid0 on IDF1** (~0.81) — global IDs come from the SGIE, so the extra ReID buys ~0. Keep only if local-track continuity matters. |

## Detector

`configs/models/nvinfer_yolov11_mmp_retailclean.yml` (used by **both** presets)

- ONNX: `models/yolov11/yolo11n_mmp_retailclean.onnx` — the **deployed** detector,
  retrained on retail-cleaned labels (precision 0.62→0.94)
- class count: 1 person class
- mode: FP16
- parser: `models/yolov8/libnvds_infercustomparser_yolov8.so`

`nvinfer_yolov11_mmp.yml` (→ `yolo11n_mmp.onnx`) is the original detector, kept for
rollback; `nvinfer_yolov11_mmp_int8.yml` is the experimental INT8 variant.

The parser filename still says YOLOv8 because the custom parser function handles
the YOLO tensor layout used here. The detector model itself is YOLO11.

## SGIE ReID

`configs/models/nvinfer_reid_swin_sgie_all.yml`

- ONNX: `models/reid/swin_tiny_mmp_reid_all.onnx`
- runs on PGIE person crops
- exports `output-tensor-meta`
- uses stretch preprocessing (`maintain-aspect-ratio: 0`), which matched the
  validated clean-crop ReID preprocessing

## Tracker

| Config | Meaning |
|--------|---------|
| `nvdcf_accuracy_mmp_recall_sgie.yaml` | Quality tracker. Keeps NvDCF internal ReID for reassociation, but does not export tracker ReID tensors. |
| `nvdcf_accuracy_mmp_recall_sgie_reid0.yaml` | Performance tracker. Disables NvDCF internal ReID; SGIE still supplies ReID embeddings for global IDs. |

The Python export/gallery path reads SGIE tensors from object metadata. Tracker
ReID tensors are intentionally disabled in both production configs.

## Sources

| File | Use |
|------|-----|
| `configs/sources/val_20cam_mixed.txt` | Current 20-camera mixed validation source list. |
| `configs/sources/rtsp_cameras.txt` | RTSP source template. |
| `configs/sources/video_files.txt` | Local file source template. |
| `configs/sources/video_files_docker.txt` | Docker-mounted source template. |

## MTMC (warehouse — separate pipeline)

The AI-City warehouse path is **kept separate** from MMP (disjoint cameras,
geometry-first cross-camera ID). Do not point MMP eval at these.

| Config | Notes |
|--------|-------|
| `pipeline_mtmc_nvdcf_online_sgie_reid0[_1280].yaml` | MTMC pipeline (YOLO11n@960/@1280 + MTMC Swin ReID). Run with `--no-tiler`. |
| `nvinfer_yolov11_mtmc[_640/_1280].yml` | MTMC detector configs. |
| `nvinfer_reid_swin_sgie_mtmc.yml` | MTMC Swin ReID. |
| `nvdcf_accuracy_mtmc_sgie_reid0.yaml` | MTMC tracker. |

## Analytics zones

`configs/analytics/nvdsanalytics_*.txt` — ROI occupancy / line-crossing /
overcrowding rules drawn in the web console's ROI editor (`webui/`). The `_w022`
files drive the warehouse custom-analytics demo.

---

Generated `*.runtime_b*_gpu*.yml` files and TensorRT `.engine` files are build
artifacts. They are gitignored and can be deleted safely.
