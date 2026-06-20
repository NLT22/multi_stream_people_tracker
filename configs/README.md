# Config Reference

The active configs are the production YOLO11 + NvDCF + SGIE ReID path.

## Pipeline Presets

| Preset | Use |
|--------|-----|
| `pipeline_mmp_nvdcf_online_sgie.yaml` | Quality default: best IDF1, meets the 20-cam 10 FPS target on the current dataset. |
| `pipeline_mmp_nvdcf_online_sgie_reid0.yaml` | Performance default: lower VRAM and more FPS, small IDF1 drop. |

## Detector

`configs/models/nvinfer_yolov11_mmp.yml`

- ONNX: `models/yolov11/yolo11n_mmp.onnx`
- class count: 1 person class
- mode: FP16
- parser: `models/yolov8/libnvds_infercustomparser_yolov8.so`

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

Generated `*.runtime_b*_gpu*.yml` files and TensorRT `.engine` files are build
artifacts. They are gitignored and can be deleted safely.
