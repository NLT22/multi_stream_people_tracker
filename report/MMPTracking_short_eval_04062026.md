# MMPTracking_short Eval Rerun — 04/06/2026

Nguon lenh: `report/Remote_04062026.md`.

Pham vi rerun:

- Baseline: `configs/pipeline_mmp_nvdcf_realtime_baseline.yaml`
- Nearline remap tren prediction baseline
- Geometry-tuned: `configs/pipeline_mmp_nvdcf_realtime_geo_tuned.yaml`

Bo qua FastReID A/B vi da quyet dinh tam bo FastReID khoi workflow chinh.

## Ghi Chu Moi Truong

Host venv thieu `pyservicemaker`, nen pipeline DeepStream duoc chay bang system `python3`.
Da cai cac package Python toi thieu vao user site cua system Python:

```bash
python3 -m pip install --user --break-system-packages \
  pandas pyyaml numpy scipy motmetrics trackeval opencv-python-headless tqdm
```

Tracker config tam thoi duoc tao tai:

```text
output/eval_configs/nvdcf_accuracy_mmp_recall_output_reid.yaml
output/eval_configs/nvdcf_accuracy_mmp_recall_output_reid_geo.yaml
```

Hai config nay chi doi `onnxFile` tu `models/reid/swin_tiny_mmp_reid.onnx`
sang `output/reid_mmp/swin_tiny_mmp_reid.onnx` vi `models/reid/` dang root-owned.

## Retail Pred-Space

`retail_0` baseline/nearline bi `metrics_mmp` auto-detect pred-space thanh `1280x720`,
lam metric sai nang. Khi ep dung pred-space `640x360`, ket qua hop ly hon va duoc dung
trong bang tong hop ben duoi:

```bash
python3 -m src.eval.metrics_mmp \
  --short-root dataset/MMPTracking_short \
  --scene retail_0 \
  --pred-dir output/eval/mmp_retail_0_nvdcf_realtime_baseline \
  --pred-width 640 --pred-height 360
```

## Baseline

| Scene | MOTA | IDF1 | Global IDF1 | Pred IDs | FN | FP |
|---|---:|---:|---:|---:|---:|---:|
| lobby_0 | 90.5% | 89.5% | 0.6976 | 12 | 3916 | 28 |
| industry_safety_0 | 91.8% | 88.8% | 0.7446 | 9 | 3107 | 316 |
| office_0 | 86.6% | 92.8% | 0.7843 | 9 | 5152 | 1755 |
| cafe_shop_0 | 94.9% | 95.2% | 0.7663 | 8 | 1722 | 320 |
| retail_0 | 62.4% | 63.7% | 0.4072 | 19 | 17818 | 5475 |

- Avg Global IDF1, 5 scenes: `0.6800`
- Avg Global IDF1, excluding retail_0: `0.7482`
- Avg MOTA, 5 scenes: `85.2%`

## Nearline

| Scene | MOTA | IDF1 | Global IDF1 | Pred IDs | FN | FP |
|---|---:|---:|---:|---:|---:|---:|
| lobby_0 | 90.5% | 84.7% | 0.7954 | 9 | 3916 | 28 |
| industry_safety_0 | 91.8% | 89.4% | 0.7505 | 8 | 3107 | 316 |
| office_0 | 86.6% | 92.8% | 0.7843 | 9 | 5152 | 1755 |
| cafe_shop_0 | 94.9% | 95.2% | 0.7663 | 8 | 1722 | 320 |
| retail_0 | 62.4% | 63.7% | 0.4072 | 19 | 17818 | 5475 |

- Avg Global IDF1, 5 scenes: `0.7007`
- Avg Global IDF1, excluding retail_0: `0.7741`
- Avg MOTA, 5 scenes: `85.2%`

## Geo-Tuned

| Scene | MOTA | IDF1 | Global IDF1 | Pred IDs | FN | FP |
|---|---:|---:|---:|---:|---:|---:|
| lobby_0 | 90.5% | 91.7% | 0.7561 | 10 | 3912 | 28 |
| industry_safety_0 | 91.8% | 88.1% | 0.7255 | 9 | 3083 | 316 |
| office_0 | 86.6% | 92.8% | 0.8142 | 9 | 5146 | 1754 |
| cafe_shop_0 | 95.0% | 93.6% | 0.7805 | 8 | 1690 | 320 |
| retail_0 | 62.3% | 63.9% | 0.4180 | 16 | 17739 | 5572 |

- Avg Global IDF1, 5 scenes: `0.6989`
- Avg Global IDF1, excluding retail_0: `0.7691`
- Avg MOTA, 5 scenes: `85.2%`

## Best Per Scene

| Scene | Best Run | Global IDF1 | Note |
|---|---|---:|---|
| lobby_0 | nearline | 0.7954 | Gan moc 0.80, sua duoc split global |
| industry_safety_0 | nearline | 0.7505 | Tang nhe so voi baseline |
| office_0 | geo_tuned | 0.8142 | Vuot 0.80 |
| cafe_shop_0 | geo_tuned | 0.7805 | Tang nhe, local IDF1 giam |
| retail_0 | geo_tuned | 0.4180 | Van la bottleneck lon |

## Nhan Xet

Nearline la preset tot nhat neu nhin trung binh 4 scene non-retail: `0.7741`.
Geo-tuned co loi cho `office_0` va `cafe_shop_0`, nhung lam `industry_safety_0`
giam va khong vuot nearline tren `lobby_0`.

`retail_0` van la diem keo tong rat manh. MOTA/IDF1 local chi quanh `62-64%`,
Global IDF1 chi `0.4072-0.4180`. Loi chinh khong chi la global merge, ma la
recall/local tracking/detection trong retail: FN rat cao, dac biet cam_5/cam_6.

Khuyen nghi hien tai:

1. Dung nearline lam baseline MTMC chinh cho non-retail.
2. Giu geo-tuned nhu ung vien theo scene, khong thay global default ngay.
3. Tach retail thanh preset rieng de debug detection/local tracker truoc khi tune ReID/global merge.

## Throughput Benchmark

Nguon video:

```text
dataset/MMPTracking_short/lobby_0/cam1.mp4
```

Lenh baseline:

```bash
python3 scripts/benchmark_throughput.py \
  --source dataset/MMPTracking_short/lobby_0/cam1.mp4 \
  --cam-counts 1 2 4 6 8 10 12 16 20 \
  --duration 30 \
  --warmup 8 \
  --target-fps 10 \
  --nvinfer-config configs/models/nvinfer_yolov11_mmp.yml \
  --output-dir output/benchmark/mmp_yolo_baseline \
  --stop-on-fail
```

CSV:

```text
output/benchmark/mmp_yolo_baseline/throughput_20260604_160159.csv
```

| Cams | Inference interval | FPS/cam | FPS total | VRAM peak | Pass 10 FPS/cam |
|---:|---:|---:|---:|---:|---|
| 1 | 0 | 375.45 | 375.45 | 1086 MB | yes |
| 2 | 0 | 110.41 | 220.83 | 1139 MB | yes |
| 4 | 0 | 27.47 | 109.87 | 1275 MB | yes |
| 6 | 0 | 12.41 | 74.44 | 1434 MB | yes |
| 8 | 0 | 6.83 | 54.68 | 1637 MB | no |

Baseline `interval=0` dat toi da `6 camera` o muc `>=10 FPS/cam`.

Lenh interval sweep:

```bash
python3 scripts/benchmark_throughput.py \
  --source dataset/MMPTracking_short/lobby_0/cam1.mp4 \
  --cam-counts 4 8 12 16 20 \
  --duration 30 \
  --warmup 8 \
  --target-fps 10 \
  --nvinfer-config configs/models/nvinfer_yolov11_mmp.yml \
  --inference-intervals 0 1 2 4 \
  --output-dir output/benchmark/mmp_yolo_interval_sweep
```

CSV:

```text
output/benchmark/mmp_yolo_interval_sweep/throughput_20260604_162052.csv
```

| Interval | Infer every | Best passing cams | 4-cam FPS/cam | 8-cam FPS/cam | 12-cam FPS/cam | 16-cam FPS/cam | 20-cam FPS/cam |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1f | 4 | 27.73 | 6.93 | 2.95 | 1.70 | 1.02 |
| 1 | 2f | 4 | 30.84 | 7.52 | 3.47 | 1.92 | 1.23 |
| 2 | 3f | 4 | 33.61 | 8.55 | 3.79 | 2.07 | 1.32 |
| 4 | 5f | 4 | 38.59 | 9.46 | 4.19 | 2.34 | 1.49 |

Nhan xet throughput:

- Interval sweep khong dat muc `8 camera x 10 FPS/cam`; tot nhat la `8 cam` voi `interval=4`, dat `9.46 FPS/cam`, thieu nhe.
- `20 cam x 10 FPS/cam` chua kha thi trong benchmark hien tai; `interval=4` chi dat `1.49 FPS/cam`.
- VRAM khong phai bottleneck: 20 cam interval 4 chi peak khoang `2816 MB / 16311 MB`.
- GPU util o interval cao khong day 100%, nen bottleneck co the nam o decode/source duplication/mux/tracker/probe Python hon la YOLO inference rieng le.

## FPS Culprit Ablation

Lenh/script dung de tach stage:

```bash
venv/bin/python scripts/benchmark_fps_ablation.py \
  --source output/_bench_loop/cam1_loop19x.mp4 \
  --cam-counts 4 8 20 \
  --variants detector_only tracker_iou tracker_perf tracker_recall full_main \
  --duration 15 \
  --warmup 5 \
  --target-fps 10 \
  --nvinfer-config configs/models/nvinfer_yolov11_mmp.yml \
  --full-tracker-config output/eval_configs/nvdcf_accuracy_mmp_recall_output_reid.yaml \
  --output-dir output/benchmark/fps_ablation
```

CSV day du:

```text
output/benchmark/fps_ablation/fps_ablation_20260604_165023.csv
```

Sau do chay lai rieng `tracker_recall/full_main` voi warmup dai va `--stop-on-fail`
de loai thoi gian build ReID engine:

```text
output/benchmark/fps_ablation/fps_ablation_20260604_170031.csv
```

Bang ablation day du:

| Variant | 4 cam FPS/cam | 8 cam FPS/cam | 20 cam FPS/cam | Nhan xet |
|---|---:|---:|---:|---|
| detector_only | 161.09 | 37.08 | 5.42 | YOLO khong phai bottleneck o 8 cam; 20 cam da nghen source/decode/mux/batch |
| tracker_iou | 160.45 | 36.89 | 5.36 | IoU tracker gan nhu khong them chi phi |
| tracker_perf | 160.28 | 36.85 | 5.34 | NvDCF perf khong them chi phi dang ke |
| tracker_recall | 20.87 | 5.45 | skipped | NvDCF recall/ReID extraction la diem roi FPS lon |
| full_main | 20.94 | 5.44 | skipped | Gallery/OSD/Python khong giam them nhieu so voi tracker_recall |

Ket luan thu pham FPS:

1. `NvDCF recall` voi ReID extraction/Swin tracker config la bottleneck chinh cho pipeline MTMC hien tai.
2. `detector_only`, `tracker_iou`, va `tracker_perf` deu du suc chay 8 cam tren 10 FPS/cam.
3. 20 cam van khong dat ke ca detector-only (`5.42 FPS/cam`), nen bai toan 20 cam con co bottleneck rieng o source/decode/mux/batch scaling.
4. Gallery/OSD/Python layer khong phai thu pham chinh: `full_main` gan bang `tracker_recall`.

Huong toi uu FPS tiep theo:

- Neu uu tien 20 cam realtime, can bo ReID extraction frame-level trong NvDCF tracker, hoac chay ReID sparse/nearline rieng.
- Dung `nvdcf_perf`/IoU cho local tracking realtime, xuat tracklets, roi nearline ReID/global merge sau.
- Tach 20 cam thanh nhieu process/nhom camera neu source/decode/mux scaling tiep tuc gioi han detector-only.

## Realtime Lite Follow-up

Sau khi ap dung huong realtime theo DeepStream skill:

- Tracker: `configs/tracker/nvdcf_perf_mmp_lite.yaml`
- Pipeline preset: `configs/pipeline_mmp_realtime_20cam.yaml`
- Detector: `configs/models/nvinfer_yolov11_mmp_iv4.yml` (`interval=4`, infer moi 5 frame)
- Runtime: headless, tat gallery, tat OSD, tat tiler, `sync=0`

Ket qua benchmark moi:

| Cams | FPS/cam | FPS total | Min | Max | Samples | Pass 10 FPS/cam |
|---:|---:|---:|---:|---:|---:|---|
| 4 | 162.1 | 648.6 | 635.8 | 662.6 | 7 | yes |
| 8 | 38.8 | 310.0 | 309.6 | 310.2 | 9 | yes |
| 12 | 16.5 | 198.2 | 197.8 | 198.4 | 9 | yes |
| 16 | 9.0 | 143.8 | 141.6 | 144.2 | 9 | no |

GPU:

| Cams | VRAM mean | VRAM peak | GPU mean | GPU peak | Temp | Power |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 974 MB | 1163 MB | 90.4% | 98% | 73 C | 153.1 W |
| 8 | 1221 MB | 1267 MB | 92.0% | 99% | 74 C | 158.1 W |
| 12 | 1413 MB | 1464 MB | 94.1% | 100% | 75 C | 162.1 W |
| 16 | 1619 MB | 1731 MB | 92.9% | 100% | 75 C | 155.8 W |

Nhan xet:

- Realtime lite tang kha nang chay tu `6 cam` baseline cu len `12 cam` dat target `>=10 FPS/cam`.
- `20 cam x 10 FPS` can it nhat `200 FPS total`; tai `12 cam` da gan muc nay (`198.2 FPS total`), nhung `16 cam` roi xuong `143.8 FPS total`.
- VRAM khong phai bottleneck: `12 cam` chi peak `1464/16311 MB` va `16 cam` chi peak `1731/16311 MB`.
- GPU da gan tran: mean `90-94%`, peak `98-100%`.
- Thu pham hien tai la tong compute/throughput cua GPU-side pipeline khi scale stream, chu khong con la ReID tracker. Can giam detector cost tiep (`interval` cao hon, input 512/416, INT8/model nho hon) hoac tach 20 cam thanh nhieu pipeline/process.
