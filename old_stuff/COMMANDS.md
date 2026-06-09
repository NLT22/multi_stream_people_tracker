# COMMANDS — Multi-Stream People Tracker

Tất cả lệnh chạy từ thư mục gốc project với venv đã activate:

```bash
source venv/bin/activate
```

---

## 0. Tạo MMPTracking_short từ raw dataset

### Yêu cầu: cấu trúc thư mục dataset gốc

Script `create_mmp_short.py` đọc thẳng từ file zip — **không cần extract trước**. Zip phải nằm đúng vị trí sau:

```
dataset/MMPTracking/
└── MMPTracking_validation/
    └── validation/
        ├── images/
        │   └── 64pm/
        │       ├── cafe_shop_0.zip
        │       ├── cafe_shop_1.zip
        │       ├── ...                 ← 24 file zip ảnh (mỗi file ~1.4–3.7 GB)
        │       └── retail_7.zip
        ├── labels/
        │   └── 64pm/
        │       ├── cafe_shop_0.zip
        │       ├── ...                 ← 24 file zip GT JSON (~140 MB tổng)
        │       └── retail_7.zip
        └── calibrations/
            ├── cafe_shop/
            │   └── calibrations.json
            ├── industry_safety/
            │   └── calibrations.json
            ├── lobby/
            │   └── calibrations.json
            ├── office/
            │   └── calibrations.json
            └── retail/
                └── calibrations.json
```

> Dataset gốc tải từ: [MMPTracking @ paperswithcode](https://paperswithcode.com/dataset/mmptracking)  
> Giải nén file tải về → đặt thư mục `MMPTracking_validation/` vào `dataset/MMPTracking/`.

### Chạy script

```bash
python scripts/create_mmp_short.py
```

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--mmp-root` | `dataset/MMPTracking` | Thư mục gốc chứa `MMPTracking_validation/` |
| `--output` | `dataset/MMPTracking_short` | Thư mục output |
| `--scenes` | tất cả 24 | Chỉ xử lý scene chỉ định |
| `--max-frames` | `1500` | Số frame mỗi camera (1500 = 60s @ 25fps) |
| `--fps` | `25` | Framerate video output |
| `--jobs` | `2` | Số scene xử lý song song (mỗi scene extract ~2GB; không nên > 3) |
| `--keep-extracted` | — | Giữ lại frames đã extract sau khi encode video |

**Ví dụ chỉ tạo một số scenes:**
```bash
python scripts/create_mmp_short.py --scenes lobby_0 lobby_1 cafe_shop_0 --jobs 3
```

**Dùng root khác (nếu đặt dataset ở nơi khác):**
```bash
python scripts/create_mmp_short.py --mmp-root /data/MMPTracking --output /data/MMPTracking_short
```

**Yêu cầu disk:**
- Input zips: ~48 GB
- Temp frames (xóa sau mỗi scene): ~2 GB/scene tại một thời điểm
- Output MMPTracking_short: ~1.4 GB

---

## 1. Chuẩn bị dataset YOLO

Chuyển `MMPTracking_short` sang format Ultralytics YOLO.

```bash
python scripts/mmp_to_yolo.py
```

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--short-root` | `dataset/MMPTracking_short` | Thư mục chứa dataset short |
| `--output-dir` | `dataset/mmp_yolo` | Nơi ghi output |
| `--sample-rate` | `5` | Lấy 1 frame mỗi N frame (5 → 5fps) |
| `--min-height` | `20` | Bỏ box thấp hơn N pixel |
| `--min-width` | `8` | Bỏ box hẹp hơn N pixel |
| `--min-vis` | `0.3` | Bỏ box có < 30% diện tích trong frame |

Output: `dataset/mmp_yolo/` gồm `images/train`, `images/val`, `labels/train`, `labels/val`, `dataset.yaml`.  
Val split = scene cuối mỗi môi trường (5 scenes); train = 19 scenes còn lại.
Nếu `dataset.yaml` được tạo trong Docker, path có thể là `/app/dataset/mmp_yolo`; script train sẽ tự tạo `dataset.local.yaml` khi chạy trực tiếp bằng venv host.

---

## 2. Train YOLO

Fine-tune YOLO11n trên MMPTracking_short để detect người trong môi trường indoor.

```bash
python scripts/train_yolo_mmp.py
```

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--data` | `dataset/mmp_yolo/dataset.yaml` | Path tới dataset.yaml |
| `--epochs` | `30` | Số epoch tối đa |
| `--batch` | `16` | Batch size |
| `--imgsz` | `640` | Kích thước ảnh đầu vào |
| `--device` | `0` | GPU id (`0`, `1`, hoặc `cpu`) |
| `--workers` | `4` | DataLoader worker threads |
| `--weights` | `yolo11n.pt` | Weights khởi đầu; dùng MTA model để warm-start: `output/train/yolo11n_mta/weights/best.pt` |
| `--freeze` | `0` | Freeze N layer backbone đầu (0 = train toàn bộ; 10 = chỉ train detection head) |
| `--patience` | `10` | Early stopping: dừng sau N epoch không cải thiện mAP50 |
| `--resume` | — | Resume từ checkpoint cuối (`output/train/yolo11n_mmp/weights/last.pt`) |
| `--project` | `output/train` | Thư mục lưu run |
| `--name` | `yolo11n_mmp` | Tên run |

Output: `output/train/yolo11n_mmp/weights/best.pt` và `models/yolov11/yolo11n_mmp.onnx`.

**Ví dụ warm-start từ MTA model (nhanh hơn, thường tốt hơn):**
```bash
python scripts/train_yolo_mmp.py \
    --weights output/train/yolo11n_mta/weights/best.pt \
    --epochs 20 --patience 8
```

---

## 3. Train ReID

Fine-tune Swin-Tiny ReID trên MMPTracking_short. Crop người trực tiếp từ video + GT.

```bash
python scripts/finetune_reid_mmp.py
```

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--short-root` | `dataset/MMPTracking_short` | Thư mục dataset short |
| `--output` | `output/reid_mmp` | Thư mục lưu checkpoint và ONNX |
| `--epochs` | `40` | Số epoch tối đa |
| `--pk-p` | `24` | P trong P×K sampler: số người mỗi batch |
| `--pk-k` | `4` | K trong P×K sampler: số ảnh mỗi người → batch size = P×K = 96 |
| `--accum-steps` | `2` | Gradient accumulation → effective batch = 192 |
| `--lr` | `3.5e-4` | Learning rate (backbone dùng lr×0.1) |
| `--sample-rate` | `5` | Lấy 1 frame mỗi N frame để crop |
| `--min-w` | `20` | Bỏ crop nhỏ hơn N pixel chiều ngang |
| `--min-h` | `40` | Bỏ crop nhỏ hơn N pixel chiều dọc |
| `--min-imgs-pid` | `4` | Bỏ person có ít hơn N crop sau filter |
| `--early-stop` | `8` | Dừng sau N epoch không cải thiện similarity gap |
| `--workers` | `4` | DataLoader workers |
| `--batches-per-epoch` | `200` | Số PK batch mỗi epoch; `0` = phủ xấp xỉ toàn bộ crop |
| `--grad-ckpt` | — | Bật gradient checkpointing (tiết kiệm ~400MB VRAM, chậm hơn ~20%) |
| `--resume` | — | Resume hoặc warm-start từ checkpoint `.pth` (tự thay classifier head nếu khác số class) |

Best checkpoint và early stopping dùng `val_gap` trên 5 scene validation của MMPTracking_short, không dùng train gap.
Output: `output/reid_mmp/best.pth` và `output/reid_mmp/swin_tiny_mmp_reid.onnx`.

**Ví dụ warm-start từ MTA ReID model:**
```bash
python scripts/finetune_reid_mmp.py \
    --resume output/reid_v2/best.pth \
    --epochs 30 --early-stop 8
```

**Nếu OOM (4GB VRAM):**
```bash
python scripts/finetune_reid_mmp.py \
    --pk-p 16 --pk-k 4 --accum-steps 4 --grad-ckpt
```

## 4. Chạy pipeline

Chạy DeepStream pipeline trên một scene MMPTracking_short.

```bash
python -m src.main \
    --config configs/pipeline_mta.yaml \
    --mmp-short-dataset dataset/MMPTracking_short:lobby_0 \
    --no-display --no-sync \
    --export-predictions output/eval/mmp_lobby0
```

| Tham số | Mô tả |
|---------|-------|
| `--config` | File YAML pipeline (detection + tracker + reid defaults) |
| `--mmp-short-dataset ROOT:SCENE` | Chạy một scene từ MMPTracking_short |
| `--mmp-dataset ROOT:SCENE` | Chạy một scene từ MMPTracking đầy đủ (cần extract trước) |
| `--nvinfer-config` | Override detector config (mặc định lấy từ `--config`) |
| `--tracker-config` | Override tracker config |
| `--no-display` | Headless, không mở cửa sổ |
| `--no-sync` | Bỏ clock sync, chạy nhanh nhất có thể |
| `--export-predictions DIR` | Ghi CSV predictions ra thư mục để eval |
| `--show-gt` | Overlay GT boxes (xanh lá) lên display |
| `--trim-seconds N` | Cắt mỗi video còn N giây trước khi chạy |
| `--max-sources N` | Chỉ load N camera đầu |
| `--save-video PATH` | Ghi output ra file MP4 |

**Dùng model MMP mới train:**
```bash
python -m src.main \
    --config configs/pipeline_mmp.yaml \
    --mmp-short-dataset dataset/MMPTracking_short:lobby_0 \
    --no-display --no-sync \
    --export-predictions output/eval/mmp_lobby0
```

---

## 5. Eval tracking (per-camera MOTA + Global IDF1)

Đánh giá kết quả tracking với MMPTracking_short.

```bash
python -m src.eval.metrics_mmp \
    --short-root dataset/MMPTracking_short \
    --scene lobby_0 \
    --pred-dir output/eval/mmp_lobby0
```

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--short-root` | `dataset/MMPTracking_short` | Thư mục MMPTracking_short |
| `--scene` | *(bắt buộc)* | Scene cần eval, vd `lobby_0` |
| `--pred-dir` | *(bắt buộc)* | Thư mục chứa `cam_N_predictions.csv` |
| `--cameras` | tất cả | Chỉ eval camera thật chỉ định, vd: `--cameras 1 2 3 4` |
| `--iou-threshold` | `0.5` | IoU tối thiểu để match GT ↔ pred |
| `--min-height` | `20` | Filter box quá nhỏ trong GT space 640×360 |
| `--min-width` | `8` | Filter box quá hẹp trong GT space 640×360 |
| `--min-visibility` | `0.3` | Filter box ngoài frame |
| `--pred-width/--pred-height` | auto | Không gian bbox prediction; mặc định auto-detect |
| `--no-filter` | — | Tắt toàn bộ difficulty filter |

---

## 6. Offline merge Global ID

Sau khi chạy pipeline, merge các global ID bị fragment bằng embedding similarity.

```bash
python -m src.eval.offline_merge \
    --pred-dir output/eval/mmp_lobby0 \
    --out-dir  output/eval/mmp_lobby0_merged \
    --threshold 0.70 \
    --margin 0.05 \
    --min-gid-embeddings 6 \
    --min-tracklet-detections 10
```

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--pred-dir` | *(bắt buộc)* | Thư mục predictions gốc |
| `--out-dir` | *(bắt buộc)* | Thư mục ghi predictions sau merge |
| `--threshold` | `0.82` | Similarity tối thiểu để merge 2 global ID (`0.70` đang tốt hơn cho MMP) |
| `--margin` | `0.05` | Best candidate phải hơn runner-up ít nhất N |
| `--min-gid-embeddings` | `12` | Bỏ qua global ID có ít hơn N embeddings |
| `--min-tracklet-detections` | `20` | Bỏ qua tracklet quá ngắn |
| `--temporal-tolerance` | `0` | Cho phép merge GID overlap N frame (0 = strict) |
| `--dry-run` | — | Chỉ in merge plan, không ghi file |

---

## 7. Tune NvDCF + online merge

### 7.0a Realtime 20-camera throughput path

Preset này theo hướng DeepStream realtime nhẹ: YOLO MMP + NvDCF lite, tắt
tracker ReID/re-assoc, tắt gallery/OSD/tiler khi chạy headless. Dùng nó để đo
khả năng đạt 20 camera @ 10 FPS trước; Global-ID/IDF1 xử lý bằng nearline ở
các mục sau.

```bash
python -m src.main \
  --config configs/pipeline_mmp_realtime_20cam.yaml \
  --mmp-short-dataset dataset/MMPTracking_short:lobby_0
```

Benchmark throughput, dừng ngay khi fail target:

```bash
python scripts/benchmark_throughput.py \
  --source dataset/MMPTracking_short/lobby_0/cam1.mp4 \
  --cam-counts 4 8 12 16 20 \
  --target-fps 10 \
  --stop-on-fail \
  --nvinfer-config configs/models/nvinfer_yolov11_mmp_iv4.yml \
  --tracker-config configs/tracker/nvdcf_perf_mmp_lite.yaml
```

Mốc hiện tại trên RTX 5060 Ti: `12 cam` đạt `16.5 FPS/cam`, `16 cam` fail ở
`9.0 FPS/cam`. VRAM chỉ peak `1.7GB/16GB`, GPU peak `100%`, nên bottleneck sau
khi bỏ ReID tracker là compute/throughput tổng thể của detector + decode/mux
scaling, không phải bộ nhớ.

Nếu cần so thủ phạm FPS:

```bash
python scripts/benchmark_fps_ablation.py \
  --source dataset/MMPTracking_short/lobby_0/cam1.mp4 \
  --variants detector_only tracker_lite full_lite tracker_recall full_main \
  --cam-counts 4 8 20 \
  --target-fps 10 \
  --stop-on-fail
```

### 7.0b Realtime simple gallery

Nếu mục tiêu là chạy realtime ổn định, ưu tiên preset này trước. Nó bỏ online duplicate merge và tắt ambiguity rejection để giảm split GID trong lúc live.

```bash
SCENE=lobby_0

python -m src.main \
  --config configs/pipeline_mmp_nvdcf_realtime_simple.yaml \
  --mmp-short-dataset dataset/MMPTracking_short:${SCENE} \
  --no-display --no-sync \
  --geo-weight 0.30 \
  --export-predictions output/eval/mmp_${SCENE}_nvdcf_realtime_simple

python -m src.eval.metrics_mmp \
  --short-root dataset/MMPTracking_short \
  --scene ${SCENE} \
  --pred-dir output/eval/mmp_${SCENE}_nvdcf_realtime_simple
```

### 7.0c Baseline frozen vs geometry-tuned

Baseline frozen giữ lại mốc tốt cũ; geometry-tuned dùng calibration thận trọng hơn: chỉ để geometry chọn giữa các candidate có ReID score gần nhau.

```bash
for SCENE in lobby_0 industry_safety_0 office_0 cafe_shop_0; do
  for PRESET in baseline geo_tuned; do
    CFG=configs/pipeline_mmp_nvdcf_realtime_${PRESET}.yaml
    OUT=output/eval/mmp_${SCENE}_nvdcf_realtime_${PRESET}

    python -m src.main \
      --config ${CFG} \
      --mmp-short-dataset dataset/MMPTracking_short:${SCENE} \
      --no-display --no-sync \
      --geo-weight 0.30 \
      --export-predictions ${OUT}

    python -m src.eval.metrics_mmp \
      --short-root dataset/MMPTracking_short \
      --scene ${SCENE} \
      --pred-dir ${OUT}
  done
done
```

### 7.0d Nearline remap events

Mô phỏng service nearline: realtime gallery xuất GID tạm, sau mỗi cửa sổ vài giây sinh event `source_gid -> target_gid`, rồi ghi predictions đã remap để eval.

```bash
SCENE=lobby_0
PRED=output/eval/mmp_${SCENE}_nvdcf_realtime_baseline
OUT=output/eval/mmp_${SCENE}_nvdcf_realtime_baseline_nearline

python -m src.eval.nearline_merge \
  --pred-dir ${PRED} \
  --out-dir ${OUT} \
  --threshold 0.65 --margin 0.03 \
  --min-gid-embeddings 6 --min-tracklet-detections 10 \
  --mmp-short-root dataset/MMPTracking_short \
  --scene ${SCENE} \
  --geo-weight 0.25 \
  --geo-sample-step 5 \
  --geo-min-overlaps 8 \
  --window-frames 125 \
  --delay-frames 50

python -m src.eval.metrics_mmp \
  --short-root dataset/MMPTracking_short \
  --scene ${SCENE} \
  --pred-dir ${OUT}
```

Output phụ:

- `remap_events.csv`: event remap theo thời gian, dùng để mô phỏng nearline service.
- `merge_map.csv`: các merge được chấp nhận.
- `global_id_remap.csv`: bảng remap cuối cùng.

Lưu ý: nearline remap tối ưu MTMC Global IDF1, có thể làm per-camera IDF1 giảm vì ID được đổi lại sau delay.

### 7.0c2 Sweep nearline để tune Global IDF1

Script sweep dùng các prediction baseline đã export, chạy `nearline_merge + metrics_mmp`, rồi sort tham số theo micro Global IDF1.

```bash
python scripts/sweep_nearline_mmp.py \
  --scenes lobby_0 industry_safety_0 office_0 cafe_shop_0 lobby_3 industry_safety_4 office_2 cafe_shop_3 \
  --thresholds 0.62,0.65 \
  --margins 0.02,0.03 \
  --geo-weights 0.25,0.35 \
  --geo-min-overlaps 8 \
  --window-frames 125 \
  --out-root output/eval/nearline_sweep_nonretail_small
```

Kết quả hiện tại trên 8 scene non-retail:

- Best: `threshold=0.62`, `margin=0.02 hoặc 0.03`, `geo_weight=0.25`, `geo_min_overlaps=8`, `window_frames=125`.
- Micro Global IDF1: `0.7444`.
- Scene đã đạt/tiệm cận mục tiêu: `lobby_0=0.8365`, `industry_safety_0=0.8360`, `office_0=0.8192`, `lobby_3=0.8082`.
- Scene kéo tụt: `cafe_shop_3=0.5549`, `industry_safety_4=0.6129`, `office_2=0.7085`.

Điều này nghĩa là merge threshold không còn là bottleneck duy nhất; muốn tổng quát `Global IDF1 > 80%` cần cải thiện baseline local/ReID trên các validation scene yếu.

### 7.0d Validation scenes: export baseline trước, nearline sau

`nearline_merge` không tự chạy detector/tracker. Nó cần thư mục prediction đã có đủ `cam_*_predictions.csv`, `tracklets.csv`, `tracklet_embeddings.npz`. Nếu báo thiếu `tracklets.csv`, nghĩa là scene đó chưa được chạy pipeline với `--export-predictions`.

Pha 1: chạy realtime baseline để sinh prediction + tracklet embedding:

```bash
for SCENE in cafe_shop_3 industry_safety_4 lobby_3 office_2; do
  OUT=output/eval/mmp_${SCENE}_nvdcf_realtime_baseline
  rm -rf ${OUT}

  python -m src.main \
    --config configs/pipeline_mmp_nvdcf_realtime_baseline.yaml \
    --mmp-short-dataset dataset/MMPTracking_short:${SCENE} \
    --no-display --no-sync \
    --geo-weight 0.30 \
    --export-predictions ${OUT}
done
```

`retail_7` nên chạy riêng bằng tracker low-memory nếu preset baseline bị OOM:

```bash
SCENE=retail_7
OUT=output/eval/mmp_${SCENE}_nvdcf_realtime_baseline
rm -rf ${OUT}

python -m src.main \
  --config configs/pipeline_mmp_nvdcf_realtime_baseline.yaml \
  --tracker-config configs/tracker/nvdcf_accuracy_mmp_retail_lowmem.yaml \
  --mmp-short-dataset dataset/MMPTracking_short:${SCENE} \
  --no-display --no-sync \
  --geo-weight 0.30 \
  --export-predictions ${OUT}
```

Pha 2: chạy nearline + eval, tự skip scene chưa có baseline export:

```bash
for SCENE in cafe_shop_3 industry_safety_4 lobby_3 office_2 retail_7; do
  PRED=output/eval/mmp_${SCENE}_nvdcf_realtime_baseline
  OUT=output/eval/mmp_${SCENE}_nvdcf_realtime_baseline_nearline

  if [ ! -f "${PRED}/tracklets.csv" ] || [ ! -f "${PRED}/tracklet_embeddings.npz" ]; then
    echo "[skip] ${SCENE}: missing baseline export in ${PRED}"
    continue
  fi

  python -m src.eval.nearline_merge \
    --pred-dir ${PRED} \
    --out-dir ${OUT} \
    --threshold 0.65 --margin 0.03 \
    --min-gid-embeddings 6 --min-tracklet-detections 10 \
    --mmp-short-root dataset/MMPTracking_short \
    --scene ${SCENE} \
    --geo-weight 0.25 \
    --geo-sample-step 5 \
    --geo-min-overlaps 8 \
    --window-frames 125 \
    --delay-frames 50

  python -m src.eval.metrics_mmp \
    --short-root dataset/MMPTracking_short \
    --scene ${SCENE} \
    --pred-dir ${OUT}
done
```

Sweep nhẹ geometry margin nếu `geo_tuned` chưa ổn:

```bash
for SCENE in lobby_0 industry_safety_0 office_0 cafe_shop_0; do
  for MARGIN in 0.04 0.08 0.12; do
    OUT=output/eval/mmp_${SCENE}_nvdcf_geo_margin${MARGIN}

    python -m src.main \
      --config configs/pipeline_mmp_nvdcf_realtime_geo_tuned.yaml \
      --mmp-short-dataset dataset/MMPTracking_short:${SCENE} \
      --no-display --no-sync \
      --geo-weight 0.30 \
      --geometry-reid-margin ${MARGIN} \
      --export-predictions ${OUT}

    python -m src.eval.metrics_mmp \
      --short-root dataset/MMPTracking_short \
      --scene ${SCENE} \
      --pred-dir ${OUT}
  done
done
```

Preset online merge:

- Pipeline: `configs/pipeline_mmp_nvdcf_online.yaml`
- Tracker: `configs/tracker/nvdcf_accuracy_mmp_recall.yaml`
- Online merge đã bật trong config, có calibration-assisted scoring khi chạy với `--mmp-short-dataset`.

### 7.1 Chạy một scene với NvDCF online merge

```bash
SCENE=lobby_0

python -m src.main \
  --config configs/pipeline_mmp_nvdcf_online.yaml \
  --mmp-short-dataset dataset/MMPTracking_short:${SCENE} \
  --no-display --no-sync \
  --geo-weight 0.25 \
  --export-predictions output/eval/mmp_${SCENE}_nvdcf_online

python -m src.eval.metrics_mmp \
  --short-root dataset/MMPTracking_short \
  --scene ${SCENE} \
  --pred-dir output/eval/mmp_${SCENE}_nvdcf_online
```

### 7.1b Retail low-memory NvDCF

`retail_0` có 6 camera và đông người, NvDCF recall preset có thể lỗi `cudaErrorMemoryAllocation` trong `cuDCFv2`. Dùng config nhẹ hơn này để chạy retail.

```bash
SCENE=retail_0
OUT=output/eval/mmp_${SCENE}_nvdcf_realtime_lowmem

rm -rf ${OUT}

python -m src.main \
  --config configs/pipeline_mmp_nvdcf_realtime_simple.yaml \
  --tracker-config configs/tracker/nvdcf_accuracy_mmp_retail_lowmem.yaml \
  --mmp-short-dataset dataset/MMPTracking_short:${SCENE} \
  --no-display --no-sync \
  --geo-weight 0.30 \
  --export-predictions ${OUT}

python -m src.eval.metrics_mmp \
  --short-root dataset/MMPTracking_short \
  --scene ${SCENE} \
  --pred-dir ${OUT}
```

### 7.2 Conservative online merge sweep

Ưu tiên chạy trước trên `lobby_0`, `industry_safety_0`, `retail_0`.

```bash
for SCENE in lobby_0 industry_safety_0 retail_0; do
  for THR in 0.72 0.74 0.76; do
    OUT=output/eval/mmp_${SCENE}_nvdcf_online_t${THR}

    python -m src.main \
      --config configs/pipeline_mmp_nvdcf_online.yaml \
      --mmp-short-dataset dataset/MMPTracking_short:${SCENE} \
      --no-display --no-sync \
      --geo-weight 0.25 \
      --global-merge-threshold ${THR} \
      --global-merge-margin 0.04 \
      --global-merge-interval 10 \
      --export-predictions ${OUT}

    python -m src.eval.metrics_mmp \
      --short-root dataset/MMPTracking_short \
      --scene ${SCENE} \
      --pred-dir ${OUT}
  done
done
```

### 7.3 A/B online merge vs offline merge

```bash
SCENE=lobby_0

# Online merge trực tiếp
python -m src.main \
  --config configs/pipeline_mmp_nvdcf_online.yaml \
  --mmp-short-dataset dataset/MMPTracking_short:${SCENE} \
  --no-display --no-sync \
  --geo-weight 0.25 \
  --global-merge-threshold 0.74 \
  --global-merge-margin 0.04 \
  --export-predictions output/eval/mmp_${SCENE}_nvdcf_online_t074

python -m src.eval.metrics_mmp \
  --short-root dataset/MMPTracking_short \
  --scene ${SCENE} \
  --pred-dir output/eval/mmp_${SCENE}_nvdcf_online_t074

# Offline geo merge từ raw NvDCF output để làm mốc so sánh
python -m src.main \
  --config configs/pipeline_mmp.yaml \
  --tracker-config configs/tracker/nvdcf_accuracy_mmp_recall.yaml \
  --mmp-short-dataset dataset/MMPTracking_short:${SCENE} \
  --no-display --no-sync \
  --disable-global-merge \
  --geo-weight 0.25 \
  --export-predictions output/eval/mmp_${SCENE}_nvdcf_raw

python -m src.eval.offline_merge \
  --pred-dir output/eval/mmp_${SCENE}_nvdcf_raw \
  --out-dir output/eval/mmp_${SCENE}_nvdcf_raw_geo_merged \
  --threshold 0.65 --margin 0.03 \
  --min-gid-embeddings 6 --min-tracklet-detections 10 \
  --mmp-short-root dataset/MMPTracking_short \
  --scene ${SCENE} \
  --geo-weight 0.25 \
  --geo-sample-step 5 \
  --geo-min-overlaps 20

python -m src.eval.metrics_mmp \
  --short-root dataset/MMPTracking_short \
  --scene ${SCENE} \
  --pred-dir output/eval/mmp_${SCENE}_nvdcf_raw_geo_merged
```

Nếu online merge làm Global IDF1 tăng nhưng Pred IDs vẫn cao, giảm threshold nhẹ (`0.72`). Nếu Global IDF1 tụt hoặc IDFP tăng mạnh, tăng threshold (`0.76`) hoặc margin (`0.06`).

---

## 8. Luồng làm việc đầy đủ

```
# Bước 1: Tạo YOLO dataset
python scripts/mmp_to_yolo.py

# Bước 2: Train YOLO
python scripts/train_yolo_mmp.py --weights output/train/yolo11n_mta/weights/best.pt

# Bước 3: Train ReID
python scripts/finetune_reid_mmp.py --resume output/reid_v2/best.pth

# Bước 4: Chạy pipeline trên từng scene
python -m src.main \
    --config configs/pipeline_mmp.yaml \
    --mmp-short-dataset dataset/MMPTracking_short:lobby_0 \
    --no-display --no-sync \
    --export-predictions output/eval/mmp_lobby0

# Bước 5: Offline merge (tuỳ chọn)
python -m src.eval.offline_merge \
    --pred-dir output/eval/mmp_lobby0 \
    --out-dir  output/eval/mmp_lobby0_merged \
    --threshold 0.70 \
    --margin 0.05 \
    --min-gid-embeddings 6 \
    --min-tracklet-detections 10

# Bước 6: Eval
python -m src.eval.metrics_mmp \
    --short-root dataset/MMPTracking_short \
    --scene lobby_0 \
    --pred-dir output/eval/mmp_lobby0_merged
```

---

## 9. Docker

### Yêu cầu

- Docker Engine + NVIDIA Container Toolkit (`nvidia-docker2`)
- `docker compose` (v2+)

### Services có sẵn

| Service | Image | Mục đích |
|---------|-------|----------|
| `yolo_train` | `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime` | Convert dataset + train YOLO |
| `reid_train_mmp` | `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime` | Train ReID trên MMPTracking_short |
| `reid_train` | `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime` | Train ReID trên MTA (legacy) |
| `tracker` | `multi_stream_people_tracker:latest` (build local) | Chạy DeepStream pipeline |

---

### Build image DeepStream

Chỉ cần build lần đầu hoặc khi thay đổi `Dockerfile` / `requirements-runtime.txt`.

```bash
docker compose build tracker
```

---

### Train YOLO (Docker)

Chạy với tham số mặc định:

```bash
docker compose run --rm yolo_train
```

Override tham số qua biến môi trường:

```bash
YOLO_EPOCHS=50 YOLO_BATCH=8 YOLO_PATIENCE=15 docker compose run --rm yolo_train
```

Warm-start từ MTA model (copy `best.pt` vào container qua volume):

```bash
# best.pt phải nằm trong output/train/yolo11n_mta/weights/ (đã mount vào /app/output/train)
YOLO_WEIGHTS=output/train/yolo11n_mta/weights/best.pt docker compose run --rm yolo_train
```

| Biến môi trường | Mặc định | Tương đương tham số |
|----------------|----------|---------------------|
| `YOLO_EPOCHS` | `30` | `--epochs` |
| `YOLO_BATCH` | `16` | `--batch` |
| `YOLO_PATIENCE` | `10` | `--patience` |
| `YOLO_WEIGHTS` | `yolo11n.pt` | `--weights` |
| `YOLO_WORKERS` | `4` | `--workers` |

Output tự động ghi ra `output/train/yolo11n_mmp/` và `models/yolov11/yolo11n_mmp.onnx` trên host.
Các service train dùng `shm_size: "16gb"` để tránh lỗi PyTorch DataLoader `Unexpected bus error` / thiếu `/dev/shm`.

Nếu đã từng chạy bằng `sudo docker compose run` rồi chuyển sang chạy trực tiếp bằng venv host, sửa owner các thư mục output/cache trước:

```bash
sudo chown -R $USER:$USER output dataset/mmp_yolo models/yolov11
```

---

### Train ReID MMPTracking (Docker)

Chạy với tham số mặc định:

```bash
docker compose run --rm reid_train_mmp
```

Warm-start từ MTA model:

```bash
# best.pth phải nằm trong output/reid_mmp/ hoặc output/reid_v2/ trên host
# Mount thêm nếu cần:
REID_RESUME=output/reid_v2/best.pth docker compose run --rm \
    -v $(pwd)/output/reid_v2:/app/output/reid_v2:ro \
    reid_train_mmp
```

Giảm memory nếu OOM (4GB VRAM):

```bash
REID_PKP=16 REID_PKK=4 docker compose run --rm reid_train_mmp \
    bash -c "pip install timm tqdm opencv-python-headless pandas -q &&
             python scripts/finetune_reid_mmp.py
               --pk-p 16 --pk-k 4 --accum-steps 4 --grad-ckpt
               --output output/reid_mmp"
```

| Biến môi trường | Mặc định | Tương đương tham số |
|----------------|----------|---------------------|
| `REID_EPOCHS` | `40` | `--epochs` |
| `REID_PKP` | `24` | `--pk-p` |
| `REID_PKK` | `4` | `--pk-k` |
| `REID_RESUME` | _(trống)_ | `--resume` |

Output tự động ghi ra `output/reid_mmp/` trên host.

---

### Chạy pipeline DeepStream (Docker)

**Scene MMPTracking_short:**

```bash
docker compose run --rm tracker \
    python3 -m src.main \
        --config configs/pipeline_mta.yaml \
        --mmp-short-dataset dataset/MMPTracking_short:lobby_0 \
        --no-display --no-sync \
        --export-predictions output/eval/mmp_lobby0
```

**Với model MMP mới train:**

```bash
docker compose run --rm tracker \
    python3 -m src.main \
        --config configs/pipeline_mmp.yaml \
        --mmp-short-dataset dataset/MMPTracking_short:lobby_0 \
        --no-display --no-sync \
        --export-predictions output/eval/mmp_lobby0
```

**Offline merge + eval trong container:**

```bash
docker compose run --rm tracker \
    python3 -m src.eval.offline_merge \
        --pred-dir output/eval/mmp_lobby0 \
        --out-dir  output/eval/mmp_lobby0_merged \
        --threshold 0.70 \
        --margin 0.05 \
        --min-gid-embeddings 6 \
        --min-tracklet-detections 10

docker compose run --rm tracker \
    python3 -m src.eval.metrics_mmp \
        --short-root dataset/MMPTracking_short \
        --scene lobby_0 \
        --pred-dir output/eval/mmp_lobby0_merged
```

---

### Luồng Docker đầy đủ

```bash
# 1. Build DeepStream image (lần đầu)
docker compose build tracker

# 2. Train YOLO
YOLO_WEIGHTS=output/train/yolo11n_mta/weights/best.pt \
YOLO_EPOCHS=30 YOLO_PATIENCE=10 \
docker compose run --rm yolo_train

# 3. Train ReID
REID_RESUME=output/reid_v2/best.pth \
REID_EPOCHS=30 \
docker compose run --rm reid_train_mmp

# 4. Chạy pipeline (lặp lại cho từng scene)
for SCENE in lobby_0 lobby_1 cafe_shop_0 office_0 retail_0; do
    docker compose run --rm tracker \
        python3 -m src.main \
            --config configs/pipeline_mmp.yaml \
            --mmp-short-dataset dataset/MMPTracking_short:${SCENE} \
            --no-display --no-sync \
            --export-predictions output/eval/mmp_${SCENE}
done

# 5. Offline merge + eval mỗi scene
for SCENE in lobby_0 lobby_1 cafe_shop_0 office_0 retail_0; do
    docker compose run --rm tracker \
        python3 -m src.eval.offline_merge \
            --pred-dir output/eval/mmp_${SCENE} \
            --out-dir  output/eval/mmp_${SCENE}_merged \
            --threshold 0.70 \
            --margin 0.05 \
            --min-gid-embeddings 6 \
            --min-tracklet-detections 10
done
```
