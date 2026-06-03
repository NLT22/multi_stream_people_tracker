# COMMANDS — Multi-Stream People Tracker

Tất cả lệnh chạy từ thư mục gốc project với venv đã activate:

```bash
source venv/bin/activate
```

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
| `--grad-ckpt` | — | Bật gradient checkpointing (tiết kiệm ~400MB VRAM, chậm hơn ~20%) |
| `--resume` | — | Resume hoặc warm-start từ checkpoint `.pth` (tự thay classifier head nếu khác số class) |

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

---

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
    --config configs/pipeline_mta.yaml \
    --mmp-short-dataset dataset/MMPTracking_short:lobby_0 \
    --nvinfer-config configs/models/nvinfer_yolov11_mmp.yml \
    --tracker-config configs/tracker/nvdeepsort_reid_swin_mmp.yaml \
    --no-display --no-sync \
    --export-predictions output/eval/mmp_lobby0
```

---

## 5. Eval tracking (per-camera MOTA + Global IDF1)

Đánh giá kết quả tracking với MTA dataset.

```bash
python -m src.eval.metrics \
    --gt-dir dataset/mta/MTA_ext_short/test \
    --pred-dir output/eval/mta_run1
```

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--gt-dir` | *(bắt buộc)* | Thư mục GT (MTA split) |
| `--pred-dir` | *(bắt buộc)* | Thư mục chứa `cam_N_predictions.csv` |
| `--cameras` | tất cả | Chỉ eval camera chỉ định, vd: `--cameras 0 1 2` |
| `--iou-threshold` | `0.5` | IoU tối thiểu để match GT ↔ pred |
| `--min-height` | `60` | Filter box quá nhỏ |
| `--min-width` | `20` | Filter box quá hẹp |
| `--min-visibility` | `0.3` | Filter box ngoài frame |
| `--no-filter` | — | Tắt toàn bộ difficulty filter |

---

## 6. Offline merge Global ID

Sau khi chạy pipeline, merge các global ID bị fragment bằng embedding similarity.

```bash
python -m src.eval.offline_merge \
    --pred-dir output/eval/mmp_lobby0 \
    --out-dir  output/eval/mmp_lobby0_merged
```

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--pred-dir` | *(bắt buộc)* | Thư mục predictions gốc |
| `--out-dir` | *(bắt buộc)* | Thư mục ghi predictions sau merge |
| `--threshold` | `0.82` | Similarity tối thiểu để merge 2 global ID |
| `--margin` | `0.05` | Best candidate phải hơn runner-up ít nhất N |
| `--min-gid-embeddings` | `12` | Bỏ qua global ID có ít hơn N embeddings |
| `--min-tracklet-detections` | `20` | Bỏ qua tracklet quá ngắn |
| `--temporal-tolerance` | `0` | Cho phép merge GID overlap N frame (0 = strict) |
| `--dry-run` | — | Chỉ in merge plan, không ghi file |

---

## 7. Luồng làm việc đầy đủ

```
# Bước 1: Tạo YOLO dataset
python scripts/mmp_to_yolo.py

# Bước 2: Train YOLO
python scripts/train_yolo_mmp.py --weights output/train/yolo11n_mta/weights/best.pt

# Bước 3: Train ReID
python scripts/finetune_reid_mmp.py --resume output/reid_v2/best.pth

# Bước 4: Tạo nvinfer config cho model mới (copy và sửa path ONNX)
cp configs/models/nvinfer_yolov11_mta.yml configs/models/nvinfer_yolov11_mmp.yml
# → sửa onnxFile: "models/yolov11/yolo11n_mmp.onnx"

# Bước 5: Tạo tracker config cho ReID mới (copy và sửa path ONNX)
cp configs/tracker/nvdeepsort_reid_swin_mta.yaml configs/tracker/nvdeepsort_reid_swin_mmp.yaml
# → sửa onnxFile: "output/reid_mmp/swin_tiny_mmp_reid.onnx"

# Bước 6: Chạy pipeline trên từng scene
python -m src.main \
    --config configs/pipeline_mta.yaml \
    --mmp-short-dataset dataset/MMPTracking_short:lobby_0 \
    --nvinfer-config configs/models/nvinfer_yolov11_mmp.yml \
    --tracker-config configs/tracker/nvdeepsort_reid_swin_mmp.yaml \
    --no-display --no-sync \
    --export-predictions output/eval/mmp_lobby0

# Bước 7: Offline merge (tuỳ chọn)
python -m src.eval.offline_merge \
    --pred-dir output/eval/mmp_lobby0 \
    --out-dir  output/eval/mmp_lobby0_merged

# Bước 8: Eval
python -m src.eval.metrics \
    --gt-dir dataset/mta/MTA_ext_short/test \
    --pred-dir output/eval/mmp_lobby0_merged
```
