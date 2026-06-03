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

---

## 8. Docker

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
        --config configs/pipeline_mta.yaml \
        --mmp-short-dataset dataset/MMPTracking_short:lobby_0 \
        --nvinfer-config  configs/models/nvinfer_yolov11_mmp.yml \
        --tracker-config  configs/tracker/nvdeepsort_reid_swin_mmp.yaml \
        --no-display --no-sync \
        --export-predictions output/eval/mmp_lobby0
```

**Offline merge + eval trong container:**

```bash
docker compose run --rm tracker \
    python3 -m src.eval.offline_merge \
        --pred-dir output/eval/mmp_lobby0 \
        --out-dir  output/eval/mmp_lobby0_merged

docker compose run --rm tracker \
    python3 -m src.eval.metrics \
        --gt-dir  dataset/mta/MTA_ext_short/test \
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
            --config configs/pipeline_mta.yaml \
            --mmp-short-dataset dataset/MMPTracking_short:${SCENE} \
            --nvinfer-config  configs/models/nvinfer_yolov11_mmp.yml \
            --tracker-config  configs/tracker/nvdeepsort_reid_swin_mmp.yaml \
            --no-display --no-sync \
            --export-predictions output/eval/mmp_${SCENE}
done

# 5. Offline merge + eval mỗi scene
for SCENE in lobby_0 lobby_1 cafe_shop_0 office_0 retail_0; do
    docker compose run --rm tracker \
        python3 -m src.eval.offline_merge \
            --pred-dir output/eval/mmp_${SCENE} \
            --out-dir  output/eval/mmp_${SCENE}_merged
done
```
