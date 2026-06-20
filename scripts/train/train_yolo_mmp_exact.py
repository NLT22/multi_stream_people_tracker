"""Train and evaluate YOLO on the official MMPTracking zip-derived dataset."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO


def _resolve_data_yaml(data_path: Path) -> Path:
    if not data_path.exists():
        raise SystemExit(f"dataset yaml not found: {data_path}")
    data = yaml.safe_load(data_path.read_text(encoding="utf-8")) or {}
    root = Path(data.get("path") or data_path.parent)
    if not root.is_absolute():
        root = (data_path.parent / root).resolve()
    train = root / data.get("train", "images/train")
    val = root / data.get("val", "images/val")
    if train.exists() and val.exists():
        return data_path

    local_root = data_path.parent.resolve()
    local_train = local_root / data.get("train", "images/train")
    local_val = local_root / data.get("val", "images/val")
    if local_train.exists() and local_val.exists():
        local_yaml = data_path.with_name(f"{data_path.stem}.local{data_path.suffix}")
        data["path"] = str(local_root)
        local_yaml.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        return local_yaml

    raise SystemExit(f"YOLO train/val folders not found from {data_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="dataset/mmp_exact_yolo/dataset.yaml")
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", default="output/train_exact")
    parser.add_argument("--name", default="yolo11n_mmp_exact")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--cache", choices=["ram", "disk"], default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--final-val", action="store_true", help="Run an extra full validation after training.")
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument("--onnx-dest", default=None)
    args = parser.parse_args()

    data_yaml = _resolve_data_yaml(Path(args.data).resolve())
    project = Path(args.project).resolve()
    run_dir = project / args.name

    if args.resume:
        last = run_dir / "weights" / "last.pt"
        if not last.exists():
            raise SystemExit(f"resume checkpoint not found: {last}")
        model = YOLO(str(last))
    else:
        model = YOLO(args.weights)

    print(f"[train] data={data_yaml}")
    print(f"[train] weights={args.weights if not args.resume else run_dir / 'weights' / 'last.pt'}")
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=str(project),
        name=args.name,
        exist_ok=True,
        patience=args.patience,
        cache=args.cache if args.cache else False,
        verbose=True,
        hsv_h=0.01,
        hsv_s=0.4,
        hsv_v=0.3,
        degrees=0.0,
        translate=0.1,
        scale=0.4,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.5,
    )
    print(f"[train] results={results}")

    best_pt = run_dir / "weights" / "best.pt"
    best = YOLO(str(best_pt))
    if args.final_val:
        print(f"[eval] validating best checkpoint: {best_pt}")
        val_results = best.val(
            data=str(data_yaml),
            split="val",
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=str(project),
            name=f"{args.name}_final_val",
            exist_ok=True,
        )
        print(f"[eval] metrics={val_results.results_dict}")

    if args.no_export:
        return

    exported = Path(str(best.export(format="onnx", imgsz=args.imgsz, opset=12, dynamic=True, simplify=True)))
    if args.onnx_dest:
        dest = Path(args.onnx_dest)
    else:
        dest = run_dir / "weights" / exported.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if exported.resolve() != dest.resolve():
        shutil.copy2(exported, dest)
    print(f"[export] onnx={dest}")


if __name__ == "__main__":
    main()
