"""Train/export FastReID on MMPTracking_short.

This wrapper keeps the remote-machine workflow in one place:

    python scripts/train/train_fastreid_mmp.py prepare
    python scripts/train/train_fastreid_mmp.py train --num-gpus 1
    python scripts/train/train_fastreid_mmp.py export

The default config is deployment-friendly ResNet50 without IBN.  The IBN model
can train, but its ONNX uses Split/InstanceNorm patterns that failed inside
DeepStream TensorRT during local tests.
"""

from __future__ import annotations

import argparse
import os
import site
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FASTREID_ROOT = ROOT / "third_party" / "fast-reid"
DEFAULT_CONFIG = ROOT / "configs" / "fastreid" / "mmp_bagtricks_R50_deploy.yml"
DEFAULT_DATASET = ROOT / "dataset" / "fastreid_mmp"
DEFAULT_OUTPUT = ROOT / "output" / "fastreid_mmp" / "bagtricks_R50_deploy"
DEFAULT_ONNX = ROOT / "models" / "reid" / "fastreid_mmp_R50_deploy.onnx"


def _run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("[fastreid-mmp]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def _python() -> str:
    return sys.executable


def _ensure_fastreid_path() -> None:
    if not FASTREID_ROOT.exists():
        raise SystemExit(
            f"Missing {FASTREID_ROOT}. Clone FastReID first:\n"
            "  git clone https://github.com/JDAI-CV/fast-reid third_party/fast-reid"
        )
    _patch_fastreid_for_python312()

    target = str(FASTREID_ROOT.resolve())
    site_pkgs = [Path(p) for p in site.getsitepackages()]
    site_pkgs.append(Path(site.getusersitepackages()))

    for site_pkg in site_pkgs:
        if site_pkg.exists():
            pth = site_pkg / "fastreid_local.pth"
            try:
                current = pth.read_text().strip() if pth.exists() else ""
                if current != target:
                    pth.write_text(target + "\n")
                    print(f"[fastreid-mmp] wrote {pth} -> {target}")
            except OSError as exc:
                print(f"[fastreid-mmp] skip writing {pth}: {exc}")
            return

    print("[fastreid-mmp] no site-packages found; will rely on PYTHONPATH.")


def _patch_fastreid_for_python312() -> None:
    """Apply tiny compatibility fixes needed by official FastReID on Python 3.12."""
    patches = {
        FASTREID_ROOT / "fastreid" / "evaluation" / "testing.py": {
            "from collections import Mapping, OrderedDict":
                "from collections import OrderedDict\nfrom collections.abc import Mapping",
        },
        FASTREID_ROOT / "fastreid" / "data" / "build.py": {
            "from collections import Mapping":
                "from collections.abc import Mapping",
        },
    }
    for path, replacements in patches.items():
        if not path.exists():
            continue
        text = path.read_text()
        updated = text
        for old, new in replacements.items():
            updated = updated.replace(old, new)
        if updated != text:
            path.write_text(updated)
            print(f"[fastreid-mmp] patched Python 3.12 compatibility: {path}")


def _env(dataset_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["FASTREID_DATASETS"] = str(dataset_root)
    old_pythonpath = env.get("PYTHONPATH")
    paths = [str(FASTREID_ROOT)]
    if old_pythonpath:
        paths.append(old_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def prepare(args: argparse.Namespace) -> None:
    _ensure_fastreid_path()
    if args.install_deps:
        _run([_python(), "-m", "pip", "install", "-r", "requirements-fastreid.txt"])

    cmd = [
        _python(),
        "scripts/datasets/mmp_to_fastreid_market.py",
        "--short-root",
        str(args.short_root),
        "--output",
        str(args.dataset_root),
        "--sample-rate",
        str(args.sample_rate),
        "--min-w",
        str(args.min_w),
        "--min-h",
        str(args.min_h),
        "--min-visible-ratio",
        str(args.min_visible_ratio),
        "--query-per-pid-cam",
        str(args.query_per_pid_cam),
        "--jpeg-quality",
        str(args.jpeg_quality),
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    _run(cmd)


def train(args: argparse.Namespace) -> None:
    _ensure_fastreid_path()
    cmd = [
        _python(),
        str(FASTREID_ROOT / "tools" / "train_net.py"),
        "--config-file",
        str(args.config),
        "--num-gpus",
        str(args.num_gpus),
    ]
    opts = list(args.opts)
    if opts and opts[0] == "--":
        opts = opts[1:]
    if args.output_dir:
        opts.extend(["OUTPUT_DIR", str(args.output_dir)])
    if args.batch_size:
        opts.extend(["SOLVER.IMS_PER_BATCH", str(args.batch_size)])
    if args.test_batch_size:
        opts.extend(["TEST.IMS_PER_BATCH", str(args.test_batch_size)])
    if args.max_epoch:
        opts.extend(["SOLVER.MAX_EPOCH", str(args.max_epoch)])
    if args.no_amp:
        opts.extend(["SOLVER.AMP.ENABLED", "False"])
    if opts:
        cmd.extend(["--opts", *opts])
    if args.resume:
        cmd.append("--resume")
    _run(cmd, env=_env(args.dataset_root))


def evaluate(args: argparse.Namespace) -> None:
    _ensure_fastreid_path()
    weights = _resolve_weights(args.weights, args.output_dir)
    cmd = [
        _python(),
        str(FASTREID_ROOT / "tools" / "train_net.py"),
        "--config-file",
        str(args.config),
        "--num-gpus",
        str(args.num_gpus),
        "--eval-only",
        "--opts",
        "MODEL.WEIGHTS",
        str(weights),
        "OUTPUT_DIR",
        str(args.output_dir),
    ]
    _run(cmd, env=_env(args.dataset_root))


def _resolve_weights(weights_arg: Path | None, output_dir: Path) -> Path:
    if weights_arg:
        return weights_arg
    for name in ("model_best.pth", "model_final.pth"):
        path = output_dir / name
        if path.exists():
            return path
    candidates = sorted(output_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime)
    if candidates:
        return candidates[-1]
    raise SystemExit(f"No checkpoint found in {output_dir}; pass --weights explicitly.")


def export(args: argparse.Namespace) -> None:
    _ensure_fastreid_path()
    weights = _resolve_weights(args.weights, args.output_dir)
    args.onnx_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.onnx_path.with_suffix(".tmp.onnx")

    code = f"""
import sys
from pathlib import Path
import torch
import onnx
from onnxsim import simplify

sys.path.insert(0, {str(FASTREID_ROOT)!r})
from fastreid.config import get_cfg
from fastreid.modeling.meta_arch import build_model
from fastreid.utils.checkpoint import Checkpointer

cfg = get_cfg()
cfg.merge_from_file({str(args.config)!r})
cfg.merge_from_list([
    "MODEL.WEIGHTS", {str(weights)!r},
    "MODEL.DEVICE", {args.device!r},
    "OUTPUT_DIR", {str(args.output_dir)!r},
])
cfg.defrost()
cfg.MODEL.BACKBONE.PRETRAIN = False
if cfg.MODEL.HEADS.POOL_LAYER == "FastGlobalAvgPool":
    cfg.MODEL.HEADS.POOL_LAYER = "GlobalAvgPool"
cfg.freeze()

model = build_model(cfg)
Checkpointer(model).load(cfg.MODEL.WEIGHTS)
model.eval()
x = torch.randn({args.batch_size}, 3, cfg.INPUT.SIZE_TEST[0], cfg.INPUT.SIZE_TEST[1], device=model.device)
with torch.no_grad():
    torch.onnx.export(
        model,
        x,
        {str(tmp)!r},
        input_names=["input"],
        output_names=["features"],
        opset_version={args.opset},
        do_constant_folding=True,
        dynamo=False,
    )
m = onnx.load({str(tmp)!r})
ms, check = simplify(m)
assert check, "ONNX simplification failed"
onnx.save(ms, {str(args.onnx_path)!r})
Path({str(tmp)!r}).unlink(missing_ok=True)
print({str(args.onnx_path)!r})
for y in ms.graph.output:
    dims = [d.dim_value or d.dim_param for d in y.type.tensor_type.shape.dim]
    print("output", y.name, dims)
"""
    _run([_python(), "-c", code])


def clean_engines(args: argparse.Namespace) -> None:
    for path in args.model_dir.glob("*.engine"):
        print(f"[fastreid-mmp] remove {path}")
        path.unlink()
    for path in args.model_dir.glob("*.onnx_*_gpu*_*.engine"):
        print(f"[fastreid-mmp] remove {path}")
        path.unlink()


def all_steps(args: argparse.Namespace) -> None:
    prepare(args)
    train(args)
    export(args)


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="Create FastReID Market1501-style crops.")
    p.set_defaults(func=prepare)
    p.add_argument("--short-root", type=Path, default=ROOT / "dataset" / "MMPTracking_short")
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--sample-rate", type=int, default=25)
    p.add_argument("--min-w", type=int, default=20)
    p.add_argument("--min-h", type=int, default=40)
    p.add_argument("--min-visible-ratio", type=float, default=0.30)
    p.add_argument("--query-per-pid-cam", type=int, default=1)
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--install-deps", action="store_true")

    p = sub.add_parser("train", help="Train or resume FastReID.")
    p.set_defaults(func=train)
    _add_common(p)
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--test-batch-size", type=int)
    p.add_argument("--max-epoch", type=int)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("opts", nargs=argparse.REMAINDER,
                   help="Extra FastReID opts after '--', e.g. -- SOLVER.BASE_LR 0.0002")

    p = sub.add_parser("eval", help="Evaluate a trained checkpoint.")
    p.set_defaults(func=evaluate)
    _add_common(p)
    p.add_argument("--weights", type=Path)
    p.add_argument("--num-gpus", type=int, default=1)

    p = sub.add_parser("export", help="Export a DeepStream-friendly legacy ONNX.")
    p.set_defaults(func=export)
    _add_common(p)
    p.add_argument("--weights", type=Path)
    p.add_argument("--onnx-path", type=Path, default=DEFAULT_ONNX)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--device", default="cuda")

    p = sub.add_parser("clean-engines", help="Remove cached TensorRT engines.")
    p.set_defaults(func=clean_engines)
    p.add_argument("--model-dir", type=Path, default=ROOT / "models" / "reid")

    p = sub.add_parser("all", help="Prepare dataset, train, then export ONNX.")
    p.set_defaults(func=all_steps)
    p.add_argument("--short-root", type=Path, default=ROOT / "dataset" / "MMPTracking_short")
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--sample-rate", type=int, default=25)
    p.add_argument("--min-w", type=int, default=20)
    p.add_argument("--min-h", type=int, default=40)
    p.add_argument("--min-visible-ratio", type=float, default=0.30)
    p.add_argument("--query-per-pid-cam", type=int, default=1)
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--install-deps", action="store_true")
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--test-batch-size", type=int)
    p.add_argument("--max-epoch", type=int)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--weights", type=Path)
    p.add_argument("--onnx-path", type=Path, default=DEFAULT_ONNX)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--device", default="cuda")
    p.add_argument("opts", nargs=argparse.REMAINDER)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
