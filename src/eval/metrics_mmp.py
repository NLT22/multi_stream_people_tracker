"""Backward-compat entry point for `python -m src.eval.metrics_mmp`.

The metric engine moved to src/eval/mmp_metrics/core.py and the CLI to
src/eval/mmp_metrics/cli.py. This shim preserves the documented module path.
"""

from src.eval.mmp_metrics.cli import main

if __name__ == "__main__":
    main()
