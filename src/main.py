"""
Multi-Stream People Tracker — Full Pipeline Entry Point

This is the full assembled pipeline. Work through the milestones/ scripts
first to understand each component, then use this to run everything together.

Usage:
    python -m src.main --config configs/pipeline.yaml
    python -m src.main  # uses default config path

Milestones covered by this entry point:
    Milestone 1-3:  use milestones/01_*.py through 03_*.py instead
    Milestone 4+:   this script is the target (with progressive TODO unlocks)
"""

import argparse
import sys

from src.config.loader import PipelineConfig
from src.pipeline.builder import PipelineBuilder
from src.pipeline.sources import load_uris
from src.utils.platform_utils import check_display, gpu_info_str


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Stream People Tracker (DeepStream 9.0 / pyservicemaker)"
    )
    parser.add_argument(
        "--config",
        default="configs/pipeline.yaml",
        help="Path to pipeline.yaml (default: configs/pipeline.yaml)",
    )
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────────────────
    try:
        config = PipelineConfig.from_yaml(args.config)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print(config.summary())
    print(f"[INFO]  GPU: {gpu_info_str()}")

    if not check_display():
        print(
            "[WARNING] No display detected (DISPLAY/WAYLAND_DISPLAY not set).\n"
            "          nveglglessink will fail. Use a local session or forward X11."
        )

    # ── Load source URIs ─────────────────────────────────────────────────────
    try:
        uris = load_uris(config.source_mode, config.active_source_config)
    except (FileNotFoundError, ValueError, NotADirectoryError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    config.num_sources = len(uris)
    print(f"[INFO]  Loaded {len(uris)} source(s):")
    for i, uri in enumerate(uris):
        print(f"          [{i}] {uri}")

    # Clamp batch_size to actual source count
    if config.batch_size != config.num_sources:
        print(
            f"[INFO]  Adjusting batch_size: {config.batch_size} → {config.num_sources} "
            "(must equal source count)"
        )
        config.batch_size = config.num_sources

    # ── Build and run pipeline ───────────────────────────────────────────────
    builder = PipelineBuilder(config, uris)
    pipeline = builder.build()

    try:
        pipeline.start()
        print("[INFO]  Pipeline running. Press Ctrl+C to stop.")
        pipeline.wait()
    except KeyboardInterrupt:
        print("\n[INFO]  Stopped by user.")


if __name__ == "__main__":
    main()
