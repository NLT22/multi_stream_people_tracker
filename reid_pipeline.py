"""Backward-compatible wrapper for the main ReID application.

The project entrypoint is now:

    python -m src.main

This file is kept so older notes/commands that call `python reid_pipeline.py`
continue to work.
"""

from src.main import main


if __name__ == "__main__":
    main()
