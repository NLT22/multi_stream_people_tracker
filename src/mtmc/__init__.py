"""MDX-style micro-batch Multi-Target Multi-Camera (MTMC) tracking.

The offline anchor-guided method (src/eval/offline_anchor_faithful.py) clusters the
WHOLE clip at once, so it can't stream. This package runs the same algorithm family
(hierarchical clustering + Hungarian reassignment) INCREMENTALLY on micro-batches of
*tracklets* (not frames), keeping persistent anchor banks across batches — the
architecture NVIDIA Metropolis MTMC uses. See docs/production_todo.md §2.

- `tracklet.Tracklet`         — the per-tracklet message (perception -> bus schema)
- `incremental_mtmc.IncrementalMTMC` — stateful per-batch cross-camera ID assignment
- `run_incremental`           — offline simulation harness: replays a scene's tracklets
                                as micro-batches and writes eval-compatible predictions
"""
from .tracklet import Tracklet
from .incremental_mtmc import IncrementalMTMC, Anchor

__all__ = ["Tracklet", "IncrementalMTMC", "Anchor"]
