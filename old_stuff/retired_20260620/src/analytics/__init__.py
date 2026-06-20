"""Spatial analytics on the multi-camera global tracks: named zones, movement
routes, entry/exit points, and per-zone occupancy over time (production_todo §4).
Operates on the world ground-plane foot points (tracklet_bev.csv) keyed by global_id.
"""
from .zones import Zone, load_zones, save_zones, assign_zone, auto_grid_zones
__all__ = ["Zone", "load_zones", "save_zones", "assign_zone", "auto_grid_zones"]
