# ReID manual identity labels

Manual grouping of scene-local `(scene, person_id)` tracks into real people,
produced by `scripts/datasets/reid_label_app.py`. Tracked in git so the labeling
is portable: commit/push here, pull on another machine, and keep editing.

- `labels_<env>.json` — `{ "scene|orig_id": "P<n>" | "JUNK", ... }` per environment.
- Track keys (`scene|orig_id`) are reproducible from the crop cache, so on another
  machine: have `dataset/MMPTracking_10minute_reid_cache/` + regenerate the proposal
  (`scripts/datasets/consolidate_reid_identities.py`), then run the labeler app.
