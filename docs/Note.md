# Notes — cross-camera ID concepts

## Q1: What is a tracklet?

A **tracklet** = the chain of detections of one person as followed by the
**single-camera tracker**, within **one camera**, under one local track ID. A
"mini-track" inside a single camera.

### Hierarchy (smallest → largest)
```
detection   →   tracklet   →   global identity (person)
(1 box,         (many boxes,    (the real person, across
 1 frame,        1 camera,       ALL cameras and ALL time)
 1 camera)       1 local_track_id)
```
- **Detection** = one bounding box, one frame, one camera.
- **Tracklet** = all detections the tracker linked as the same object in that one
  camera = all rows sharing the same `(cam_id, local_track_id)`. Spans many frames,
  one camera.
- **Global identity** = the actual person — should be one ID across all cameras and
  the whole clip — usually made of several tracklets.

### Concrete example
Person **A** appears as:
- cam1: tracklet `t1` (frames 0–40) → occlusion → tracklet `t5` (frames 60–100).
  Two tracklets, same person, same camera (**fragmentation**).
- cam2: tracklet `t9` (frames 0–100).

So person A = **3 tracklets** (`t1`, `t5`, `t9`). The cross-camera job: group those
3 tracklets → one global ID.
```
cam1:  t1 ████████        t5 ███████████        ← 2 tracklets (A fragmented)
cam2:  t9 ██████████████████████████            ← 1 tracklet (A)
                    ↑ all 3 are really person A → want SAME global ID
```

### Where it lives in our data
In `cam_0_predictions.csv`, each **row = one detection**:
`frame_no_cam, cam_id, local_track_id, global_id, box`. All rows with the same
`(cam_id, local_track_id)` = **one tracklet**. `tracklets.csv` = one row per tracklet
(`start_frame, end_frame, num_detections`, ...).

---

## Q2: The three association strategies (with end-to-end pipelines)

Shared mini-example: 2 cameras, 3 people (A,B,C). Person A is fragmented in cam1
(`t1` frames 0–40, then `t5` frames 60–100) and seen in cam2 as `t9`. Goal: t1, t5,
t9 all get the same global ID.

Each method differs in **(a) what embedding, (b) when, (c) how the global ID is decided.**

### Method 1 — Online gallery (project, realtime)
- **Embedding**: per detection, live; NvDCF emits a ReID vector for only ~16.8% of
  detections (sparse coverage).
- **Structure**: gallery `{global_id → running-mean prototype}` (one averaged vector
  per person) + `track_to_gid = {(cam, local_track_id) → global_id}`.
- **Decision (causal, greedy)**: when a new local track appears, match its embedding
  vs every prototype; assign best if ≥ threshold else mint a new GID; fold embedding
  into that GID's running mean.
```
frame 0:  cam1 t1 new → no match → GID 1, prototype₁ = emb(t1)
          cam2 t9 new → match prototype₁? yes → t9 = GID 1 (prototype updated)
frame 60: cam1 t5 NEW → match vs prototype₁ →
            ≥thr → t5 = GID 1 ✅ (A recovered)
            <thr → t5 = GID 2 ❌ (A split → ID explosion)
export per frame: global_id = track_to_gid[(cam, local_track_id)]
```
Realtime, but the decision is made once at track birth from past only — can't be revised.

### Method 2 — Offline anchor, tracklet-mean (project offline)
- **Embedding**: after the clip, average all embeddings of each tracklet → one mean
  vector per tracklet (~150 vectors/scene).
- **Decision**: cluster those ~150 means (agglomerative) into k identities; every
  detection inherits its tracklet's cluster label.
```
1. m(t1)=mean(t1 embs), m(t5)=..., m(t9)=...        (3 of ~150 points)
2. cluster ~150 means into k=7 → cluster 3 = {t1, t5, t9, ...}  (= person A)
3. assign: every detection of t1,t5,t9 → GID 3
```
Offline (full hindsight, so it can regroup t1+t5+t9), robust (averaging kills noise),
but coarse: ~150 decisions, no per-frame reasoning → two look-alikes' means can merge.

### Method 3 — OSNet per-detection (breakthrough, offline)
- **Embedding**: after the clip, run OSNet on every detection crop → one 512-d vector
  per detection (~148k, dense 100%).
- **Anchors**: cluster a sample of detection embeddings into k feature-banks (each
  anchor = a SET of exemplar embeddings for one identity).
- **Decision (fine-grained)**:
  1. every detection → cost = avg cosine distance to each anchor bank (k costs).
  2. per camera, per frame → Hungarian assigns each detection to a DISTINCT anchor
     (mutual exclusion: 2 people in one camera-frame can't be the same ID).
  3. sliding-window(15) majority vote per local track → stable ID.
```
per frame/camera (cam1 f30 has d_t1, d_B):
   Hungarian over {d_t1,d_B} × 7 anchors → d_t1→anchor_A, d_B→anchor_B (distinct)
cam1 f70: d_t5→anchor_A ; cam2: d_t9→anchor_A
vote per track: t1→A, t5→A, t9→A → all = global ID A ✅
```
Every detection decided individually (then voted), using dense strong embeddings +
per-frame mutual exclusion. Beats tracklet-mean (0.93 vs 0.75) because it fixes
ID-switches frame-by-frame and can't put two co-visible people on one ID. Cost: must
embed every crop + process the whole clip → offline, slow.

### Side-by-side
| | embedding unit | #decisions/scene | when | strength | cost |
|---|---|---|---|---|---|
| Online gallery | per-GID running mean | streaming (per new track) | live, causal, greedy | realtime | can't revise; sparse embeddings |
| Offline tracklet-mean | per-tracklet mean (~150) | once, offline | full hindsight | robust, cheap | coarse; merges look-alikes |
| OSNet per-detection | per-detection (~148k) | every det + window vote | full hindsight | per-frame mutual exclusion, finest | dense ReID on every crop → offline |

One line: **online = realtime but greedy/coarse; tracklet-mean = robust but coarse;
per-detection (dense + in-domain ReID) = most accurate but offline + expensive.**

---

## Q3: How is `k` (the number of anchors / identities) identified?

`k` = how many distinct people (identities) the scene has = how many anchor feature
banks / clusters to form. The clustering needs `k`. Three ways it's determined:

### A. The paper's way (estimate from appearance, per-scene tuned)
In `aic_hungarian_cluster.py` `get_people`:
1. Sample appearance embeddings from a few "anchor frames" across the clip.
2. `AgglomerativeClustering(distance_threshold=D, n_clusters=None)` → `k = #clusters`.
3. `D` (`distance_thers`) is **hand-tuned per scene** (e.g. 13, 19.5, 16, …).

So they pick a distance cutoff; everything closer than `D` is the same person. `k`
falls out of where the cutoff lands. Then `get_anchor` re-clusters into exactly `k`
to build the `k` feature banks.

### B. Our project's auto-`k` (unsupervised, no tuning) — `src/eval/offline_anchor.py`
We avoid a hand-tuned threshold by combining two signals and taking the larger:
```
k = max( dendrogram_gap , concurrency_floor )    # floor-preferred
```
- **dendrogram-gap** (`estimate_k`): build the agglomerative merge tree; sort the
  merge distances; the **largest jump** between consecutive merges marks the boundary
  between "within-person" and "between-person" merges → cut there → `k`. Appearance-
  based; **overshoots on look-alike scenes** (industry: gap=9 vs true 7).
- **concurrency-floor** (`concurrency_floor`): a geometry/physics lower bound — the
  **95th-percentile number of people simultaneously visible in any single camera**
  (counted per-frame from BEV; one camera can't show one person as two tracks). So
  `#people ≥ this`. Reliable; on every MMP scene tested it equalled the true count.
- We made the **floor primary** (gap only a fallback) because the floor was correct
  on all scenes while the gap overshoots. e.g. industry_3: gap=9, floor=7 → k=7.

### C. Oracle `k` (used in the OSNet breakthrough experiments)
For the "exact paper + fine-tuned OSNet" numbers (0.91–0.94) we passed `k =` GT
person count (= 7 for MMP) via `--oracle-k` / `--num-people`. This isolates the ReID
+ association quality from `k`-estimation error. In deployment you'd use auto-`k` (B)
or set `k` to the known headcount.

### Summary
| method | how k is found | tuning | reliability on MMP |
|---|---|---|---|
| Paper | agglomerative cut at a distance threshold | per-scene hand-tuned `D` | n/a (their data) |
| Project auto-k | `max(dendrogram-gap, concurrency-floor)`, floor-preferred | none | lands on true k=7 every scene |
| Oracle | GT headcount (`--num-people`) | n/a | exact (validation only) |

Key idea: the **concurrency floor** (max people seen at once in one camera) is the
most trustworthy `k` signal here — appearance-only estimation (gap / distance
threshold) is fooled by look-alikes.

---

## Q4: Which algorithm does the clustering?

There are **two different algorithms** doing two different jobs:

### 1. Grouping identities → Agglomerative (hierarchical) clustering
Both the paper and our project use **Agglomerative clustering** (`sklearn
AgglomerativeClustering`) to form anchors / group tracklets / estimate `k`.

How it works (bottom-up):
1. start with every point (embedding) as its own cluster;
2. repeatedly **merge the two closest clusters** (closeness = the *linkage*);
3. stop when there are `k` clusters (`n_clusters=k`) **or** when the nearest pair is
   farther than a cutoff (`distance_threshold=D`).

Parameters:
- **linkage** = how cluster–cluster distance is defined:
  - `ward` (default, what we use): merge the pair that least increases within-cluster
    variance (euclidean). Robust; best on MMP in our tests.
  - `average`: mean pairwise distance (this is what the paper's anchor *cost*, eq.1,
    effectively uses — avg cosine to the bank). We tried `cosine+average`: slightly
    worse than ward here (0.740 vs 0.754).
  - `complete`/`single`: max / min pairwise distance (we don't use).
- **metric** = euclidean on **L2-normalized** vectors ≈ cosine distance.
- `n_clusters=k` (we know k) **vs** `distance_threshold=D` (paper estimates k from D).

Why agglomerative, not k-means:
- can cut by a **distance threshold** to *discover* `k` (k-means needs `k` upfront);
- **deterministic** (k-means depends on random init);
- works with **cosine / precomputed distances** and non-spherical clusters
  (k-means assumes spherical euclidean blobs).

Used in: `estimate_k` (full tree for the gap), anchor building (`build_anchors`,
`get_anchor`), and project tracklet clustering (`cluster_anchors`).
Our experimental **cannot-link** variant is a *constrained* agglomerative (same
algorithm, forbids merging two tracklets co-visible in one camera-frame).

### 2. Per-frame assignment → Hungarian (NOT clustering)
The per-detection method also uses the **Hungarian algorithm**
(`scipy.optimize.linear_sum_assignment`) — but that is **assignment, not
clustering**: within one camera-frame it optimally matches the detections to the `k`
anchors so each detection gets a **distinct** anchor (mutual exclusion). Clustering
*creates* the `k` identities (anchors); Hungarian then *assigns* each detection to
one of them per frame; a sliding-window vote stabilizes it.

### Summary
| step | algorithm | role |
|---|---|---|
| estimate k / build anchors / group tracklets | **Agglomerative (hierarchical)** | create the `k` identities |
| per-frame detection→anchor | **Hungarian (linear_sum_assignment)** | assign, with mutual exclusion |
| smooth per-track | majority vote over a 15-frame window | stabilize IDs |

---

## Q5: What is a "prototype"?

A **prototype** = the single embedding that *represents one global identity* in the
**online gallery** — the running (incremental) **mean** of the good appearance
embeddings seen so far for that person. Think "template / centroid for the person."

- Gallery = `{global_id → prototype}` (one vector per person).
- When a new track appears, its embedding is compared (cosine) to **each prototype**;
  best match ≥ threshold → that global ID.
- On a match, the prototype is updated to fold in the new embedding (stays a mean).

So: **anchor (offline)** and **prototype (online)** are the same idea — a per-identity
appearance representative. Difference: an *anchor* is usually a **set of exemplars**
(feature bank) built offline from the whole clip; a *prototype* is **one running mean**
maintained online.

---

## Glossary — terms to know

**Detection** — one bounding box of a person, in one frame, in one camera (one row in
`cam_*_predictions.csv`).

**Tracklet** — all detections the single-camera tracker linked as the same object in
**one** camera = rows with the same `(cam_id, local_track_id)`. One person can be
several tracklets (fragmentation).

**local_track_id (SCT id)** — the per-camera track id from single-camera tracking. Only
unique *within* a camera; not a cross-camera identity.

**SCT (Single-Camera Tracking)** — tracking within one camera (here: NvDCF). Produces
tracklets / local_track_ids. (Paper uses BoT-SORT.)

**Global ID / global identity** — the real person, one id across **all cameras and all
time**. The output we solve for (column `global_id`).

**Cross-camera / MTMC association** — the step that groups tracklets across cameras into
global IDs.

**ReID (Re-Identification) model** — a CNN that maps a person crop → an appearance
**embedding**, trained so the same person’s crops are close, different people far.
(Ours: Swin-Tiny; paper: OSNet.)

**Embedding (appearance feature)** — the ReID output vector (e.g. 256-d Swin, 512-d
OSNet). Compared by **cosine** distance.

**Dense vs sparse (coverage)** — *dense* = an embedding for every detection (100%);
*sparse* = only some detections get one (our in-tracker ReID emits ~16.8%).

**Prototype** — online per-GID running-mean embedding (see Q5).

**Anchor / feature bank** — offline per-identity representative; a *set* of exemplar
embeddings for one person (captures pose/lighting variation). `k` anchors = `k`
identities.

**k** — number of distinct people / identities / anchors in the scene (see Q3).

**Agglomerative (hierarchical) clustering** — bottom-up clustering: merge closest
clusters until `k` (or a distance cutoff). Used to build anchors / group tracklets.

**Linkage** — how cluster–cluster distance is measured in agglomerative clustering:
`ward` (variance, our default), `average` (mean pairwise = paper cost), complete, single.

**Hungarian algorithm (linear_sum_assignment)** — optimal one-to-one bipartite
matching. Here: assign detections in a camera-frame to distinct anchors (mutual
exclusion). Assignment, **not** clustering.

**Sliding-window majority vote** — per track, take the per-frame anchor assignments in a
window (15 frames) and pick the most common → stabilizes IDs / fixes ID-switches.

**Mutual exclusion** — two people visible in the same camera-frame cannot be the same
global ID (enforced by per-frame Hungarian, or by a cannot-link constraint).

**Cannot-link** — a clustering constraint forbidding two items from being merged (we
tried: forbid merging two tracklets co-visible in one camera-frame).

**Online vs offline** — *online* = decide per frame as the stream arrives (causal,
realtime, can't revise); *offline* = post-process the whole clip with full hindsight
(can revise; not realtime).

**Greedy assignment** — commit a global ID the moment a track appears, never revised
(the online gallery). Cause of ID errors accumulating over time.

**Fragmentation** — one real person split into multiple tracklets/IDs (occlusion, missed
detections, re-entry).

**ID-switch** — a track’s id jumps to a different person (e.g. two look-alikes cross).

**Look-alike** — different people with near-identical appearance (e.g. industry safety
uniforms) → ambiguous embeddings → ID-switches.

**ID explosion** — the count of distinct global IDs growing unbounded over time (re-
entries/fragmentation spawning new IDs beyond the true headcount).

**Concurrency floor** — robust lower bound on `k`: 95th-pct of people simultaneously
visible in any single camera (from BEV per-frame counts).

**Dendrogram gap** — the largest jump in agglomerative merge distances; an
appearance-based `k` estimate (overshoots on look-alikes).

**Gallery** — online store `{global_id → prototype}` matched against per camera.

**BEV (Bird’s-Eye View) / world / top-down** — the ground-plane coordinate; foot points
projected via calibration. Used by geometry/STCRA and by TrackTacular.

**Calibration (intrinsic/extrinsic, K/R/t)** — camera parameters mapping 3D world ↔
image pixels (MMP provides them).

**STCRA (Spatio-Temporal Consistency ID Reassignment)** — paper’s geometry stage:
reassign IDs so one person isn’t in two far-apart world locations at once.

**IDF1** — identity F1: how consistently predicted IDs match GT identities over the
whole clip (the main cross-camera metric). **Global IDF1** = across all cameras.

**MOTA / MOTP** — tracking accuracy (FP+FN+IDsw) / localization precision.

**Anchor-guided clustering** — the paper’s method: build `k` anchor feature banks →
per-frame Hungarian assign detections to anchors → sliding-window vote.
