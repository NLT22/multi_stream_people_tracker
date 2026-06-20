"""Incremental, stateful cross-camera ID assignment over micro-batches — BANK version.

Each tracklet is a BANK of k per-crop embeddings (not a mean), so matching keeps
per-crop discrimination (the offline strength) while staying cheap. Per batch:
  1. within-batch HIERARCHICAL clustering of tracklets (avg-link on bank-cost) with a
     CANNOT-LINK constraint (same-sensor tracklets are different people);
  2. match each group's pooled bank to the persistent ANCHOR banks via Hungarian,
     gated by max cost -> existing global id, or spawn a new anchor;
  3. append the group's crops to the matched/new anchor bank (capped exemplars);
  4. age out anchors not seen within `ttl` (MDX "behavior state management").

Bank cost (per-crop, robust): cost(Q, A) = 1 - mean_i max_j cos(q_i, a_j) — each query
crop matched to its best anchor exemplar, averaged. This rewards a tracklet whose
crops each find a close match among the anchor's diverse exemplars (an occluded crop
doesn't drag the match), unlike centroid-cosine which collapses to the mean.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from scipy.optimize import linear_sum_assignment


def bank_cost(q: np.ndarray, a: np.ndarray) -> float:
    """1 - mean over q-crops of (max cosine to any a-crop). q,a are (n,D) L2-normed."""
    if len(q) == 0 or len(a) == 0:
        return 1.0
    sim = q @ a.T                       # (nq, na) cosine
    return float(1.0 - sim.max(axis=1).mean())


@dataclass
class Anchor:
    gid: int
    bank: np.ndarray               # (n, D) capped exemplar crops
    last_seen: float
    n_obs: int = 0
    sensors: set = field(default_factory=set)


class IncrementalMTMC:
    def __init__(self, merge_thr: float = 0.30, assign_thr: float = 0.35,
                 bank_cap: int = 64, ttl: float | None = None,
                 max_anchors: int | None = None):
        self.merge_thr = merge_thr        # within-batch avg-link bank-cost merge threshold
        self.assign_thr = assign_thr      # max bank-cost to bind a group to an anchor
        self.bank_cap = bank_cap
        self.ttl = ttl
        self.max_anchors = max_anchors
        self.anchors: list[Anchor] = []
        self._next_gid = 1
        self._rng = np.random.default_rng(0)

    # ---------------------------------------------------------------- public
    def ingest(self, batch, t_now: float) -> dict[tuple[int, int], int]:
        """batch: list[Tracklet] (each with .bank). Returns {key: gid}."""
        if not batch:
            self._age(t_now)
            return {}
        groups = self._cluster_batch(batch)                 # list[list[idx]]
        gbanks = [np.vstack([batch[i].bank for i in g]) for g in groups]
        for k, gb in enumerate(gbanks):                     # cap pooled group bank
            if len(gb) > self.bank_cap:
                gbanks[k] = gb[self._rng.choice(len(gb), self.bank_cap, replace=False)]
        assign = self._match(gbanks)
        out: dict[tuple[int, int], int] = {}
        for gi, g in enumerate(groups):
            anchor = self.anchors[assign[gi]] if assign[gi] is not None else self._new_anchor(t_now)
            self._update(anchor, gbanks[gi], [batch[i] for i in g], t_now)
            for i in g:
                out[batch[i].key] = anchor.gid
        self._age(t_now)
        return out

    # ----------------------------------------------- step 1: within-batch cluster
    def _cluster_batch(self, batch) -> list[list[int]]:
        clusters = {i: {"idx": [i], "sensors": {batch[i].sensor_id},
                        "bank": batch[i].bank} for i in range(len(batch))}
        nxt = len(batch)
        while True:
            best, best_d = None, self.merge_thr
            for a, b in combinations(list(clusters), 2):
                if clusters[a]["sensors"] & clusters[b]["sensors"]:
                    continue                                # cannot-link: same camera
                d = bank_cost(clusters[a]["bank"], clusters[b]["bank"])
                if d < best_d:
                    best_d, best = d, (a, b)
            if best is None:
                break
            a, b = best
            merged = np.vstack([clusters[a]["bank"], clusters[b]["bank"]])
            if len(merged) > self.bank_cap:
                merged = merged[self._rng.choice(len(merged), self.bank_cap, replace=False)]
            clusters[nxt] = {"idx": clusters[a]["idx"] + clusters[b]["idx"],
                             "sensors": clusters[a]["sensors"] | clusters[b]["sensors"],
                             "bank": merged}
            del clusters[a], clusters[b]
            nxt += 1
        return [c["idx"] for c in clusters.values()]

    # ------------------------------------------------- step 2: match to anchors
    def _match(self, gbanks: list[np.ndarray]) -> list[int | None]:
        assign: list[int | None] = [None] * len(gbanks)
        if not self.anchors:
            return assign
        cost = np.array([[bank_cost(gb, a.bank) for a in self.anchors] for gb in gbanks])
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols):
            if cost[r, c] <= self.assign_thr:
                assign[r] = int(c)
        return assign

    # --------------------------------------------------- step 3: update anchors
    def _new_anchor(self, t_now: float) -> Anchor:
        a = Anchor(gid=self._next_gid, bank=np.empty((0, 0), np.float32), last_seen=t_now)
        self._next_gid += 1
        self.anchors.append(a)
        return a

    def _update(self, anchor: Anchor, gbank: np.ndarray, members, t_now: float) -> None:
        anchor.bank = gbank if anchor.bank.size == 0 else np.vstack([anchor.bank, gbank])
        if len(anchor.bank) > self.bank_cap:
            anchor.bank = anchor.bank[self._rng.choice(len(anchor.bank), self.bank_cap, replace=False)]
        anchor.last_seen = t_now
        anchor.n_obs += sum(max(1, m.n_obs) for m in members)
        anchor.sensors.update(m.sensor_id for m in members)

    # ------------------------------------------------------- step 4: TTL aging
    def _age(self, t_now: float) -> None:
        if self.ttl is not None:
            self.anchors = [a for a in self.anchors if t_now - a.last_seen <= self.ttl]
        if self.max_anchors is not None and len(self.anchors) > self.max_anchors:
            self.anchors.sort(key=lambda a: a.n_obs, reverse=True)
            self.anchors = self.anchors[:self.max_anchors]

    @property
    def num_identities(self) -> int:
        return len(self.anchors)
