"""Minimal Tracklet container so the AIC23 authors' aic_hungarian_cluster main
loop (`for feat in trk.features`) runs verbatim on our MMP-built tracklet pkls."""


class Tracklet:
    def __init__(self, features):
        self.features = features      # list of per-detection embeddings (frame order)
