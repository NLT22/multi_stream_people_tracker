"""Pure embedding-quality gate — no DeepStream / gallery state.

A low-quality crop (frame edge, tiny, wrong aspect, heavily overlapped) can still
be drawn, but it must NOT update long-term identity memory. These are plain
functions so the gate can be unit-tested without running the pipeline.
Extracted from src/reid/gallery.py.
"""


def rect_iou(a: dict, b: dict) -> float:
    """IoU of two bbox dicts with keys left/top/width/height."""
    ax1, ay1 = a["left"], a["top"]
    ax2, ay2 = ax1 + a["width"], ay1 + a["height"]
    bx1, by1 = b["left"], b["top"]
    bx2, by2 = bx1 + b["width"], by1 + b["height"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a["width"]) * max(0.0, a["height"])
    area_b = max(0.0, b["width"]) * max(0.0, b["height"])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0.0 else 0.0


def embedding_quality(
    row: dict,
    rows: list[dict],
    *,
    enabled: bool,
    edge_margin_ratio: float,
    min_height_ratio: float,
    min_area_ratio: float,
    min_aspect: float,
    max_aspect: float,
    max_overlap_iou: float,
) -> tuple[bool, str]:
    """Return (ok, reason). Same-camera rows are used for the overlap check."""
    if not row["raw_embedding"]:
        return False, "no_embedding"
    if not enabled:
        return True, "disabled"

    rect = row["rect"]
    frame_w = max(1.0, rect["frame_w"])
    frame_h = max(1.0, rect["frame_h"])
    left = rect["left"]
    top = rect["top"]
    width = max(0.0, rect["width"])
    height = max(0.0, rect["height"])
    right = left + width
    bottom = top + height

    margin_x = frame_w * edge_margin_ratio
    margin_y = frame_h * edge_margin_ratio
    if (
        left <= margin_x
        or top <= margin_y
        or right >= frame_w - margin_x
        or bottom >= frame_h - margin_y
    ):
        return False, "edge_crop"

    if height / frame_h < min_height_ratio:
        return False, "small_height"
    if (width * height) / (frame_w * frame_h) < min_area_ratio:
        return False, "small_area"

    aspect = width / height if height > 0.0 else 999.0
    if aspect < min_aspect:
        return False, "thin_crop"
    if aspect > max_aspect:
        return False, "wide_or_merged_crop"

    max_iou = 0.0
    for other in rows:
        if other is row or other["src"] != row["src"]:
            continue
        max_iou = max(max_iou, rect_iou(rect, other["rect"]))
    if max_iou > max_overlap_iou:
        return False, f"overlap_iou={max_iou:.2f}"

    return True, "ok"
