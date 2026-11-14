"""Conservative car/van birth competition for VisDrone tracking."""

import numpy as np


COMPETING_CLASSES = {3: 4, 4: 3}


def box_iou(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def ambiguous_birth(cls_id, tlbr, score, detections, class_thresholds,
                    min_iou=0.80, score_tolerance=0.10):
    """Require more evidence only when a competing high-score peak is co-located."""
    other = COMPETING_CLASSES.get(int(cls_id))
    if other is None:
        return False
    for detection in detections.get(other, ()):
        if detection[4] <= class_thresholds[other]:
            continue
        if detection[4] < float(score) - score_tolerance:
            continue
        if box_iou(tlbr, detection[:4]) >= min_iou:
            return True
    return False


def mature_competitor(track, tracks_by_class, frame_id, quality_fn,
                      min_iou=0.75, min_observations=4,
                      min_quality=0.60, min_score=0.45):
    """Find an independently established alternative-class track seen now."""
    other = COMPETING_CLASSES.get(int(track.cls_id))
    if other is None:
        return None
    for competitor in tracks_by_class.get(other, ()):
        if not competitor.is_activated or competitor.frame_id != frame_id:
            continue
        if getattr(competitor, 'last_update_source', None) != 'high':
            continue
        if len(getattr(competitor, 'tlwh_deque', ())) < min_observations:
            continue
        if float(competitor.score) < min_score:
            continue
        if quality_fn(competitor) < min_quality:
            continue
        if box_iou(track.tlbr, competitor.tlbr) >= min_iou:
            return competitor
    return None
