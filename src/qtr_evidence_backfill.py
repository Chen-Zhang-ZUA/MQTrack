"""Opt-in, annotation-free restoration of rejected real recovery detections."""

from bisect import bisect_left
from collections import Counter, defaultdict

import numpy as np


PROFILES = {
    'strict': dict(max_gap=4, min_iou=0.50, max_center=0.25, max_scale_log=0.35, min_reid=0.75),
    'balanced': dict(max_gap=6, min_iou=0.35, max_center=0.40, max_scale_log=0.50, min_reid=0.65),
}
PROFILES.update({
    'balanced_gap8': dict(PROFILES['balanced'], max_gap=8),
    'balanced_gap10': dict(PROFILES['balanced'], max_gap=10),
})
SHARED = dict(min_quality=0.60, min_anchors=4, min_anchor_score=0.45,
              max_step=1.20, duplicate_iou=0.50, min_score=0.18)


def valid_feature(row, qtr, independent=False):
    if independent:
        feature = row[12] if len(row) > 12 else None
    else:
        feature = qtr.row_feature(row)
    if feature is None:
        return None
    feature = np.asarray(feature, dtype=np.float64)
    if feature.ndim != 1 or not np.isfinite(feature).all():
        return None
    norm = np.linalg.norm(feature)
    return feature / norm if norm > qtr.EPS else None


def restore_recovery_rows(rows, rejected, args, profile, trace=None):
    # This function accepts predictions and frozen configuration only; no GT paths.
    import qhit_postprocess as qtr

    if profile not in PROFILES:
        raise ValueError('Unknown evidence-backfill profile: ' + str(profile))
    settings = dict(SHARED, **PROFILES[profile])
    independent = bool(getattr(args, 'evidence_feature_dir', None))

    def record(candidate, reason, track=None, anchors=(), **details):
        if trace is None:
            return
        frame = int(candidate[0])
        frames = [int(row[0]) for row in anchors]
        at = bisect_left(frames, frame)
        trace.append(dict(frame=frame, track_id=qtr.track_id(candidate),
            class_id=qtr.class_id(candidate), score=qtr.det_score(candidate),
            width=candidate[4], height=candidate[5], reason=reason,
            track_quality=track.quality if track is not None else -1,
            anchors=len(anchors), left_anchors=at, right_anchors=len(anchors) - at,
            anchor_gap=frames[at] - frames[at - 1] if 0 < at < len(frames) else -1,
            **details))
    if not rows or not rejected:
        return list(rows), dict(added=0, considered=0, rejected={})
    tracks = {(tr.cls_id, tr.track_id): tr for tr in qtr.build_tracklets(rows)}
    occupied = {(int(row[0]), qtr.class_id(row), qtr.track_id(row)) for row in rows}
    by_frame_class = defaultdict(list)
    for row in rows:
        by_frame_class[int(row[0]), qtr.class_id(row)].append(row)
    pools = defaultdict(list)
    for row in rejected:
        pools[qtr.class_id(row), qtr.track_id(row)].append(row)
    added = []
    reasons = Counter()
    considered = 0
    for key, pool in pools.items():
        track = tracks.get(key)
        if track is None or track.quality < settings['min_quality']:
            reasons['no_retained_quality_track'] += len(pool)
            for candidate in pool:
                record(candidate, 'no_retained_quality_track', track)
            continue
        anchors = [row for row in track.rows if qtr.row_source(row) == qtr.SOURCE_HIGH
                   and qtr.det_score(row) >= settings['min_anchor_score']]
        if len(anchors) < settings['min_anchors']:
            reasons['too_few_high_anchors'] += len(pool)
            for candidate in pool:
                record(candidate, 'too_few_high_anchors', track, anchors)
            continue
        anchor_frames = [int(row[0]) for row in anchors]
        pool_counts = Counter(int(row[0]) for row in pool)
        for candidate in sorted(pool, key=lambda row: row[0]):
            considered += 1
            frame = int(candidate[0])
            row_key = (frame, *key)
            if row_key in occupied or pool_counts[frame] != 1:
                reasons['occupied_or_ambiguous'] += 1
                record(candidate, 'occupied_or_ambiguous', track, anchors)
                continue
            minimum_score = max(settings['min_score'], qtr.class_setting(args, key[0], 'recovery_confirm_min_score'))
            if qtr.row_source(candidate) != qtr.SOURCE_RECOVERY or qtr.det_score(candidate) < minimum_score:
                reasons['source_or_score'] += 1
                record(candidate, 'source_or_score', track, anchors)
                continue
            at = bisect_left(anchor_frames, frame)
            if at < 2 or len(anchors) - at < 2:
                reasons['missing_two_sided_evidence'] += 1
                record(candidate, 'missing_two_sided_evidence', track, anchors)
                continue
            left, right = anchors[at - 1], anchors[at]
            gap = right[0] - left[0]
            if not 0 < frame - left[0] < gap <= settings['max_gap']:
                reasons['anchor_gap'] += 1
                record(candidate, 'anchor_gap', track, anchors)
                continue
            left_box = np.asarray(left[2:6], dtype=np.float64)
            right_box = np.asarray(right[2:6], dtype=np.float64)
            box = np.asarray(candidate[2:6], dtype=np.float64)
            if not np.isfinite(box).all() or min(box[2:]) <= 0:
                reasons['invalid_box'] += 1
                record(candidate, 'invalid_box', track, anchors)
                continue
            expected = left_box + ((frame - left[0]) / gap) * (right_box - left_box)
            scale = max(np.sqrt(qtr.bbox_area_xywh(expected)), 4.0)
            center_error = np.linalg.norm(qtr.bbox_center_xywh(box) - qtr.bbox_center_xywh(expected)) / scale
            step = np.linalg.norm(qtr.bbox_center_xywh(right_box) - qtr.bbox_center_xywh(left_box)) / (gap * scale)
            scale_error = abs(np.log(max(qtr.bbox_area_xywh(box), 1.0) / max(qtr.bbox_area_xywh(expected), 1.0)))
            geometry = dict(motion_iou=qtr.bbox_iou_xywh(box, expected),
                            center_error=float(center_error), scale_error=float(scale_error))
            if (qtr.bbox_iou_xywh(box, expected) < settings['min_iou']
                    or center_error > settings['max_center'] or step > settings['max_step']
                    or scale_error > settings['max_scale_log']):
                reasons['motion_or_scale'] += 1
                record(candidate, 'motion_or_scale', track, anchors, **geometry)
                continue
            feature = valid_feature(candidate, qtr, independent)
            left_features = [valid_feature(row, qtr, independent) for row in anchors[at - 2:at]]
            right_features = [valid_feature(row, qtr, independent) for row in anchors[at:at + 2]]
            if feature is None or any(f is None or f.shape != feature.shape for f in left_features + right_features):
                reasons['missing_valid_reid'] += 1
                record(candidate, 'missing_valid_reid', track, anchors, **geometry)
                continue
            similarities = []
            for evidence in (left_features, right_features):
                mean = np.mean(evidence, axis=0)
                norm = np.linalg.norm(mean)
                similarities.append(float(np.dot(feature, mean / norm)) if norm > qtr.EPS else -1.0)
            if min(similarities) < settings['min_reid']:
                reasons['reid_disagreement'] += 1
                record(candidate, 'reid_disagreement', track, anchors,
                       min_reid=min(similarities), **geometry)
                continue
            if any(qtr.track_id(other) != key[1]
                   and qtr.bbox_iou_xywh(box, other[2:6]) > settings['duplicate_iou']
                   for other in by_frame_class[frame, key[0]]):
                reasons['other_track_duplicate'] += 1
                record(candidate, 'other_track_duplicate', track, anchors,
                       min_reid=min(similarities), **geometry)
                continue
            restored = list(candidate)
            added.append(restored)
            occupied.add(row_key)
            by_frame_class[frame, key[0]].append(restored)
            record(candidate, 'added', track, anchors, min_reid=min(similarities), **geometry)
    return list(rows) + added, dict(added=len(added), considered=considered,
                                   pool_rows=len(rejected), rejected=dict(reasons),
                                   feature_kind='current_frame' if independent else 'track_memory')
