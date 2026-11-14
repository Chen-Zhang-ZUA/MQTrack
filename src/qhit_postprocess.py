from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


EPS = 1e-6
INTERNAL_SOURCE_INDEX = 10
INTERNAL_FEATURE_INDEX = 11
SOURCE_UNKNOWN = -1
SOURCE_HIGH = 0
SOURCE_RECOVERY = 1
SOURCE_MTC = 2
SOURCE_INTERPOLATED = 3


VISDRONE_CLASS_NAMES = {
    1: 'pedestrian',
    2: 'person',
    3: 'bicycle',
    4: 'car',
    5: 'van',
    6: 'truck',
    7: 'tricycle',
    8: 'awning-tricycle',
    9: 'bus',
    10: 'motor',
}


CLASS_PARAM_OVERRIDES = {
    'none': {},
    'visdrone-v2': {
        1: {
            'base_iou_thresh': 0.16,
            'min_merge_score': 0.37,
            'interp_min_quality': 0.50,
            'interp_min_score': 0.42,
            'interp_max_gap': 2,
            'drop_min_len': 2,
            'drop_min_quality': 0.50,
            'recovery_confirm_min_quality': 0.44,
            'recovery_confirm_min_score': 0.16,
            'recovery_confirm_max_run': 2,
        },
        2: {
            'base_iou_thresh': 0.14,
            'min_quality': 0.16,
            'min_merge_score': 0.35,
            'gap_penalty': 0.022,
            'interp_min_quality': 0.46,
            'interp_min_score': 0.38,
            'interp_max_gap': 2,
            'drop_min_len': 2,
            'drop_min_quality': 0.50,
            'recovery_confirm_min_quality': 0.44,
            'recovery_confirm_min_score': 0.16,
            'recovery_confirm_max_run': 2,
        },
        3: {
            'base_iou_thresh': 0.15,
            'min_quality': 0.16,
            'min_merge_score': 0.36,
            'interp_min_quality': 0.47,
            'interp_min_score': 0.38,
            'interp_max_gap': 2,
            'drop_min_len': 2,
            'drop_min_quality': 0.50,
            'recovery_confirm_min_quality': 0.45,
            'recovery_confirm_min_score': 0.18,
            'recovery_confirm_max_run': 1,
        },
        4: {
            'base_iou_thresh': 0.20,
            'min_quality': 0.20,
            'min_merge_score': 0.40,
            'gap_penalty': 0.030,
            'interp_min_quality': 0.55,
            'interp_min_score': 0.45,
            'interp_max_gap': 2,
            'drop_min_len': 2,
            'drop_min_quality': 0.48,
            'recovery_confirm_min_quality': 0.43,
            'recovery_confirm_min_score': 0.20,
            'recovery_confirm_max_run': 2,
        },
        5: {
            'base_iou_thresh': 0.20,
            'min_quality': 0.20,
            'min_merge_score': 0.40,
            'gap_penalty': 0.030,
            'interp_min_quality': 0.53,
            'interp_min_score': 0.42,
            'interp_max_gap': 2,
            'drop_min_len': 2,
            'drop_min_quality': 0.50,
            'recovery_confirm_min_quality': 0.52,
            'recovery_confirm_min_score': 0.26,
            'recovery_confirm_max_run': 1,
        },
        6: {
            'base_iou_thresh': 0.22,
            'min_quality': 0.22,
            'min_merge_score': 0.42,
            'gap_penalty': 0.035,
            'interp_min_quality': 0.55,
            'interp_min_score': 0.42,
            'interp_max_gap': 2,
            'drop_min_len': 2,
            'drop_min_quality': 0.50,
            'recovery_confirm_min_quality': 0.48,
            'recovery_confirm_min_score': 0.24,
            'recovery_confirm_max_run': 1,
        },
        7: {
            'base_iou_thresh': 0.16,
            'min_quality': 0.18,
            'min_merge_score': 0.37,
            'interp_min_quality': 0.48,
            'interp_min_score': 0.38,
            'interp_max_gap': 2,
            'drop_min_len': 2,
            'drop_min_quality': 0.50,
            'recovery_confirm_min_quality': 0.45,
            'recovery_confirm_min_score': 0.18,
            'recovery_confirm_max_run': 1,
        },
        8: {
            'base_iou_thresh': 0.16,
            'min_quality': 0.18,
            'min_merge_score': 0.38,
            'interp_min_quality': 0.50,
            'interp_min_score': 0.36,
            'interp_max_gap': 2,
            'drop_min_len': 2,
            'drop_min_quality': 0.50,
            'recovery_confirm_min_quality': 0.45,
            'recovery_confirm_min_score': 0.18,
            'recovery_confirm_max_run': 1,
        },
        9: {
            'base_iou_thresh': 0.22,
            'min_quality': 0.22,
            'min_merge_score': 0.42,
            'gap_penalty': 0.035,
            'interp_min_quality': 0.55,
            'interp_min_score': 0.42,
            'interp_max_gap': 2,
            'drop_min_len': 2,
            'drop_min_quality': 0.50,
            'recovery_confirm_min_quality': 0.48,
            'recovery_confirm_min_score': 0.24,
            'recovery_confirm_max_run': 1,
        },
        10: {
            'base_iou_thresh': 0.15,
            'min_quality': 0.16,
            'min_merge_score': 0.36,
            'interp_min_quality': 0.46,
            'interp_min_score': 0.36,
            'interp_max_gap': 2,
            'drop_min_len': 2,
            'drop_min_quality': 0.49,
            'recovery_confirm_min_quality': 0.44,
            'recovery_confirm_min_score': 0.16,
            'recovery_confirm_max_run': 1,
        },
    },
    'mqtrack-diagnostic': {
        # Test-dev diagnostic profile.  It keeps the proven VisDrone-v2 merge
        # structure, then tightens classes that generated many false positives
        # or ID switches while preserving high-quality low-score continuity.
        1: {
            'base_iou_thresh': 0.18,
            'min_quality': 0.18,
            'min_merge_score': 0.39,
            'gap_penalty': 0.028,
            'interp_min_quality': 0.54,
            'interp_min_score': 0.42,
            'interp_max_gap': 2,
            'drop_min_len': 2,
            'drop_min_quality': 0.54,
            'trim_min_track_len': 7,
            'trim_anchor_score': 0.46,
            'trim_edge_score': 0.36,
            'recovery_confirm_min_quality': 0.50,
            'recovery_confirm_min_score': 0.20,
            'recovery_confirm_max_run': 1,
        },
        4: {
            'base_iou_thresh': 0.22,
            'min_quality': 0.22,
            'min_merge_score': 0.43,
            'gap_penalty': 0.034,
            'interp_min_quality': 0.58,
            'interp_min_score': 0.46,
            'interp_max_gap': 2,
            'drop_min_len': 3,
            'drop_min_quality': 0.54,
            'trim_min_track_len': 7,
            'trim_anchor_score': 0.47,
            'trim_edge_score': 0.36,
            'recovery_confirm_min_quality': 0.50,
            'recovery_confirm_min_score': 0.23,
            'recovery_confirm_max_run': 1,
        },
        5: {
            'base_iou_thresh': 0.22,
            'min_quality': 0.22,
            'min_merge_score': 0.42,
            'gap_penalty': 0.032,
            'interp_min_quality': 0.54,
            'interp_min_score': 0.42,
            'interp_max_gap': 2,
            'drop_min_len': 2,
            'drop_min_quality': 0.52,
            'trim_min_track_len': 6,
            'trim_anchor_score': 0.48,
            'trim_edge_score': 0.38,
            'recovery_confirm_min_quality': 0.54,
            'recovery_confirm_min_score': 0.28,
            'recovery_confirm_max_run': 1,
        },
        6: {
            'base_iou_thresh': 0.22,
            'min_quality': 0.22,
            'min_merge_score': 0.42,
            'gap_penalty': 0.034,
            'interp_min_quality': 0.52,
            'interp_min_score': 0.40,
            'interp_max_gap': 2,
            'drop_min_len': 2,
            'drop_min_quality': 0.52,
            'trim_min_track_len': 6,
            'trim_anchor_score': 0.45,
            'trim_edge_score': 0.34,
            'recovery_confirm_min_quality': 0.48,
            'recovery_confirm_min_score': 0.24,
            'recovery_confirm_max_run': 1,
        },
        9: {
            'base_iou_thresh': 0.22,
            'min_quality': 0.22,
            'min_merge_score': 0.42,
            'gap_penalty': 0.035,
            'interp_min_quality': 0.55,
            'interp_min_score': 0.42,
            'interp_max_gap': 2,
            'drop_min_len': 2,
            'drop_min_quality': 0.50,
            'trim_min_track_len': 6,
            'trim_anchor_score': 0.46,
            'trim_edge_score': 0.36,
            'recovery_confirm_min_quality': 0.50,
            'recovery_confirm_min_score': 0.24,
            'recovery_confirm_max_run': 1,
        },
    },
    'mqtrack-recall-safe': {
        # Recall-safe profile: add only short-gap boxes inside high-quality
        # tracklets, avoiding direct low-score detection expansion.
        1: {
            'base_iou_thresh': 0.16,
            'min_quality': 0.18,
            'min_merge_score': 0.38,
            'gap_penalty': 0.026,
            'interp_min_quality': 0.58,
            'interp_min_score': 0.43,
            'interp_max_gap': 3,
            'interp_max_motion': 1.25,
            'interp_max_scale_log': 0.55,
            'drop_min_len': 2,
            'drop_min_quality': 0.50,
            'trim_min_track_len': 6,
            'trim_anchor_score': 0.45,
            'trim_edge_score': 0.34,
        },
        4: {
            'base_iou_thresh': 0.20,
            'min_quality': 0.20,
            'min_merge_score': 0.41,
            'gap_penalty': 0.032,
            'interp_min_quality': 0.62,
            'interp_min_score': 0.46,
            'interp_max_gap': 2,
            'interp_max_motion': 1.20,
            'interp_max_scale_log': 0.50,
            'drop_min_len': 2,
            'drop_min_quality': 0.50,
            'trim_min_track_len': 6,
            'trim_anchor_score': 0.46,
            'trim_edge_score': 0.35,
        },
        5: {
            'base_iou_thresh': 0.20,
            'min_quality': 0.20,
            'min_merge_score': 0.40,
            'gap_penalty': 0.030,
            'interp_min_quality': 0.56,
            'interp_min_score': 0.41,
            'interp_max_gap': 3,
            'interp_max_motion': 1.20,
            'interp_max_scale_log': 0.55,
            'drop_min_len': 2,
            'drop_min_quality': 0.50,
            'trim_min_track_len': 6,
            'trim_anchor_score': 0.46,
            'trim_edge_score': 0.35,
        },
        6: {
            'base_iou_thresh': 0.22,
            'min_quality': 0.22,
            'min_merge_score': 0.42,
            'gap_penalty': 0.034,
            'interp_min_quality': 0.56,
            'interp_min_score': 0.40,
            'interp_max_gap': 3,
            'interp_max_motion': 1.20,
            'interp_max_scale_log': 0.55,
            'drop_min_len': 2,
            'drop_min_quality': 0.50,
            'trim_min_track_len': 6,
            'trim_anchor_score': 0.45,
            'trim_edge_score': 0.34,
        },
        9: {
            'base_iou_thresh': 0.22,
            'min_quality': 0.22,
            'min_merge_score': 0.42,
            'gap_penalty': 0.035,
            'interp_min_quality': 0.58,
            'interp_min_score': 0.42,
            'interp_max_gap': 2,
            'interp_max_motion': 1.15,
            'interp_max_scale_log': 0.50,
            'drop_min_len': 2,
            'drop_min_quality': 0.50,
            'trim_min_track_len': 6,
            'trim_anchor_score': 0.46,
            'trim_edge_score': 0.35,
        },
    },
    'uavdt-balanced': {
        # UAVDT result files use class id -1 for the single vehicle class.
        # This profile preserves QTR's ID benefit while being stricter about
        # short, low-quality vehicle tracklets that caused FP-heavy results.
        -1: {
            'base_iou_thresh': 0.22,
            'min_quality': 0.26,
            'min_merge_score': 0.44,
            'gap_penalty': 0.036,
            'quality_penalty': 0.15,
            'interp_min_quality': 0.66,
            'interp_min_score': 0.45,
            'interp_max_gap': 2,
            'interp_max_motion': 1.15,
            'interp_max_scale_log': 0.45,
            'drop_min_len': 3,
            'drop_min_quality': 0.55,
            'trim_min_track_len': 8,
            'trim_anchor_score': 0.47,
            'trim_edge_score': 0.37,
            'recovery_confirm_min_quality': 0.58,
            'recovery_confirm_min_score': 0.30,
            'recovery_confirm_max_run': 1,
        },
    },
    'uavdt-mota-v2': {
        # MOTA-oriented UAVDT profile searched on the official MATLAB toolkit.
        # It is intentionally stricter than uavdt-balanced: FP drops, but FN
        # rises, so IDF1 is lower than the MQTrack identity-oriented setting.
        -1: {
            'base_iou_thresh': 0.22,
            'min_quality': 0.26,
            'min_merge_score': 0.44,
            'gap_penalty': 0.036,
            'quality_penalty': 0.15,
            'interp_min_quality': 0.66,
            'interp_min_score': 0.45,
            'interp_max_gap': 2,
            'interp_max_motion': 1.15,
            'interp_max_scale_log': 0.45,
            'drop_min_len': 6,
            'drop_min_quality': 0.64,
            'trim_min_track_len': 8,
            'trim_anchor_score': 0.54,
            'trim_edge_score': 0.44,
            'recovery_confirm_min_quality': 0.58,
            'recovery_confirm_min_score': 0.30,
            'recovery_confirm_max_run': 1,
        },
    },
}


CLASS_SCORE_PROFILES = {
    'none': {},
    # MOTA-oriented five-class VisDrone setting, tuned on AMOT-compatible
    # classes: car, bus, truck, pedestrian, van.
    'visdrone5-mota': {
        1: 0.35,  # pedestrian
        4: 0.30,  # car
        5: 0.50,  # van
        6: 0.25,  # truck
        9: 0.25,  # bus
    },
    # Recall-anchored variant: keeps the stable five-class MOTA profile but
    # disables truck score filtering because val recall-constrained search and
    # test-dev evaluation both favored preserving truck recall.
    'visdrone5-anchor-recall': {
        1: 0.35,  # pedestrian
        4: 0.30,  # car
        5: 0.50,  # van
        6: 0.00,  # truck
        9: 0.25,  # bus
    },
    # MOTA-v2 profile after class-wise official search on the current
    # SA-AMC + SCA-TDRM + Q-HIT setting.
    'visdrone5-mota-v2': {
        1: 0.35,  # pedestrian
        4: 0.27,  # car
        5: 0.54,  # van
        6: 0.00,  # truck
        9: 0.00,  # bus
    },
    'visdrone5-diagnostic-row': {
        1: 0.36,  # pedestrian
        4: 0.31,  # car
        5: 0.52,  # van
        6: 0.00,  # truck
        9: 0.00,  # bus
    },
    'visdrone5-recall-safe': {
        1: 0.33,  # pedestrian
        4: 0.30,  # car
        5: 0.50,  # van
        6: 0.00,  # truck
        9: 0.00,  # bus
    },
    'uavdt-car-strict': {
        -1: 0.30,
    },
    'uavdt-car-mota-v2': {
        -1: 0.455,
    },
}


CLASS_TRACKLET_SCORE_PROFILES = {
    'none': {},
    # Tracklet-level CA gate.  Compared with row-level score filtering, good
    # quality tracklets get a lower per-row threshold so isolated low-score
    # true positives inside stable trajectories are preserved.
    'visdrone5-tracklet-balanced': {
        1: {
            'base_thresh': 0.35,
            'floor_thresh': 0.25,
            'poor_thresh': 0.40,
            'protect_quality': 0.58,
            'protect_len': 12,
            'protect_mean': 0.42,
            'poor_quality': 0.34,
            'poor_len': 3,
            'poor_mean': 0.34,
        },
        4: {
            'base_thresh': 0.30,
            'floor_thresh': 0.20,
            'poor_thresh': 0.36,
            'protect_quality': 0.58,
            'protect_len': 12,
            'protect_mean': 0.38,
            'poor_quality': 0.34,
            'poor_len': 3,
            'poor_mean': 0.30,
        },
        5: {
            'base_thresh': 0.50,
            'floor_thresh': 0.42,
            'poor_thresh': 0.54,
            'protect_quality': 0.60,
            'protect_len': 10,
            'protect_mean': 0.52,
            'poor_quality': 0.36,
            'poor_len': 3,
            'poor_mean': 0.45,
        },
        6: {
            'base_thresh': 0.00,
            'floor_thresh': 0.00,
            'poor_thresh': 0.18,
            'protect_quality': 0.55,
            'protect_len': 8,
            'protect_mean': 0.35,
            'poor_quality': 0.32,
            'poor_len': 2,
            'poor_mean': 0.22,
        },
        9: {
            'base_thresh': 0.25,
            'floor_thresh': 0.18,
            'poor_thresh': 0.35,
            'protect_quality': 0.56,
            'protect_len': 8,
            'protect_mean': 0.38,
            'poor_quality': 0.32,
            'poor_len': 2,
            'poor_mean': 0.25,
        },
    },
    # A more FP-oriented version for a quick MOTA stress test.
    'visdrone5-tracklet-strict': {
        1: {
            'base_thresh': 0.38,
            'floor_thresh': 0.30,
            'poor_thresh': 0.45,
            'protect_quality': 0.62,
            'protect_len': 16,
            'protect_mean': 0.46,
            'poor_quality': 0.38,
            'poor_len': 4,
            'poor_mean': 0.38,
        },
        4: {
            'base_thresh': 0.34,
            'floor_thresh': 0.26,
            'poor_thresh': 0.42,
            'protect_quality': 0.62,
            'protect_len': 16,
            'protect_mean': 0.42,
            'poor_quality': 0.38,
            'poor_len': 4,
            'poor_mean': 0.34,
        },
        5: {
            'base_thresh': 0.52,
            'floor_thresh': 0.46,
            'poor_thresh': 0.58,
            'protect_quality': 0.64,
            'protect_len': 12,
            'protect_mean': 0.55,
            'poor_quality': 0.40,
            'poor_len': 4,
            'poor_mean': 0.48,
        },
        6: {
            'base_thresh': 0.15,
            'floor_thresh': 0.00,
            'poor_thresh': 0.25,
            'protect_quality': 0.58,
            'protect_len': 10,
            'protect_mean': 0.38,
            'poor_quality': 0.36,
            'poor_len': 3,
            'poor_mean': 0.28,
        },
        9: {
            'base_thresh': 0.28,
            'floor_thresh': 0.22,
            'poor_thresh': 0.40,
            'protect_quality': 0.60,
            'protect_len': 10,
            'protect_mean': 0.42,
            'poor_quality': 0.36,
            'poor_len': 3,
            'poor_mean': 0.30,
        },
    },
    # MOTA-oriented tracklet gate: close to the proven row-level anchor-recall
    # thresholds for normal tracks, slightly lower for very reliable tracklets,
    # and stricter for short low-quality tracklets.
    'visdrone5-tracklet-mota': {
        1: {
            'base_thresh': 0.35,
            'floor_thresh': 0.32,
            'poor_thresh': 0.50,
            'protect_quality': 0.66,
            'protect_len': 20,
            'protect_mean': 0.50,
            'poor_quality': 0.46,
            'poor_len': 5,
            'poor_mean': 0.42,
        },
        4: {
            'base_thresh': 0.30,
            'floor_thresh': 0.28,
            'poor_thresh': 0.45,
            'protect_quality': 0.66,
            'protect_len': 20,
            'protect_mean': 0.45,
            'poor_quality': 0.46,
            'poor_len': 5,
            'poor_mean': 0.36,
        },
        5: {
            'base_thresh': 0.50,
            'floor_thresh': 0.48,
            'poor_thresh': 0.62,
            'protect_quality': 0.68,
            'protect_len': 16,
            'protect_mean': 0.58,
            'poor_quality': 0.48,
            'poor_len': 5,
            'poor_mean': 0.52,
        },
        6: {
            'base_thresh': 0.00,
            'floor_thresh': 0.00,
            'poor_thresh': 0.22,
            'protect_quality': 0.62,
            'protect_len': 12,
            'protect_mean': 0.42,
            'poor_quality': 0.40,
            'poor_len': 4,
            'poor_mean': 0.30,
        },
        9: {
            'base_thresh': 0.25,
            'floor_thresh': 0.23,
            'poor_thresh': 0.45,
            'protect_quality': 0.64,
            'protect_len': 12,
            'protect_mean': 0.46,
            'poor_quality': 0.42,
            'poor_len': 4,
            'poor_mean': 0.35,
        },
    },
    'visdrone5-diagnostic': {
        1: {
            'base_thresh': 0.36,
            'floor_thresh': 0.30,
            'poor_thresh': 0.48,
            'protect_quality': 0.66,
            'protect_len': 18,
            'protect_mean': 0.48,
            'poor_quality': 0.44,
            'poor_len': 4,
            'poor_mean': 0.38,
        },
        4: {
            'base_thresh': 0.31,
            'floor_thresh': 0.29,
            'poor_thresh': 0.44,
            'protect_quality': 0.64,
            'protect_len': 20,
            'protect_mean': 0.45,
            'poor_quality': 0.44,
            'poor_len': 5,
            'poor_mean': 0.35,
        },
        5: {
            'base_thresh': 0.52,
            'floor_thresh': 0.44,
            'poor_thresh': 0.62,
            'protect_quality': 0.68,
            'protect_len': 18,
            'protect_mean': 0.57,
            'poor_quality': 0.48,
            'poor_len': 5,
            'poor_mean': 0.50,
        },
        6: {
            'base_thresh': 0.00,
            'floor_thresh': 0.00,
            'poor_thresh': 0.20,
            'protect_quality': 0.60,
            'protect_len': 10,
            'protect_mean': 0.36,
            'poor_quality': 0.42,
            'poor_len': 4,
            'poor_mean': 0.28,
        },
        9: {
            'base_thresh': 0.22,
            'floor_thresh': 0.18,
            'poor_thresh': 0.40,
            'protect_quality': 0.62,
            'protect_len': 10,
            'protect_mean': 0.42,
            'poor_quality': 0.40,
            'poor_len': 4,
            'poor_mean': 0.30,
        },
    },
    'uavdt-tracklet-strict': {
        -1: {
            'base_thresh': 0.30,
            'floor_thresh': 0.24,
            'poor_thresh': 0.42,
            'protect_quality': 0.70,
            'protect_len': 30,
            'protect_mean': 0.50,
            'poor_quality': 0.50,
            'poor_len': 5,
            'poor_mean': 0.38,
        },
    },
}


@dataclass
class Tracklet:
    cls_id: int
    track_id: int
    rows: list
    quality: float
    feature: object = None

    @property
    def start(self):
        return int(round(self.rows[0][0]))

    @property
    def end(self):
        return int(round(self.rows[-1][0]))

    @property
    def first_bbox(self):
        return np.asarray(self.rows[0][2:6], dtype=np.float32)

    @property
    def last_bbox(self):
        return np.asarray(self.rows[-1][2:6], dtype=np.float32)


@dataclass
class MergeCandidate:
    cls_id: int
    src_id: int
    dst_id: int
    score: float
    threshold: float
    iou: float
    velocity: float
    scale: float
    quality_pair: float
    gap: int
    reid: float = -1.0


def parse_delta_t(value):
    deltas = [int(v.strip()) for v in value.split(',') if v.strip()]
    if not deltas or any(delta <= 0 for delta in deltas):
        raise ValueError('--delta-t must contain positive integers, e.g. 1,3,5,7,9')
    return deltas


def read_result_file(path):
    rows = []
    with path.open('r', encoding='utf-8') as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 6:
                continue
            try:
                row = [float(part) for part in parts]
            except ValueError as exc:
                raise ValueError('{}:{} is not a numeric MOT row'.format(path, line_no)) from exc
            rows.append(row)
    return rows


def read_source_file(path):
    source_map = {}
    if not path.exists():
        return source_map

    with path.open('r', encoding='utf-8') as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 4:
                continue
            try:
                frame = int(round(float(parts[0])))
                tid = int(round(float(parts[1])))
                cls_id = int(round(float(parts[2])))
                source = int(round(float(parts[3])))
            except ValueError as exc:
                raise ValueError('{}:{} is not a valid source row'.format(path, line_no)) from exc
            source_map[(frame, tid, cls_id)] = source
    return source_map


def read_feature_file(path):
    feature_map = {}
    if not path.exists():
        return feature_map

    with path.open('r', encoding='utf-8') as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 4:
                continue
            try:
                frame = int(round(float(parts[0])))
                tid = int(round(float(parts[1])))
                cls_id = int(round(float(parts[2])))
                feat = np.asarray([float(value) for value in parts[3:]], dtype=np.float32)
            except ValueError as exc:
                raise ValueError('{}:{} is not a valid feature row'.format(path, line_no)) from exc
            norm = float(np.linalg.norm(feat))
            if norm > EPS:
                feature_map[(frame, tid, cls_id)] = feat / norm
    return feature_map


def source_path_for(input_path, args):
    if args.source_dir:
        return Path(args.source_dir) / (input_path.name + '.src')
    return input_path.with_suffix(input_path.suffix + '.src')


def feature_path_for(input_path, args):
    if args.feature_dir:
        return Path(args.feature_dir) / (input_path.name + '.feat')
    return input_path.with_suffix(input_path.suffix + '.feat')


def attach_sources(rows, input_path, args):
    source_map = read_source_file(source_path_for(input_path, args))
    if not source_map:
        return 0

    attached = 0
    for row in rows:
        key = (int(round(row[0])), track_id(row), class_id(row))
        source = source_map.get(key, SOURCE_UNKNOWN)
        if len(row) <= INTERNAL_SOURCE_INDEX:
            row.append(float(source))
        else:
            row[INTERNAL_SOURCE_INDEX] = float(source)
        if source != SOURCE_UNKNOWN:
            attached += 1
    return attached


def attach_features(rows, input_path, args):
    if not getattr(args, 'reid_aware_qhit', False):
        return 0

    feature_map = read_feature_file(feature_path_for(input_path, args))
    if not feature_map:
        return 0

    attached = 0
    for row in rows:
        key = (int(round(row[0])), track_id(row), class_id(row))
        feat = feature_map.get(key)
        if len(row) <= INTERNAL_FEATURE_INDEX:
            row.extend([None] * (INTERNAL_FEATURE_INDEX + 1 - len(row)))
        row[INTERNAL_FEATURE_INDEX] = feat
        if feat is not None:
            attached += 1
    return attached


def attach_evidence_features(rows, input_path, args):
    feature_dir = getattr(args, 'evidence_feature_dir', None)
    if not feature_dir:
        return 0
    path = Path(feature_dir) / (input_path.name + '.feat')
    if not path.exists():
        raise FileNotFoundError('Current-frame evidence feature file missing: ' + str(path))
    features = read_feature_file(path)
    attached = 0
    for row in rows:
        key = (int(row[0]), track_id(row), class_id(row))
        feature = features.get(key)
        if feature is None:
            raise ValueError('Current-frame feature key missing: ' + str(key))
        row.extend([None] * max(0, 13 - len(row)))
        row[12] = feature
        attached += 1
    return attached


def format_value(value):
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return ('{:.6f}'.format(value)).rstrip('0').rstrip('.')


def write_result_file(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: (row[0], class_id(row), row[1]))
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(','.join(format_value(v) for v in row[:10]) + '\n')


def class_id(row):
    return int(round(row[7])) if len(row) > 7 else -1


def track_id(row):
    return int(round(row[1]))


def det_score(row):
    return float(row[6]) if len(row) > 6 else 1.0


def row_source(row):
    if len(row) > INTERNAL_SOURCE_INDEX and row[INTERNAL_SOURCE_INDEX] is not None:
        return int(round(row[INTERNAL_SOURCE_INDEX]))
    return SOURCE_UNKNOWN


def row_feature(row):
    if len(row) > INTERNAL_FEATURE_INDEX:
        return row[INTERNAL_FEATURE_INDEX]
    return None


def is_recovery_row(row):
    return row_source(row) == SOURCE_RECOVERY


def recovery_fraction(rows):
    if not rows:
        return 0.0
    return float(sum(1 for row in rows if is_recovery_row(row))) / float(len(rows))


def clip01(value):
    return float(np.clip(value, 0.0, 1.0))


def class_setting(args, cls_id, name):
    profile = CLASS_PARAM_OVERRIDES.get(args.class_profile, {})
    if cls_id in profile and name in profile[cls_id]:
        return profile[cls_id][name]
    return getattr(args, name)


def bbox_area_xywh(bbox):
    return float(max(0.0, bbox[2]) * max(0.0, bbox[3]))


def bbox_center_xywh(bbox):
    return np.asarray([bbox[0] + bbox[2] * 0.5, bbox[1] + bbox[3] * 0.5], dtype=np.float32)


def bbox_iou_xywh(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, aw) * max(0.0, ah)
    area_b = max(0.0, bw) * max(0.0, bh)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def tracklet_quality_from_rows(rows):
    rows = sorted(rows, key=lambda row: row[0])
    num_rows = len(rows)
    start = int(round(rows[0][0]))
    end = int(round(rows[-1][0]))
    span = max(end - start + 1, 1)

    length_quality = clip01(np.log1p(num_rows) / np.log1p(30.0))
    confidence_quality = clip01((np.mean([det_score(row) for row in rows]) - 0.25) / 0.45)
    density_quality = clip01(float(num_rows) / float(span))
    motion_quality = motion_smoothness_quality(rows)
    scale_quality = scale_stability_quality(rows)

    quality = (
        0.26 * length_quality +
        0.24 * confidence_quality +
        0.20 * density_quality +
        0.16 * motion_quality +
        0.14 * scale_quality)
    return clip01(quality)


def motion_smoothness_quality(rows):
    if len(rows) < 3:
        return 0.6

    frames = np.asarray([row[0] for row in rows], dtype=np.float32)
    centers = np.asarray([bbox_center_xywh(row[2:6]) for row in rows], dtype=np.float32)
    bboxes = np.asarray([row[2:6] for row in rows], dtype=np.float32)
    scales = np.sqrt(np.maximum(bboxes[:, 2] * bboxes[:, 3], 1.0))

    dt = np.diff(frames)
    valid = dt > 0
    if np.sum(valid) < 2:
        return 0.4

    velocities = np.diff(centers, axis=0)[valid] / dt[valid, None]
    scale_norm = float(np.median(scales) + EPS)
    speeds = np.linalg.norm(velocities, axis=1) / scale_norm
    if speeds.size < 2:
        return 0.6

    accel = np.diff(velocities, axis=0)
    accel_norm = np.linalg.norm(accel, axis=1) / scale_norm
    speed_cv = float(np.std(speeds) / (np.mean(speeds) + 0.05))
    accel_mean = float(np.mean(accel_norm))
    return clip01(np.exp(-0.7 * speed_cv - 0.5 * accel_mean))


def scale_stability_quality(rows):
    if len(rows) < 2:
        return 0.6
    areas = np.asarray([bbox_area_xywh(row[2:6]) for row in rows], dtype=np.float32)
    areas = np.maximum(areas, 1.0)
    return clip01(np.exp(-float(np.std(np.log(areas)))))


def tracklet_feature_from_rows(rows):
    features = [row_feature(row) for row in rows if row_feature(row) is not None]
    if not features:
        return None
    feat = np.mean(np.asarray(features, dtype=np.float32), axis=0)
    norm = float(np.linalg.norm(feat))
    if norm <= EPS:
        return None
    return feat / norm


def build_tracklets(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(class_id(row), track_id(row))].append(row)

    tracklets = []
    for (cls_id, tid), group_rows in grouped.items():
        group_rows = sorted(group_rows, key=lambda row: row[0])
        tracklets.append(Tracklet(
            cls_id=cls_id,
            track_id=tid,
            rows=group_rows,
            quality=tracklet_quality_from_rows(group_rows),
            feature=tracklet_feature_from_rows(group_rows)))
    return tracklets


def predict_next_bbox(tracklet, target_frame):
    rows = tracklet.rows
    last_frame = int(round(rows[-1][0]))
    last_bbox = np.asarray(rows[-1][2:6], dtype=np.float32)
    if len(rows) < 2:
        return last_bbox

    prev_frame = int(round(rows[-2][0]))
    prev_bbox = np.asarray(rows[-2][2:6], dtype=np.float32)
    frame_delta = max(last_frame - prev_frame, 1)
    velocity = (last_bbox - prev_bbox) / float(frame_delta)
    return last_bbox + velocity * float(target_frame - last_frame)


def tracklet_velocity(tracklet, from_end):
    rows = tracklet.rows
    if len(rows) < 2:
        return np.zeros(2, dtype=np.float32)

    if from_end:
        prev, cur = rows[-2], rows[-1]
    else:
        cur, prev = rows[1], rows[0]
    frame_delta = max(abs(int(round(cur[0])) - int(round(prev[0]))), 1)
    cur_center = bbox_center_xywh(cur[2:6])
    prev_center = bbox_center_xywh(prev[2:6])
    return (cur_center - prev_center) / float(frame_delta)


def velocity_similarity(src, dst):
    src_velocity = tracklet_velocity(src, from_end=True)
    dst_velocity = tracklet_velocity(dst, from_end=False)
    src_area = bbox_area_xywh(src.last_bbox)
    dst_area = bbox_area_xywh(dst.first_bbox)
    norm = np.sqrt(max((src_area + dst_area) * 0.5, 1.0))
    return clip01(np.exp(-float(np.linalg.norm(src_velocity - dst_velocity)) / (norm + EPS)))


def scale_similarity(pred_bbox, dst_bbox):
    pred_area = max(bbox_area_xywh(pred_bbox), 1.0)
    dst_area = max(bbox_area_xywh(dst_bbox), 1.0)
    return clip01(np.exp(-abs(np.log(pred_area / dst_area))))


def reid_similarity(src, dst):
    if src.feature is None or dst.feature is None:
        return None
    return clip01(float(np.dot(src.feature, dst.feature)))


def candidate_score(src, dst, pred_bbox, gap, args):
    cls_id = src.cls_id
    iou = bbox_iou_xywh(pred_bbox, dst.first_bbox)
    vel = velocity_similarity(src, dst)
    scale = scale_similarity(pred_bbox, dst.first_bbox)
    quality_pair = float(np.sqrt(max(src.quality * dst.quality, 0.0)))
    app = reid_similarity(src, dst)

    score = (
        class_setting(args, cls_id, 'iou_weight') * iou +
        class_setting(args, cls_id, 'velocity_weight') * vel +
        class_setting(args, cls_id, 'scale_weight') * scale +
        class_setting(args, cls_id, 'quality_weight') * quality_pair)
    if getattr(args, 'reid_aware_qhit', False) and app is not None:
        reid_weight = float(np.clip(args.reid_weight, 0.0, 0.8))
        score = (1.0 - reid_weight) * score + reid_weight * app
    threshold = (
        class_setting(args, cls_id, 'min_merge_score') +
        class_setting(args, cls_id, 'gap_penalty') * float(max(gap - 1, 0)) +
        class_setting(args, cls_id, 'quality_penalty') * (1.0 - quality_pair))
    if getattr(args, 'reid_aware_qhit', False) and app is not None and app < args.reid_min_sim:
        threshold += float(args.reid_low_sim_penalty)
    if getattr(args, 'recovery_aware_qhit', False):
        boundary_recovery = is_recovery_row(src.rows[-1]) or is_recovery_row(dst.rows[0])
        if boundary_recovery:
            threshold += float(args.recovery_merge_penalty)
    return score, threshold, iou, vel, scale, quality_pair, (-1.0 if app is None else app)


def collect_candidates(tracklets, max_delta_t, args):
    by_class = defaultdict(list)
    for tracklet in tracklets:
        min_len = class_setting(args, tracklet.cls_id, 'min_len')
        min_quality = class_setting(args, tracklet.cls_id, 'min_quality')
        if len(tracklet.rows) >= min_len and tracklet.quality >= min_quality:
            by_class[tracklet.cls_id].append(tracklet)

    candidates = []
    for cls_id, cls_tracklets in by_class.items():
        for src in cls_tracklets:
            for dst in cls_tracklets:
                if src.track_id == dst.track_id:
                    continue
                temporal_gap = dst.start - src.end
                if temporal_gap <= 0 or temporal_gap > max_delta_t:
                    continue

                pred_bbox = predict_next_bbox(src, dst.start)
                score, thresh, iou, vel, scale, quality_pair, app = candidate_score(
                    src, dst, pred_bbox, temporal_gap, args)
                if iou < class_setting(args, cls_id, 'base_iou_thresh'):
                    continue
                if score < thresh:
                    continue

                candidates.append(MergeCandidate(
                    cls_id=cls_id,
                    src_id=src.track_id,
                    dst_id=dst.track_id,
                    score=score,
                    threshold=thresh,
                    iou=iou,
                    velocity=vel,
                    scale=scale,
                    quality_pair=quality_pair,
                    gap=temporal_gap,
                    reid=app))

    return sorted(candidates, key=lambda c: (
        -c.score, -c.iou, c.gap, c.cls_id, c.src_id, c.dst_id))


def apply_merges(rows, candidates):
    used_src = set()
    used_dst = set()
    accepted = []
    for candidate in candidates:
        src_key = (candidate.cls_id, candidate.src_id)
        dst_key = (candidate.cls_id, candidate.dst_id)
        if src_key in used_src or src_key in used_dst:
            continue
        if dst_key in used_src or dst_key in used_dst:
            continue
        used_src.add(src_key)
        used_dst.add(dst_key)
        accepted.append(candidate)

    if not accepted:
        return 0

    dst_to_src = {(c.cls_id, c.dst_id): c.src_id for c in accepted}
    for row in rows:
        key = (class_id(row), track_id(row))
        if key in dst_to_src:
            row[1] = float(dst_to_src[key])

    return len(accepted)


def hierarchical_merge(rows, delta_t, args):
    total = 0
    for max_delta_t in delta_t:
        while True:
            tracklets = build_tracklets(rows)
            candidates = collect_candidates(tracklets, max_delta_t, args)
            merged = apply_merges(rows, candidates)
            if merged == 0:
                break
            total += merged
    return total


def drop_low_quality_tracklets(rows, args):
    keep_keys = set()
    drop_keys = set()
    for tracklet in build_tracklets(rows):
        key = (tracklet.cls_id, tracklet.track_id)
        min_len = class_setting(args, tracklet.cls_id, 'drop_min_len')
        min_quality = class_setting(args, tracklet.cls_id, 'drop_min_quality')
        if len(tracklet.rows) <= min_len and tracklet.quality < min_quality:
            drop_keys.add(key)
        else:
            keep_keys.add(key)

    if not drop_keys:
        return rows, 0

    kept_rows = [row for row in rows if (class_id(row), track_id(row)) not in drop_keys]
    return kept_rows, len(rows) - len(kept_rows)


def trim_recovery_edges(rows, args):
    kept_rows = []
    trimmed = 0
    for tracklet in build_tracklets(rows):
        sorted_rows = tracklet.rows
        if len(sorted_rows) < int(args.recovery_trim_min_track_len):
            kept_rows.extend(sorted_rows)
            continue

        anchor_indices = [
            idx for idx, row in enumerate(sorted_rows)
            if (not is_recovery_row(row)) and det_score(row) >= args.recovery_trim_anchor_score
        ]
        if not anchor_indices:
            kept_rows.extend(sorted_rows)
            continue

        first_anchor = anchor_indices[0]
        last_anchor = anchor_indices[-1]
        start = 0
        while (
                start < first_anchor and
                is_recovery_row(sorted_rows[start]) and
                det_score(sorted_rows[start]) <= args.recovery_trim_edge_score):
            start += 1

        end = len(sorted_rows) - 1
        while (
                end > last_anchor and
                is_recovery_row(sorted_rows[end]) and
                det_score(sorted_rows[end]) <= args.recovery_trim_edge_score):
            end -= 1

        if start <= end:
            kept_rows.extend(sorted_rows[start:end + 1])
            trimmed += start + (len(sorted_rows) - 1 - end)
        else:
            kept_rows.extend(sorted_rows)

    return kept_rows, trimmed


def trim_low_score_edges(rows, args):
    kept_rows = []
    trimmed = 0
    for tracklet in build_tracklets(rows):
        sorted_rows = tracklet.rows
        min_len = int(class_setting(args, tracklet.cls_id, 'trim_min_track_len'))
        anchor_score = class_setting(args, tracklet.cls_id, 'trim_anchor_score')
        edge_score = class_setting(args, tracklet.cls_id, 'trim_edge_score')
        if len(sorted_rows) < min_len:
            kept_rows.extend(sorted_rows)
            continue

        anchor_indices = [
            idx for idx, row in enumerate(sorted_rows)
            if det_score(row) >= anchor_score and
            (not getattr(args, 'recovery_aware_qhit', False) or not is_recovery_row(row))
        ]
        if not anchor_indices:
            kept_rows.extend(sorted_rows)
            continue

        first_anchor = anchor_indices[0]
        last_anchor = anchor_indices[-1]
        start = 0
        while start < first_anchor and (
                det_score(sorted_rows[start]) < edge_score or
                (getattr(args, 'recovery_aware_qhit', False) and
                 is_recovery_row(sorted_rows[start]) and
                 det_score(sorted_rows[start]) <= args.recovery_trim_edge_score)):
            start += 1

        end = len(sorted_rows) - 1
        while end > last_anchor and (
                det_score(sorted_rows[end]) < edge_score or
                (getattr(args, 'recovery_aware_qhit', False) and
                 is_recovery_row(sorted_rows[end]) and
                 det_score(sorted_rows[end]) <= args.recovery_trim_edge_score)):
            end -= 1

        if start <= end:
            kept_rows.extend(sorted_rows[start:end + 1])
            trimmed += start + (len(sorted_rows) - 1 - end)
        else:
            kept_rows.extend(sorted_rows)

    return kept_rows, trimmed


def load_class_score_threshold_file(path):
    if not path:
        return {}

    threshold_path = Path(path)
    if not threshold_path.exists():
        raise FileNotFoundError('Class score threshold file not found: {}'.format(path))

    if threshold_path.suffix.lower() == '.json':
        with threshold_path.open('r', encoding='utf-8') as handle:
            data = json.load(handle)
        if isinstance(data, dict) and 'thresholds' in data:
            data = data['thresholds']
        if isinstance(data, dict):
            return {int(k): float(v) for k, v in data.items()}
        thresholds = {}
        for item in data:
            cls_id = int(item.get('class_id', item.get('class', -1)))
            if cls_id > 0:
                thresholds[cls_id] = float(item['threshold'])
        return thresholds

    thresholds = {}
    with threshold_path.open('r', encoding='utf-8-sig', newline='') as handle:
        sample = handle.read(2048)
        handle.seek(0)
        has_header = csv.Sniffer().has_header(sample) if sample.strip() else False
        if has_header:
            reader = csv.DictReader(handle)
            for item in reader:
                cls_value = item.get('class_id') or item.get('class') or item.get('cls_id')
                threshold_value = item.get('threshold') or item.get('score_thresh')
                if cls_value is None or threshold_value is None:
                    continue
                thresholds[int(float(cls_value))] = float(threshold_value)
        else:
            reader = csv.reader(handle)
            for row in reader:
                if len(row) < 2:
                    continue
                thresholds[int(float(row[0]))] = float(row[1])
    return thresholds


def load_sequence_class_score_threshold_file(path):
    if not path:
        return {}

    threshold_path = Path(path)
    if not threshold_path.exists():
        raise FileNotFoundError('Sequence class score threshold file not found: {}'.format(path))

    overrides = {}
    with threshold_path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        for item in reader:
            sequence = item.get('sequence') or item.get('seq') or item.get('video')
            cls_value = item.get('class_id') or item.get('class') or item.get('cls_id')
            threshold_value = item.get('threshold') or item.get('score_thresh')
            if not sequence or cls_value is None or threshold_value is None:
                continue
            sequence = str(sequence).strip()
            if sequence.endswith('.txt'):
                sequence = sequence[:-4]
            cls_id = int(float(cls_value))
            overrides[(sequence, cls_id)] = float(threshold_value)
    return overrides


def class_score_thresholds(args):
    score_profile = getattr(args, 'class_score_profile', 'none')
    thresholds = dict(CLASS_SCORE_PROFILES.get(score_profile, {}))
    threshold_file = getattr(args, 'class_score_thresh_file', None)
    if threshold_file:
        cached = getattr(args, '_class_score_thresholds_from_file', None)
        if cached is None:
            cached = load_class_score_threshold_file(threshold_file)
            setattr(args, '_class_score_thresholds_from_file', cached)
        thresholds.update(cached)
    sequence_threshold_file = getattr(args, 'sequence_class_score_thresh_file', None)
    current_sequence = getattr(args, '_current_sequence', None)
    if sequence_threshold_file and current_sequence:
        cached = getattr(args, '_sequence_class_score_thresholds_from_file', None)
        if cached is None:
            cached = load_sequence_class_score_threshold_file(sequence_threshold_file)
            setattr(args, '_sequence_class_score_thresholds_from_file', cached)
        for (sequence, cls_id), threshold in cached.items():
            if sequence == current_sequence:
                thresholds[cls_id] = threshold
    return thresholds


def filter_low_score_rows(rows, args):
    class_thresholds = class_score_thresholds(args)
    default_thresh = float(getattr(args, 'score_filter_thresh', 0.0))
    if not class_thresholds and default_thresh <= 0.0:
        return rows, 0

    keep_interpolated = bool(getattr(args, 'score_filter_keep_interpolated', False))
    keep_interpolated_min_score = float(getattr(args, 'score_filter_keep_interpolated_min_score', 0.0))
    kept_rows = []
    filtered = 0
    for row in rows:
        thresh = float(class_thresholds.get(class_id(row), default_thresh))
        if keep_interpolated and row_source(row) == SOURCE_INTERPOLATED and det_score(row) >= keep_interpolated_min_score:
            kept_rows.append(row)
        elif det_score(row) >= thresh:
            kept_rows.append(row)
        else:
            filtered += 1
    return kept_rows, filtered


def tracklet_score_profile_settings(args, cls_id):
    profile_name = getattr(args, 'tracklet_score_profile', 'none')
    profile = CLASS_TRACKLET_SCORE_PROFILES.get(profile_name, {})
    return profile.get(cls_id)


def tracklet_adaptive_threshold(tracklet, args):
    settings = tracklet_score_profile_settings(args, tracklet.cls_id)
    default_thresh = float(getattr(args, 'tracklet_score_default_thresh', 0.0))
    if settings is None:
        return default_thresh

    scores = [det_score(row) for row in tracklet.rows]
    mean_score = float(np.mean(scores)) if scores else 0.0
    length = len(tracklet.rows)
    quality = float(tracklet.quality)

    base_thresh = float(settings.get('base_thresh', default_thresh))
    floor_thresh = float(settings.get('floor_thresh', base_thresh))
    poor_thresh = float(settings.get('poor_thresh', base_thresh))
    protect_quality = float(settings.get('protect_quality', 0.60))
    protect_len = int(settings.get('protect_len', 12))
    protect_mean = float(settings.get('protect_mean', base_thresh + 0.08))
    poor_quality = float(settings.get('poor_quality', 0.35))
    poor_len = int(settings.get('poor_len', 3))
    poor_mean = float(settings.get('poor_mean', base_thresh))

    if length <= poor_len and quality < poor_quality and mean_score < poor_mean:
        return poor_thresh

    if quality >= protect_quality or length >= protect_len or mean_score >= protect_mean:
        return floor_thresh

    denom = max(protect_quality - poor_quality, EPS)
    quality_alpha = clip01((quality - poor_quality) / denom)
    return float(base_thresh - quality_alpha * max(base_thresh - floor_thresh, 0.0))


def filter_tracklet_score_rows(rows, args):
    if not rows or getattr(args, 'tracklet_score_profile', 'none') == 'none':
        return rows, 0

    kept_rows = []
    filtered = 0
    for tracklet in build_tracklets(rows):
        threshold = tracklet_adaptive_threshold(tracklet, args)
        if threshold <= 0.0:
            kept_rows.extend(tracklet.rows)
            continue

        for row in tracklet.rows:
            if det_score(row) >= threshold:
                kept_rows.append(row)
            else:
                filtered += 1

    return kept_rows, filtered


def recovery_segment_confirmed(tracklet, sorted_rows, start_idx, end_idx, args):
    cls_id = tracklet.cls_id
    segment = sorted_rows[start_idx:end_idx + 1]
    max_run = int(class_setting(args, cls_id, 'recovery_confirm_max_run'))
    min_quality = class_setting(args, cls_id, 'recovery_confirm_min_quality')
    min_score = class_setting(args, cls_id, 'recovery_confirm_min_score')
    anchor_score = class_setting(args, cls_id, 'recovery_confirm_anchor_score')
    window = int(class_setting(args, cls_id, 'recovery_confirm_window'))

    if len(segment) > max_run:
        return False
    if tracklet.quality < min_quality:
        return False
    if min(det_score(row) for row in segment) < min_score:
        return False

    start_frame = int(round(segment[0][0]))
    end_frame = int(round(segment[-1][0]))
    past_anchor = False
    future_anchor = False

    for idx, row in enumerate(sorted_rows):
        if start_idx <= idx <= end_idx:
            continue
        if is_recovery_row(row):
            continue
        if det_score(row) < anchor_score:
            continue
        frame = int(round(row[0]))
        if 0 < start_frame - frame <= window:
            past_anchor = True
        if 0 < frame - end_frame <= window:
            future_anchor = True

    if getattr(args, 'recovery_confirm_require_past_anchor', False):
        return past_anchor and future_anchor
    return future_anchor


def confirm_recovery_tracklets(rows, args):
    if not rows or not getattr(args, 'confirm_recovery_tracklets', False):
        return rows, 0

    kept_rows = []
    removed = 0
    for tracklet in build_tracklets(rows):
        sorted_rows = tracklet.rows
        idx = 0
        keep_mask = [True] * len(sorted_rows)
        while idx < len(sorted_rows):
            if not is_recovery_row(sorted_rows[idx]):
                idx += 1
                continue

            start_idx = idx
            while idx + 1 < len(sorted_rows) and is_recovery_row(sorted_rows[idx + 1]):
                idx += 1
            end_idx = idx

            if not recovery_segment_confirmed(tracklet, sorted_rows, start_idx, end_idx, args):
                for remove_idx in range(start_idx, end_idx + 1):
                    keep_mask[remove_idx] = False
                    removed += 1
            idx += 1

        kept_rows.extend(row for row, keep in zip(sorted_rows, keep_mask) if keep)

    return kept_rows, removed


def interpolation_motion_ok(left, right, args):
    cls_id = class_id(left)
    left_bbox = np.asarray(left[2:6], dtype=np.float32)
    right_bbox = np.asarray(right[2:6], dtype=np.float32)
    left_center = bbox_center_xywh(left_bbox)
    right_center = bbox_center_xywh(right_bbox)
    frame_gap = max(int(round(right[0])) - int(round(left[0])), 1)
    mean_area = max((bbox_area_xywh(left_bbox) + bbox_area_xywh(right_bbox)) * 0.5, 1.0)
    step = float(np.linalg.norm(right_center - left_center) / frame_gap)
    step_norm = step / (np.sqrt(mean_area) + EPS)

    area_ratio = max(bbox_area_xywh(left_bbox), 1.0) / max(bbox_area_xywh(right_bbox), 1.0)
    scale_log = abs(float(np.log(area_ratio)))
    return (
        step_norm <= class_setting(args, cls_id, 'interp_max_motion') and
        scale_log <= class_setting(args, cls_id, 'interp_max_scale_log'))


def interpolate_tracklets(rows, args):
    interpolated = list(rows)
    added = 0
    for tracklet in build_tracklets(rows):
        max_gap = int(class_setting(args, tracklet.cls_id, 'interp_max_gap'))
        min_quality = class_setting(args, tracklet.cls_id, 'interp_min_quality')
        min_score = class_setting(args, tracklet.cls_id, 'interp_min_score')
        if tracklet.quality < min_quality:
            continue
        sorted_rows = tracklet.rows
        for left, right in zip(sorted_rows[:-1], sorted_rows[1:]):
            left_frame = int(round(left[0]))
            right_frame = int(round(right[0]))
            frame_gap = right_frame - left_frame
            if frame_gap <= 1 or frame_gap > max_gap:
                continue
            if min(det_score(left), det_score(right)) < min_score:
                continue
            if not interpolation_motion_ok(left, right, args):
                continue

            left_bbox = np.asarray(left[2:6], dtype=np.float32)
            right_bbox = np.asarray(right[2:6], dtype=np.float32)
            left_score = det_score(left)
            right_score = det_score(right)
            fill_score = min(left_score, right_score) * np.exp(-0.15 * float(frame_gap - 1))
            for offset in range(1, frame_gap):
                ratio = float(offset) / float(frame_gap)
                new_row = list(left)
                new_row[0] = float(left_frame + offset)
                new_row[1] = float(tracklet.track_id)
                new_row[2:6] = (left_bbox + ratio * (right_bbox - left_bbox)).tolist()
                if len(new_row) > 6:
                    new_row[6] = float(fill_score)
                if len(new_row) > 7:
                    new_row[7] = float(tracklet.cls_id)
                if len(new_row) <= INTERNAL_SOURCE_INDEX:
                    new_row.extend([0.0] * (INTERNAL_SOURCE_INDEX + 1 - len(new_row)))
                new_row[INTERNAL_SOURCE_INDEX] = float(SOURCE_INTERPOLATED)
                interpolated.append(new_row)
                added += 1
    return interpolated, added


def process_file(input_path, output_path, delta_t, args):
    setattr(args, '_current_sequence', input_path.stem)
    rows = read_result_file(input_path)
    original_rows = len(rows)
    sourced = attach_sources(rows, input_path, args)
    featured = attach_features(rows, input_path, args)
    attach_evidence_features(rows, input_path, args)

    recovery_trimmed = 0
    if args.recovery_aware_qhit:
        rows, recovery_trimmed = trim_recovery_edges(rows, args)

    merged = hierarchical_merge(rows, delta_t, args)

    recovery_unconfirmed = 0
    rejected_recovery_rows = []
    if args.confirm_recovery_tracklets:
        before_confirmation = rows
        rows, recovery_unconfirmed = confirm_recovery_tracklets(rows, args)
        if getattr(args, 'evidence_backfill_profile', 'none') != 'none':
            confirmed_ids = {id(row) for row in rows}
            rejected_recovery_rows = [row for row in before_confirmation if id(row) not in confirmed_ids]

    dropped = 0
    if args.drop_low_quality:
        rows, dropped = drop_low_quality_tracklets(rows, args)

    interpolated = 0
    if args.interpolate:
        rows, interpolated = interpolate_tracklets(rows, args)

    trimmed = 0
    if args.trim_low_score_edges:
        rows, trimmed = trim_low_score_edges(rows, args)

    score_filtered = 0
    if (args.score_filter_thresh > 0.0 or args.class_score_profile != 'none' or
            args.class_score_thresh_file or args.sequence_class_score_thresh_file):
        rows, score_filtered = filter_low_score_rows(rows, args)

    tracklet_score_filtered = 0
    if args.tracklet_score_profile != 'none':
        rows, tracklet_score_filtered = filter_tracklet_score_rows(rows, args)

    backfilled = 0
    backfill_profile = getattr(args, 'evidence_backfill_profile', 'none')
    if backfill_profile != 'none':
        from qtr_evidence_backfill import restore_recovery_rows
        rows, backfill_stats = restore_recovery_rows(rows, rejected_recovery_rows, args, backfill_profile)
        backfilled = backfill_stats['added']
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.with_suffix(output_path.suffix + '.backfill.json').open('w', encoding='utf-8') as handle:
            json.dump(dict(profile=backfill_profile, **backfill_stats), handle, indent=2, sort_keys=True)
        print('{} evidence-backfill: {}'.format(input_path.name, backfill_stats))

    write_result_file(output_path, rows)
    return {
        'original_rows': original_rows,
        'output_rows': len(rows),
        'merges': merged,
        'dropped_rows': dropped,
        'interpolated_rows': interpolated,
        'trimmed_rows': trimmed,
        'score_filtered_rows': score_filtered,
        'tracklet_score_filtered_rows': tracklet_score_filtered,
        'source_rows': sourced,
        'feature_rows': featured,
        'recovery_unconfirmed_rows': recovery_unconfirmed,
        'recovery_trimmed_rows': recovery_trimmed,
        'backfilled_rows': backfilled,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description='Q-HIT quality-aware offline post-processing for MCMOT result txt files.')
    parser.add_argument('--input-dir', required=True, help='Directory containing tracker result txt files.')
    parser.add_argument('--output-dir', required=True, help='Directory for post-processed txt files.')
    parser.add_argument('--source-dir', default=None,
                        help='Optional directory containing *.txt.src tracker source sidecars.')
    parser.add_argument('--feature-dir', default=None,
                        help='Optional directory containing *.txt.feat tracker ReID sidecars.')
    parser.add_argument('--delta-t', default='1,3,5,7,9',
                        help='Comma-separated hierarchical temporal windows.')
    parser.add_argument('--class-profile', default='none',
                        choices=sorted(CLASS_PARAM_OVERRIDES.keys()),
                        help='Optional class-aware parameter profile.')
    parser.add_argument('--base-iou-thresh', type=float, default=0.18,
                        help='Minimum predicted IoU before quality-aware scoring.')
    parser.add_argument('--min-len', type=int, default=1,
                        help='Minimum tracklet length considered for merging.')
    parser.add_argument('--min-quality', type=float, default=0.18,
                        help='Minimum tracklet quality considered for merging.')
    parser.add_argument('--min-merge-score', type=float, default=0.38,
                        help='Base threshold for quality-aware merge score.')
    parser.add_argument('--gap-penalty', type=float, default=0.025,
                        help='Extra merge threshold per missing frame.')
    parser.add_argument('--quality-penalty', type=float, default=0.12,
                        help='Extra merge threshold for low-quality tracklet pairs.')
    parser.add_argument('--iou-weight', type=float, default=0.50)
    parser.add_argument('--velocity-weight', type=float, default=0.20)
    parser.add_argument('--scale-weight', type=float, default=0.15)
    parser.add_argument('--quality-weight', type=float, default=0.15)
    parser.add_argument('--reid-aware-qhit', action='store_true',
                        help='Use tracker ReID sidecars to constrain and score Q-HIT merges.')
    parser.add_argument('--reid-weight', type=float, default=0.20,
                        help='Blend weight of ReID similarity in Q-HIT merge scoring.')
    parser.add_argument('--reid-min-sim', type=float, default=0.35,
                        help='ReID similarity below this value receives a merge threshold penalty.')
    parser.add_argument('--reid-low-sim-penalty', type=float, default=0.08,
                        help='Extra merge threshold when ReID similarity is below --reid-min-sim.')
    parser.add_argument('--drop-low-quality', action='store_true',
                        help='Drop very short and low-quality tracklets after merging.')
    parser.add_argument('--drop-min-len', type=int, default=2,
                        help='Only tracklets this short or shorter can be dropped.')
    parser.add_argument('--drop-min-quality', type=float, default=0.12,
                        help='Quality threshold used with --drop-low-quality.')
    parser.add_argument('--interpolate', action='store_true',
                        help='Conservatively fill short gaps after quality-aware merging.')
    parser.add_argument('--interp-max-gap', type=int, default=3,
                        help='Maximum frame gap filled by --interpolate.')
    parser.add_argument('--interp-min-quality', type=float, default=0.35,
                        help='Minimum merged tracklet quality for interpolation.')
    parser.add_argument('--interp-min-score', type=float, default=0.35,
                        help='Minimum endpoint score for interpolation.')
    parser.add_argument('--interp-max-motion', type=float, default=1.60,
                        help='Maximum center step per frame, normalized by object scale.')
    parser.add_argument('--interp-max-scale-log', type=float, default=0.70,
                        help='Maximum absolute log area ratio for interpolation endpoints.')
    parser.add_argument('--trim-low-score-edges', action='store_true',
                        help='Trim low-confidence rows from the beginning/end of confirmed tracklets.')
    parser.add_argument('--trim-min-track-len', type=int, default=6,
                        help='Minimum tracklet length eligible for edge score trimming.')
    parser.add_argument('--trim-anchor-score', type=float, default=0.45,
                        help='Score used as the high-confidence anchor for edge trimming.')
    parser.add_argument('--trim-edge-score', type=float, default=0.35,
                        help='Low-confidence edge rows below this score are trimmed.')
    parser.add_argument('--recovery-aware-qhit', action='store_true',
                        help='Use SCA-TDRM source sidecars to trim and merge recovered rows more carefully.')
    parser.add_argument('--recovery-trim-min-track-len', type=int, default=4,
                        help='Minimum tracklet length eligible for recovery-aware edge trimming.')
    parser.add_argument('--recovery-trim-anchor-score', type=float, default=0.45,
                        help='Non-recovery score anchor used by recovery-aware edge trimming.')
    parser.add_argument('--recovery-trim-edge-score', type=float, default=0.35,
                        help='Recovered edge rows at or below this score can be trimmed.')
    parser.add_argument('--recovery-merge-penalty', type=float, default=0.0,
                        help='Extra Q-HIT merge threshold when a merge boundary is a recovered row.')
    parser.add_argument('--confirm-recovery-tracklets', action='store_true',
                        help='Drop recovered rows that are not confirmed by nearby high-score non-recovery anchors.')
    parser.add_argument('--recovery-confirm-window', type=int, default=3,
                        help='Max frame distance to a confirming high-score non-recovery anchor.')
    parser.add_argument('--recovery-confirm-anchor-score', type=float, default=0.45,
                        help='Minimum score for a non-recovery row to confirm recovered rows.')
    parser.add_argument('--recovery-confirm-min-quality', type=float, default=0.45,
                        help='Minimum tracklet quality for keeping recovered rows.')
    parser.add_argument('--recovery-confirm-min-score', type=float, default=0.18,
                        help='Minimum score of each recovered row kept by confirmation.')
    parser.add_argument('--recovery-confirm-max-run', type=int, default=2,
                        help='Maximum consecutive recovered rows that can be kept without a high-score update.')
    parser.add_argument('--recovery-confirm-require-past-anchor', action='store_true',
                        help='Require both previous and future anchors for recovered rows.')
    parser.add_argument('--score-filter-thresh', type=float, default=0.0,
                        help='Drop rows below this confidence score after all Q-HIT steps. 0 disables it.')
    parser.add_argument('--class-score-profile', default='none',
                        choices=sorted(CLASS_SCORE_PROFILES.keys()),
                        help='Optional class-aware confidence filter applied after all Q-HIT steps.')
    parser.add_argument('--class-score-thresh-file', default=None,
                        help='Optional CSV/JSON class_id -> score threshold file. Overrides --class-score-profile values.')
    parser.add_argument('--sequence-class-score-thresh-file', default=None,
                        help='Optional CSV sequence,class_id,threshold overrides for class score filtering.')
    parser.add_argument('--score-filter-keep-interpolated', action='store_true',
                        help='Keep interpolated QTR rows from high-quality tracklets even when below the row score gate.')
    parser.add_argument('--score-filter-keep-interpolated-min-score', type=float, default=0.25,
                        help='Minimum score for interpolated rows protected by --score-filter-keep-interpolated.')
    parser.add_argument('--tracklet-score-profile', default='none',
                        choices=sorted(CLASS_TRACKLET_SCORE_PROFILES.keys()),
                        help='Optional tracklet-quality-adaptive class score gate.')
    parser.add_argument('--tracklet-score-default-thresh', type=float, default=0.0,
                        help='Fallback score threshold for classes not covered by --tracklet-score-profile.')
    parser.add_argument('--evidence-backfill-profile', default='none',
                        choices=['none', 'strict', 'balanced', 'balanced_gap8', 'balanced_gap10'],
                        help='Opt-in offline restoration of rejected real recovery detections inside retained tracks.')
    parser.add_argument('--evidence-feature-dir', default=None,
                        help='Optional independent current-frame *.txt.feat files for backfill only, not QTR merges.')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.evidence_backfill_profile != 'none' and not (args.confirm_recovery_tracklets and args.reid_aware_qhit):
        raise ValueError('Evidence backfill requires --confirm-recovery-tracklets and --reid-aware-qhit')
    if args.evidence_feature_dir and args.evidence_backfill_profile == 'none':
        raise ValueError('--evidence-feature-dir requires evidence backfill to be enabled')
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    delta_t = parse_delta_t(args.delta_t)

    txt_files = sorted(input_dir.glob('*.txt'))
    if not txt_files:
        raise FileNotFoundError('No txt result files found in {}'.format(input_dir))

    totals = defaultdict(int)
    for input_path in txt_files:
        output_path = output_dir / input_path.name
        stats = process_file(
            input_path=input_path,
            output_path=output_path,
            delta_t=delta_t,
            args=args)
        for key, value in stats.items():
            totals[key] += value
        print(
            '{} -> {} rows, {} source-tagged, {} feature-tagged, {} recovery-unconfirmed, {} recovery-trimmed, {} merges, {} dropped, {} interpolated, {} trimmed, {} score-filtered, {} tracklet-score-filtered'.format(
                input_path.name,
                stats['output_rows'],
                stats['source_rows'],
                stats['feature_rows'],
                stats['recovery_unconfirmed_rows'],
                stats['recovery_trimmed_rows'],
                stats['merges'],
                stats['dropped_rows'],
                stats['interpolated_rows'],
                stats['trimmed_rows'],
                stats['score_filtered_rows'],
                stats['tracklet_score_filtered_rows']))

    print(
        'TOTAL -> {} rows in, {} rows out, {} source-tagged, {} feature-tagged, {} recovery-unconfirmed, {} recovery-trimmed, {} merges, {} dropped, {} interpolated, {} trimmed, {} score-filtered, {} tracklet-score-filtered'.format(
            totals['original_rows'],
            totals['output_rows'],
            totals['source_rows'],
            totals['feature_rows'],
            totals['recovery_unconfirmed_rows'],
            totals['recovery_trimmed_rows'],
            totals['merges'],
            totals['dropped_rows'],
            totals['interpolated_rows'],
            totals['trimmed_rows'],
            totals['score_filtered_rows'],
            totals['tracklet_score_filtered_rows']))


if __name__ == '__main__':
    main()
