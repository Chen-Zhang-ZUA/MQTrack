import lap
import numpy as np
import scipy
import torch
import torch.nn.functional as F
from cython_bbox import bbox_overlaps as bbox_ious
from scipy.spatial.distance import cdist
from lib.tracking_utils import kalman_filter


EPS = 1e-12


SCA_TDRM_VISDRONE_PROFILE = {
    # Internal class ids are zero-based. Result files are written with +1 ids.
    0: {'low_conf': 0.10, 'match_thresh': 0.38, 'center_gate': 0.95, 'scale_gate': 0.58, 'min_quality': 0.34},
    1: {'low_conf': 0.10, 'match_thresh': 0.38, 'center_gate': 1.00, 'scale_gate': 0.58, 'min_quality': 0.34},
    2: {'low_conf': 0.12, 'match_thresh': 0.35, 'center_gate': 0.95, 'scale_gate': 0.52, 'min_quality': 0.36},
    3: {'low_conf': 0.18, 'match_thresh': 0.33, 'center_gate': 1.05, 'scale_gate': 0.46, 'min_quality': 0.38},
    4: {'low_conf': 0.18, 'match_thresh': 0.33, 'center_gate': 1.05, 'scale_gate': 0.46, 'min_quality': 0.38},
    5: {'low_conf': 0.20, 'match_thresh': 0.31, 'center_gate': 1.08, 'scale_gate': 0.44, 'min_quality': 0.40},
    6: {'low_conf': 0.12, 'match_thresh': 0.35, 'center_gate': 0.95, 'scale_gate': 0.52, 'min_quality': 0.36},
    7: {'low_conf': 0.14, 'match_thresh': 0.34, 'center_gate': 1.00, 'scale_gate': 0.50, 'min_quality': 0.36},
    8: {'low_conf': 0.20, 'match_thresh': 0.31, 'center_gate': 1.08, 'scale_gate': 0.44, 'min_quality': 0.40},
    9: {'low_conf': 0.10, 'match_thresh': 0.36, 'center_gate': 0.90, 'scale_gate': 0.55, 'min_quality': 0.34},
}


SCA_TDRM_VISDRONE_QC_PROFILE = {
    # A stricter quality-confirmed profile for UAV small targets.  It keeps
    # SCA-TDRM useful for recall, but limits consecutive low-score recoveries
    # and raises the quality bar for classes that created many FPs on VisDrone.
    0: {
        'low_conf': 0.16,
        'match_thresh': 0.32,
        'center_gate': 0.82,
        'scale_gate': 0.46,
        'min_quality': 0.42,
        'quality_bonus': 0.03,
        'max_recovery_streak': 1,
        'confirm_window': 4,
        'pending_penalty': 0.06,
    },
    1: {
        'low_conf': 0.16,
        'match_thresh': 0.32,
        'center_gate': 0.85,
        'scale_gate': 0.46,
        'min_quality': 0.42,
        'quality_bonus': 0.03,
        'max_recovery_streak': 1,
        'confirm_window': 4,
        'pending_penalty': 0.06,
    },
    2: {
        'low_conf': 0.18,
        'match_thresh': 0.30,
        'center_gate': 0.78,
        'scale_gate': 0.42,
        'min_quality': 0.44,
        'quality_bonus': 0.025,
        'max_recovery_streak': 1,
        'confirm_window': 3,
        'pending_penalty': 0.08,
    },
    3: {
        'low_conf': 0.20,
        'match_thresh': 0.30,
        'center_gate': 0.92,
        'scale_gate': 0.38,
        'min_quality': 0.43,
        'quality_bonus': 0.025,
        'max_recovery_streak': 2,
        'confirm_window': 4,
        'pending_penalty': 0.05,
    },
    4: {
        'low_conf': 0.26,
        'match_thresh': 0.27,
        'center_gate': 0.78,
        'scale_gate': 0.34,
        'min_quality': 0.52,
        'quality_bonus': 0.015,
        'max_recovery_streak': 1,
        'confirm_window': 3,
        'pending_penalty': 0.10,
    },
    5: {
        'low_conf': 0.24,
        'match_thresh': 0.28,
        'center_gate': 0.88,
        'scale_gate': 0.36,
        'min_quality': 0.48,
        'quality_bonus': 0.02,
        'max_recovery_streak': 1,
        'confirm_window': 3,
        'pending_penalty': 0.08,
    },
    6: {
        'low_conf': 0.18,
        'match_thresh': 0.30,
        'center_gate': 0.78,
        'scale_gate': 0.42,
        'min_quality': 0.44,
        'quality_bonus': 0.025,
        'max_recovery_streak': 1,
        'confirm_window': 3,
        'pending_penalty': 0.08,
    },
    7: {
        'low_conf': 0.18,
        'match_thresh': 0.30,
        'center_gate': 0.80,
        'scale_gate': 0.40,
        'min_quality': 0.45,
        'quality_bonus': 0.02,
        'max_recovery_streak': 1,
        'confirm_window': 3,
        'pending_penalty': 0.08,
    },
    8: {
        'low_conf': 0.24,
        'match_thresh': 0.28,
        'center_gate': 0.88,
        'scale_gate': 0.36,
        'min_quality': 0.48,
        'quality_bonus': 0.02,
        'max_recovery_streak': 1,
        'confirm_window': 3,
        'pending_penalty': 0.08,
    },
    9: {
        'low_conf': 0.16,
        'match_thresh': 0.31,
        'center_gate': 0.76,
        'scale_gate': 0.44,
        'min_quality': 0.42,
        'quality_bonus': 0.025,
        'max_recovery_streak': 1,
        'confirm_window': 3,
        'pending_penalty': 0.07,
    },
}


SCA_TDRM_MQTRACK_QC_PROFILE = {
    # Balanced profile for MQTrack ablations. It keeps CAST's recall gain for
    # reliable classes, while tightening pedestrian/van recovery where false
    # positives dominated the test-dev ablation.
    0: {
        'low_conf': 0.14,
        'match_thresh': 0.34,
        'center_gate': 0.84,
        'scale_gate': 0.48,
        'min_quality': 0.40,
        'quality_bonus': 0.025,
        'max_recovery_streak': 1,
        'confirm_window': 4,
        'pending_penalty': 0.06,
        'lost_max_gap': 6,
        'lost_min_score': 0.20,
        'lost_match_thresh': 0.28,
        'lost_min_quality': 0.45,
    },
    1: {
        'low_conf': 0.16,
        'match_thresh': 0.32,
        'center_gate': 0.86,
        'scale_gate': 0.46,
        'min_quality': 0.42,
        'quality_bonus': 0.025,
        'max_recovery_streak': 1,
        'confirm_window': 4,
        'pending_penalty': 0.06,
    },
    2: {
        'low_conf': 0.18,
        'match_thresh': 0.30,
        'center_gate': 0.80,
        'scale_gate': 0.42,
        'min_quality': 0.44,
        'quality_bonus': 0.02,
        'max_recovery_streak': 1,
        'confirm_window': 3,
        'pending_penalty': 0.08,
    },
    3: {
        'low_conf': 0.18,
        'match_thresh': 0.31,
        'center_gate': 0.96,
        'scale_gate': 0.42,
        'min_quality': 0.40,
        'quality_bonus': 0.02,
        'max_recovery_streak': 2,
        'confirm_window': 4,
        'pending_penalty': 0.04,
        'lost_max_gap': 6,
        'lost_min_score': 0.20,
        'lost_match_thresh': 0.28,
        'lost_min_quality': 0.42,
    },
    4: {
        'low_conf': 0.32,
        'match_thresh': 0.24,
        'center_gate': 0.70,
        'scale_gate': 0.30,
        'min_quality': 0.58,
        'quality_bonus': 0.01,
        'max_recovery_streak': 1,
        'confirm_window': 3,
        'pending_penalty': 0.12,
        'lost_max_gap': 3,
        'lost_min_score': 0.34,
        'lost_match_thresh': 0.22,
        'lost_min_quality': 0.60,
    },
    5: {
        'low_conf': 0.20,
        'match_thresh': 0.30,
        'center_gate': 0.92,
        'scale_gate': 0.38,
        'min_quality': 0.42,
        'quality_bonus': 0.02,
        'max_recovery_streak': 1,
        'confirm_window': 3,
        'pending_penalty': 0.07,
        'lost_max_gap': 6,
        'lost_min_score': 0.20,
        'lost_match_thresh': 0.29,
        'lost_min_quality': 0.42,
    },
    6: {
        'low_conf': 0.18,
        'match_thresh': 0.30,
        'center_gate': 0.80,
        'scale_gate': 0.42,
        'min_quality': 0.44,
        'quality_bonus': 0.02,
        'max_recovery_streak': 1,
        'confirm_window': 3,
        'pending_penalty': 0.08,
    },
    7: {
        'low_conf': 0.18,
        'match_thresh': 0.30,
        'center_gate': 0.82,
        'scale_gate': 0.40,
        'min_quality': 0.45,
        'quality_bonus': 0.02,
        'max_recovery_streak': 1,
        'confirm_window': 3,
        'pending_penalty': 0.08,
    },
    8: {
        'low_conf': 0.28,
        'match_thresh': 0.25,
        'center_gate': 0.82,
        'scale_gate': 0.34,
        'min_quality': 0.54,
        'quality_bonus': 0.01,
        'max_recovery_streak': 1,
        'confirm_window': 3,
        'pending_penalty': 0.10,
        'lost_max_gap': 3,
        'lost_min_score': 0.30,
        'lost_match_thresh': 0.22,
        'lost_min_quality': 0.55,
    },
    9: {
        'low_conf': 0.16,
        'match_thresh': 0.31,
        'center_gate': 0.78,
        'scale_gate': 0.44,
        'min_quality': 0.42,
        'quality_bonus': 0.02,
        'max_recovery_streak': 1,
        'confirm_window': 3,
        'pending_penalty': 0.07,
    },
}


SCA_TDRM_MQTRACK_DIAGNOSTIC_PROFILE = {
    # Diagnostic profile from test-dev error analysis.  Pedestrian recovery is
    # still allowed because FN dominates, but low-score updates must come from
    # stable tracks.  Car recovery is stricter to reduce FP/ID churn.  Van and
    # truck are recall-starved, so only high-quality tracks may absorb low-score
    # detections instead of globally raising recall.
    0: {
        'low_conf': 0.14,
        'match_thresh': 0.31,
        'center_gate': 0.88,
        'scale_gate': 0.44,
        'min_quality': 0.44,
        'quality_bonus': 0.020,
        'max_recovery_streak': 1,
        'confirm_window': 4,
        'pending_penalty': 0.07,
        'lost_max_gap': 5,
        'lost_min_score': 0.22,
        'lost_match_thresh': 0.27,
        'lost_min_quality': 0.48,
    },
    1: {
        'low_conf': 0.16,
        'match_thresh': 0.31,
        'center_gate': 0.84,
        'scale_gate': 0.44,
        'min_quality': 0.44,
        'quality_bonus': 0.020,
        'max_recovery_streak': 1,
        'confirm_window': 3,
        'pending_penalty': 0.08,
    },
    2: {
        'low_conf': 0.18,
        'match_thresh': 0.30,
        'center_gate': 0.80,
        'scale_gate': 0.40,
        'min_quality': 0.46,
        'quality_bonus': 0.015,
        'max_recovery_streak': 1,
        'confirm_window': 3,
        'pending_penalty': 0.09,
    },
    3: {
        'low_conf': 0.20,
        'match_thresh': 0.28,
        'center_gate': 0.92,
        'scale_gate': 0.38,
        'min_quality': 0.46,
        'quality_bonus': 0.012,
        'max_recovery_streak': 1,
        'confirm_window': 3,
        'pending_penalty': 0.09,
        'lost_max_gap': 4,
        'lost_min_score': 0.24,
        'lost_match_thresh': 0.25,
        'lost_min_quality': 0.50,
    },
    4: {
        'low_conf': 0.26,
        'match_thresh': 0.24,
        'center_gate': 0.74,
        'scale_gate': 0.32,
        'min_quality': 0.56,
        'quality_bonus': 0.010,
        'max_recovery_streak': 1,
        'confirm_window': 3,
        'pending_penalty': 0.12,
        'lost_max_gap': 4,
        'lost_min_score': 0.30,
        'lost_match_thresh': 0.22,
        'lost_min_quality': 0.58,
    },
    5: {
        'low_conf': 0.20,
        'match_thresh': 0.29,
        'center_gate': 0.94,
        'scale_gate': 0.38,
        'min_quality': 0.44,
        'quality_bonus': 0.018,
        'max_recovery_streak': 1,
        'confirm_window': 3,
        'pending_penalty': 0.08,
        'lost_max_gap': 6,
        'lost_min_score': 0.22,
        'lost_match_thresh': 0.28,
        'lost_min_quality': 0.44,
    },
    6: {
        'low_conf': 0.18,
        'match_thresh': 0.30,
        'center_gate': 0.80,
        'scale_gate': 0.40,
        'min_quality': 0.46,
        'quality_bonus': 0.015,
        'max_recovery_streak': 1,
        'confirm_window': 3,
        'pending_penalty': 0.09,
    },
    7: {
        'low_conf': 0.18,
        'match_thresh': 0.30,
        'center_gate': 0.82,
        'scale_gate': 0.40,
        'min_quality': 0.46,
        'quality_bonus': 0.015,
        'max_recovery_streak': 1,
        'confirm_window': 3,
        'pending_penalty': 0.09,
    },
    8: {
        'low_conf': 0.26,
        'match_thresh': 0.25,
        'center_gate': 0.84,
        'scale_gate': 0.34,
        'min_quality': 0.54,
        'quality_bonus': 0.010,
        'max_recovery_streak': 1,
        'confirm_window': 3,
        'pending_penalty': 0.11,
        'lost_max_gap': 3,
        'lost_min_score': 0.30,
        'lost_match_thresh': 0.22,
        'lost_min_quality': 0.56,
    },
    9: {
        'low_conf': 0.16,
        'match_thresh': 0.30,
        'center_gate': 0.78,
        'scale_gate': 0.42,
        'min_quality': 0.44,
        'quality_bonus': 0.015,
        'max_recovery_streak': 1,
        'confirm_window': 3,
        'pending_penalty': 0.08,
    },
}


SCA_TDRM_UAVDT_STRICT_PROFILE = {
    # UAVDT uses one vehicle class.  Low-score recovery is useful for identity
    # continuity, but too much recovery quickly turns into false positives, so
    # this profile only admits near-threshold detections matched to reliable
    # motion-consistent tracklets.
    0: {
        'low_conf': 0.30,
        'match_thresh': 0.23,
        'center_gate': 0.66,
        'scale_gate': 0.30,
        'min_quality': 0.58,
        'quality_bonus': 0.010,
        'max_recovery_streak': 1,
        'confirm_window': 2,
        'pending_penalty': 0.12,
        'lost_max_gap': 3,
        'lost_min_score': 0.32,
        'lost_match_thresh': 0.20,
        'lost_min_quality': 0.62,
    },
}


def sca_tdrm_class_param(cls_id, profile, name, default):
    if profile == 'visdrone':
        return SCA_TDRM_VISDRONE_PROFILE.get(int(cls_id), {}).get(name, default)
    if profile in ('visdrone-qc', 'visdrone-v3'):
        return SCA_TDRM_VISDRONE_QC_PROFILE.get(int(cls_id), {}).get(name, default)
    if profile in ('mqtrack-qc', 'visdrone5-mqtrack-qc'):
        return SCA_TDRM_MQTRACK_QC_PROFILE.get(int(cls_id), {}).get(name, default)
    if profile in ('mqtrack-diagnostic', 'visdrone5-diagnostic'):
        return SCA_TDRM_MQTRACK_DIAGNOSTIC_PROFILE.get(int(cls_id), {}).get(name, default)
    if profile == 'uavdt-strict':
        return SCA_TDRM_UAVDT_STRICT_PROFILE.get(int(cls_id), {}).get(name, default)
    return default


def merge_matches(m1, m2, shape):
    O, P, Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1 * M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    """
    :param cost_matrix:
    :param thresh:
    :return:
    """
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), \
               tuple(range(cost_matrix.shape[0])), \
               tuple(range(cost_matrix.shape[1]))

    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)

    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])

    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)

    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float64)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float64),
        np.ascontiguousarray(btlbrs, dtype=np.float64)
    )

    return ious


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
            len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]

    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


# TODO: using GIOU, DIOU, CIOU... to replace IOU

def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    if cost_matrix.size == 0:
        return cost_matrix

    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float32)
    # for i, track in enumerate(tracks):
    # cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float32)

    # 默认计算两个特征向量之间的夹角余弦
    # Nomalized features
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))

    return cost_matrix


def normalize_feature_array(features):
    features = np.asarray(features, dtype=np.float32)
    if features.size == 0:
        return features
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norm, EPS)


def temporal_boost_features(curr_features,
                            prev_id_feature,
                            alpha=0.2,
                            temperature=0.07):
    """STCMOT-lite temporal embedding boost from the previous ID feature map."""
    curr_features = normalize_feature_array(curr_features)
    if curr_features.size == 0 or prev_id_feature is None or alpha <= 0:
        return curr_features

    with torch.no_grad():
        feat_map = prev_id_feature[0]
        _, h, w = feat_map.shape
        query = torch.from_numpy(curr_features).to(feat_map.device).float()
        query = F.normalize(query, dim=1)
        memory = feat_map.view(feat_map.shape[0], -1).transpose(0, 1)
        memory = F.normalize(memory, dim=1)
        attn = torch.softmax(torch.matmul(query, memory.t()) / temperature, dim=1)
        context = torch.matmul(attn, memory).detach().cpu().numpy()

    boosted = (1.0 - alpha) * curr_features + alpha * context
    return normalize_feature_array(boosted)


def map_points_to_orig(points, h_out, w_out, h_orig, w_orig):
    """Map center points from output feature coordinates to original image coordinates."""
    points = np.asarray(points, dtype=np.float32).copy()
    if points.size == 0:
        return points.reshape(-1, 2)

    ratio = min(float(w_out) / float(w_orig), float(h_out) / float(h_orig))
    new_w = max(round(w_orig * ratio), 1)
    new_h = max(round(h_orig * ratio), 1)
    pad_x = (w_out - new_w) * 0.5
    pad_y = (h_out - new_h) * 0.5

    points[:, 0] = (points[:, 0] - pad_x) / float(new_w) * float(w_orig)
    points[:, 1] = (points[:, 1] - pad_y) / float(new_h) * float(h_orig)
    points[:, 0] = np.clip(points[:, 0], 0, max(w_orig - 1, 0))
    points[:, 1] = np.clip(points[:, 1], 0, max(h_orig - 1, 0))
    return points


def _attention_points(query_features,
                      id_feature,
                      reg,
                      h_out,
                      w_out,
                      height,
                      width,
                      temperature=0.07):
    query_features = normalize_feature_array(query_features)
    if query_features.size == 0 or id_feature is None:
        return np.zeros((query_features.shape[0], 2), dtype=np.float32)

    with torch.no_grad():
        feat_map = id_feature[0]
        c, h, w = feat_map.shape
        query = torch.from_numpy(query_features).to(feat_map.device).float()
        query = F.normalize(query, dim=1)
        memory = feat_map.view(c, -1).transpose(0, 1)
        memory = F.normalize(memory, dim=1)
        attn = torch.softmax(torch.matmul(query, memory.t()) / temperature, dim=1)
        max_inds = torch.argmax(attn, dim=1)
        ys = (max_inds // w).float()
        xs = (max_inds % w).float()

        if reg is not None:
            reg_flat = reg[0].view(2, -1).transpose(0, 1)
            offsets = reg_flat[max_inds]
            xs = xs + offsets[:, 0]
            ys = ys + offsets[:, 1]

        points = torch.stack((xs, ys), dim=1).detach().cpu().numpy()

    return map_points_to_orig(points, h_out, w_out, height, width)


def _track_centers(tracks, prefer_observed=True):
    centers = []
    for track in tracks:
        if prefer_observed and hasattr(track, 'curr_tlwh'):
            x, y, w, h = track.curr_tlwh
            centers.append([x + w * 0.5, y + h * 0.5])
        else:
            x1, y1, x2, y2 = track.tlbr
            centers.append([(x1 + x2) * 0.5, (y1 + y2) * 0.5])
    return np.asarray(centers, dtype=np.float32)


def _det_centers(detections):
    centers = []
    for det in detections:
        x1, y1, x2, y2 = det.tlbr
        centers.append([(x1 + x2) * 0.5, (y1 + y2) * 0.5])
    return np.asarray(centers, dtype=np.float32)


def _track_tlwhs(tracks, prefer_observed=False):
    tlwhs = []
    for track in tracks:
        if prefer_observed and hasattr(track, 'curr_tlwh'):
            tlwh = track.curr_tlwh
        else:
            tlwh = track.tlwh
        tlwhs.append(np.asarray(tlwh, dtype=np.float32))
    return np.asarray(tlwhs, dtype=np.float32).reshape(-1, 4)


def _det_tlwhs(detections):
    tlwhs = []
    for det in detections:
        x1, y1, x2, y2 = det.tlbr
        tlwhs.append([x1, y1, x2 - x1, y2 - y1])
    return np.asarray(tlwhs, dtype=np.float32).reshape(-1, 4)


def _bbox_scales(tlwhs):
    tlwhs = np.asarray(tlwhs, dtype=np.float32).reshape(-1, 4)
    areas = np.maximum(tlwhs[:, 2], 1.0) * np.maximum(tlwhs[:, 3], 1.0)
    return np.sqrt(np.maximum(areas, 1.0))


def _pair_scales(track_tlwhs, det_tlwhs, min_scale=16.0):
    track_scales = _bbox_scales(track_tlwhs)
    det_scales = _bbox_scales(det_tlwhs)
    pair_scales = np.sqrt(
        np.maximum(track_scales[:, None], EPS) *
        np.maximum(det_scales[None, :], EPS))
    return np.maximum(pair_scales, float(min_scale))


def _scale_consistency_cost(track_tlwhs, det_tlwhs, sigma=0.6):
    track_wh = np.maximum(np.asarray(track_tlwhs, dtype=np.float32)[:, 2:4], 1.0)
    det_wh = np.maximum(np.asarray(det_tlwhs, dtype=np.float32)[:, 2:4], 1.0)
    log_wh = np.log(det_wh[None, :, :] / track_wh[:, None, :])
    sigma = max(float(sigma), EPS)
    cost = 1.0 - np.exp(-np.sum(log_wh * log_wh, axis=2) / (2.0 * sigma * sigma))
    return np.clip(cost, 0.0, 1.0)


def _scale_aware_app_weight(track_tlwhs,
                            det_tlwhs,
                            min_weight=0.25,
                            max_weight=0.85,
                            scale_mid=52.0,
                            gamma=16.0,
                            min_scale=16.0):
    min_weight = float(np.clip(min_weight, 0.0, 1.0))
    max_weight = float(np.clip(max_weight, min_weight, 1.0))
    pair_scales = _pair_scales(track_tlwhs, det_tlwhs, min_scale=min_scale)
    gamma = max(float(gamma), EPS)
    logits = np.clip((pair_scales - float(scale_mid)) / gamma, -50.0, 50.0)
    weights = 1.0 / (1.0 + np.exp(-logits))
    return min_weight + (max_weight - min_weight) * weights


def amc_distance(tracks,
                 detections,
                 id_feature_history,
                 reg_history,
                 h_out,
                 w_out,
                 height,
                 width,
                 sigma=5.0,
                 temperature=0.07):
    """AMOT-lite appearance-guided bidirectional spatial consistency cost."""
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    if cost_matrix.size == 0:
        return cost_matrix
    if len(id_feature_history) < 2:
        return None

    prev_id_feature = id_feature_history[-2]
    curr_id_feature = id_feature_history[-1]
    prev_reg = reg_history[-2] if len(reg_history) >= 2 else None
    curr_reg = reg_history[-1] if len(reg_history) >= 1 else None

    track_features = np.asarray(
        [getattr(track, 'curr_feat', track.smooth_feat) for track in tracks],
        dtype=np.float32)
    det_features = np.asarray([det.curr_feat for det in detections], dtype=np.float32)

    track_to_curr = _attention_points(track_features, curr_id_feature, curr_reg,
                                      h_out, w_out, height, width, temperature)
    det_to_prev = _attention_points(det_features, prev_id_feature, prev_reg,
                                    h_out, w_out, height, width, temperature)

    track_centers = _track_centers(tracks, prefer_observed=True)
    det_centers = _det_centers(detections)
    forward_dist = np.linalg.norm(det_centers[None, :, :] - track_to_curr[:, None, :], axis=2)
    backward_dist = np.linalg.norm(det_to_prev[None, :, :] - track_centers[:, None, :], axis=2)

    sigma = max(float(sigma), EPS)
    cost_matrix = 1.0 - np.exp(
        -(forward_dist * forward_dist + backward_dist * backward_dist) / (2.0 * sigma * sigma))
    return np.clip(cost_matrix, 0.0, 1.0)


def sa_amc_distance(tracks,
                    detections,
                    id_feature_history,
                    reg_history,
                    h_out,
                    w_out,
                    height,
                    width,
                    norm_sigma=0.75,
                    min_scale=16.0,
                    scale_weight=0.25,
                    scale_sigma=0.6,
                    temperature=0.07):
    """Scale-aware appearance-guided motion consistency cost for UAV targets."""
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    if cost_matrix.size == 0:
        return cost_matrix

    track_tlwhs = _track_tlwhs(tracks, prefer_observed=False)
    det_tlwhs = _det_tlwhs(detections)
    track_centers = _track_centers(tracks, prefer_observed=True)
    det_centers = _det_centers(detections)

    if len(id_feature_history) >= 2:
        prev_id_feature = id_feature_history[-2]
        curr_id_feature = id_feature_history[-1]
        prev_reg = reg_history[-2] if len(reg_history) >= 2 else None
        curr_reg = reg_history[-1] if len(reg_history) >= 1 else None

        track_features = np.asarray(
            [getattr(track, 'curr_feat', track.smooth_feat) for track in tracks],
            dtype=np.float32)
        det_features = np.asarray([det.curr_feat for det in detections], dtype=np.float32)

        track_to_curr = _attention_points(track_features, curr_id_feature, curr_reg,
                                          h_out, w_out, height, width, temperature)
        det_to_prev = _attention_points(det_features, prev_id_feature, prev_reg,
                                        h_out, w_out, height, width, temperature)
    else:
        track_to_curr = _track_centers(tracks, prefer_observed=False)
        det_to_prev = det_centers

    pair_scales = _pair_scales(track_tlwhs, det_tlwhs, min_scale=min_scale)
    forward_dist = np.linalg.norm(det_centers[None, :, :] - track_to_curr[:, None, :], axis=2)
    backward_dist = np.linalg.norm(det_to_prev[None, :, :] - track_centers[:, None, :], axis=2)
    forward_norm = forward_dist / pair_scales
    backward_norm = backward_dist / pair_scales

    norm_sigma = max(float(norm_sigma), EPS)
    center_cost = 1.0 - np.exp(
        -(forward_norm * forward_norm + backward_norm * backward_norm) /
        (2.0 * norm_sigma * norm_sigma))

    scale_cost = _scale_consistency_cost(track_tlwhs, det_tlwhs, sigma=scale_sigma)
    scale_weight = float(np.clip(scale_weight, 0.0, 1.0))
    cost_matrix = (1.0 - scale_weight) * center_cost + scale_weight * scale_cost
    return np.clip(cost_matrix, 0.0, 1.0)


def predict_center_from_feature(track,
                                id_feature_history,
                                reg_history,
                                h_out,
                                w_out,
                                height,
                                width,
                                temperature=0.07):
    if len(id_feature_history) < 1:
        return None
    reg = reg_history[-1] if len(reg_history) >= 1 else None
    feat = np.asarray([getattr(track, 'curr_feat', track.smooth_feat)], dtype=np.float32)
    return _attention_points(feat, id_feature_history[-1], reg,
                             h_out, w_out, height, width, temperature)[0]


def fuse_iou_amc_embedding(iou_cost,
                           amc_cost,
                           embedding_cost,
                           amc_weight=0.5):
    if amc_cost is None:
        spatial_cost = iou_cost
    else:
        amc_weight = float(np.clip(amc_weight, 0.0, 1.0))
        spatial_cost = (1.0 - amc_weight) * iou_cost + amc_weight * amc_cost

    spatial_sim = 1.0 - np.clip(spatial_cost, 0.0, 1.0)
    embedding_sim = 1.0 - np.clip(embedding_cost, 0.0, 1.0)
    return 1.0 - spatial_sim * embedding_sim


def fuse_sa_amc_embedding(iou_cost,
                          sa_amc_cost,
                          embedding_cost,
                          tracks,
                          detections,
                          amc_weight=0.5,
                          app_min_weight=0.45,
                          app_max_weight=1.0,
                          app_scale_mid=52.0,
                          app_scale_gamma=16.0,
                          min_scale=16.0):
    if sa_amc_cost is None:
        spatial_cost = iou_cost
    else:
        amc_weight = float(np.clip(amc_weight, 0.0, 1.0))
        spatial_cost = (1.0 - amc_weight) * iou_cost + amc_weight * sa_amc_cost

    track_tlwhs = _track_tlwhs(tracks, prefer_observed=False)
    det_tlwhs = _det_tlwhs(detections)
    app_weight = _scale_aware_app_weight(
        track_tlwhs,
        det_tlwhs,
        min_weight=app_min_weight,
        max_weight=app_max_weight,
        scale_mid=app_scale_mid,
        gamma=app_scale_gamma,
        min_scale=min_scale)

    spatial_sim = 1.0 - np.clip(spatial_cost, 0.0, 1.0)
    embedding_sim = 1.0 - np.clip(embedding_cost, 0.0, 1.0)
    appearance_sim = np.power(np.clip(embedding_sim, EPS, 1.0), app_weight)
    joint_sim = spatial_sim * appearance_sim
    return 1.0 - np.clip(joint_sim, 0.0, 1.0)


def _tracklet_motion_quality(track):
    if not hasattr(track, 'tlwh_deque') or len(track.tlwh_deque) < 3:
        return 0.6

    frames = np.asarray([item[0] for item in track.tlwh_deque], dtype=np.float32)
    tlwhs = np.asarray([item[1] for item in track.tlwh_deque], dtype=np.float32)
    centers = np.stack((tlwhs[:, 0] + tlwhs[:, 2] * 0.5,
                        tlwhs[:, 1] + tlwhs[:, 3] * 0.5), axis=1)
    scales = _bbox_scales(tlwhs)
    dt = np.diff(frames)
    valid = dt > 0
    if np.sum(valid) < 2:
        return 0.5

    velocities = np.diff(centers, axis=0)[valid] / dt[valid, None]
    scale_norm = float(np.median(scales) + EPS)
    speeds = np.linalg.norm(velocities, axis=1) / scale_norm
    if speeds.size < 2:
        return 0.6

    accel = np.diff(velocities, axis=0)
    speed_cv = float(np.std(speeds) / (np.mean(speeds) + 0.05))
    accel_norm = float(np.mean(np.linalg.norm(accel, axis=1) / scale_norm))
    return float(np.clip(np.exp(-0.7 * speed_cv - 0.5 * accel_norm), 0.0, 1.0))


def _tracklet_scale_quality(track):
    if not hasattr(track, 'tlwh_deque') or len(track.tlwh_deque) < 2:
        return 0.6
    tlwhs = np.asarray([item[1] for item in track.tlwh_deque], dtype=np.float32)
    areas = np.maximum(tlwhs[:, 2], 1.0) * np.maximum(tlwhs[:, 3], 1.0)
    return float(np.clip(np.exp(-float(np.std(np.log(np.maximum(areas, 1.0))))), 0.0, 1.0))


def track_recovery_quality(track):
    history_len = len(track.tlwh_deque) if hasattr(track, 'tlwh_deque') else 1
    length_quality = np.log1p(max(history_len, 1)) / np.log1p(20.0)
    score_quality = float(np.clip(getattr(track, 'score', 0.5), 0.0, 1.0))
    motion_quality = _tracklet_motion_quality(track)
    scale_quality = _tracklet_scale_quality(track)
    quality = (
        0.30 * length_quality +
        0.25 * score_quality +
        0.25 * motion_quality +
        0.20 * scale_quality)
    return float(np.clip(quality, 0.0, 1.0))


def sca_tdrm_distance(tracks,
                      detections,
                      embedding_cost,
                      iou_cost,
                      cls_id,
                      current_frame=None,
                      profile='visdrone',
                      min_scale=16.0,
                      center_gate=1.30,
                      scale_gate=0.70,
                      min_quality=0.25,
                      iou_weight=0.30,
                      emb_weight=0.30,
                      center_weight=0.25,
                      scale_weight=0.15,
                      quality_bonus=0.08,
                      max_recovery_streak=0,
                      confirm_window=6,
                      pending_penalty=0.0):
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    if cost_matrix.size == 0:
        return cost_matrix

    center_gate = float(sca_tdrm_class_param(cls_id, profile, 'center_gate', center_gate))
    scale_gate = float(sca_tdrm_class_param(cls_id, profile, 'scale_gate', scale_gate))
    min_quality = float(sca_tdrm_class_param(cls_id, profile, 'min_quality', min_quality))
    quality_bonus = float(sca_tdrm_class_param(cls_id, profile, 'quality_bonus', quality_bonus))
    max_recovery_streak = int(round(sca_tdrm_class_param(
        cls_id, profile, 'max_recovery_streak', max_recovery_streak)))
    confirm_window = int(round(sca_tdrm_class_param(
        cls_id, profile, 'confirm_window', confirm_window)))
    pending_penalty = float(sca_tdrm_class_param(
        cls_id, profile, 'pending_penalty', pending_penalty))

    track_tlwhs = _track_tlwhs(tracks, prefer_observed=False)
    det_tlwhs = _det_tlwhs(detections)
    track_centers = _track_centers(tracks, prefer_observed=False)
    det_centers = _det_centers(detections)
    pair_scales = _pair_scales(track_tlwhs, det_tlwhs, min_scale=min_scale)

    center_dist = np.linalg.norm(
        det_centers[None, :, :] - track_centers[:, None, :], axis=2) / pair_scales

    track_area = np.maximum(track_tlwhs[:, 2], 1.0) * np.maximum(track_tlwhs[:, 3], 1.0)
    det_area = np.maximum(det_tlwhs[:, 2], 1.0) * np.maximum(det_tlwhs[:, 3], 1.0)
    scale_dist = np.abs(np.log(det_area[None, :] / np.maximum(track_area[:, None], 1.0)))

    track_quality = np.asarray([track_recovery_quality(track) for track in tracks], dtype=np.float32)
    recovery_streaks = np.zeros((len(tracks),), dtype=np.float32)
    pending_ages = np.zeros((len(tracks),), dtype=np.float32)
    if current_frame is not None:
        for idx, track in enumerate(tracks):
            last_recovery = getattr(track, 'sca_tdrm_last_recovery_frame', -1)
            if last_recovery == current_frame - 1:
                recovery_streaks[idx] = float(getattr(track, 'sca_tdrm_recovery_streak', 0))

            if getattr(track, 'sca_tdrm_pending', False):
                pending_start = getattr(track, 'sca_tdrm_pending_start_frame', -1)
                if pending_start >= 0:
                    pending_ages[idx] = max(float(current_frame - pending_start + 1), 0.0)

    valid = np.ones(cost_matrix.shape, dtype=bool)
    valid &= center_dist <= max(center_gate, EPS)
    valid &= scale_dist <= max(scale_gate, EPS)
    valid &= track_quality[:, None] >= min_quality
    if max_recovery_streak > 0:
        valid &= recovery_streaks[:, None] < max(float(max_recovery_streak), 1.0)

    center_cost = np.clip(center_dist / max(center_gate, EPS), 0.0, 1.0)
    scale_cost = np.clip(scale_dist / max(scale_gate, EPS), 0.0, 1.0)

    weights = np.asarray([iou_weight, emb_weight, center_weight, scale_weight], dtype=np.float32)
    weights = weights / max(float(np.sum(weights)), EPS)
    cost_matrix = (
        weights[0] * np.clip(iou_cost, 0.0, 1.0) +
        weights[1] * np.clip(embedding_cost, 0.0, 1.0) +
        weights[2] * center_cost +
        weights[3] * scale_cost -
        float(quality_bonus) * track_quality[:, None])
    if current_frame is not None:
        age_norm = pending_ages[:, None] / max(float(confirm_window), 1.0)
        streak_norm = recovery_streaks[:, None] / max(float(max_recovery_streak), 1.0)
        cost_matrix += float(pending_penalty) * np.clip(age_norm + streak_norm, 0.0, 2.0)
    cost_matrix = np.clip(cost_matrix, 0.0, 1.0)
    cost_matrix[~valid] = 1e5
    return cost_matrix.astype(np.float32)


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    """
    :param kf:
    :param cost_matrix:
    :param tracks:
    :param detections:
    :param only_position:
    :return:
    """
    if cost_matrix.size == 0:
        return cost_matrix

    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])

    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf

    return cost_matrix


def fuse_motion(kf,
                cost_matrix,
                tracks,
                detections,
                only_position=False,
                lambda_=0.98):
    """
    :param kf:
    :param cost_matrix:
    :param tracks:
    :param detections:
    :param only_position:
    :param lambda_:
    :return:
    """
    if cost_matrix.size == 0:
        return cost_matrix

    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])

    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(track.mean,
                                             track.covariance,
                                             measurements,
                                             only_position,
                                             metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance

    return cost_matrix
