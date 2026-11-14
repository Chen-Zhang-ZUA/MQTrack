import csv
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

from lib.models import *
from lib.models.decode import mot_decode
from lib.models.model import create_model, load_model
from lib.models.utils import _tranpose_and_gather_feat
from lib.tracker import matching
from lib.tracker.birth_competition import ambiguous_birth, mature_competitor
from lib.tracker.uav_gmc import UAVGMC, apply_gmc_to_tracks
from lib.tracking_utils.kalman_filter import KalmanFilter
from lib.tracking_utils.log import logger
from lib.tracking_utils.utils import *
from lib.utils.post_process import ctdet_post_process
from .basetrack import BaseTrack, MCBaseTrack, TrackState
from gen_dataset_visdrone import cls2id, id2cls  # visdrone
# from gen_labels_detrac_mcmot import cls2id, id2cls  # mcmot_c5


CLASS_CONF_PROFILES = {
    'none': {},
    # 1-based VisDrone class ids.  Truck benefits from a lower online
    # activation threshold; the other classes keep opt.conf_thres.
    'visdrone5-truck-recall': {
        6: 0.35,
    },
    # Test-dev diagnostic profile: keep truck recall, but make car online
    # activation slightly stricter because car contributed many FPs/ID switches.
    'mqtrack-diagnostic': {
        4: 0.42,
        6: 0.35,
    },
}


SAGE_PROFILES = {
    'none': {},
    # Quality-control profile for the five AMOT-compatible VisDrone classes.
    # It tightens pedestrian/van continuation, which were the main FP sources
    # in the SAGE-only ablation, while keeping car/truck/bus less constrained.
    'visdrone5-qc': {
        0: {
            'match_thresh': 0.62,
            'mtc_center_thresh': 10.0,
            'mtc_min_quality': 0.46,
            'mtc_min_score': 0.42,
            'mtc_min_history': 4,
        },
        3: {
            'match_thresh': 0.68,
            'mtc_center_thresh': 13.0,
            'mtc_min_quality': 0.40,
            'mtc_min_score': 0.36,
            'mtc_min_history': 4,
        },
        4: {
            'match_thresh': 0.56,
            'mtc_center_thresh': 8.0,
            'mtc_min_quality': 0.56,
            'mtc_min_score': 0.50,
            'mtc_min_history': 5,
        },
        5: {
            'match_thresh': 0.64,
            'mtc_center_thresh': 12.0,
            'mtc_min_quality': 0.43,
            'mtc_min_score': 0.38,
            'mtc_min_history': 4,
        },
        8: {
            'match_thresh': 0.62,
            'mtc_center_thresh': 10.0,
            'mtc_min_quality': 0.48,
            'mtc_min_score': 0.44,
            'mtc_min_history': 4,
        },
    },
    # UAVDT is a one-class vehicle-tracking setting.  The baseline already has
    # strong recall, so this profile keeps SAGE conservative: it allows
    # scale-aware association, but only lets MTC continue stable, high-score
    # tracklets.
    'uavdt-strict': {
        0: {
            'match_thresh': 0.64,
            'mtc_center_thresh': 7.0,
            'mtc_min_quality': 0.62,
            'mtc_min_score': 0.55,
            'mtc_min_history': 8,
        },
    },
}


def load_class_conf_threshold_file(path):
    if not path:
        return {}
    threshold_path = Path(path)
    if not threshold_path.exists():
        raise FileNotFoundError('Class confidence threshold file not found: {}'.format(path))

    if threshold_path.suffix.lower() == '.json':
        with threshold_path.open('r', encoding='utf-8') as handle:
            data = json.load(handle)
        if isinstance(data, dict) and 'thresholds' in data:
            data = data['thresholds']
        if isinstance(data, dict):
            return {int(k): float(v) for k, v in data.items()}
        thresholds = {}
        for item in data:
            cls_id = int(item.get('class_id', item.get('class', item.get('cls_id', -1))))
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
                threshold_value = item.get('threshold') or item.get('conf_thres') or item.get('score_thresh')
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


# TODO: Multi-class Track class
class MCTrack(MCBaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, temp_feat, num_classes, cls_id, buff_size=30):
        """
        :param tlwh:
        :param score:
        :param temp_feat:
        :param num_classes:
        :param cls_id:
        :param buff_size:
        """
        # object class id
        self.cls_id = cls_id

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.track_len = 0

        self.smooth_feat = None
        self.features = deque([], maxlen=buff_size)
        self.alpha = 0.9
        self.curr_tlwh = self._tlwh.copy()
        self.tlwh_deque = deque([], maxlen=buff_size)
        self.sca_tdrm_recovery_streak = 0
        self.sca_tdrm_pending = False
        self.sca_tdrm_pending_start_frame = -1
        self.sca_tdrm_last_recovery_frame = -1
        self.sca_tdrm_last_high_frame = -1
        self.last_update_source = 'high'
        self.birth_required_hits = 2
        self.update_features(temp_feat)

    def mark_high_conf_update(self, frame_id):
        self.sca_tdrm_recovery_streak = 0
        self.sca_tdrm_pending = False
        self.sca_tdrm_pending_start_frame = -1
        self.sca_tdrm_last_high_frame = frame_id
        self.last_update_source = 'high'

    def mark_low_recovery_update(self, frame_id):
        if self.sca_tdrm_last_recovery_frame == frame_id - 1:
            self.sca_tdrm_recovery_streak += 1
        else:
            self.sca_tdrm_recovery_streak = 1
        if not self.sca_tdrm_pending:
            self.sca_tdrm_pending_start_frame = frame_id
        self.sca_tdrm_pending = True
        self.sca_tdrm_last_recovery_frame = frame_id
        self.last_update_source = 'recovery'

    def update_features(self, feat):
        # L2 normalizing
        feat = np.asarray(feat, dtype=np.float32)
        feat /= max(np.linalg.norm(feat), 1e-12)

        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1.0 - self.alpha) * feat

        self.features.append(feat)

        # L2 normalizing
        self.smooth_feat /= max(np.linalg.norm(self.smooth_feat), 1e-12)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(tracks):
        if len(tracks) > 0:
            multi_mean = np.asarray([track.mean.copy() for track in tracks])
            multi_covariance = np.asarray([track.covariance for track in tracks])

            for i, st in enumerate(tracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0

            multi_mean, multi_covariance = MCTrack.shared_kalman.multi_predict(multi_mean, multi_covariance)

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                tracks[i].mean = mean
                tracks[i].covariance = cov

    def reset_track_id(self):
        self.reset_track_count(self.cls_id)

    def activate(self, kalman_filter, frame_id):
        """Start a new track"""
        self.kalman_filter = kalman_filter  # assign a filter to each track?

        # update track id for the object class
        self.track_id = self.next_id(self.cls_id)

        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        self.curr_tlwh = self._tlwh.copy()

        self.track_len = 0
        self.state = TrackState.Tracked  # set flag 'tracked'
        self.tlwh_deque.append((frame_id, self.curr_tlwh.copy()))

        # self.is_activated = True
        if frame_id == 1:  # to record the first frame's detection result
            self.is_activated = True

        self.frame_id = frame_id
        self.start_frame = frame_id
        self.mark_high_conf_update(frame_id)

    def re_activate(self, new_track, frame_id, new_id=False):
        # kalman update
        self.mean, self.covariance = self.kalman_filter.update(self.mean,
                                                               self.covariance,
                                                               self.tlwh_to_xyah(new_track.tlwh))

        # feature vector update
        self.update_features(new_track.curr_feat)
        self.curr_tlwh = new_track.tlwh.copy()
        self.tlwh_deque.append((frame_id, self.curr_tlwh.copy()))

        self.track_len = 0
        self.frame_id = frame_id

        self.state = TrackState.Tracked  # set flag 'tracked'
        self.is_activated = True
        self.mark_high_conf_update(frame_id)

        if new_id:  # update track id for the object class
            self.track_id = self.next_id(self.cls_id)

    def update(self, new_track, frame_id, update_feature=True, recovery=False):
        """
        Update a matched track
        :type new_track: Track
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.track_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(self.mean,
                                                               self.covariance,
                                                               self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked  # set flag 'tracked'
        self.is_activated = True  # set flag 'activated'

        self.score = new_track.score
        self.curr_tlwh = new_tlwh.copy()
        self.tlwh_deque.append((frame_id, self.curr_tlwh.copy()))
        if update_feature:
            self.update_features(new_track.curr_feat)
        if recovery:
            self.mark_low_recovery_update(frame_id)
        else:
            self.mark_high_conf_update(frame_id)

    def update_retrack(self, curr_tlwh, frame_id):
        self.frame_id = frame_id
        self.track_len += 1
        self.state = TrackState.Tracked
        self.is_activated = True
        self.curr_tlwh = np.asarray(curr_tlwh, dtype=np.float32)
        self.tlwh_deque.append((frame_id, self.curr_tlwh.copy()))
        self.last_update_source = 'mtc'

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()

        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()  # numpy中的.copy()是深拷贝
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_center(tlwh):
        x, y, w, h = tlwh
        return np.asarray([x + w * 0.5, y + h * 0.5], dtype=np.float32)

    def __repr__(self):
        return 'OT_({}-{})_({}-{})'.format(self.cls_id, self.track_id, self.start_frame, self.end_frame)


class Track(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, temp_feat, buff_size=30):
        """
        :param tlwh:
        :param score:
        :param temp_feat:
        :param buff_size:
        """

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.features = deque([], maxlen=buff_size)
        self.alpha = 0.9
        self.curr_tlwh = self._tlwh.copy()
        self.tlwh_deque = deque([], maxlen=buff_size)
        self.update_features(temp_feat)

    def update_features(self, feat):
        feat = np.asarray(feat, dtype=np.float32)
        feat /= max(np.linalg.norm(feat), 1e-12)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * \
                               self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= max(np.linalg.norm(self.smooth_feat), 1e-12)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(
            mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0

            multi_mean, multi_covariance = Track.shared_kalman.multi_predict(multi_mean, multi_covariance)

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def reset_track_id(self):
        self.reset_track_count()

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        self.curr_tlwh = self._tlwh.copy()
        self.tlwh_deque.append((frame_id, self.curr_tlwh.copy()))

        self.tracklet_len = 0
        self.state = TrackState.Tracked

        # to return the first frame's detection result
        if frame_id == 1:  # activate the first frame's detection result
            self.is_activated = True
        # self.is_activated = True

        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(self.mean,
                                                               self.covariance,
                                                               self.tlwh_to_xyah(new_track.tlwh))

        self.update_features(new_track.curr_feat)
        self.curr_tlwh = new_track.tlwh.copy()
        self.tlwh_deque.append((frame_id, self.curr_tlwh.copy()))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id

        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track, frame_id, update_feature=True):
        """
        Update a matched track
        :type new_track: Track
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(self.mean,
                                                               self.covariance,
                                                               self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.curr_tlwh = new_tlwh.copy()
        self.tlwh_deque.append((frame_id, self.curr_tlwh.copy()))
        if update_feature:
            self.update_features(new_track.curr_feat)

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()

        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()  # numpy中的.copy()是深拷贝
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


# rewrite a post processing(without using affine matrix)
def map2orig(dets, h_out, w_out, h_orig, w_orig, num_classes):
    """
    :param dets:
    :param h_out:
    :param w_out:
    :param h_orig:
    :param w_orig:
    :param num_classes:
    :return: dict of detections(key: cls_id)
    """

    def get_padding():
        """
        :return: pad_1, pad_2, pad_type('pad_x' or 'pad_y'), new_shape(w, h)
        """
        ratio_x = float(w_out) / w_orig
        ratio_y = float(h_out) / h_orig
        ratio = min(ratio_x, ratio_y)
        new_shape = (round(w_orig * ratio), round(h_orig * ratio))  # new_w, new_h

        pad_x = (w_out - new_shape[0]) * 0.5  # width padding
        pad_y = (h_out - new_shape[1]) * 0.5  # height padding
        top, bottom = round(pad_y - 0.1), round(pad_y + 0.1)
        left, right = round(pad_x - 0.1), round(pad_x + 0.1)
        if ratio == ratio_x:  # pad_y
            return top, bottom, 'pad_y', new_shape
        else:  # pad_x
            return left, right, 'pad_x', new_shape

    pad_1, pad_2, pad_type, new_shape = get_padding()

    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])  # default: 1×128×6
    dets = dets[0]  # 128×6

    dets_dict = {}

    if pad_type == 'pad_x':
        dets[:, 0] = (dets[:, 0] - pad_1) / new_shape[0] * w_orig  # x1
        dets[:, 2] = (dets[:, 2] - pad_1) / new_shape[0] * w_orig  # x2
        dets[:, 1] = dets[:, 1] / h_out * h_orig  # y1
        dets[:, 3] = dets[:, 3] / h_out * h_orig  # y2
    else:  # 'pad_y'
        dets[:, 0] = dets[:, 0] / w_out * w_orig  # x1
        dets[:, 2] = dets[:, 2] / w_out * w_orig  # x2
        dets[:, 1] = (dets[:, 1] - pad_1) / new_shape[1] * h_orig  # y1
        dets[:, 3] = (dets[:, 3] - pad_1) / new_shape[1] * h_orig  # y2

    classes = dets[:, -1]
    for cls_id in range(num_classes):
        inds = (classes == cls_id)
        dets_dict[cls_id] = dets[inds, :]

    return dets_dict


class MCJDETracker(object):
    def __init__(self, opt, frame_rate=30):
        self.opt = opt
        self.class_conf_thresholds = dict(
            CLASS_CONF_PROFILES.get(getattr(opt, 'class_conf_profile', 'none'), {}))
        class_conf_file = getattr(opt, 'class_conf_thresh_file', None)
        if class_conf_file:
            self.class_conf_thresholds.update(load_class_conf_threshold_file(class_conf_file))
        if self.class_conf_thresholds:
            logger.info('class-adaptive conf thresholds: {}'.format(self.class_conf_thresholds))

        # ----- init model
        print('Creating model...')
        self.model = create_model(opt.arch, opt.heads, opt.head_conv)
        self.model = load_model(self.model, opt.load_model)  # load specified checkpoint
        self.model = self.model.to(opt.device)
        self.model.eval()

        # ----- track_lets
        self.tracked_tracks_dict = defaultdict(list)  # value type: list[STrack]
        self.lost_tracks_dict = defaultdict(list)  # value type: list[STrack]
        self.removed_tracks_dict = defaultdict(list)  # value type: list[STrack]

        self.frame_id = 0
        self.det_thresh = opt.conf_thres
        self.buffer_size = int(frame_rate / 30.0 * opt.track_buffer)  # int(frame_rate / 30.0 * opt.track_buffer)
        self.max_time_lost = self.buffer_size
        self.max_per_image = self.opt.K  # max objects per image
        self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)

        # ----- using kalman filter to stabilize tracking
        self.kalman_filter = KalmanFilter()
        self.id_feature_history = deque([], maxlen=2)
        self.reg_history = deque([], maxlen=2)
        self.gmc = UAVGMC(opt)
        self.birth_competition_enabled = bool(getattr(opt, 'birth_competition', False))
        self.birth_confirm_hits = int(getattr(opt, 'birth_confirm_hits', 2))
        if self.birth_confirm_hits < 2:
            raise ValueError('birth_confirm_hits must be at least 2')
        self.birth_competition_stats = {'ambiguous': 0, 'suppressed': 0}
        self.tiled_inference = None
        if getattr(opt, 'tile_inference', False):
            from lib.tracker.tiled_inference import TiledInference
            if getattr(opt, 'tile_supported', False):
                from lib.tracker.supported_tiled_inference import SupportedTiledInference
                self.tiled_inference = SupportedTiledInference(opt)
            else:
                self.tiled_inference = TiledInference(opt)
            logger.info('Tiled inference enabled: fraction={}, overlap={}, embedding={}'.format(
                opt.tile_fraction, opt.tile_overlap, opt.tile_embedding))
            if getattr(opt, 'tile_supported', False):
                logger.info('Supported-only tiling: tile_score={}, full_score={}, class_ratio={}'.format(
                    opt.tile_support_tile_score, opt.tile_support_full_score, opt.tile_support_class_ratio))
        if self.gmc.enabled:
            logger.info('UAV-GMC enabled: mode={}'.format(self.gmc.mode))

    def class_conf_thresh(self, cls_id):
        return float(self.class_conf_thresholds.get(int(cls_id) + 1, self.opt.conf_thres))

    def _sage_param(self, cls_id, name, default):
        profile = SAGE_PROFILES.get(getattr(self.opt, 'sage_profile', 'none'), {})
        return profile.get(int(cls_id), {}).get(name, default)

    def _cast_param(self, cls_id, name, default):
        profile = getattr(self.opt, 'sca_tdrm_profile', 'none')
        if not getattr(self.opt, 'sca_tdrm', False):
            profile = 'none'
        return matching.sca_tdrm_class_param(cls_id, profile, name, default)

    def _lost_recovery_tracks(self, tracks, cls_id=None):
        if not getattr(self.opt, 'lost_track_recovery', False):
            return []
        max_gap = int(getattr(self.opt, 'lost_track_recovery_max_gap', 8))
        min_quality = float(getattr(self.opt, 'lost_track_recovery_min_quality', 0.40))
        if cls_id is not None:
            max_gap = int(self._cast_param(cls_id, 'lost_max_gap', max_gap))
            min_quality = float(self._cast_param(cls_id, 'lost_min_quality', min_quality))
        candidates = []
        for track in tracks:
            gap = self.frame_id - track.end_frame
            if gap < 1 or gap > max_gap:
                continue
            if matching.track_recovery_quality(track) < min_quality:
                continue
            candidates.append(track)
        return candidates

    def reset(self):
        """
        :return:
        """
        # Reset tracks dict
        self.tracked_tracks_dict = defaultdict(list)  # value type: list[Track]
        self.lost_tracks_dict = defaultdict(list)  # value type: list[Track]
        self.removed_tracks_dict = defaultdict(list)  # value type: list[Track]

        # Reset frame id
        self.frame_id = 0

        # Reset kalman filter to stabilize tracking
        self.kalman_filter = KalmanFilter()
        self.id_feature_history = deque([], maxlen=2)
        self.reg_history = deque([], maxlen=2)
        self.gmc.reset()

    def _try_mtc_continue(self, track, detections, h_out, w_out, height, width):
        if not (self.opt.amot_lite or self.opt.amot_mtc):
            return False
        if len(self.id_feature_history) < 1:
            return False
        cls_id = int(getattr(track, 'cls_id', -1))
        min_history = int(self._sage_param(
            cls_id, 'mtc_min_history', self.opt.amot_mtc_min_history))
        if len(track.tlwh_deque) < min_history:
            return False
        if self.frame_id - track.end_frame != 1:
            return False
        min_quality = float(self._sage_param(
            cls_id, 'mtc_min_quality', getattr(self.opt, 'sage_mtc_min_quality', 0.0)))
        if min_quality > 0.0 and matching.track_recovery_quality(track) < min_quality:
            return False
        min_score = float(self._sage_param(
            cls_id, 'mtc_min_score', getattr(self.opt, 'sage_mtc_min_score', 0.0)))
        if min_score > 0.0 and float(getattr(track, 'score', 0.0)) < min_score:
            return False

        attention_center = matching.predict_center_from_feature(
            track,
            self.id_feature_history,
            self.reg_history,
            h_out,
            w_out,
            height,
            width)
        if attention_center is None:
            return False

        pred_center = MCTrack.tlwh_to_center(track.tlwh)
        center_thresh = float(self._sage_param(
            cls_id, 'mtc_center_thresh', self.opt.amot_mtc_center_thresh))
        if np.linalg.norm(pred_center - attention_center) > center_thresh:
            return False

        if len(detections) > 0:
            max_iou = 1.0 - np.min(matching.iou_distance([track], detections))
            if max_iou > 0.3:
                return False

        track.update_retrack(track.tlwh, self.frame_id)
        return True

    def post_process(self, dets, meta):
        """
        2D bbox检测结果后处理
        :param dets:
        :param meta:
        :return:
        """
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])  # default: 1×128×6

        # affine transform
        dets = ctdet_post_process(dets.copy(),
                                  [meta['c']], [meta['s']],
                                  meta['out_height'],
                                  meta['out_width'],
                                  self.opt.num_classes)

        # detection dict(cls_id as key)
        dets = dets[0]  # fetch the first image dets results(batch_size = 1 by default)

        return dets

    def merge_outputs(self, detections):
        """
        :param detections:
        :return:
        """
        results = {}
        for j in range(1, self.opt.num_classes + 1):
            results[j] = np.concatenate([detection[j] for detection in detections],
                                        axis=0).astype(np.float32)

        scores = np.hstack([results[j][:, 4] for j in range(1, self.opt.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.opt.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]

        return results

    def update_detection(self, im_blob, img_0):
        """
        更新视频序列或图片序列的检测结果
        :rtype: dict
        :param im_blob:
        :param img_0:
        :return:
        """
        height, width = img_0.shape[0], img_0.shape[1]  # H, W of original input image
        net_height, net_width = im_blob.shape[2], im_blob.shape[3]  # H, W of net input

        c = np.array([width * 0.5, height * 0.5], dtype=np.float32)  # image center
        s = max(float(net_width) / float(net_height) * height, width) * 1.0

        h_out = net_height // self.opt.down_ratio
        w_out = net_width // self.opt.down_ratio

        # ----- get detections
        with torch.no_grad():
            dets_dict = defaultdict(list)

            # --- network output
            output = self.model.forward(im_blob)[-1]

            # --- detection outputs
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            reg = output['reg'] if self.opt.reg_offset else None

            # --- decode results of detection
            dets, inds, cls_inds_mask = mot_decode(heatmap=hm,
                                                   wh=wh,
                                                   reg=reg,
                                                   num_classes=self.opt.num_classes,
                                                   cat_spec_wh=self.opt.cat_spec_wh,
                                                   K=self.opt.K)

            # --- map to original image coordinate system
            dets = map2orig(dets, h_out, w_out, height, width, self.opt.num_classes)  # translate and scale
            if self.tiled_inference is not None:
                id_feature = F.normalize(output['id'], dim=1)
                features = _tranpose_and_gather_feat(id_feature, inds)[0].cpu().numpy()
                class_features = [features[cls_inds_mask[c].cpu().numpy()]
                                  for c in range(self.opt.num_classes)]
                dets, _ = self.tiled_inference.fuse(
                    self.model, im_blob, img_0, dets, class_features, id_feature,
                    full_heatmap=hm,
                    class_thresholds=[self.class_conf_thresh(c) for c in range(self.opt.num_classes)])

            # --- parse detections of each class
            for cls_id in range(self.opt.num_classes):  # cls_id start from index 0
                cls_dets = dets[cls_id]

                # filter out low conf score dets
                remain_inds = cls_dets[:, 4] > self.class_conf_thresh(cls_id)
                cls_dets = cls_dets[remain_inds]
                dets_dict[cls_id] = cls_dets

        return dets_dict

    def update_tracking(self, im_blob, img_0):
        """
        :param im_blob:
        :param img_0:
        :return:
        """
        # update frame id
        self.frame_id += 1

        # ----- reset the track ids for all object classes in the first frame
        if self.frame_id == 1:
            MCTrack.init_count(self.opt.num_classes)
        # -----

        # record tracking results, key: class_id
        activated_tracks_dict = defaultdict(list)
        refined_tracks_dict = defaultdict(list)
        lost_tracks_dict = defaultdict(list)
        removed_tracks_dict = defaultdict(list)
        output_tracks_dict = defaultdict(list)

        height, width = img_0.shape[0], img_0.shape[1]  # H, W of original input image
        net_height, net_width = im_blob.shape[2], im_blob.shape[3]  # H, W of net input

        c = np.array([width * 0.5, height * 0.5], dtype=np.float32)
        s = max(float(net_width) / float(net_height) * height, width) * 1.0
        h_out = net_height // self.opt.down_ratio
        w_out = net_width // self.opt.down_ratio

        ''' Step 1: Network forward, get detections & embeddings'''
        with torch.no_grad():
            output = self.model.forward(im_blob)[-1]

            hm = output['hm'].sigmoid_()
            wh = output['wh']
            reg = output['reg'] if self.opt.reg_offset else None
            id_feature = output['id']

            # L2 normalize the reid feature vector
            id_feature = F.normalize(id_feature, dim=1)
            self.id_feature_history.append(id_feature.detach())
            self.reg_history.append(reg.detach() if reg is not None else None)

            #  detection decoding
            dets, inds, cls_inds_mask = mot_decode(heatmap=hm,
                                                   wh=wh,
                                                   reg=reg,
                                                   num_classes=self.opt.num_classes,
                                                   cat_spec_wh=self.opt.cat_spec_wh,
                                                   K=self.opt.K)

            # ----- get ReID feature vector by object class
            cls_id_feats = []  # topK feature vectors of each object class
            for cls_id in range(self.opt.num_classes):  # cls_id starts from 0
                # get inds of each object class
                cls_inds = inds[:, cls_inds_mask[cls_id]]

                # gather feats for each object class
                cls_id_feature = _tranpose_and_gather_feat(id_feature, cls_inds)  # inds: 1×128
                cls_id_feature = cls_id_feature.squeeze(0)  # n × FeatDim
                cls_id_feature = cls_id_feature.cpu().numpy()
                cls_id_feats.append(cls_id_feature)

        # translate and scale
        dets = map2orig(dets, h_out, w_out, height, width, self.opt.num_classes)
        if self.tiled_inference is not None:
            dets, cls_id_feats = self.tiled_inference.fuse(
                self.model, im_blob, img_0, dets, cls_id_feats, id_feature,
                full_heatmap=hm,
                class_thresholds=[self.class_conf_thresh(c) for c in range(self.opt.num_classes)])
            if self.frame_id == 1 or self.frame_id % 300 == 0:
                logger.info('Tile fusion frame {}: {}'.format(
                    self.frame_id, self.tiled_inference.last_stats))
        gmc_warp = self.gmc.estimate(img_0)
        class_thresholds = {
            cls_id: self.class_conf_thresh(cls_id)
            for cls_id in range(self.opt.num_classes)
        }

        # ----- parse each object class
        for cls_id in range(self.opt.num_classes):  # cls_id从0开始
            cls_dets = dets[cls_id]

            cls_all_dets = cls_dets
            cls_all_features = cls_id_feats[cls_id]
            if self.opt.stcmot_tebm and len(self.id_feature_history) >= 2:
                cls_all_features = matching.temporal_boost_features(
                    cls_all_features,
                    self.id_feature_history[-2],
                    alpha=self.opt.stcmot_tebm_alpha)

            # High-score detections drive normal online tracking.
            class_conf_thres = self.class_conf_thresh(cls_id)
            remain_inds = cls_all_dets[:, 4] > class_conf_thres
            cls_dets = cls_all_dets[remain_inds]
            cls_id_feature = cls_all_features[remain_inds]

            # TDRM recovers low-score candidates only for unmatched tracks.
            if self.opt.stcmot_tdrm or self.opt.sca_tdrm:
                if self.opt.sca_tdrm:
                    low_conf = matching.sca_tdrm_class_param(
                        cls_id,
                        self.opt.sca_tdrm_profile,
                        'low_conf',
                        self.opt.sca_tdrm_low_conf)
                else:
                    low_conf = self.opt.stcmot_tdrm_low_conf
                low_inds = np.logical_and(
                    cls_all_dets[:, 4] > low_conf,
                    cls_all_dets[:, 4] <= class_conf_thres)
                cls_dets_low = cls_all_dets[low_inds]
                cls_id_feature_low = cls_all_features[low_inds]
            else:
                feat_dim = cls_all_features.shape[1] if cls_all_features.ndim == 2 else self.opt.reid_dim
                cls_dets_low = np.empty((0, cls_all_dets.shape[1]), dtype=cls_all_dets.dtype)
                cls_id_feature_low = np.empty((0, feat_dim), dtype=cls_all_features.dtype)

            if len(cls_dets) > 0:
                '''Detections, tlbrs: top left bottom right score'''
                cls_detects = [
                    MCTrack(MCTrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], feat, self.opt.num_classes, cls_id, 30)
                    for (tlbrs, feat) in zip(cls_dets[:, :5], cls_id_feature)
                ]
            else:
                cls_detects = []

            if len(cls_dets_low) > 0:
                cls_detects_low = [
                    MCTrack(MCTrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], feat, self.opt.num_classes, cls_id, 30)
                    for (tlbrs, feat) in zip(cls_dets_low[:, :5], cls_id_feature_low)
                ]
            else:
                cls_detects_low = []

            ''' Add newly detected tracks to tracked_tracks'''
            unconfirmed_dict = defaultdict(list)
            tracked_tracks_dict = defaultdict(list)
            for track in self.tracked_tracks_dict[cls_id]:
                if not track.is_activated:
                    unconfirmed_dict[cls_id].append(track)
                else:
                    tracked_tracks_dict[cls_id].append(track)

            ''' Step 2: First association, with embedding'''
            # building tracking pool for the current frame
            track_pool_dict = defaultdict(list)
            track_pool_dict[cls_id] = join_tracks(tracked_tracks_dict[cls_id], self.lost_tracks_dict[cls_id])

            # Predict the current location with KF
            MCTrack.multi_predict(track_pool_dict[cls_id])
            apply_gmc_to_tracks(track_pool_dict[cls_id], gmc_warp)
            emb_dists = matching.embedding_distance(track_pool_dict[cls_id], cls_detects)
            if self.opt.amot_lite or self.opt.sa_amc:
                iou_dists = matching.iou_distance(track_pool_dict[cls_id], cls_detects)
                if self.opt.sa_amc:
                    amc_dists = matching.sa_amc_distance(
                        track_pool_dict[cls_id],
                        cls_detects,
                        self.id_feature_history,
                        self.reg_history,
                        h_out,
                        w_out,
                        height,
                        width,
                        norm_sigma=self.opt.sa_amc_norm_sigma,
                        min_scale=self.opt.sa_amc_min_scale,
                        scale_weight=self.opt.sa_amc_scale_weight,
                        scale_sigma=self.opt.sa_amc_scale_sigma)
                    dists = matching.fuse_sa_amc_embedding(
                        iou_dists,
                        amc_dists,
                        emb_dists,
                        track_pool_dict[cls_id],
                        cls_detects,
                        amc_weight=self.opt.amot_amc_weight,
                        app_min_weight=self.opt.sa_amc_app_min_weight,
                        app_max_weight=self.opt.sa_amc_app_max_weight,
                        app_scale_mid=self.opt.sa_amc_app_scale_mid,
                        app_scale_gamma=self.opt.sa_amc_app_scale_gamma,
                        min_scale=self.opt.sa_amc_min_scale)
                else:
                    amc_dists = matching.amc_distance(
                        track_pool_dict[cls_id],
                        cls_detects,
                        self.id_feature_history,
                        self.reg_history,
                        h_out,
                        w_out,
                        height,
                        width,
                        sigma=self.opt.amot_amc_sigma)
                    dists = matching.fuse_iou_amc_embedding(
                        iou_dists,
                        amc_dists,
                        emb_dists,
                        amc_weight=self.opt.amot_amc_weight)
            else:
                dists = emb_dists
            dists = matching.fuse_motion(self.kalman_filter, dists, track_pool_dict[cls_id], cls_detects)
            first_match_thresh = 0.7
            if self.opt.sa_amc:
                first_match_thresh = float(self._sage_param(
                    cls_id, 'match_thresh', getattr(self.opt, 'sage_match_thresh', 0.7)))
            matches, u_track, u_detection = matching.linear_assignment(
                dists, thresh=first_match_thresh)

            for i_tracked, i_det in matches:
                track = track_pool_dict[cls_id][i_tracked]
                det = cls_detects[i_det]
                if track.state == TrackState.Tracked:
                    track.update(cls_detects[i_det], self.frame_id)
                    activated_tracks_dict[cls_id].append(track)  # for multi-class
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refined_tracks_dict[cls_id].append(track)

            lost_recovery_tracks = self._lost_recovery_tracks(
                [track_pool_dict[cls_id][i] for i in u_track
                 if track_pool_dict[cls_id][i].state == TrackState.Lost],
                cls_id)

            ''' Step 3: Second association, with IOU'''
            cls_detects = [cls_detects[i] for i in u_detection]
            r_tracked_tracks = [track_pool_dict[cls_id][i]
                                 for i in u_track if track_pool_dict[cls_id][i].state == TrackState.Tracked]
            dists = matching.iou_distance(r_tracked_tracks, cls_detects)
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)  # thresh=0.5

            for i_tracked, i_det in matches:
                track = r_tracked_tracks[i_tracked]
                det = cls_detects[i_det]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_tracks_dict[cls_id].append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refined_tracks_dict[cls_id].append(track)

            u_low_detection = list(range(len(cls_detects_low)))
            if (self.opt.stcmot_tdrm or self.opt.sca_tdrm) and len(cls_detects_low) > 0 and len(u_track) > 0:
                tdrm_tracks = [r_tracked_tracks[i] for i in u_track]
                low_emb_dists = matching.embedding_distance(tdrm_tracks, cls_detects_low)
                low_iou_dists = matching.iou_distance(tdrm_tracks, cls_detects_low)
                if self.opt.sca_tdrm:
                    low_dists = matching.sca_tdrm_distance(
                        tdrm_tracks,
                        cls_detects_low,
                        low_emb_dists,
                        low_iou_dists,
                        cls_id,
                        current_frame=self.frame_id,
                        profile=self.opt.sca_tdrm_profile,
                        min_scale=self.opt.sca_tdrm_min_scale,
                        center_gate=self.opt.sca_tdrm_center_gate,
                        scale_gate=self.opt.sca_tdrm_scale_gate,
                        min_quality=self.opt.sca_tdrm_min_track_quality,
                        iou_weight=self.opt.sca_tdrm_iou_weight,
                        emb_weight=self.opt.sca_tdrm_emb_weight,
                        center_weight=self.opt.sca_tdrm_center_weight,
                        scale_weight=self.opt.sca_tdrm_scale_weight,
                        quality_bonus=self.opt.sca_tdrm_quality_bonus,
                        max_recovery_streak=self.opt.sca_tdrm_max_recovery_streak,
                        confirm_window=self.opt.sca_tdrm_confirm_window,
                        pending_penalty=self.opt.sca_tdrm_pending_penalty)
                    low_match_thresh = matching.sca_tdrm_class_param(
                        cls_id,
                        self.opt.sca_tdrm_profile,
                        'match_thresh',
                        self.opt.sca_tdrm_match_thresh)
                elif self.opt.amot_lite:
                    low_amc_dists = matching.amc_distance(
                        tdrm_tracks,
                        cls_detects_low,
                        self.id_feature_history,
                        self.reg_history,
                        h_out,
                        w_out,
                        height,
                        width,
                        sigma=self.opt.amot_amc_sigma)
                    low_dists = matching.fuse_iou_amc_embedding(
                        low_iou_dists,
                        low_amc_dists,
                        low_emb_dists,
                        amc_weight=self.opt.amot_amc_weight)
                    low_match_thresh = self.opt.stcmot_tdrm_match_thresh
                else:
                    low_dists = 0.5 * low_iou_dists + 0.5 * low_emb_dists
                    low_match_thresh = self.opt.stcmot_tdrm_match_thresh

                low_matches, u_tdrm_track, u_low_detection = matching.linear_assignment(
                    low_dists, thresh=low_match_thresh)
                for i_tracked, i_det in low_matches:
                    track = tdrm_tracks[i_tracked]
                    det = cls_detects_low[i_det]
                    track.update(det, self.frame_id, recovery=self.opt.sca_tdrm)
                    activated_tracks_dict[cls_id].append(track)

                u_track = [u_track[i] for i in u_tdrm_track]

            if self.opt.lost_track_recovery and len(lost_recovery_tracks) > 0 and len(u_low_detection) > 0:
                min_score = float(self._cast_param(
                    cls_id, 'lost_min_score', self.opt.lost_track_recovery_min_score))
                low_candidates = [
                    cls_detects_low[i] for i in u_low_detection
                    if cls_detects_low[i].score >= min_score
                ]
                if len(low_candidates) > 0:
                    lost_emb_dists = matching.embedding_distance(lost_recovery_tracks, low_candidates)
                    lost_iou_dists = matching.iou_distance(lost_recovery_tracks, low_candidates)
                    center_gate = matching.sca_tdrm_class_param(
                        cls_id, self.opt.sca_tdrm_profile, 'center_gate', self.opt.sca_tdrm_center_gate)
                    scale_gate = matching.sca_tdrm_class_param(
                        cls_id, self.opt.sca_tdrm_profile, 'scale_gate', self.opt.sca_tdrm_scale_gate)
                    min_quality = max(
                        float(self._cast_param(
                            cls_id, 'lost_min_quality', self.opt.lost_track_recovery_min_quality)),
                        float(matching.sca_tdrm_class_param(
                            cls_id, self.opt.sca_tdrm_profile, 'min_quality',
                            self.opt.sca_tdrm_min_track_quality)))
                    lost_dists = matching.sca_tdrm_distance(
                        lost_recovery_tracks,
                        low_candidates,
                        lost_emb_dists,
                        lost_iou_dists,
                        cls_id,
                        current_frame=self.frame_id,
                        profile='none',
                        min_scale=self.opt.sca_tdrm_min_scale,
                        center_gate=center_gate,
                        scale_gate=scale_gate,
                        min_quality=min_quality,
                        iou_weight=self.opt.sca_tdrm_iou_weight,
                        emb_weight=self.opt.sca_tdrm_emb_weight,
                        center_weight=self.opt.sca_tdrm_center_weight,
                        scale_weight=self.opt.sca_tdrm_scale_weight,
                        quality_bonus=self.opt.sca_tdrm_quality_bonus,
                        max_recovery_streak=1,
                        confirm_window=self.opt.sca_tdrm_confirm_window,
                        pending_penalty=self.opt.sca_tdrm_pending_penalty)
                    lost_match_thresh = float(self._cast_param(
                        cls_id, 'lost_match_thresh', self.opt.lost_track_recovery_match_thresh))
                    lost_matches, _, _ = matching.linear_assignment(
                        lost_dists, thresh=lost_match_thresh)
                    for i_tracked, i_det in lost_matches:
                        track = lost_recovery_tracks[i_tracked]
                        det = low_candidates[i_det]
                        track.re_activate(det, self.frame_id, new_id=False)
                        track.mark_low_recovery_update(self.frame_id)
                        refined_tracks_dict[cls_id].append(track)

            for it in u_track:
                track = r_tracked_tracks[it]
                if not track.state == TrackState.Lost:
                    remaining_dets = cls_detects + cls_detects_low
                    if self._try_mtc_continue(track, remaining_dets, h_out, w_out, height, width):
                        activated_tracks_dict[cls_id].append(track)
                    else:
                        track.mark_lost()
                        lost_tracks_dict[cls_id].append(track)

            '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
            cls_detects = [cls_detects[i] for i in u_detection]
            dists = matching.iou_distance(unconfirmed_dict[cls_id], cls_detects)
            matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
            for i_tracked, i_det in matches:
                track = unconfirmed_dict[cls_id][i_tracked]
                track.update(cls_detects[i_det], self.frame_id)
                if self.birth_competition_enabled and len(track.tlwh_deque) < track.birth_required_hits:
                    track.is_activated = False
                activated_tracks_dict[cls_id].append(track)
            for it in u_unconfirmed:
                track = unconfirmed_dict[cls_id][it]
                track.mark_removed()
                removed_tracks_dict[cls_id].append(track)

            """ Step 4: Init new tracks"""
            for i_new in u_detection:
                track = cls_detects[i_new]

                if track.score < class_conf_thres:
                    continue

                track.activate(self.kalman_filter, self.frame_id)
                if self.birth_competition_enabled and ambiguous_birth(
                        cls_id, track.tlbr, track.score, dets, class_thresholds):
                    track.birth_required_hits = self.birth_confirm_hits
                    if track.birth_required_hits > 2:
                        track.is_activated = False
                    self.birth_competition_stats['ambiguous'] += 1
                activated_tracks_dict[cls_id].append(track)

            """ Step 5: Update state"""
            for track in self.lost_tracks_dict[cls_id]:
                if self.frame_id - track.end_frame > self.max_time_lost:
                    track.mark_removed()
                    removed_tracks_dict[cls_id].append(track)

            # print('Ramained match {} s'.format(t4-t3))
            self.tracked_tracks_dict[cls_id] = [t for t in self.tracked_tracks_dict[cls_id] if
                                                t.state == TrackState.Tracked]
            self.tracked_tracks_dict[cls_id] = join_tracks(self.tracked_tracks_dict[cls_id],
                                                           activated_tracks_dict[cls_id])
            self.tracked_tracks_dict[cls_id] = join_tracks(self.tracked_tracks_dict[cls_id],
                                                           refined_tracks_dict[cls_id])
            self.lost_tracks_dict[cls_id] = sub_tracks(self.lost_tracks_dict[cls_id],
                                                       self.tracked_tracks_dict[cls_id])
            self.lost_tracks_dict[cls_id].extend(lost_tracks_dict[cls_id])
            self.lost_tracks_dict[cls_id] = sub_tracks(self.lost_tracks_dict[cls_id],
                                                       self.removed_tracks_dict[cls_id])
            self.removed_tracks_dict[cls_id].extend(removed_tracks_dict[cls_id])
            self.tracked_tracks_dict[cls_id], self.lost_tracks_dict[cls_id] = remove_duplicate_tracks(
                self.tracked_tracks_dict[cls_id],
                self.lost_tracks_dict[cls_id])

            # get scores of lost tracks
            output_tracks_dict[cls_id] = [track for track in self.tracked_tracks_dict[cls_id] if track.is_activated]

            logger.debug('===========Frame {}=========='.format(self.frame_id))
            logger.debug('Activated: {}'.format(
                [track.track_id for track in activated_tracks_dict[cls_id]]))
            logger.debug('Refind: {}'.format(
                [track.track_id for track in refined_tracks_dict[cls_id]]))
            logger.debug('Lost: {}'.format(
                [track.track_id for track in lost_tracks_dict[cls_id]]))
            logger.debug('Removed: {}'.format(
                [track.track_id for track in removed_tracks_dict[cls_id]]))

        if self.birth_competition_enabled:
            for cls_id in (3, 4):
                kept = []
                for track in self.tracked_tracks_dict[cls_id]:
                    if self.frame_id - track.start_frame >= self.birth_confirm_hits:
                        kept.append(track)
                        continue
                    competitor = mature_competitor(
                        track, self.tracked_tracks_dict, self.frame_id,
                        matching.track_recovery_quality)
                    if competitor is None:
                        kept.append(track)
                        continue
                    track.mark_removed()
                    self.removed_tracks_dict[cls_id].append(track)
                    self.birth_competition_stats['suppressed'] += 1
                self.tracked_tracks_dict[cls_id] = kept
                output_tracks_dict[cls_id] = [track for track in kept if track.is_activated]

        return output_tracks_dict


class JDETracker(object):
    def __init__(self, opt, frame_rate=30):
        self.opt = opt

        # ----- init model
        print('Creating model...')
        self.model = create_model(opt.arch, opt.heads, opt.head_conv)
        self.model = load_model(self.model, opt.load_model)  # load specified checkpoint
        self.model = self.model.to(opt.device)
        self.model.eval()

        # ----- track_lets: value type: list[Track]
        self.tracked_tracks_dict = defaultdict(list)
        self.lost_tracks_dict = defaultdict(list)
        self.removed_tracks_dict = defaultdict(list)

        self.frame_id = 0
        self.det_thresh = opt.conf_thres
        self.buffer_size = int(frame_rate / 30.0 * opt.track_buffer)  # int(frame_rate / 30.0 * opt.track_buffer)
        self.max_time_lost = self.buffer_size
        self.max_per_image = self.opt.K  # max objects per image
        self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)

        # ----- using kalman filter to stabilize tracking
        self.kalman_filter = KalmanFilter()

    def reset(self):
        """
        :return:
        """
        # Reset tracks dict
        self.tracked_tracks_dict = defaultdict(list)  # value type: list[Track]
        self.lost_tracks_dict = defaultdict(list)  # value type: list[Track]
        self.removed_tracks_dict = defaultdict(list)  # value type: list[Track]

        # Reset frame id
        self.frame_id = 0

        # Reset kalman filter to stabilize tracking
        self.kalman_filter = KalmanFilter()

    def post_process(self, dets, meta):
        """
        2D bbox检测结果后处理
        :param dets:
        :param meta:
        :return:
        """
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])  # default: 1×128×6

        # affine transform
        dets = ctdet_post_process(dets.copy(),
                                  [meta['c']], [meta['s']],
                                  meta['out_height'],
                                  meta['out_width'],
                                  self.opt.num_classes)

        # detection dict(cls_id as key)
        dets = dets[0]  # fetch the first image dets results(batch_size = 1 by default)

        return dets

    def merge_outputs(self, detections):
        """
        :param detections:
        :return:
        """
        results = {}
        for j in range(1, self.opt.num_classes + 1):
            results[j] = np.concatenate([detection[j] for detection in detections],
                                        axis=0).astype(np.float32)

        scores = np.hstack([results[j][:, 4] for j in range(1, self.opt.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.opt.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]

        return results

    def update_detection(self, im_blob, img_0):
        """
        更新视频序列或图片序列的检测结果
        :rtype: dict
        :param im_blob:
        :param img_0:
        :return:
        """
        height, width = img_0.shape[0], img_0.shape[1]  # H, W of original input image
        net_height, net_width = im_blob.shape[2], im_blob.shape[3]  # H, W of net input

        c = np.array([width * 0.5, height * 0.5], dtype=np.float32)  # image center
        s = max(float(net_width) / float(net_height) * height, width) * 1.0

        h_out = net_height // self.opt.down_ratio
        w_out = net_width // self.opt.down_ratio

        # ----- get detections
        with torch.no_grad():
            dets_dict = defaultdict(list)

            # --- network output
            output = self.model.forward(im_blob)[-1]

            # --- detection outputs
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            reg = output['reg'] if self.opt.reg_offset else None

            # --- decode results of detection
            dets, inds, cls_inds_mask = mot_decode(heatmap=hm,
                                                   wh=wh,
                                                   reg=reg,
                                                   num_classes=self.opt.num_classes,
                                                   cat_spec_wh=self.opt.cat_spec_wh,
                                                   K=self.opt.K)

            # --- map to original image coordinate system
            # meta = {'c': c,
            #         's': s,
            #         'out_height': h_out,
            #         'out_width': w_out}
            # dets = self.post_process(dets, meta)  # using affine matrix
            dets = map2orig(dets, h_out, w_out, height, width, self.opt.num_classes)  # translate and scale
            # dets = self.merge_outputs([dets])

            # --- parse detections of each class
            for cls_id in range(self.opt.num_classes):  # cls_id start from index 0
                cls_dets = dets[cls_id]

                # filter out low conf score dets
                remain_inds = cls_dets[:, 4] > self.opt.conf_thres
                cls_dets = cls_dets[remain_inds]
                dets_dict[cls_id] = cls_dets

        return dets_dict

    def update_tracking(self, im_blob, img_0):
        """
        :param im_blob:
        :param img_0:
        :return:
        """
        # update frame id
        self.frame_id += 1

        # record tracking results, key: class_id
        activated_tracks_dict = defaultdict(list)
        refined_tracks_dict = defaultdict(list)
        lost_tracks_dict = defaultdict(list)
        removed_tracks_dict = defaultdict(list)
        output_tracks_dict = defaultdict(list)

        height, width = img_0.shape[0], img_0.shape[1]  # H, W of original input image
        net_height, net_width = im_blob.shape[2], im_blob.shape[3]  # H, W of net input

        c = np.array([width * 0.5, height * 0.5], dtype=np.float32)
        s = max(float(net_width) / float(net_height) * height, width) * 1.0
        h_out = net_height // self.opt.down_ratio
        w_out = net_width // self.opt.down_ratio

        ''' Step 1: Network forward, get detections & embeddings'''
        with torch.no_grad():
            output = self.model.forward(im_blob)[-1]

            hm = output['hm'].sigmoid_()
            wh = output['wh']
            reg = output['reg'] if self.opt.reg_offset else None
            id_feature = output['id']

            # L2 normalize the reid feature vector
            id_feature = F.normalize(id_feature, dim=1)

            #  detection decoding
            dets, inds, cls_inds_mask = mot_decode(heatmap=hm,
                                                   wh=wh,
                                                   reg=reg,
                                                   num_classes=self.opt.num_classes,
                                                   cat_spec_wh=self.opt.cat_spec_wh,
                                                   K=self.opt.K)

            # ----- get ReID feature vector by object class
            cls_id_feats = []  # topK feature vectors of each object class
            for cls_id in range(self.opt.num_classes):  # cls_id starts from 0
                # get inds of each object class
                cls_inds = inds[:, cls_inds_mask[cls_id]]

                # gather feats for each object class
                cls_id_feature = _tranpose_and_gather_feat(id_feature, cls_inds)  # inds: 1×128
                cls_id_feature = cls_id_feature.squeeze(0)  # n × FeatDim
                cls_id_feature = cls_id_feature.cpu().numpy()
                cls_id_feats.append(cls_id_feature)

        # 检测结果后处理
        # meta = {'c': c,
        #         's': s,
        #         'out_height': h_out,
        #         'out_width': w_out}
        # dets = self.post_process(dets, meta)  # using affine matrix
        # dets = self.merge_outputs([dets])

        # translate and scale
        dets = map2orig(dets, h_out, w_out, height, width, self.opt.num_classes)

        # ----- parse each object class
        for cls_id in range(self.opt.num_classes):  # cls_id从0开始
            cls_dets = dets[cls_id]

            '''
            # visualize each class
            for i in range(0, cls_dets.shape[0]):
                bbox = cls_dets[i][0:4]
                cv2.rectangle(img0,
                              (bbox[0], bbox[1]),  # left-top point
                              (bbox[2], bbox[3]),  # right-down point
                              [0, 255, 255],  # yellow
                              2)
                cv2.putText(img0,
                            id2cls[cls_id],
                            (bbox[0], bbox[1]),
                            cv2.FONT_HERSHEY_PLAIN,
                            1.3,
                            [0, 0, 255],  # red
                            2)
            cv2.imshow('{}'.format(id2cls[cls_id]), img0)
            cv2.waitKey(0)
            '''

            # filter out low confidence detections
            remain_inds = cls_dets[:, 4] > self.opt.conf_thres
            cls_dets = cls_dets[remain_inds]
            cls_id_feature = cls_id_feats[cls_id][remain_inds]

            if len(cls_dets) > 0:
                '''Detections, tlbrs: top left bottom right score'''
                cls_detections = [Track(Track.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], feat, buff_size=30)
                                  for (tlbrs, feat) in zip(cls_dets[:, :5], cls_id_feature)]
            else:
                cls_detections = []

            # reset the track ids for each different object class
            if self.frame_id == 1:
                for track in cls_detections:
                    track.reset_track_id()

            ''' Add newly detected tracklets to tracked_stracks'''
            unconfirmed_dict = defaultdict(list)
            tracked_stracks_dict = defaultdict(list)
            for track in self.tracked_tracks_dict[cls_id]:
                if not track.is_activated:
                    unconfirmed_dict[cls_id].append(track)
                else:
                    tracked_stracks_dict[cls_id].append(track)

            ''' Step 2: First association, with embedding'''
            strack_pool_dict = defaultdict(list)
            strack_pool_dict[cls_id] = join_tracks(tracked_stracks_dict[cls_id], self.lost_tracks_dict[cls_id])

            # Predict the current location with KF
            # for strack in strack_pool:
            Track.multi_predict(strack_pool_dict[cls_id])
            dists = matching.embedding_distance(strack_pool_dict[cls_id], cls_detections)
            dists = matching.fuse_motion(self.kalman_filter, dists, strack_pool_dict[cls_id], cls_detections)
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.7)  # thresh=0.7

            for i_tracked, i_det in matches:
                track = strack_pool_dict[cls_id][i_tracked]
                det = cls_detections[i_det]
                if track.state == TrackState.Tracked:
                    track.update(cls_detections[i_det], self.frame_id)
                    activated_tracks_dict[cls_id].append(track)  # for multi-class
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refined_tracks_dict[cls_id].append(track)

            ''' Step 3: Second association, with IOU'''
            cls_detections = [cls_detections[i] for i in u_detection]
            r_tracked_stracks = [strack_pool_dict[cls_id][i]
                                 for i in u_track if strack_pool_dict[cls_id][i].state == TrackState.Tracked]
            dists = matching.iou_distance(r_tracked_stracks, cls_detections)
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)  # thresh=0.5

            for i_tracked, i_det in matches:
                track = r_tracked_stracks[i_tracked]
                det = cls_detections[i_det]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_tracks_dict[cls_id].append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refined_tracks_dict[cls_id].append(track)

            for it in u_track:
                track = r_tracked_stracks[it]
                if not track.state == TrackState.Lost:
                    track.mark_lost()
                    lost_tracks_dict[cls_id].append(track)

            '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
            cls_detections = [cls_detections[i] for i in u_detection]
            dists = matching.iou_distance(unconfirmed_dict[cls_id], cls_detections)
            matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
            for i_tracked, i_det in matches:
                unconfirmed_dict[cls_id][i_tracked].update(cls_detections[i_det], self.frame_id)
                activated_tracks_dict[cls_id].append(unconfirmed_dict[cls_id][i_tracked])
            for it in u_unconfirmed:
                track = unconfirmed_dict[cls_id][it]
                track.mark_removed()
                removed_tracks_dict[cls_id].append(track)

            """ Step 4: Init new tracks"""
            for i_new in u_detection:
                track = cls_detections[i_new]

                if track.score < self.det_thresh:
                    continue

                track.activate(self.kalman_filter, self.frame_id)
                activated_tracks_dict[cls_id].append(track)

            """ Step 5: Update state"""
            for track in self.lost_tracks_dict[cls_id]:
                if self.frame_id - track.end_frame > self.max_time_lost:
                    track.mark_removed()
                    removed_tracks_dict[cls_id].append(track)

            # print('Ramained match {} s'.format(t4-t3))
            self.tracked_tracks_dict[cls_id] = [t for t in self.tracked_tracks_dict[cls_id] if
                                                t.state == TrackState.Tracked]
            self.tracked_tracks_dict[cls_id] = join_tracks(self.tracked_tracks_dict[cls_id],
                                                           activated_tracks_dict[cls_id])
            self.tracked_tracks_dict[cls_id] = join_tracks(self.tracked_tracks_dict[cls_id],
                                                           refined_tracks_dict[cls_id])
            self.lost_tracks_dict[cls_id] = sub_tracks(self.lost_tracks_dict[cls_id],
                                                       self.tracked_tracks_dict[cls_id])
            self.lost_tracks_dict[cls_id].extend(lost_tracks_dict[cls_id])
            self.lost_tracks_dict[cls_id] = sub_tracks(self.lost_tracks_dict[cls_id],
                                                       self.removed_tracks_dict[cls_id])
            self.removed_tracks_dict[cls_id].extend(removed_tracks_dict[cls_id])
            self.tracked_tracks_dict[cls_id], self.lost_tracks_dict[cls_id] = remove_duplicate_tracks(
                self.tracked_tracks_dict[cls_id],
                self.lost_tracks_dict[cls_id])

            # get scores of lost tracks
            output_tracks_dict[cls_id] = [track for track in self.tracked_tracks_dict[cls_id] if track.is_activated]

            logger.debug('===========Frame {}=========='.format(self.frame_id))
            logger.debug('Activated: {}'.format(
                [track.track_id for track in activated_tracks_dict[cls_id]]))
            logger.debug('Refind: {}'.format(
                [track.track_id for track in refined_tracks_dict[cls_id]]))
            logger.debug('Lost: {}'.format(
                [track.track_id for track in lost_tracks_dict[cls_id]]))
            logger.debug('Removed: {}'.format(
                [track.track_id for track in removed_tracks_dict[cls_id]]))

        return output_tracks_dict


def join_tracks(t_list_a, t_list_b):
    """
    join two track lists
    :param t_list_a:
    :param t_list_b:
    :return:
    """
    exists = {}
    res = []
    for t in t_list_a:
        exists[t.track_id] = 1
        res.append(t)
    for t in t_list_b:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_tracks(t_list_a, t_list_b):
    tracks = {}
    for t in t_list_a:
        tracks[t.track_id] = t
    for t in t_list_b:
        tid = t.track_id
        if tracks.get(tid, 0):
            del tracks[tid]
    return list(tracks.values())


def remove_duplicate_tracks(tracks_a, tracks_b):
    p_dist = matching.iou_distance(tracks_a, tracks_b)
    pairs = np.where(p_dist < 0.15)
    dup_a, dup_b = list(), list()

    for p, q in zip(*pairs):
        time_p = tracks_a[p].frame_id - tracks_a[p].start_frame
        time_q = tracks_b[q].frame_id - tracks_b[q].start_frame
        if time_p > time_q:
            dup_b.append(q)
        else:
            dup_a.append(p)

    res_a = [t for i, t in enumerate(tracks_a) if not i in dup_a]
    res_b = [t for i, t in enumerate(tracks_b) if not i in dup_b]

    return res_a, res_b
