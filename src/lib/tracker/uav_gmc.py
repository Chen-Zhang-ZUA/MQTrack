import cv2
import numpy as np


class UAVGMC:
    """Lightweight global motion compensation for UAV video tracking."""

    def __init__(self, opt):
        self.enabled = bool(getattr(opt, 'uav_gmc', False))
        self.mode = getattr(opt, 'uav_gmc_mode', 'translation')
        self.max_features = int(getattr(opt, 'uav_gmc_max_features', 600))
        self.quality_level = float(getattr(opt, 'uav_gmc_quality_level', 0.01))
        self.min_distance = float(getattr(opt, 'uav_gmc_min_distance', 8.0))
        self.ransac_thresh = float(getattr(opt, 'uav_gmc_ransac_thresh', 3.0))
        self.min_inliers = int(getattr(opt, 'uav_gmc_min_inliers', 24))
        self.max_translation_ratio = float(getattr(opt, 'uav_gmc_max_translation_ratio', 0.18))
        self.max_scale_delta = float(getattr(opt, 'uav_gmc_max_scale_delta', 0.08))
        self.max_rotation_deg = float(getattr(opt, 'uav_gmc_max_rotation_deg', 8.0))
        self.downscale_max_side = int(getattr(opt, 'uav_gmc_downscale_max_side', 720))
        self.prev_gray = None

    def reset(self):
        self.prev_gray = None

    def estimate(self, image):
        if not self.enabled:
            return None

        gray, scale = self._to_gray(image)
        if self.prev_gray is None:
            self.prev_gray = gray
            return np.eye(2, 3, dtype=np.float32)

        warp = self._estimate_from_gray(self.prev_gray, gray, scale)
        self.prev_gray = gray
        return warp

    def _to_gray(self, image):
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        h, w = gray.shape[:2]
        max_side = max(h, w)
        if self.downscale_max_side > 0 and max_side > self.downscale_max_side:
            scale = float(self.downscale_max_side) / float(max_side)
            gray = cv2.resize(gray, (int(round(w * scale)), int(round(h * scale))),
                              interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0

        return gray, scale

    def _estimate_from_gray(self, prev_gray, curr_gray, scale):
        identity = np.eye(2, 3, dtype=np.float32)
        if prev_gray.shape != curr_gray.shape:
            return identity

        prev_pts = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=self.max_features,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=3)
        if prev_pts is None or len(prev_pts) < self.min_inliers:
            return identity

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            curr_gray,
            prev_pts,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        if curr_pts is None or status is None:
            return identity

        status = status.reshape(-1).astype(bool)
        prev_match = prev_pts.reshape(-1, 2)[status]
        curr_match = curr_pts.reshape(-1, 2)[status]
        if len(prev_match) < self.min_inliers:
            return identity

        affine, inliers = cv2.estimateAffinePartial2D(
            prev_match,
            curr_match,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_thresh,
            maxIters=2000,
            confidence=0.99,
            refineIters=10)
        if affine is None or inliers is None:
            return identity

        inlier_count = int(inliers.sum())
        if inlier_count < self.min_inliers:
            return identity

        warp = np.asarray(affine, dtype=np.float32)
        if scale > 0:
            warp[:, 2] /= scale

        if self.mode == 'translation':
            trans = warp[:, 2].copy()
            warp = identity.copy()
            warp[:, 2] = trans
        elif self.mode not in ('similarity', 'affine'):
            return identity

        if not self._valid_warp(warp, curr_gray.shape, scale):
            return identity
        return warp

    def _valid_warp(self, warp, shape, scale):
        h, w = shape[:2]
        if scale > 0:
            h = h / scale
            w = w / scale

        translation = float(np.linalg.norm(warp[:, 2]))
        if translation > self.max_translation_ratio * float(np.hypot(h, w)):
            return False

        linear = warp[:, :2]
        det = float(np.linalg.det(linear))
        if det <= 0:
            return False
        scale_est = float(np.sqrt(det))
        if abs(scale_est - 1.0) > self.max_scale_delta:
            return False

        rotation = float(np.degrees(np.arctan2(linear[1, 0], linear[0, 0])))
        if abs(rotation) > self.max_rotation_deg:
            return False
        return True


def apply_gmc_to_tracks(tracks, warp):
    if warp is None or len(tracks) == 0:
        return

    warp = np.asarray(warp, dtype=np.float32)
    if warp.shape != (2, 3):
        return

    linear = warp[:, :2]
    trans = warp[:, 2]
    det = float(np.linalg.det(linear))
    height_scale = float(np.sqrt(det)) if det > 0 else 1.0
    if not np.isfinite(height_scale) or height_scale <= 0:
        height_scale = 1.0

    for track in tracks:
        if track.mean is None:
            continue
        center = np.asarray(track.mean[:2], dtype=np.float32)
        track.mean[:2] = np.dot(linear, center) + trans
        track.mean[3] = max(1.0, track.mean[3] * height_scale)
        if track.mean.shape[0] >= 6:
            velocity = np.asarray(track.mean[4:6], dtype=np.float32)
            track.mean[4:6] = np.dot(linear, velocity)
