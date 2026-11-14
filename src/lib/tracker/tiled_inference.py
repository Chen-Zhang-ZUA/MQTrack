"""Opt-in full-frame + overlapping crop inference without changing the model."""

import math
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import nms

from lib.models.decode import mot_decode
from lib.models.utils import _tranpose_and_gather_feat


def axis_starts(length, crop_size, overlap):
    if length <= 0 or crop_size <= 0 or not 0 <= overlap < 1:
        raise ValueError('Invalid crop geometry')
    crop_size = min(crop_size, length)
    step = max(1, int(round(crop_size * (1 - overlap))))
    starts = list(range(0, length - crop_size + 1, step))
    if starts[-1] != length - crop_size:
        starts.append(length - crop_size)
    return starts


def tile_windows(height, width, fraction, overlap):
    if not 0 < fraction <= 1:
        raise ValueError('tile_fraction must be in (0, 1]')
    crop_h = max(1, int(math.ceil(height * fraction)))
    crop_w = max(1, int(math.ceil(width * fraction)))
    if crop_h == height and crop_w == width:
        return []
    return [(x, y, x + crop_w, y + crop_h)
            for y in axis_starts(height, crop_h, overlap)
            for x in axis_starts(width, crop_w, overlap)]


@dataclass(frozen=True)
class LetterboxGeometry:
    scale_x: float
    scale_y: float
    left: int
    top: int

    @classmethod
    def create(cls, height, width, net_h, net_w):
        if min(height, width, net_h, net_w) <= 0:
            raise ValueError('Image dimensions must be positive')
        ratio = min(net_h / height, net_w / width)
        resized_w = max(1, round(width * ratio))
        resized_h = max(1, round(height * ratio))
        return cls(resized_w / width, resized_h / height,
                   round((net_w - resized_w) / 2 - 0.1),
                   round((net_h - resized_h) / 2 - 0.1))

    def invert_boxes(self, rows, stride):
        rows = np.asarray(rows, dtype=np.float32).copy()
        rows[:, [0, 2]] = (rows[:, [0, 2]] * stride - self.left) / self.scale_x
        rows[:, [1, 3]] = (rows[:, [1, 3]] * stride - self.top) / self.scale_y
        return rows

    def preprocess(self, image, net_h, net_w):
        h, w = image.shape[:2]
        resized = cv2.resize(image, (round(w * self.scale_x), round(h * self.scale_y)),
                             interpolation=cv2.INTER_AREA)
        return cv2.copyMakeBorder(resized, self.top, net_h - resized.shape[0] - self.top,
                                  self.left, net_w - resized.shape[1] - self.left,
                                  cv2.BORDER_CONSTANT, value=(127.5, 127.5, 127.5))


def filter_crop_boxes(rows, window, image_shape, min_score, max_side, edge_margin):
    """Reject padding peaks and internally truncated boxes before clipping."""
    h, w = image_shape[:2]
    x, y, right, bottom = window
    crop_w, crop_h = right - x, bottom - y
    centers = (rows[:, :2] + rows[:, 2:4]) * 0.5
    size = rows[:, 2:4] - rows[:, :2]
    keep = (np.isfinite(rows).all(axis=1) & (rows[:, 4] >= min_score)
            & (size > 0).all(axis=1)
            & (centers >= 0).all(axis=1)
            & (centers[:, 0] < crop_w) & (centers[:, 1] < crop_h))
    if max_side > 0:
        keep &= size.max(axis=1) <= max_side
    if x > 0:
        keep &= rows[:, 0] > edge_margin
    if y > 0:
        keep &= rows[:, 1] > edge_margin
    if right < w:
        keep &= rows[:, 2] < crop_w - edge_margin
    if bottom < h:
        keep &= rows[:, 3] < crop_h - edge_margin
    mapped = rows[keep].copy()
    mapped[:, [0, 2]] = np.clip(mapped[:, [0, 2]], 0, crop_w) + x
    mapped[:, [1, 3]] = np.clip(mapped[:, [1, 3]], 0, crop_h) + y
    return mapped, keep


def sample_full_embeddings(rows, feature_map, image_shape, net_shape, stride):
    if not len(rows):
        return np.empty((0, feature_map.shape[1]), dtype=np.float32)
    geometry = LetterboxGeometry.create(*image_shape[:2], *net_shape)
    centers = (rows[:, :2] + rows[:, 2:4]) * 0.5
    xs = np.floor((centers[:, 0] * geometry.scale_x + geometry.left) / stride)
    ys = np.floor((centers[:, 1] * geometry.scale_y + geometry.top) / stride)
    xs = np.clip(xs, 0, feature_map.shape[3] - 1).astype(np.int64)
    ys = np.clip(ys, 0, feature_map.shape[2] - 1).astype(np.int64)
    indices = torch.as_tensor(ys * feature_map.shape[3] + xs,
                              device=feature_map.device).unsqueeze(0)
    features = _tranpose_and_gather_feat(feature_map, indices)[0]
    return F.normalize(features, dim=1).cpu().numpy()


def merge_detections(rows, features, num_classes, image_shape, min_score, iou, max_detections):
    """NMS indices select both rows and features; never average ReID views."""
    if len(rows) != len(features):
        raise ValueError('Detection/embedding count mismatch')
    h, w = image_shape[:2]
    rows = np.asarray(rows, dtype=np.float32).copy()
    rows[:, [0, 2]] = np.clip(rows[:, [0, 2]], 0, w)
    rows[:, [1, 3]] = np.clip(rows[:, [1, 3]], 0, h)
    valid = (np.isfinite(rows).all(axis=1) & np.isfinite(features).all(axis=1)
             & (rows[:, 4] >= min_score)
             & (rows[:, 2] > rows[:, 0]) & (rows[:, 3] > rows[:, 1])
             & (rows[:, 5] >= 0) & (rows[:, 5] < num_classes))
    selected = []
    for cls_id in range(num_classes):
        indices = np.flatnonzero(valid & (rows[:, 5] == cls_id))
        if not len(indices):
            continue
        keep = nms(torch.from_numpy(rows[indices, :4].copy()),
                   torch.from_numpy(rows[indices, 4].copy()), iou).numpy()
        selected.extend(indices[keep].tolist())
    selected = np.asarray(selected, dtype=np.int64)
    selected = selected[np.argsort(-rows[selected, 4], kind='stable')[:max_detections]]
    return ({cls_id: rows[selected[rows[selected, 5] == cls_id]] for cls_id in range(num_classes)},
            [features[selected[rows[selected, 5] == cls_id]] for cls_id in range(num_classes)],
            selected)


class TiledInference:
    def __init__(self, opt):
        self.opt = opt
        for key in ('tile_fraction', 'tile_overlap', 'tile_nms_iou', 'tile_min_score',
                    'tile_max_side', 'tile_edge_margin'):
            value = float(getattr(opt, key))
            if not math.isfinite(value):
                raise ValueError('{} must be finite'.format(key))
        if not 0 < opt.tile_fraction <= 1 or not 0 <= opt.tile_overlap < 1:
            raise ValueError('Invalid tile fraction/overlap')
        if not 0 <= opt.tile_nms_iou <= 1 or not 0 <= opt.tile_min_score <= 1:
            raise ValueError('Invalid tile score/IoU threshold')
        if opt.tile_max_detections < 1 or opt.tile_edge_margin < 0 or opt.tile_max_side < 0:
            raise ValueError('Invalid tile detection limit/size/margin')
        self.last_stats = {}

    @torch.no_grad()
    def fuse(self, model, blob, image, full_rows, full_features, full_feature_map,
             full_heatmap=None, class_thresholds=None):
        opt = self.opt
        net_h, net_w = blob.shape[2:]
        windows = tile_windows(*image.shape[:2], opt.tile_fraction, opt.tile_overlap)
        row_parts = [full_rows[c] for c in range(opt.num_classes)]
        feat_parts = list(full_features)
        full_count = sum(len(part) for part in row_parts)
        tile_count = 0
        for window in windows:
            x, y, right, bottom = window
            crop = image[y:bottom, x:right]
            geometry = LetterboxGeometry.create(*crop.shape[:2], net_h, net_w)
            resized = geometry.preprocess(crop, net_h, net_w)
            tensor = np.ascontiguousarray(resized[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / 255.0
            tile_blob = torch.from_numpy(tensor).unsqueeze(0).to(blob.device)
            output = model(tile_blob)[-1]
            decoded, indices, _ = mot_decode(
                output['hm'].sigmoid(), output['wh'],
                reg=output['reg'] if opt.reg_offset else None,
                num_classes=opt.num_classes, cat_spec_wh=opt.cat_spec_wh, K=opt.K)
            rows = geometry.invert_boxes(decoded[0].cpu().numpy(), opt.down_ratio)
            rows, keep = filter_crop_boxes(rows, window, image.shape, opt.tile_min_score,
                                           opt.tile_max_side, opt.tile_edge_margin)
            if not len(rows):
                continue
            if opt.tile_embedding == 'full':
                features = sample_full_embeddings(rows, full_feature_map, image.shape,
                                                   (net_h, net_w), opt.down_ratio)
            else:
                normalized = F.normalize(output['id'], dim=1)
                features = _tranpose_and_gather_feat(normalized, indices)[0].cpu().numpy()[keep]
            row_parts.append(rows)
            feat_parts.append(features)
            tile_count += len(rows)
        rows = np.concatenate(row_parts, axis=0)
        features = np.concatenate(feat_parts, axis=0)
        detections, embeddings, selected = merge_detections(
            rows, features, opt.num_classes, image.shape, opt.tile_min_score,
            opt.tile_nms_iou, opt.tile_max_detections)
        self.last_stats = dict(crops=len(windows), full_candidates=full_count,
                               tile_candidates=tile_count, fused=len(selected),
                               tile_winners=int(np.sum(selected >= full_count)))
        return detections, embeddings
