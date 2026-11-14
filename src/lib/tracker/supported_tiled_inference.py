"""Conservative crop supplementation backed by the unchanged full-frame heatmap."""

import math

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import box_iou

from lib.tracker.tiled_inference import LetterboxGeometry, TiledInference, merge_detections


def supported_mask(rows, full_rows, full_heatmap, image_shape, net_shape, stride,
                   class_thresholds, tile_score, support_score, class_ratio, duplicate_iou,
                   center_ratio=0.25):
    if not len(rows):
        return np.zeros(0, dtype=bool)
    if full_heatmap is None or full_heatmap.shape[0] != 1:
        raise ValueError('Supported tiling requires the full-frame probability heatmap')
    geometry = LetterboxGeometry.create(*image_shape[:2], *net_shape)
    centers = (rows[:, :2] + rows[:, 2:4]) * 0.5
    heat = F.max_pool2d(full_heatmap, 3, stride=1, padding=1)[0].detach().cpu().numpy()
    xs = np.floor((centers[:, 0] * geometry.scale_x + geometry.left) / stride)
    ys = np.floor((centers[:, 1] * geometry.scale_y + geometry.top) / stride)
    xs = np.clip(xs, 0, heat.shape[2] - 1).astype(np.int64)
    ys = np.clip(ys, 0, heat.shape[1] - 1).astype(np.int64)
    classes = rows[:, 5].astype(np.int64)
    support = heat[classes, ys, xs]
    strongest_class = heat[:, ys, xs].max(axis=0)
    keep = ((rows[:, 4] >= tile_score) & (support >= support_score)
            & (support >= class_ratio * strongest_class))
    for cls_id, threshold in enumerate(class_thresholds):
        candidates = np.flatnonzero(keep & (classes == cls_id))
        reliable = full_rows[cls_id].copy()
        h, w = image_shape[:2]
        reliable[:, [0, 2]] = np.clip(reliable[:, [0, 2]], 0, w)
        reliable[:, [1, 3]] = np.clip(reliable[:, [1, 3]], 0, h)
        reliable = reliable[(reliable[:, 4] > threshold)
                            & np.isfinite(reliable).all(axis=1)
                            & (reliable[:, 2] > reliable[:, 0]) & (reliable[:, 3] > reliable[:, 1])]
        if not len(candidates) or not len(reliable):
            continue
        overlaps = box_iou(torch.from_numpy(rows[candidates, :4].copy()),
                           torch.from_numpy(reliable[:, :4].copy())).numpy()
        keep[candidates] &= overlaps.max(axis=1) <= duplicate_iou
        if center_ratio > 0:
            full_centers = (reliable[:, :2] + reliable[:, 2:4]) * 0.5
            full_size = reliable[:, 2:4] - reliable[:, :2]
            near_center = (np.abs(centers[candidates, None] - full_centers[None])
                           <= center_ratio * full_size[None]).all(axis=2)
            keep[candidates] &= ~near_center.any(axis=1)
    return keep


class SupportedTiledInference(TiledInference):
    def __init__(self, opt):
        super().__init__(opt)
        for name in ('tile_support_tile_score', 'tile_support_full_score',
                     'tile_support_class_ratio', 'tile_support_duplicate_iou', 'tile_support_center_ratio'):
            value = getattr(opt, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('{} must be finite and in [0, 1]'.format(name))

    @torch.no_grad()
    def fuse(self, model, blob, image, full_rows, full_features, full_feature_map,
             full_heatmap=None, class_thresholds=None):
        # The first merge supplies crop proposals, not the final replacements.
        fused, embeddings = super().fuse(model, blob, image, full_rows, full_features, full_feature_map)
        opt = self.opt
        proposals = np.concatenate([fused[c] for c in range(opt.num_classes)], axis=0)
        proposal_features = np.concatenate(embeddings, axis=0)
        if class_thresholds is None:
            class_thresholds = [opt.conf_thres] * opt.num_classes
        keep = supported_mask(proposals, full_rows, full_heatmap, image.shape, blob.shape[2:],
                              opt.down_ratio, class_thresholds, opt.tile_support_tile_score,
                              opt.tile_support_full_score, opt.tile_support_class_ratio,
                              opt.tile_support_duplicate_iou, opt.tile_support_center_ratio)
        rows = np.concatenate([*[full_rows[c] for c in range(opt.num_classes)], proposals[keep]], axis=0)
        features = np.concatenate([*full_features, proposal_features[keep]], axis=0)
        result, features, selected = merge_detections(
            rows, features, opt.num_classes, image.shape, opt.tile_min_score,
            opt.tile_nms_iou, opt.tile_max_detections)
        self.last_stats.update(raw_fused=self.last_stats['fused'],
                               supported_additions=int(keep.sum()),
                               supported_rejected=int(len(keep) - keep.sum()), fused=len(selected))
        self.last_stats['tile_winners'] = int(np.sum(selected >= sum(len(full_rows[c])
                                                                  for c in range(opt.num_classes))))
        return result, features
