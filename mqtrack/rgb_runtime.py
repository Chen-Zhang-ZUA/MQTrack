"""Original VT-MOT-UAV RGB sequence export."""
from configparser import ConfigParser
import cv2
import numpy as np
import torch
from lib.datasets.dataset.jde import letterbox
from .runtime import source_code

def run_sequence(seq, files, source, tracker, path):
    info = ConfigParser()
    with (source / 'test' / seq['name'] / 'seqinfo.ini').open(encoding='utf-8-sig') as handle:
        info.read_file(handle)
    fps = info['Sequence'].getfloat('frameRate')
    tracker.reset()
    tracker.buffer_size = tracker.max_time_lost = int(fps / 30 * tracker.opt.track_buffer)
    width, height = seq['size']
    ids, counts = {}, [0, 0]
    with path.open('w', encoding='ascii') as tracks, \
            path.with_suffix('.txt.src').open('w', encoding='ascii') as sources, \
            path.with_suffix('.txt.feat').open('w', encoding='ascii') as features:
        for frame, filename in enumerate(files, 1):
            frame_path = source / seq['image_dir'] / filename
            bgr = cv2.imread(str(frame_path))
            if bgr is None or bgr.shape[:2] != (height, width):
                raise ValueError(f'Cannot read frame or image size changed: {frame_path}')
            image, *_ = letterbox(bgr, height=608, width=1088)
            rgb = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / 255.
            active = tracker.update_tracking(torch.from_numpy(rgb).unsqueeze(0).to(tracker.opt.device), bgr)
            for cls, targets in active.items():
                for track in targets:
                    x, y, w, h = map(float, track.tlwh)
                    score = float(track.score)
                    if not np.isfinite([x, y, w, h, score]).all():
                        raise ValueError(f'{frame_path}: non-finite track')
                    if w <= 0 or h <= 0 or w * h <= tracker.opt.min_box_area:
                        continue
                    xx, yy = np.clip([x, x+w], 0, width), np.clip([y, y+h], 0, height)
                    if xx[1] <= xx[0] or yy[1] <= yy[0]:
                        continue
                    tid = ids.setdefault((cls, track.track_id), len(ids) + 1)
                    feat = track.smooth_feat
                    if feat.shape != (128,) or not np.isfinite(feat).all():
                        raise ValueError(f'{frame_path}: invalid track ReID feature')
                    tracks.write(f'{frame},{tid},{xx[0]:.6f},{yy[0]:.6f},{xx[1]-xx[0]:.6f},'
                                 f'{yy[1]-yy[0]:.6f},{score:.8f},{cls+1},1,-1\n')
                    code = source_code(track.last_update_source)
                    sources.write(f'{frame},{tid},{cls+1},{code}\n')
                    features.write(f'{frame},{tid},{cls+1},' + ','.join(f'{v:.6f}' for v in feat) + '\n')
                    counts[cls] += 1
    return {'sequence': seq['name'], 'frames': len(files), 'fps_from_seqinfo': fps,
            'online_by_class': counts, 'online_rows': sum(counts)}
