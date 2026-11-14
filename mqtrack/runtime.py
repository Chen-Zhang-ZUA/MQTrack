"""Original tracking entry and output serialization."""
import os
import cv2
import numpy as np
import torch
from numpy import ndarray
from collections import defaultdict
from lib.tracker.multitracker import MCJDETracker
from lib.tracking_utils.log import logger
from lib.tracking_utils.timer import Timer
from lib.tracking_utils import visualization as vis
TRACK_SOURCE_CODES = {"unknown": -1, "high": 0, "recovery": 1, "mtc": 2}

def source_code(source):
    return TRACK_SOURCE_CODES.get(str(source), TRACK_SOURCE_CODES['unknown'])

def write_results_dict(file_name,
                       results_dict,
                       data_type,
                       num_classes=5,
                       cls_id_offset=0,
                       visdrone_format=False):
    """
    :param file_name:
    :param results_dict:
    :param data_type:
    :param num_classes:
    :return:
    """
    if data_type == 'mot':
        # save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
        if visdrone_format:
            save_format = '{frame},{id},{x1},{y1},{w},{h},{score},{cls_id},-1,-1\n'
        else:
            save_format = '{frame},{id},{x1},{y1},{w},{h},{score},{cls_id},1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    write_sources = any(
        len(entry) >= 5
        for cls_results in results_dict.values()
        for entry in cls_results)
    source_file_name = file_name + '.src'
    source_handle = open(source_file_name, 'w') if write_sources else None
    write_features = any(
        len(entry) >= 6
        for cls_results in results_dict.values()
        for entry in cls_results)
    feature_file_name = file_name + '.feat'
    feature_handle = open(feature_file_name, 'w') if write_features else None

    with open(file_name, 'w') as f:
        for cls_id in range(num_classes):  # process each object class
            cls_results = results_dict[cls_id]
            for entry in cls_results:
                frame_id, tlwhs, track_ids, scores = entry[:4]
                sources = entry[4] if len(entry) >= 5 else ['unknown'] * len(track_ids)
                features = entry[5] if len(entry) >= 6 else [None] * len(track_ids)
                if data_type == 'kitti':
                    frame_id -= 1

                for tlwh, track_id, score, source, feat in zip(tlwhs, track_ids, scores, sources, features):
                    if track_id < 0:
                        continue

                    x1, y1, w, h = tlwh
                    # x2, y2 = x1 + w, y1 + h
                    # line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                    line = save_format.format(frame=frame_id,
                                              id=track_id,
                                              x1=x1, y1=y1, w=w, h=h,
                                              score=score,  # detection score
                                              cls_id=cls_id + cls_id_offset)
                    f.write(line)
                    if source_handle is not None:
                        source_handle.write('{},{},{},{},{},{}\n'.format(
                            frame_id,
                            track_id,
                            cls_id + cls_id_offset,
                            source_code(source),
                            source,
                            score))
                    if feature_handle is not None and feat is not None:
                        feat_values = np.asarray(feat, dtype=np.float32).reshape(-1)
                        feature_handle.write('{},{},{}'.format(
                            frame_id,
                            track_id,
                            cls_id + cls_id_offset))
                        for value in feat_values:
                            feature_handle.write(',{:.6f}'.format(float(value)))
                        feature_handle.write('\n')

    if source_handle is not None:
        source_handle.close()
        logger.info('save result sources to {}'.format(source_file_name))
    if feature_handle is not None:
        feature_handle.close()
        logger.info('save result features to {}'.format(feature_file_name))

    logger.info('save results to {}'.format(file_name))

def eval_seq(opt,
             data_loader,
             data_type,
             result_f_name,
             save_dir=None,
             show_image=True,
             frame_rate=30,
             mode='track',
             cls_id_offset=0,
             visdrone_format=False):
    """
    :param opt:
    :param data_loader:
    :param data_type:
    :param result_f_name:
    :param save_dir:
    :param show_image:
    :param frame_rate:
    :param mode: track or detect
    :return:
    """
    if save_dir:
        mkdir_if_missing(save_dir)

    # tracker = JDETracker(opt, frame_rate)
    tracker = MCJDETracker(opt, frame_rate)

    timer = Timer()

    results_dict = defaultdict(list)

    frame_id = 0  # frame index
    for path, img, img0 in data_loader:
        if frame_id % 30 == 0 and frame_id != 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1.0 / max(1e-5, timer.average_time)))

        # --- run tracking
        blob = torch.from_numpy(img).unsqueeze(0).to(opt.device)

        if mode == 'track':  # process tracking
            # ----- track updates of each frame
            timer.tic()

            online_targets_dict = tracker.update_tracking(blob, img0)

            timer.toc()
            # -----

            # collect current frame's result
            online_tlwhs_dict = defaultdict(list)
            online_ids_dict = defaultdict(list)
            online_scores_dict = defaultdict(list)
            online_sources_dict = defaultdict(list)
            online_features_dict = defaultdict(list)
            for cls_id in range(opt.num_classes):  # process each class id
                online_targets = online_targets_dict[cls_id]
                for track in online_targets:
                    tlwh = track.tlwh
                    t_id = track.track_id
                    score = track.score
                    source = getattr(track, 'last_update_source', 'unknown')
                    feature = getattr(track, 'smooth_feat', None)
                    if tlwh[2] * tlwh[3] > opt.min_box_area:  # and not vertical:
                        online_tlwhs_dict[cls_id].append(tlwh)
                        online_ids_dict[cls_id].append(t_id)
                        online_scores_dict[cls_id].append(score)
                        online_sources_dict[cls_id].append(source)
                        if opt.save_reid_features:
                            online_features_dict[cls_id].append(feature)

            # collect result
            for cls_id in range(opt.num_classes):
                entry = (frame_id + 1,
                         online_tlwhs_dict[cls_id],
                         online_ids_dict[cls_id],
                         online_scores_dict[cls_id],
                         online_sources_dict[cls_id])
                if opt.save_reid_features:
                    entry = entry + (online_features_dict[cls_id],)
                results_dict[cls_id].append(entry)

            # draw track/detection
            if show_image or save_dir is not None:
                if frame_id > 0:
                    online_im: ndarray = vis.plot_tracks(image=img0,
                                                         tlwhs_dict=online_tlwhs_dict,
                                                         obj_ids_dict=online_ids_dict,
                                                         num_classes=opt.num_classes,
                                                         frame_id=frame_id,
                                                         fps=1.0 / timer.average_time)

        elif mode == 'detect':  # process detections
            timer.tic()

            # update detection results of this frame(or image)
            dets_dict = tracker.update_detection(blob, img0)

            timer.toc()

            # plot detection results
            if show_image or save_dir is not None:
                online_im = vis.plot_detects(image=img0,
                                             dets_dict=dets_dict,
                                             num_classes=opt.num_classes,
                                             frame_id=frame_id,
                                             fps=1.0 / max(1e-5, timer.average_time))
        else:
            print('[Err]: un-recognized mode.')

        if frame_id > 0:
            if show_image:
                cv2.imshow('online_im', online_im)
            if save_dir is not None:
                cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)

        # update frame id
        frame_id += 1

    if getattr(opt, 'birth_competition', False):
        logger.info('Birth competition: {}'.format(tracker.birth_competition_stats))

    # write track/detection results
    write_results_dict(result_f_name,
                       results_dict,
                       data_type,
                       num_classes=opt.num_classes,
                       cls_id_offset=cls_id_offset,
                       visdrone_format=visdrone_format)

    return frame_id, timer.average_time, timer.calls
