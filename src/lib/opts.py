from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os


class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # basic experiment setting
        self.parser.add_argument('--task', default='mot', help='mot')
        self.parser.add_argument('--dataset', default='jde', help='jde')
        self.parser.add_argument('--exp_id', default='default')
        self.parser.add_argument('--test', action='store_true')
        self.parser.add_argument('--load_model',
                                 default='',
                                 help='path to pretrained or trained model')
        self.parser.add_argument('--resume',
                                 action='store_true',
                                 help='resume an experiment. '
                                      'Reloaded the optimizer parameter and '
                                      'set load_model to model_last.pth '
                                      'in the exp dir if load_model is empty.')

        # system
        self.parser.add_argument('--gpus',
                                 default='0',
                                 help='-1 for CPU, use comma for multiple gpus')
        self.parser.add_argument('--num_workers',
                                 type=int,
                                 default=4,  # 8, 6, 4
                                 help='dataloader threads. 0 for single-thread.')
        self.parser.add_argument('--not_cuda_benchmark', action='store_true',
                                 help='disable when the input size is not fixed.')
        self.parser.add_argument('--seed', type=int, default=317,
                                 help='random seed')  # from CornerNet
        self.parser.add_argument('--gen-scale',
                                 type=bool,
                                 default=True,
                                 help='Whether to generate multi-scales')
        self.parser.add_argument('--is_debug',
                                 type=bool,
                                 default=False,  # 是否使用多线程加载数据, default: False
                                 help='whether in debug mode or not')  # debug模式下只能使用单进程

        # log
        self.parser.add_argument('--print_iter', type=int, default=0,
                                 help='disable progress bar and print to screen.')
        self.parser.add_argument('--hide_data_time', action='store_true',
                                 help='not display time during training.')
        self.parser.add_argument('--save_all', action='store_true',
                                 help='save model to disk every 5 epochs.')
        self.parser.add_argument('--metric', default='loss',
                                 help='main metric to save best model')
        self.parser.add_argument('--vis_thresh', type=float, default=0.5,
                                 help='visualization threshold.')

        # model: backbone and so on...
        self.parser.add_argument('--arch',
                                 default='resdcn_18',
                                 help='model architecture. Currently tested'
                                      'resdcn_18 |resdcn_34 | resdcn_50 | resfpndcn_34 |'
                                      'dla_34 | hrnet_32 | hrnet_18 | cspdarknet_53')
        self.parser.add_argument('--head_conv',
                                 type=int,
                                 default=-1,
                                 help='conv layer channels for output head'
                                      '0 for no conv layer'
                                      '-1 for default setting: '
                                      '256 for resnets and 256 for dla.')
        self.parser.add_argument('--down_ratio',
                                 type=int,
                                 default=4,  # 输出特征图的下采样率 H=H_image/4 and W=W_image/4
                                 help='output stride. Currently only supports 4.')

        # input
        self.parser.add_argument('--input_res',
                                 type=int,
                                 default=-1,
                                 help='input height and width. -1 for default from '
                                      'dataset. Will be overriden by input_h | input_w')
        self.parser.add_argument('--input_h',
                                 type=int,
                                 default=-1,
                                 help='input height. -1 for default from dataset.')
        self.parser.add_argument('--input_w',
                                 type=int,
                                 default=-1,
                                 help='input width. -1 for default from dataset.')

        # train
        self.parser.add_argument('--lr',
                                 type=float,
                                 default=7e-5,  # 1e-4, 7e-5, 5e-5, 3e-5
                                 help='learning rate for batch size 32.')
        self.parser.add_argument('--lr_step',
                                 type=str,
                                 default='10,20',  # 20,27
                                 help='drop learning rate by 10.')
        self.parser.add_argument('--num_epochs',
                                 type=int,
                                 default=30,  # 30, 10, 3, 1
                                 help='total training epochs.')
        self.parser.add_argument('--batch-size',
                                 type=int,
                                 default=10,  # 18, 16, 14, 12, 10, 8, 4
                                 help='batch size')
        self.parser.add_argument('--grad_accum_steps',
                                 type=int,
                                 default=1,
                                 help='gradient accumulation steps; effective batch size = batch_size * grad_accum_steps')
        self.parser.add_argument('--freeze_backbone',
                                 action='store_true',
                                 help='freeze backbone and neck modules during training; useful for conservative head-only fine-tuning')
        self.parser.add_argument('--freeze_id_head',
                                 action='store_true',
                                 help='freeze the ReID embedding head during training')
        self.parser.add_argument('--train_only_hm_final',
                                 action='store_true',
                                 help='train only the final heatmap conv layer hm.2; all other model parameters are frozen')
        self.parser.add_argument('--master_batch_size', type=int, default=-1,
                                 help='batch size on the master gpu.')
        self.parser.add_argument('--num_iters', type=int, default=-1,
                                 help='default: #samples / batch_size.')
        self.parser.add_argument('--val_intervals', type=int, default=10,
                                 help='number of epochs to run validation.')
        self.parser.add_argument('--trainval',
                                 action='store_true',
                                 help='include validation in training and '
                                      'test on test set')

        # test
        self.parser.add_argument('--K',
                                 type=int,
                                 default=200,  # 128
                                 help='max number of output objects.')  # 一张图输出检测目标最大数量
        self.parser.add_argument('--not_prefetch_test',
                                 action='store_true',
                                 help='not use parallal data pre-processing.')
        self.parser.add_argument('--fix_res',
                                 action='store_true',
                                 help='fix testing resolution or keep '
                                      'the original resolution')
        self.parser.add_argument('--keep_res',
                                 action='store_true',
                                 help='keep the original resolution'
                                      ' during validation.')
        # tracking
        self.parser.add_argument(
            '--test_mot16', default=False, help='test mot16')
        self.parser.add_argument(
            '--val_mot15', default=False, help='val mot15')
        self.parser.add_argument(
            '--test_mot15', default=False, help='test mot15')
        self.parser.add_argument(
            '--val_mot16', default=False, help='val mot16 or mot15')
        self.parser.add_argument(
            '--test_mot17', default=False, help='test mot17')
        self.parser.add_argument(
            '--val_mot17', default=False, help='val mot17')
        self.parser.add_argument(
            '--val_mot20', default=False, help='val mot20')
        self.parser.add_argument(
            '--test_mot20', default=False, help='test mot20')
        self.parser.add_argument(
            '--val_visdrone', action='store_true', help='run VisDrone2019-MOT val split')
        self.parser.add_argument(
            '--test_visdrone', action='store_true', help='run VisDrone2019-MOT test-dev split')
        self.parser.add_argument(
            '--val_uavdt', action='store_true', help='run UAVDT-MOT sequences')
        self.parser.add_argument(
            '--test_uavdt', action='store_true', help='run official UAVDT-MOT test sequences')
        self.parser.add_argument(
            '--uavdt-seqs',
            type=str,
            default='',
            help='optional comma-separated UAVDT sequence names, or a txt file with one sequence per line')
        self.parser.add_argument(
            '--conf_thres',
            type=float,
            default=0.4,  # 0.6, 0.4
            help='confidence thresh for tracking')  # heat-map置信度阈值
        self.parser.add_argument('--class-conf-profile',
                                 type=str,
                                 default='none',
                                 help='optional class-adaptive tracking confidence profile')
        self.parser.add_argument('--class-conf-thresh-file',
                                 type=str,
                                 default=None,
                                 help='optional CSV/JSON class_id -> tracking confidence threshold')
        self.parser.add_argument('--save-reid-features',
                                 action='store_true',
                                 help='save per-output ReID features to a .feat sidecar for ReID-aware Q-HIT')
        self.parser.add_argument('--tile-inference', action='store_true',
                                 help='fuse full-frame and overlapping crop detections')
        self.parser.add_argument('--tile-fraction', type=float, default=0.6,
                                 help='crop width/height relative to the image; 1 means NMS only')
        self.parser.add_argument('--tile-overlap', type=float, default=0.25)
        self.parser.add_argument('--tile-nms-iou', type=float, default=0.5)
        self.parser.add_argument('--tile-min-score', type=float, default=0.10)
        self.parser.add_argument('--tile-max-side', type=float, default=128,
                                 help='max original-image box side for crop detections; 0 disables')
        self.parser.add_argument('--tile-edge-margin', type=float, default=2,
                                 help='reject detections touching internal crop boundaries')
        self.parser.add_argument('--tile-max-detections', type=int, default=1000)
        self.parser.add_argument('--tile-embedding', choices=['full', 'native'], default='full',
                                 help='sample full-frame ReID features, or use crop features')
        self.parser.add_argument('--tile-supported', action='store_true',
                                 help='only supplement missed detections backed by full-frame class heatmaps')
        self.parser.add_argument('--tile-support-tile-score', type=float, default=0.75)
        self.parser.add_argument('--tile-support-full-score', type=float, default=0.15)
        self.parser.add_argument('--tile-support-class-ratio', type=float, default=0.80)
        self.parser.add_argument('--tile-support-duplicate-iou', type=float, default=0.30)
        self.parser.add_argument('--tile-support-center-ratio', type=float, default=0.25)
        self.parser.add_argument('--det_thres',
                                 type=float,
                                 default=0.3,
                                 help='confidence thresh for detection')
        self.parser.add_argument('--nms_thres',
                                 type=float,
                                 default=0.4,
                                 help='iou thresh for nms')
        self.parser.add_argument('--track_buffer',
                                 type=int,
                                 default=30,  # 30
                                 help='tracking buffer')
        self.parser.add_argument('--min-box-area',
                                 type=float,
                                 default=100,
                                 help='filter out tiny boxes')

        # UAV tracking innovation switches.
        self.parser.add_argument('--uav-gmc',
                                 action='store_true',
                                 help='enable UAV global motion compensation before association')
        self.parser.add_argument('--uav-gmc-mode',
                                 type=str,
                                 default='translation',
                                 choices=['translation', 'similarity', 'affine'],
                                 help='global motion model used by UAV-GMC')
        self.parser.add_argument('--uav-gmc-max-features',
                                 type=int,
                                 default=600,
                                 help='max Shi-Tomasi corners for UAV-GMC optical flow')
        self.parser.add_argument('--uav-gmc-quality-level',
                                 type=float,
                                 default=0.01,
                                 help='corner quality level for UAV-GMC')
        self.parser.add_argument('--uav-gmc-min-distance',
                                 type=float,
                                 default=8.0,
                                 help='minimum distance between UAV-GMC corners')
        self.parser.add_argument('--uav-gmc-ransac-thresh',
                                 type=float,
                                 default=3.0,
                                 help='RANSAC reprojection threshold for UAV-GMC')
        self.parser.add_argument('--uav-gmc-min-inliers',
                                 type=int,
                                 default=24,
                                 help='minimum inlier count required to trust UAV-GMC')
        self.parser.add_argument('--uav-gmc-max-translation-ratio',
                                 type=float,
                                 default=0.18,
                                 help='max frame-to-frame global translation as image diagonal ratio')
        self.parser.add_argument('--uav-gmc-max-scale-delta',
                                 type=float,
                                 default=0.08,
                                 help='max allowed global scale deviation for UAV-GMC')
        self.parser.add_argument('--uav-gmc-max-rotation-deg',
                                 type=float,
                                 default=8.0,
                                 help='max allowed global rotation for UAV-GMC')
        self.parser.add_argument('--uav-gmc-downscale-max-side',
                                 type=int,
                                 default=720,
                                 help='downscale max side for fast UAV-GMC estimation; <=0 disables')
        self.parser.add_argument('--amot-lite',
                                 action='store_true',
                                 help='enable AMOT-lite AMC association and MTC continuation')
        self.parser.add_argument('--birth-competition',
                                 action='store_true',
                                 help='opt-in VisDrone car/van cross-class birth competition')
        self.parser.add_argument('--birth-confirm-hits',
                                 type=int,
                                 default=2,
                                 help='observed hits needed for ambiguous car/van birth confirmation')
        self.parser.add_argument('--amot-amc-weight',
                                 type=float,
                                 default=0.5,
                                 help='weight of AMC spatial consistency in first association')
        self.parser.add_argument('--amot-amc-sigma',
                                 type=float,
                                 default=8.0,
                                 help='spatial sigma for AMOT-lite AMC cost')
        self.parser.add_argument('--sa-amc',
                                 action='store_true',
                                 help='enable scale-aware AMC association for UAV small objects')
        self.parser.add_argument('--sage-profile',
                                 type=str,
                                 default='none',
                                 choices=['none', 'visdrone5-qc', 'uavdt-strict'],
                                 help='quality-control profile for SAGE association and MTC continuation')
        self.parser.add_argument('--sage-match-thresh',
                                 type=float,
                                 default=0.7,
                                 help='fallback first-association threshold used by SAGE')
        self.parser.add_argument('--sage-mtc-min-quality',
                                 type=float,
                                 default=0.0,
                                 help='fallback minimum track quality for SAGE MTC continuation')
        self.parser.add_argument('--sage-mtc-min-score',
                                 type=float,
                                 default=0.0,
                                 help='fallback minimum track score for SAGE MTC continuation')
        self.parser.add_argument('--sa-amc-norm-sigma',
                                 type=float,
                                 default=0.75,
                                 help='normalized center-distance sigma for SA-AMC')
        self.parser.add_argument('--sa-amc-min-scale',
                                 type=float,
                                 default=16.0,
                                 help='minimum bbox scale used by SA-AMC normalization')
        self.parser.add_argument('--sa-amc-scale-weight',
                                 type=float,
                                 default=0.25,
                                 help='weight of scale-continuity cost in SA-AMC')
        self.parser.add_argument('--sa-amc-scale-sigma',
                                 type=float,
                                 default=0.6,
                                 help='log width-height sigma for SA-AMC scale continuity')
        self.parser.add_argument('--sa-amc-app-min-weight',
                                 type=float,
                                 default=0.45,
                                 help='minimum appearance weight for small targets in SA-AMC fusion')
        self.parser.add_argument('--sa-amc-app-max-weight',
                                 type=float,
                                 default=1.0,
                                 help='maximum appearance weight for large targets in SA-AMC fusion')
        self.parser.add_argument('--sa-amc-app-scale-mid',
                                 type=float,
                                 default=52.0,
                                 help='bbox scale where SA-AMC appearance weight reaches the midpoint')
        self.parser.add_argument('--sa-amc-app-scale-gamma',
                                 type=float,
                                 default=16.0,
                                 help='smoothness of SA-AMC scale-aware appearance weighting')
        self.parser.add_argument('--amot-mtc',
                                 action='store_true',
                                 help='enable AMOT-lite motion-aware track continuation')
        self.parser.add_argument('--amot-mtc-center-thresh',
                                 type=float,
                                 default=16.0,
                                 help='max center distance for MTC continuation in original pixels')
        self.parser.add_argument('--amot-mtc-min-history',
                                 type=int,
                                 default=3,
                                 help='min consecutive observations before MTC can continue a track')
        self.parser.add_argument('--stcmot-tebm',
                                 action='store_true',
                                 help='enable STCMOT-lite temporal embedding boosting')
        self.parser.add_argument('--stcmot-tebm-alpha',
                                 type=float,
                                 default=0.2,
                                 help='blend weight for temporal embedding boosting')
        self.parser.add_argument('--stcmot-tdrm',
                                 action='store_true',
                                 help='enable STCMOT-lite low-score temporal detection recovery')
        self.parser.add_argument('--stcmot-tdrm-low-conf',
                                 type=float,
                                 default=0.1,
                                 help='low score threshold for TDRM recovery detections')
        self.parser.add_argument('--stcmot-tdrm-match-thresh',
                                 type=float,
                                 default=0.6,
                                 help='matching threshold for TDRM recovery detections')
        self.parser.add_argument('--sca-tdrm',
                                 action='store_true',
                                 help='enable scale-class-aware temporal detection recovery')
        self.parser.add_argument('--sca-tdrm-profile',
                                 type=str,
                                 default='visdrone',
                                 choices=['none', 'visdrone', 'visdrone-qc', 'visdrone-v3',
                                          'mqtrack-qc', 'mqtrack-diagnostic', 'uavdt-strict'],
                                 help='class-aware SCA-TDRM parameter profile')
        self.parser.add_argument('--sca-tdrm-low-conf',
                                 type=float,
                                 default=0.1,
                                 help='fallback low score threshold for SCA-TDRM candidates')
        self.parser.add_argument('--sca-tdrm-match-thresh',
                                 type=float,
                                 default=0.52,
                                 help='fallback matching threshold for SCA-TDRM recovery')
        self.parser.add_argument('--sca-tdrm-min-scale',
                                 type=float,
                                 default=16.0,
                                 help='minimum bbox scale used by SCA-TDRM normalization')
        self.parser.add_argument('--sca-tdrm-center-gate',
                                 type=float,
                                 default=1.3,
                                 help='fallback scale-normalized center gate for SCA-TDRM')
        self.parser.add_argument('--sca-tdrm-scale-gate',
                                 type=float,
                                 default=0.7,
                                 help='fallback absolute log-area gate for SCA-TDRM')
        self.parser.add_argument('--sca-tdrm-min-track-quality',
                                 type=float,
                                 default=0.25,
                                 help='fallback minimum track quality for SCA-TDRM recovery')
        self.parser.add_argument('--sca-tdrm-quality-bonus',
                                 type=float,
                                 default=0.04,
                                 help='cost reduction for high-quality tracks in SCA-TDRM')
        self.parser.add_argument('--sca-tdrm-max-recovery-streak',
                                 type=int,
                                 default=0,
                                 help='maximum consecutive low-score recoveries for one track; 0 disables this gate')
        self.parser.add_argument('--sca-tdrm-confirm-window',
                                 type=int,
                                 default=6,
                                 help='soft confirmation window for low-score recovered tracks')
        self.parser.add_argument('--sca-tdrm-pending-penalty',
                                 type=float,
                                 default=0.0,
                                 help='extra cost for tracks still pending high-score confirmation')
        self.parser.add_argument('--sca-tdrm-iou-weight',
                                 type=float,
                                 default=0.30)
        self.parser.add_argument('--sca-tdrm-emb-weight',
                                 type=float,
                                 default=0.30)
        self.parser.add_argument('--sca-tdrm-center-weight',
                                 type=float,
                                 default=0.25)
        self.parser.add_argument('--sca-tdrm-scale-weight',
                                 type=float,
                                 default=0.15)
        self.parser.add_argument('--lost-track-recovery',
                                 action='store_true',
                                 help='recover short-gap lost tracks with remaining low-score detections')
        self.parser.add_argument('--lost-track-recovery-max-gap',
                                 type=int,
                                 default=8,
                                 help='maximum missing-frame gap for lost-track recovery')
        self.parser.add_argument('--lost-track-recovery-min-score',
                                 type=float,
                                 default=0.18,
                                 help='minimum low detection score for lost-track recovery')
        self.parser.add_argument('--lost-track-recovery-match-thresh',
                                 type=float,
                                 default=0.30,
                                 help='matching threshold for lost-track recovery')
        self.parser.add_argument('--lost-track-recovery-min-quality',
                                 type=float,
                                 default=0.40,
                                 help='minimum track quality for lost-track recovery')

        # 测试阶段的输入数据模式: video or image dir
        self.parser.add_argument('--input-mode',
                                 type=str,
                                 default='video',  # video or image_dir or img_path_list_txt
                                 help='input data type(video or image dir)')

        # 输入的video文件路径
        self.parser.add_argument('--input-video',
                                 type=str,
                                 default='../videos/uav_339.mp4',
                                 help='path to the input video')

        # 输入的image目录
        self.parser.add_argument('--input-img',
                                 type=str,
                                 default='/users/duanyou/c5/all_pretrain/test.txt',  # ../images/
                                 help='path to the input image directory or image file list(.txt)')

        self.parser.add_argument('--output-format',
                                 type=str,
                                 default='video',
                                 help='video or text')
        self.parser.add_argument('--output-root',
                                 type=str,
                                 default='../results',
                                 help='expected output root path')

        # mot: 选择数据集的配置文件
        # self.parser.add_argument('--data_cfg', type=str,
        #                          default='../src/lib/cfg/visdrone.json',  # 'mcmot_det.json', 'visdrone.json'
        #                          help='load data from cfg')
        self.parser.add_argument('--data_cfg', type=str,
                                 default=os.path.join(os.path.dirname(__file__), 'cfg', 'visdrone.json'),
                                 help='load data from cfg')
        self.parser.add_argument('--data_dir',
                                 type=str,
                                 default='./datasets')

        # loss
        self.parser.add_argument('--mse_loss',  # default: false
                                 action='store_true',
                                 help='use mse loss or focal loss to train '
                                      'keypoint heatmaps.')
        self.parser.add_argument('--reg_loss',
                                 default='l1',
                                 help='regression loss: sl1 | l1 | l2')  # sl1: smooth L1 loss
        self.parser.add_argument('--hm_weight',
                                 type=float,
                                 default=1,
                                 help='loss weight for keypoint heatmaps.')
        self.parser.add_argument('--off_weight',
                                 type=float,
                                 default=1,
                                 help='loss weight for keypoint local offsets.')
        self.parser.add_argument('--wh_weight',
                                 type=float,
                                 default=0.1,
                                 help='loss weight for bounding box size.')
        self.parser.add_argument('--id_loss',
                                 default='ce',
                                 help='reid loss: ce | triplet')
        self.parser.add_argument('--id_weight',
                                 type=float,
                                 default=1,  # 0 for detection only and 1 for detection and re-id
                                 help='loss weight for id')  # ReID feature extraction or not
        self.parser.add_argument('--reid_dim',
                                 type=int,
                                 default=128,  # 128, 256, 512
                                 help='feature dim for reid')
        self.parser.add_argument('--input-wh',
                                 type=tuple,
                                 default=(1088, 608),  # (768, 448) or (1088, 608)
                                 help='net input resplution')
        self.parser.add_argument('--multi-scale',
                                 type=bool,
                                 default=True,
                                 help='Whether to use multi-scale training or not')

        # ----------------------1~10 object classes are what we need
        # pedestrian      (1),  --> 0
        # people          (2),  --> 1
        # bicycle         (3),  --> 2
        # car             (4),  --> 3
        # van             (5),  --> 4
        # truck           (6),  --> 5
        # tricycle        (7),  --> 6
        # awning-tricycle (8),  --> 7
        # bus             (9),  --> 8
        # motor           (10), --> 9
        # ----------------------

        # others          (11)
        self.parser.add_argument('--reid_cls_ids',
                                 default='0,1,2,3,4,5,6,7,8,9',  # '0,1,2,3,4' or '0,1,2,3,4,5,6,7,8,9'
                                 help='')  # the object classes need to do reid

        self.parser.add_argument('--norm_wh', action='store_true',
                                 help=r'L1(\hat(y) / y, 1) or L1(\hat(y), y)')
        self.parser.add_argument('--dense_wh', action='store_true',
                                 help='apply weighted regression near center or '
                                      'just apply regression on center point.')
        self.parser.add_argument('--cat_spec_wh',
                                 action='store_true',
                                 help='category specific bounding box size.')
        self.parser.add_argument('--not_reg_offset',
                                 action='store_true',
                                 help='not regress local offset.')

    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        if opt.tile_supported and not opt.tile_inference:
            self.parser.error('--tile-supported requires --tile-inference')

        opt.gpus_str = opt.gpus
        opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
        # opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >= 0 else [-1]
        # print("opt.gpus", opt.gpus)
        opt.lr_step = [int(i) for i in opt.lr_step.split(',')]

        opt.fix_res = not opt.keep_res
        print('Fix size testing.' if opt.fix_res else 'Keep resolution testing.')

        opt.reg_offset = not opt.not_reg_offset

        if opt.head_conv == -1:  # init default head_conv
            opt.head_conv = 256 if 'dla' in opt.arch else 256
        opt.pad = 31
        opt.num_stacks = 1

        if opt.trainval:
            opt.val_intervals = 100000000

        if opt.master_batch_size == -1:
            opt.master_batch_size = opt.batch_size // len(opt.gpus)
        rest_batch_size = (opt.batch_size - opt.master_batch_size)
        opt.chunk_sizes = [opt.master_batch_size]
        for i in range(len(opt.gpus) - 1):
            slave_chunk_size = rest_batch_size // (len(opt.gpus) - 1)
            if i < rest_batch_size % (len(opt.gpus) - 1):
                slave_chunk_size += 1
            opt.chunk_sizes.append(slave_chunk_size)
        print('training chunk_sizes:', opt.chunk_sizes)

        opt.root_dir = os.path.join(os.path.dirname(__file__), '..', '..')
        opt.root_dir = os.path.abspath(opt.root_dir)
        if not os.path.isabs(opt.data_cfg):
            opt.data_cfg = os.path.abspath(os.path.join(opt.root_dir, opt.data_cfg))
        if opt.load_model != '' and not os.path.isabs(opt.load_model):
            opt.load_model = os.path.abspath(os.path.join(opt.root_dir, opt.load_model))
        opt.exp_dir = os.path.join(opt.root_dir, 'exp', opt.task)
        opt.save_dir = os.path.join(opt.exp_dir, opt.exp_id)
        opt.debug_dir = os.path.join(opt.save_dir, 'debug')
        print('The output will be saved to ', opt.save_dir)

        if opt.resume and opt.load_model == '':
            model_path = opt.save_dir[:-4] if opt.save_dir.endswith('TEST') \
                else opt.save_dir
            opt.load_model = os.path.join(model_path, 'model_last.pth')
        return opt

    def update_dataset_info_and_set_heads(self, opt, dataset):
        """
        :param opt:
        :param dataset:
        :return:
        """
        input_h, input_w = dataset.default_input_wh  # 图片的高和宽
        opt.mean, opt.std = dataset.mean, dataset.std  # 均值 方差
        opt.num_classes = dataset.num_classes  # 类别数

        for reid_id in opt.reid_cls_ids.split(','):
            if int(reid_id) > opt.num_classes - 1:
                print('[Err]: configuration conflict of reid_cls_ids and num_classes!')
                return

        # input_h(w): opt.input_h overrides opt.input_res overrides dataset default
        input_h = opt.input_res if opt.input_res > 0 else input_h
        input_w = opt.input_res if opt.input_res > 0 else input_w
        opt.input_h = opt.input_h if opt.input_h > 0 else input_h
        opt.input_w = opt.input_w if opt.input_w > 0 else input_w
        opt.output_h = opt.input_h // opt.down_ratio  # 输出特征图的宽高
        opt.output_w = opt.input_w // opt.down_ratio
        opt.input_res = max(opt.input_h, opt.input_w)
        opt.output_res = max(opt.output_h, opt.output_w)

        if opt.task == 'mot':
            opt.heads = {'hm': opt.num_classes,
                         'wh': 2 if not opt.cat_spec_wh else 2 * opt.num_classes,
                         'id': opt.reid_dim}
            if opt.reg_offset:
                opt.heads.update({'reg': 2})

            # opt.nID = dataset.nID

            # @even: 用nID_dict取代nID
            if opt.id_weight > 0:
                opt.nID_dict = dataset.nID_dict

            # opt.img_size = (640, 320)  # (1088, 608)
        else:
            assert 0, 'task not defined!'

        print('heads: ', opt.heads)
        return opt

    def init(self, args=''):
        opt = self.parse(args)

        default_dataset_info = {
            'mot': {'default_input_wh': [opt.input_wh[1], opt.input_wh[0]],  # [608, 1088], [320, 640]
                    'num_classes': len(opt.reid_cls_ids.split(',')),  # 1
                    'mean': [0.408, 0.447, 0.470],
                    'std': [0.289, 0.274, 0.278],
                    'dataset': 'jde',
                    'nID': 14455,
                    'nID_dict': {}},
        }

        class Struct:
            def __init__(self, entries):
                for k, v in entries.items():
                    self.__setattr__(k, v)

        h_w = default_dataset_info[opt.task]['default_input_wh']
        opt.img_size = (h_w[1], h_w[0])
        print('Net input image size: {:d}×{:d}'.format(h_w[1], h_w[0]))

        dataset = Struct(default_dataset_info[opt.task])
        opt.dataset = dataset.dataset
        opt = self.update_dataset_info_and_set_heads(opt, dataset)
        opt.img_size = (opt.input_w, opt.input_h)
        print('Final net input image size: {:d}x{:d}'.format(opt.img_size[0], opt.img_size[1]))

        return opt
