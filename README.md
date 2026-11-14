# MQTrack: Scale-Aware Association and Quality-Guided Tracklet Refinement for UAV Multi-Class Multi-Object Tracking

<p align="center">
  Yu Chen, Chen Zhang (Student Member, IEEE), Zhongliang Deng (Fellow, IEEE),<br>
  Zhaoyu Liu, Xinling Wen, Yuxin Qin, Shuai Huo
</p>

<p align="center">
  School of Electronics and Information,<br>
  Zhengzhou University of Aeronautics, Zhengzhou 450046, China
</p>

The paper has not yet been published; please stay tuned</em>.

## Overview

Scale variation, occlusion, and camera motion complicate association and trajectory maintenance in aerial multi-class multi-object tracking. We present MQTrack, which coordinates scale-aware association, temporal recovery, and quality-guided refinement using dataset-trained detection and ReID representations that remain fixed during tracking inference. Scale-Aware Guided Embedding Association combines bidirectional errors relative to object size, size consistency, and scale-conditioned appearance fusion. Category-Aware Spatio-Temporal Recovery combines temporal appearance context, constraints for each category, and track reliability for short-term recovery. Quality-Guided Tracklet Refinement conditions offline linking, filtering, and interpolation on quality, with recovery-source confirmation when enabled. Online camera-motion compensation is also included. Compared with the corresponding ablation baselines, MQTrack improves MOTA and IDF1 by 3.8 and 10.6 percentage points on VisDrone2019-MOT test-dev under the five-class protocol, 8.4 and 3.6 points on UAVDT test, and 2.8 and 5.5 points on VT-MOT-UAV test, respectively. Grouped ablations across all three benchmarks and controlled refinement comparisons demonstrate the complementary contributions of the proposed components, while revealing dataset-dependent performance variations.

## Framework

<p align="center">
  <img src="images/fig.png" alt="Overall architecture of MQTrack" width="100%">
</p>

<p align="center"><em>Overall architecture of MQTrack.</em></p>




