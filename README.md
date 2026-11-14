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

Multi-class multi-object tracking (MCMOT) in UAV videos requires reliable identity decisions under severe scale variation, uncertain observations, and fragmented trajectories. These challenges affect tracking evidence differently across frame-level association, short-term recovery, and tracklet-level refinement, while errors introduced at earlier stages can influence subsequent decisions. To address these interrelated challenges, we propose MQTrack, a stage-aware tracking framework that coordinates scale-conditioned association, reliability-constrained recovery, and quality-guided tracklet refinement. Specifically, Scale-Aware Guided Embedding Association (SAGE) adapts geometric and appearance matching using a shared pair-specific scale. Category-Aware Spatio-Temporal Recovery (CAST) incorporates temporal appearance context and recent-track reliability to recover uncertain observations while preserving recoverysource information. Quality-Guided Tracklet Refinement (QTR) leverages accumulated tracklet quality to regulate offline linking, retention, and interpolation, with optional validation of recoverysupported segments. On VisDrone2019-MOT test-dev under the five-class protocol, MQTrack achieves 44.4% MOTA and 61.5% IDF1, outperforming its corresponding ablation baseline by 3.8 and 10.6 percentage points, respectively. It further achieves 54.6% MOTA and 72.3% IDF1 on UAVDT test, and 35.8% MOTA and 56.5% IDF1 on VT-MOT-UAV test using RGB observations only. Comprehensive ablation studies across three benchmarks demonstrate the complementary contributions of the proposed components.

## Framework

<p align="center">
  <img src="images/fig.png" alt="Overall architecture of MQTrack" width="100%">
</p>

<p align="center"><em>Overall architecture of MQTrack.</em></p>

## Environment

The validated environment uses Python 3.11, PyTorch 2.8.0, torchvision 0.23.0, and an NVIDIA RTX 5060. Install matching PyTorch and torchvision wheels for your CUDA installation, then install the remaining requirements.

```bash
pip install -r requirements.txt
python run.py --help
```

Deformable convolution uses `torchvision.ops.deform_conv2d`. A compatible torchvision build with compiled operators is required. The `cython-bbox` package may require a compiler when a wheel is unavailable. The runner uses deterministic FP32 inference, disables TF32, and fixes the random seed to 317.

## Checkpoints

Supply a trained DLA-34 detection/ReID checkpoint with `--checkpoint`. Model parameters remain fixed during tracking. VisDrone uses a ten-class detection head, UAVDT a one-class head, and RGB VT-MOT-UAV a two-class head for person and vehicle. Inference tensors must fully match the selected architecture. The reference checkpoint hashes are recorded in `configs/*.json` for identifying the corresponding original weights.

## Dataset layout

```text
VisDrone2019-MOT/
  VisDrone2019-MOT-test-dev/sequences/<sequence>/*.jpg

UAVDT-MOT/
  <sequence>/*.jpg

VT-MOT-UAV/
  data1/Datasets/Tracking/MOT/VTMOT/images/
    test/<sequence>/visible/*.jpg
    test/<sequence>/seqinfo.ini
```

For VT-MOT-UAV, `--data-root` also accepts the `images` directory directly. UAVDT images are read directly from each sequence directory, as in the original pipeline. Ground-truth annotations are not read by the inference pipeline.

## Run

From the repository directory, use a new output directory for each invocation.

```bash
python run.py --dataset visdrone --data-root /path/to/VisDrone2019-MOT --checkpoint /path/to/visdrone.pth --output outputs/visdrone
python run.py --dataset uavdt --data-root /path/to/UAVDT-MOT --checkpoint /path/to/uavdt.pth --output outputs/uavdt
python run.py --dataset vtmot --data-root /path/to/VT-MOT-UAV --checkpoint /path/to/vtmot_rgb.pth --output outputs/vtmot
```

Use `--sequences name1,name2` to select complete sequences. Use `--online-only` to obtain the single SAGE+CAST+GMC online stream. For UAVDT, the default complete pipeline runs the GMC-only reference stream first, selects profiles using prediction statistics, and executes additional online branches where required. It then runs QTR and ROI assembly. Selection does not inspect annotations.

The frozen input size is 1088 by 608. Decoding retains at most 500 candidates for VisDrone/UAVDT and 200 for VT-MOT-UAV. Online thresholds, SAGE/CAST profiles, offline parameters, and VisDrone score configuration files are included under `configs/`.

## Outputs

Online trajectories are written beneath `online/`, with source and ReID feature sidecars. Full-pipeline trajectories are written beneath `predictions/`. UAVDT also writes the selected profiles and a routing record.

VisDrone rows use `frame,id,x,y,w,h,score,class,-1,-1` with original 1-based VisDrone classes. UAVDT rows use the inherited single-class output convention. RGB VT-MOT-UAV rows use `frame,id,x,y,w,h,score,class,1,-1`, with class 1 for person and 2 for vehicle. Track identities and temporal state are reset between sequences.

## Source layout

```text
run.py                     Portable inference entry
configs/                   Frozen dataset and QTR profiles
mqtrack/runtime.py         Original native sequence loop and writer
mqtrack/rgb_runtime.py     Original RGB VT-MOT-UAV sequence loop
mqtrack/routing.py         UAVDT sequence selection and ROI processing
src/lib/models/            DLA-34, detection decoding, checkpoint loading
src/lib/tracker/           SAGE, CAST, GMC, and track management
src/qhit_postprocess.py    QTR refinement
src/qtr_evidence_backfill.py  Optional recovery-evidence support
```

`SOURCE_PROVENANCE.json` records hashes of unchanged core sources and describes the packaging adaptations. The model factory in this distribution exposes DLA, and the image loader retains the inference-specific routines. Parameters and association/refinement formulas are preserved.


