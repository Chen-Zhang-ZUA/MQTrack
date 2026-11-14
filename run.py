"""Run the released MQTrack pipeline with an external checkpoint and RGB videos."""
import argparse
import json
import logging
import os
from pathlib import Path
import random
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
import _init_paths
import cv2
import numpy as np
import torch
from lib.opts import opts
from lib.models.model import _extract_state_dict, _torch_load
from lib.tracker.multitracker import MCJDETracker
from lib.datasets.dataset.jde import LoadImages
from lib.tracking_utils.log import logger
import qhit_postprocess as qtr
from mqtrack import runtime, rgb_runtime, routing


def frame_files(folder):
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.tif'))
    if not files:
        raise ValueError(f'No RGB images found in {folder}')
    return files


def find_sequences(dataset, data_root, config, selected=None):
    if dataset == 'visdrone':
        base = data_root / 'VisDrone2019-MOT-test-dev' / 'sequences'
        if data_root.name == 'VisDrone2019-MOT-test-dev':
            base = data_root / 'sequences'
        if not base.is_dir():
            raise FileNotFoundError(base)
        names = sorted(p.name for p in base.iterdir() if p.is_dir())
    elif dataset == 'uavdt':
        base = data_root
        names = config['test_sequences']
    else:
        candidates = [data_root, data_root / 'data1/Datasets/Tracking/MOT/VTMOT/images']
        base = next((p for p in candidates if (p / 'test').is_dir()), None)
        if base is None:
            raise FileNotFoundError(f'VT-MOT-UAV images/test directory not found beneath {data_root}')
        base = base / 'test'
        names = sorted(p.name for p in base.iterdir() if p.is_dir())
    if selected:
        unknown = set(selected) - set(names)
        if unknown:
            raise ValueError(f'Unknown sequences: {sorted(unknown)}')
        names = selected
    sequences = []
    for name in names:
        folder = base / name
        if dataset == 'vtmot':
            folder = folder / 'visible'
        files = frame_files(folder)
        sequence = dict(name=name, files=files, folder=folder)
        if dataset == 'vtmot':
            first = cv2.imread(str(files[0]))
            if first is None:
                raise ValueError(f'Cannot decode {files[0]}')
            sequence.update(source=base.parent, image_dir=str(folder.relative_to(base.parent)),
                            size=(first.shape[1], first.shape[0]))
        sequences.append(sequence)
    if not sequences:
        raise ValueError('No sequences selected')
    return sequences


def expand(flags, **values):
    values['root'] = ROOT.as_posix()
    return [str(flag).format(**values) for flag in flags]


def validate_checkpoint(tracker, checkpoint):
    source = _extract_state_dict(_torch_load(checkpoint))
    source = {(k[7:] if k.startswith('module') and not k.startswith('module_list') else k): v
              for k, v in source.items() if isinstance(k, str) and hasattr(v, 'shape')}
    missing = [k for k, value in tracker.model.state_dict().items()
               if k not in source or not torch.equal(value.cpu(), source[k].cpu())]
    if missing:
        raise ValueError(f'Checkpoint does not fully match this detector/ReID architecture: {missing[:8]}')


def run_online(dataset, config, checkpoint, sequences, output, branch='mqtrack', device='cuda:0'):
    output.mkdir(parents=True, exist_ok=False)
    option = opts().init(expand(config['online_branches'][branch], checkpoint=checkpoint))
    option.device = torch.device(device)
    tracker = MCJDETracker(option)
    validate_checkpoint(tracker, checkpoint)
    if dataset != 'vtmot':
        del tracker
    for sequence in sequences:
        torch.manual_seed(option.seed)
        np.random.seed(option.seed)
        random.seed(option.seed)
        cv2.setRNGSeed(option.seed)
        destination = output / (sequence['name'] + '.txt')
        if dataset == 'vtmot':
            with torch.no_grad():
                rgb_runtime.run_sequence(sequence, [p.name for p in sequence['files']], sequence['source'], tracker, destination)
        else:
            # The original entry constructs a fresh tracker per complete sequence.
            with torch.no_grad():
                runtime.eval_seq(option, LoadImages([str(p) for p in sequence['files']], option.img_size),
                                 'mot', str(destination), show_image=False, frame_rate=30,
                                 cls_id_offset=1 if dataset == 'visdrone' else -1,
                                 visdrone_format=dataset == 'visdrone')
        print(f'Completed {branch}: {sequence["name"]} ({len(sequence["files"])} RGB frames)', flush=True)


def run_qtr(flags, source, destination):
    destination.mkdir(parents=True, exist_ok=False)
    old_args = sys.argv
    try:
        sys.argv = ['qhit_postprocess.py', *expand(flags, input=source, output=destination)]
        options = qtr.parse_args()
    finally:
        sys.argv = old_args
    for path in sorted(source.glob('*.txt')):
        qtr.process_file(path, destination / path.name, qtr.parse_delta_t(options.delta_t), options)


def run_uavdt(config, checkpoint, sequences, data_root, output, device):
    base = output / 'online' / 'qtr'
    run_online('uavdt', config, checkpoint, sequences, base, 'qtr', device)
    decisions, jobs = [], {}
    for path in sorted(base.glob('*.txt')):
        features = routing.sequence_features(path.stem, routing.read_rows(path))
        profile, reason = routing.choose_profile(features)
        source_profile = config['special_profile'].get(profile, profile)
        jobs.setdefault(source_profile, []).append(path.stem)
        decisions.append(dict(sequence=path.stem, profile=profile, reason=reason))
    required = {}
    for profile, names in jobs.items():
        branch = config['source_branch'][profile]
        if branch != 'qtr':
            required.setdefault(branch, set()).update(names)
    for branch, names in required.items():
        run_online('uavdt', config, checkpoint, [s for s in sequences if s['name'] in names],
                   output / 'online' / branch, branch, device)
    profiles = {}
    for profile, names in jobs.items():
        source = output / 'online' / config['source_branch'][profile]
        target = output / 'profiles' / profile
        # Process only the complete sequences used by this fixed profile.
        target.mkdir(parents=True, exist_ok=False)
        saved = sys.argv
        try:
            sys.argv = ['qhit_postprocess.py', *expand(config['qtr_profiles'][profile], input=source, output=target)]
            arguments = qtr.parse_args()
        finally:
            sys.argv = saved
        for name in names:
            qtr.process_file(source / (name + '.txt'), target / (name + '.txt'), qtr.parse_delta_t(arguments.delta_t), arguments)
        profiles[profile] = target
    for profile in config['qtr_profiles']:
        if profile not in profiles:
            profiles[profile] = output / 'profiles' / profile
            profiles[profile].mkdir(parents=True, exist_ok=False)
    routing.ROOT = routing.RESULT_ROOT = output
    routing.BASE_STAT_DIR = base
    routing.IMAGE_ROOT = data_root
    routing.PROFILE_DIRS = profiles
    routing.OUTPUT_NAME = 'predictions'
    assert not (output / routing.OUTPUT_NAME).exists()
    routing.build_output()
    (output / 'routing.json').write_text(json.dumps(decisions, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('visdrone', 'uavdt', 'vtmot'), required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='A new output directory')
    parser.add_argument('--sequences', help='Optional comma-separated sequence names')
    parser.add_argument('--device', default='auto', help='auto, cpu, or a CUDA device such as cuda:0')
    parser.add_argument('--online-only', action='store_true', help='Run the SAGE+CAST+GMC online stage only')
    args = parser.parse_args()
    data_root, checkpoint, output = args.data_root.resolve(), args.checkpoint.resolve(), args.output.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    config = json.loads((ROOT / 'configs' / (args.dataset + '.json')).read_text(encoding='utf-8'))
    seqs = find_sequences(args.dataset, data_root, config, args.sequences.split(',') if args.sequences else None)
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.manual_seed(317)
    np.random.seed(317)
    random.seed(317)
    cv2.setRNGSeed(317)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    logger.setLevel(logging.WARNING)
    output.mkdir(parents=True)
    if args.dataset == 'uavdt' and not args.online_only:
        run_uavdt(config, checkpoint, seqs, data_root, output, device)
    else:
        source = output / 'online' / 'mqtrack'
        run_online(args.dataset, config, checkpoint, seqs, source, device=device)
        if not args.online_only:
            run_qtr(config['qtr_profiles']['default'], source, output / 'predictions')
    print(f'Finished. Outputs are in {output}', flush=True)


if __name__ == '__main__':
    main()
