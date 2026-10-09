"""Optional Stage1 auxiliary against a frozen diagonal numerical reference.

This is an experimental addition to TF-AnyFlow, not the official loss and not
DMD. Reference trajectories are not GT. The normal finite-map sampler stays
unchanged. Clean history uses the trainer's existing full-gradient path.
"""
import hashlib
import json
from pathlib import Path

import torch


def tensor_sha(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(str((tuple(value.shape), str(value.dtype))).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_reference(case, chunk, directory):
    directory = Path(directory)
    report_path = directory / f'probe_{case["label"]}.json'
    report = json.loads(report_path.read_text())
    if (report['status'] != 'complete' or not report['parameter_versions_unchanged']
            or len(report['cache_checks']) != 3
            or not all(x['unchanged'] for x in report['cache_checks'])):
        raise ValueError('A completed, immutable-reference audit is required')
    for key, actual in [('clean_sha256', tensor_sha(case['clean'])),
                        ('prompt_sha256', tensor_sha(case['prompt'])),
                        ('audio_sha256', tensor_sha(case['audio']))]:
        if report[key] != actual:
            raise ValueError(f'Current clean conditioning differs from reference: {key}')
    if report['anchor_sha256'] != [tensor_sha(x) for x in case['anchors']]:
        raise ValueError('Reference anchors changed')
    action_hash = tensor_sha(case['actions']) if case['actions'] is not None else None
    if report['action_sha256'] != action_hash:
        raise ValueError('Reference actions changed')
    matches = [x for x in report['records'] if x['chunk'] == chunk]
    if len(matches) != 1:
        raise ValueError('Expected exactly one interval reference per chunk')
    record = matches[0]
    if record['subdivisions'] != [8, 16] or record['target_sigma'] != 0.:
        raise ValueError('Expected the reviewed 8/16 endpoint reference')
    path = Path(record['target_file'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != record['target_sha256']:
        raise ValueError('Frozen reference tensor file changed')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    tensors = payload['tensors']
    if payload['action'] != case['label'] or payload['checkpoint'] != report['checkpoint']:
        raise ValueError('Reference provenance does not match the report')
    if tensor_sha(tensors['initial']) != record['initial_sha256']:
        raise ValueError('Reference initial-state tensor differs from the measured state')
    if tensors['initial'].shape != tensors['reference_velocity'].shape:
        raise ValueError('Reference shapes differ')
    if not all(torch.isfinite(tensors[key]).all() for key in ('initial', 'reference_velocity')):
        raise FloatingPointError('Nonfinite reference')
    return tensors, record, hashlib.sha256(report_path.read_bytes()).hexdigest()


def interval_consistency_loss(velocity, case, chunk, args):
    tensors, record, report_sha = load_reference(case, chunk, args.interval_reference_dir)
    clean = case['clean'][:, :, chunk*args.chunk_frames:(chunk+1)*args.chunk_frames]
    initial = tensors['initial'].to(clean)
    target = tensors['reference_velocity'].to(clean).detach()
    if initial.shape != clean.shape:
        raise ValueError('Reference does not correspond to the current chunk')
    prediction = velocity(initial, record['sigma'], record['target_sigma']).float()
    raw = (prediction-target.float()).square().mean()
    if not torch.isfinite(raw):
        raise FloatingPointError('Nonfinite interval consistency loss')
    stats = dict(raw_loss=float(raw.detach()), sigma=record['sigma'],
        target_sigma=record['target_sigma'], reference_report_sha256=report_sha,
        reference_file_sha256=record['target_sha256'],
        initial_sha256=record['initial_sha256'],
        reference_refinement_rmse=record['diagonal_refinement_endpoint_rmse'],
        reference_is_ground_truth=False, model_evaluations=1)
    return raw, stats
