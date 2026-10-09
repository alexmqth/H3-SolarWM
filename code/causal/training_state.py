"""Exact Stage1 continuation: bind Adam/RNG state to its adapter checkpoint."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch


def serialize_config(args):
    return {k: ([str(x) for x in v] if isinstance(v, list)
                else str(v) if isinstance(v, Path) else v)
            for k, v in vars(args).items()}


def load_training_state(directory, config):
    directory = Path(directory)
    file = directory / 'trainer_state.pt'
    if not file.exists():
        raise ValueError('Checkpoint lacks optimizer/RNG state; weights-only loading is not exact resume')
    state = torch.load(file, map_location='cpu', weights_only=True)
    if state.get('format') != 'h3world_stage1_training_state_v1':
        raise ValueError('Unsupported Stage1 training state')
    step = state['optimizer_step']
    if not isinstance(step, int) or step < 0 or config['steps'] <= step:
        raise ValueError('--steps is the target TOTAL update count and must exceed the resumed step')
    if [x['step'] for x in state['updates']] != list(range(1, step + 1)):
        raise ValueError('Checkpoint update history does not match its optimizer step')
    # Everything affecting data, noise, model, loss and Adam is fixed on an
    # exact continuation. A changed LR/objective is a separate experiment.
    mutable = {'out_dir', 'steps', 'checkpoint_every', 'resume_from', 'device'}
    old = state['config']
    config = dict(config)
    config.setdefault('precision_profile', 'legacy')
    compare_old = dict(old)
    compare_old.setdefault('precision_profile', 'legacy')
    for key,value in [('adapter_scope','tail_qkv'),('bank_rank',8),('bank_alpha',8.),('history_gradient_mode','detached'),
                      ('training_timestep_shift',None),('validation_timestep_shift',None),
                      ('training_weight_shift',None),('validation_weight_shift',None)]:
        config.setdefault(key,value);compare_old.setdefault(key,value)
    mismatches = [k for k in set(config) | set(old)
                  if k not in mutable and config.get(k) != compare_old.get(k)]
    if mismatches:
        raise ValueError(f'Resume protocol mismatch: {sorted(mismatches)}')
    required = {'causal_adapter.pt'}
    if config.get('adapter_scope') == 'all_qkvo_ffn':
        required.add('stage1_lora.pt')
    if config['objective'] == 'anyflow':
        required.add('anyflow_adapter.pt')
    if config.get('action_adapter'):
        required.add('action_adapter.pt')
    if set(state['weight_sha256']) != required:
        raise ValueError('Incomplete or unexpected adapter set in training state')
    for name, expected in state['weight_sha256'].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Checkpoint adapter hash mismatch: {name}')
        metadata = torch.load(directory / name, map_location='cpu', weights_only=True)['metadata']
        if metadata.get('optimizer_step') != step or metadata.get('config') != old:
            raise ValueError(f'Checkpoint metadata mismatch: {name}')
    for name in ('logical_rng_state', 'cpu_rng_state'):
        if not torch.is_tensor(state[name]) or state[name].dtype != torch.uint8:
            raise ValueError(f'Invalid RNG state: {name}')
    return state


def restore_random_state(state, generator, *, device):
    generator.set_state(state['logical_rng_state'])
    torch.set_rng_state(state['cpu_rng_state'])
    if torch.device(device).type == 'cuda':
        if state['cuda_rng_state'] is None:
            raise ValueError('CUDA resume requires the saved CUDA RNG state')
        torch.cuda.set_rng_state(state['cuda_rng_state'], device=device)
